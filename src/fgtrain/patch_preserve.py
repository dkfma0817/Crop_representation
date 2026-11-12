"""P2/P3 only: exact rectangle-union geometry and fixed-reference patch loss.

Nothing in this module is installed in the encoder or used at inference.
"""
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

from src.fgtrain.data import Data
from src.g1x.data import roi_geom
from src.viz.c0_pca import LB


def union_coverage(boxes, content, size=384, patch=16):
    """Exact per-cell rectangle union, by inclusion/exclusion (four ROIs/image).

    Float64 geometry, continuous canvas coordinates, valid-content intersection.
    Repeated/overlapping rectangles count once. Not sum/max of ROI coverages.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    assert len(boxes) <= 8, 'This pilot expects four existing B0 ROIs/image'
    c = np.asarray(content, dtype=np.float64)
    assert np.isfinite(boxes).all() and np.isfinite(c).all()
    assert ((boxes[:, 2:] - boxes[:, :2]) >= 0).all()
    g = size // patch
    yy, xx = np.divmod(np.arange(g*g), g)
    lo = np.maximum(np.stack([xx, yy], -1)*patch, c[:2])
    hi = np.minimum(np.stack([xx+1, yy+1], -1)*patch, c[2:])
    valid_area = np.maximum(hi-lo, 0).prod(-1)
    area = np.zeros(g*g, dtype=np.float64)
    for n in range(1, len(boxes)+1):
        for subset in combinations(range(len(boxes)), n):
            chosen = boxes[list(subset)]
            a = np.maximum(lo, chosen[:, :2].max(0))
            b = np.minimum(hi, chosen[:, 2:].min(0))
            area += (1 if n % 2 else -1)*np.maximum(b-a, 0).prod(-1)
    assert (area >= -1e-7).all() and (area <= valid_area+1e-7).all()
    area = np.clip(area, 0, valid_area)
    m = np.divide(area, valid_area, out=np.zeros_like(area), where=valid_area > 0)
    return (valid_area/(patch*patch)).astype(np.float32), m.astype(np.float32)


def weights(coverage, union, selective):
    v, m = coverage.detach().float(), union.detach().float()
    assert v.shape == m.shape and torch.isfinite(v).all() and torch.isfinite(m).all()
    assert ((v >= 0) & (v <= 1)).all() and ((m >= 0) & (m <= 1)).all()
    raw = 1-.75*m if selective else torch.ones_like(v)
    den = v.sum(-1, keepdim=True)
    mean = (v*raw).sum(-1, keepdim=True)/den.clamp_min(1e-12)
    return torch.where(v > 0, raw/mean.clamp_min(1e-12), 0.)


def patch_loss(student, teacher, coverage, union, selective):
    assert student.shape == teacher.shape and coverage.shape == student.shape[:2]
    with torch.autocast(device_type=student.device.type, enabled=False):
        v = coverage.detach().float()
        valid = v > 0
        s = F.normalize(torch.where(valid[..., None], student.float(), 0.), dim=-1)
        t = F.normalize(torch.where(valid[..., None], teacher.detach().float(), 0.), dim=-1)
        distance = 1-(s*t).sum(-1)
        w = weights(v, union, selective)
        den = v.sum(-1)
        per = (v*w*distance).sum(-1)/den.clamp_min(1e-12)
        images = den > 0
        loss = per[images].mean() if bool(images.any()) else s.sum()*0.
    return loss, per, distance


def drift_stats(distance, coverage, union):
    d, v, m = [x.detach().float() for x in (distance, coverage, union)]
    def group(a):
        den = a.sum(-1)
        val = (a*d).sum(-1)/den.clamp_min(1e-12)
        return float(val[den > 0].mean()) if bool((den > 0).any()) else None
    return dict(patch_distance=group(v), roi_inside_distance=group(v*m),
                roi_outside_distance=group(v*(1-m)),
                roi_union_fraction=float((v*m).sum()/v.sum().clamp_min(1e-12)))


def weight_stats(v, m, selective=True):
    w = weights(v, m, selective)
    den = v.sum(-1).clamp_min(1e-12)
    mean = (v*w).sum(-1)/den
    rows = []
    for a, b, c, avg in zip(v, m, w, mean):
        valid = a > 0
        def group(mask):
            z = a*mask
            return float((z*c).sum()/z.sum()) if float(z.sum()) > 0 else None
        rows.append(dict(union_fraction=float((a*b).sum()/a.sum()),
                         min=float(c[valid].min()), mean=float(avg), max=float(c[valid].max()),
                         std=float(((a*(c-avg)**2).sum()/a.sum()).sqrt()),
                         inside_weight=group(b), outside_weight=group(1-b),
                         boundary_cells=int(((b > 0) & (b < 1) & valid).sum()),
                         p2_p3_mean_absolute_difference=float((a*(c-1).abs()).sum()/a.sum())))
    return rows


class PatchData(Data):
    """Same B0 tensors/cache/ROI stream; append deterministic training-only masks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_cache = {}

    def geometry(self, iid):
        if iid not in self.geometry_cache:
            rows = self.by[iid]
            r = rows.iloc[0]
            lb = LB(int(r.W), int(r.H), 384, 16)
            boxes = np.stack([roi_geom(int(q.W), int(q.H), (q.l,q.t,q.r,q.b), 384, 16)['box_c'] for q in rows.itertuples()])
            content = [lb.ox, lb.oy, lb.ox+lb.nw, lb.oy+lb.nh]
            v, m = union_coverage(boxes, content)
            assert np.array_equal(v, lb.coverage())
            self.geometry_cache[iid] = (v, m, boxes)
        return self.geometry_cache[iid]

    def __getitem__(self, item):
        b = super().__getitem__(item)
        areas, unions = [], []
        for i, iid in enumerate(b['image_ids'].tolist()):
            v, m, boxes = self.geometry(int(iid))
            # Use float64 version of the same coordinates before B0 float32 cast.
            assert np.array_equal(boxes.astype(np.float32), b['box_c'][b['roi_img'] == i].numpy())
            areas.append(v); unions.append(m)
        b['patch_area'] = torch.from_numpy(np.stack(areas))
        b['roi_union'] = torch.from_numpy(np.stack(unions))
        assert torch.equal(b['patch_area'] > 0, b['fvalid'])
        return b


def gradient_trace(crop, preserve, enc, weight):
    # The existing per-block norm implementation is shared, not changed.
    from src.fgtrain.gram import gradient_trace as trace
    r = trace(crop, preserve, enc, weight)
    return dict(crop=r['crop'], preserve=r['gram'],
                weighted_preservation_norm=r['weighted_gram_norm'],
                preservation_crop_ratio=r['gram_crop_ratio'], crop_preservation_cosine=r['crop_gram_cosine'])
