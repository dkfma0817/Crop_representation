"""VAW inputs for the frozen-encoder R2 probe (DINOv3-B/16 @ 384 letterbox, 24 x 24 grid). NO feature interpolation.

Box handling (all in ORIGINAL VG pixel coordinates; VAW instance_bbox = [x, y, w, h], float):
  gt box   = [x, y, x + w, y + h] clipped to the image [0, W] x [0, H]
  canvas source rect (the pixels that are letterboxed to 384):
    full   : whole image (0, 0, W, H)
    tight  : C0 tight crop = gt box, integer pixel rect floor(x0), floor(y0), ceil(x1), ceil(y1)
    expand : C0 expanded crop (VAW-paper-text based re-implementation, NOT the official code): e = 0.15 * min(w, h)
             added to EACH side (= +0.3 * min(w, h) to width and height), clipped to the image, then integer rect
  R2 geometry (identical rule in every mode = common_rr.roi_geo / src.g1x.data.full_r2_geo convention): the ORIGINAL gt
    box is mapped exactly into canvas coordinates (X = ox + (x - sx0) * nw / sw), then per cell: rel = cell centre
    relative to the box, cov = area(box ∩ cell) / P^2, log(bw / P), log(bh / P); box4 = box / S; R2 mask = cov > 0.
    -> full / expand: R2 reads only the cells of the original object region; tight: the object fills the content
       rect, so cov == the letterbox content coverage (= the C0 convention of src.ivc.bbdata.crop_geo).
Horizontal flip (train only, p = 0.5, = PACO probe protocol; VAW has no left/right attributes): the whole image is
flipped before any crop, box x -> W - x.
"""
import math

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.viz.c0_pca import LB

G = 24
CI, CJ = np.divmod(np.arange(G * G), G)


def gt_box(bb, W, H):
    x, y, w, h = (float(v) for v in bb)
    x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
    assert x1 - x0 > 0 and y1 - y0 > 0, (bb, W, H)
    return x0, y0, x1, y1


def source_rect(mode, box, W, H):
    x0, y0, x1, y1 = box
    if mode == "full":
        return 0, 0, W, H
    if mode == "expand":
        e = 0.15 * min(x1 - x0, y1 - y0)
        x0, y0, x1, y1 = max(0.0, x0 - e), max(0.0, y0 - e), min(float(W), x1 + e), min(float(H), y1 + e)
    else:
        assert mode == "tight", mode
    l, t, r, b = int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))
    return max(0, l), max(0, t), min(W, max(r, l + 1)), min(H, max(b, t + 1))


def r2_geo(box, rect, S=384, P=16):
    """-> geo (576,5), box4 (4,), cov (576,), lb.  box / rect in original pixels."""
    sx0, sy0, sx1, sy1 = rect
    lb = LB(sx1 - sx0, sy1 - sy0, S, P)
    kx, ky = lb.nw / (sx1 - sx0), lb.nh / (sy1 - sy0)
    X0, X1 = lb.ox + (box[0] - sx0) * kx, lb.ox + (box[2] - sx0) * kx
    Y0, Y1 = lb.oy + (box[1] - sy0) * ky, lb.oy + (box[3] - sy0) * ky
    e = np.arange(G + 1) * P
    ox = np.clip(np.minimum(e[1:], X1) - np.maximum(e[:-1], X0), 0, None) / P
    oy = np.clip(np.minimum(e[1:], Y1) - np.maximum(e[:-1], Y0), 0, None) / P
    cov = (oy[:, None] * ox[None, :]).reshape(-1)
    bw, bh = X1 - X0, Y1 - Y0
    cx, cy = (CJ + 0.5) * P, (CI + 0.5) * P
    geo = np.stack([(cx - X0) / bw, (cy - Y0) / bh, cov, np.full(G * G, math.log(bw / P)), np.full(G * G, math.log(bh / P))], 1)
    return geo.astype(np.float32), np.array([X0 / S, Y0 / S, bw / S, bh / S], np.float32), cov.astype(np.float32), lb


def image_groups(img_ids):
    """rows grouped by image, first-appearance order (= C.image_groups rule)."""
    order, by = [], {}
    for r, i in enumerate(img_ids):
        if i not in by:
            by[i] = []
            order.append(i)
        by[i].append(r)
    return [np.array(by[i]) for i in order]


class VawDS(Dataset):
    """item = list of image-group indices (one rank's share of a step) -> all ROIs of those images."""

    def __init__(self, inst, Y, paths, WH, flip, mode, S=384, P=16):
        assert mode in ("full", "tight", "expand")
        self.img = np.array([int(x["image_id"]) for x in inst])
        self.bb = np.array([x["instance_bbox"] for x in inst], np.float64)
        self.groups = image_groups(self.img.tolist())
        self.paths, self.WH = paths, WH            # dicts keyed by VG image id
        self.Y, self.flip, self.mode, self.S, self.P = torch.as_tensor(Y), flip, mode, S, P
        self.n_roi = np.array([len(g) for g in self.groups])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        xs, img_index, rows_all = [], [], []
        per = {k: [] for k in ("cov", "geo", "box4")}
        for k, gi in enumerate(idx):
            rows = self.groups[gi]
            iid = int(self.img[rows[0]])
            img = Image.open(self.paths[iid]).convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[iid]), (iid, (W, H), self.WH[iid])
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            if f:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.mode == "full":
                xs.append(LB(W, H, self.S, self.P).canvas(img))
            for r in rows:
                x0, y0, x1, y1 = gt_box(self.bb[r], W, H)
                if f:
                    x0, x1 = W - x1, W - x0
                box = (x0, y0, x1, y1)
                rect = source_rect(self.mode, box, W, H)
                geo, box4, cov, lb = r2_geo(box, rect, self.S, self.P)
                if self.mode != "full":
                    xs.append(lb.canvas(img.crop(rect)))
                for kk, v in (("geo", geo), ("box4", box4), ("cov", cov)):
                    per[kk].append(v)
                img_index.append(len(xs) - 1)
            rows_all.append(rows)
        rows_all = np.concatenate(rows_all)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items()}
        out.update(side=self.S, x=torch.stack(xs), img_index=torch.as_tensor(img_index, dtype=torch.long),
                   rows=torch.from_numpy(rows_all), lv=torch.zeros(len(rows_all), dtype=torch.long), Y=self.Y[rows_all])
        return out


def check(n=300):
    """(1) full-mode geometry == src.g1x.data.full_r2_geo(roi_geom) for integer boxes; (2) tight-mode cov == C0 letterbox
    content coverage and geo == src.ivc.bbdata.crop_geo for integer boxes; (3) expanded rect contains the box and the
    mapped box lies inside the content rect."""
    from src.g1x.data import full_r2_geo, roi_geom
    from src.ivc.bbdata import crop_geo
    rng = np.random.default_rng(0)
    for _ in range(n):
        W, H = int(rng.integers(200, 1281)), int(rng.integers(200, 1281))
        l, t = int(rng.integers(0, W - 5)), int(rng.integers(0, H - 5))
        r, b = int(rng.integers(l + 2, W + 1)), int(rng.integers(t + 2, H + 1))
        g = roi_geom(W, H, (l, t, r, b), 384, 16)
        geo0, box40, cov0 = full_r2_geo(g, 384, 16)
        geo, box4, cov, _ = r2_geo((l, t, r, b), (0, 0, W, H))
        assert np.allclose(geo, geo0, atol=1e-5) and np.allclose(box4, box40, atol=1e-6) and np.allclose(cov, cov0, atol=1e-6)
        geo1, box41, cov1 = crop_geo(g["crop_lb"], 384, 16)
        geo, box4, cov, lb = r2_geo((l, t, r, b), source_rect("tight", (l, t, r, b), W, H))
        assert np.allclose(cov, lb.coverage(), atol=1e-6) and np.allclose(geo, geo1, atol=1e-5) and np.allclose(box4, box41, atol=1e-6)
        bx = (l + rng.random(), t + rng.random(), r - rng.random(), b - rng.random())
        rect = source_rect("expand", bx, W, H)
        assert rect[0] <= bx[0] and rect[1] <= bx[1] and rect[2] >= bx[2] and rect[3] >= bx[3]
        geo, box4, cov, lb = r2_geo(bx, rect)
        X0, Y0 = box4[0] * 384, box4[1] * 384
        assert X0 >= lb.ox - 1e-3 and Y0 >= lb.oy - 1e-3 and X0 + box4[2] * 384 <= lb.ox + lb.nw + 1e-3
        assert abs(cov.sum() - box4[2] * box4[3] * 384 * 384 / 256) < 1e-2
    print(f"vaw data geometry check OK ({n} random boxes): full == full_r2_geo, tight == C0 crop_geo, expand contains box")


if __name__ == "__main__":
    check()
