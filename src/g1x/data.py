"""Data for the G1-pool / G1-cross pilot (DINOv3-B/16 @ 384, 24 x 24 grid). Letterbox / crop canvases = src.viz.c0_pca.LB
(= data.preprocessing Letterbox rule at any S, P). NO feature interpolation: every ROI quantity below is geometry only.

Per ROI (both datasets):
  crop      C0 teacher input (original-pixel GT crop -> 384 letterbox)            [COCO only]
  ccov      (576,) C0 crop-grid content coverage (teacher valid mask / weights)
  qrel      (576, 2) crop-grid cell centre relative to the crop content rect (decoder query coordinates)
  box_c     (4,) ROI box on the FULL canvas, px [X0, Y0, X1, Y1]
  wfull     (576,) area of (box on full canvas) ∩ (full cell), in cell units (= R2 full-canvas cov, box_cell_weights rule)
  roi_img   ROI -> image index;  per image: full (3, S, S) canvas, fvalid (576,) full-canvas content cells

CocoDS: existing 12k COCO stream rows (cache/feat/stream_rois.parquet, 16 train2017 images x 4 ROIs, no flip) and the
COCO dev ROIs (cache/feat/dev_rois.parquet). PACO attribute labels are never touched.
PacoDS: PACO rows (same manifest / groups / flip rule as src.ivc.bbdata.BBImageDS) for the R2 probe
  mode 'full': R2 reads the full-image tokens with the R2 full-canvas geometry (common_rr.roi_geo convention: rel = cell
               centre relative to the box on the canvas, cov = box ∩ cell area, log(bw/P), log(bh/P); box4 = box / S)
  mode 'dec' : R2 reads the decoder output on the C0 crop grid with the C0 crop-grid geometry (= bbdata.crop_geo, the
               same R2 geometry as the C0 oracle probe)
"""
import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.readout.common import C
from src.ivc.bbdata import crop_geo
from src.viz.c0_pca import LB

G = 24


def roi_geom(W, H, ltrb, S, P):
    """-> dict of numpy arrays for one ROI (see module docstring) + the crop / full LB objects."""
    l, t, r, b = ltrb
    crop, full = LB(r - l, b - t, S, P), LB(W, H, S, P)
    ci, cj = np.divmod(np.arange(G * G), G)
    qrel = np.stack([((cj + 0.5) * P - crop.ox) / crop.nw, ((ci + 0.5) * P - crop.oy) / crop.nh], 1).astype(np.float32)
    X0, X1 = full.ox + l * full.nw / W, full.ox + r * full.nw / W
    Y0, Y1 = full.oy + t * full.nh / H, full.oy + b * full.nh / H
    e = np.arange(G + 1) * P
    ox = np.clip(np.minimum(e[1:], X1) - np.maximum(e[:-1], X0), 0, None) / P
    oy = np.clip(np.minimum(e[1:], Y1) - np.maximum(e[:-1], Y0), 0, None) / P
    wfull = (oy[:, None] * ox[None, :]).reshape(-1).astype(np.float32)
    return dict(crop_lb=crop, full_lb=full, ccov=crop.coverage().astype(np.float32), qrel=qrel,
                box_c=np.array([X0, Y0, X1, Y1], np.float64), wfull=wfull)


def full_r2_geo(g, S, P):
    """R2 full-canvas geometry (common_rr.roi_geo rule at canvas S, patch P) from roi_geom output."""
    X0, Y0, X1, Y1 = (float(v) for v in g["box_c"])
    bw, bh = X1 - X0, Y1 - Y0
    ci, cj = np.divmod(np.arange(G * G), G)
    cx, cy = (cj + 0.5) * P, (ci + 0.5) * P
    geo = np.stack([(cx - X0) / bw, (cy - Y0) / bh, g["wfull"], np.full(G * G, math.log(bw / P)), np.full(G * G, math.log(bh / P))], 1)
    return geo.astype(np.float32), np.array([X0 / S, Y0 / S, bw / S, bh / S], np.float32), g["wfull"]


class CocoDS(Dataset):
    def __init__(self, rois, image_dir, S=384, P=16, want_crops=True):
        self.by = {int(i): g.sort_values("k") for i, g in rois.groupby("image_id")}
        self.dir, self.S, self.P, self.want_crops = image_dir, S, P, want_crops

    def __len__(self):
        return len(self.by)

    def __getitem__(self, item):
        ids, n_glob = item if isinstance(item, tuple) else (item, None)
        full, fvalid, crops, roi_img, per = [], [], [], [], {k: [] for k in ("ccov", "qrel", "box_c", "wfull")}
        ltrb = []
        for k, iid in enumerate(ids):
            g = self.by[int(iid)]
            img = Image.open(os.path.join(self.dir, f"{int(iid):012d}.jpg")).convert("RGB")
            W, H = img.size
            assert (W, H) == (int(g.iloc[0].W), int(g.iloc[0].H)), iid
            flb = LB(W, H, self.S, self.P)
            full.append(flb.canvas(img))
            fvalid.append(flb.coverage() > 0)
            for r in g.itertuples():
                bx = (int(r.l), int(r.t), int(r.r), int(r.b))
                gg = roi_geom(W, H, bx, self.S, self.P)
                if self.want_crops:
                    crops.append(gg["crop_lb"].canvas(img.crop(bx)))
                for kk in per:
                    per[kk].append(gg[kk].astype(np.float32))
                roi_img.append(k)
                ltrb.append(bx)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items()}
        out.update(full=torch.stack(full), fvalid=torch.from_numpy(np.stack(fvalid)), roi_img=torch.as_tensor(roi_img),
                   ltrb=torch.as_tensor(ltrb), image_ids=torch.as_tensor([int(i) for i in ids]))
        if self.want_crops:
            out["crops"] = torch.stack(crops)
        if n_glob is not None:
            out["n_img_global"] = torch.tensor(float(n_glob))
        return out


class PacoDS(Dataset):
    """PACO probe input for modes 'full' / 'dec' (item = list of image-group indices, = BBImageDS item rule)."""

    def __init__(self, df, Y, flip, mode, S=384, P=16):
        assert mode in ("full", "dec")
        self.groups = C.image_groups(df)
        self.p, self.box = df.path.values, df[["l", "t", "r", "b"]].values.astype(int)
        self.WH = df[["W", "H"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.Y, self.flip, self.mode, self.S, self.P = torch.as_tensor(Y), flip, mode, S, P
        self.n_roi = np.array([len(g) for g in self.groups])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        xs, fvalid, img_index, rows_all = [], [], [], []
        per = {k: [] for k in ("cov", "geo", "box4", "qrel", "box_c")}
        for k, gi in enumerate(idx):
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[rows[0]]), self.p[rows[0]]
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            if f:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flb = LB(W, H, self.S, self.P)
            xs.append(flb.canvas(img))
            fvalid.append(flb.coverage() > 0)
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                g = roi_geom(W, H, bx, self.S, self.P)
                if self.mode == "full":
                    geo, box4, cov = full_r2_geo(g, self.S, self.P)
                else:
                    geo, box4, cov = crop_geo(g["crop_lb"], self.S, self.P)
                for kk, v in (("geo", geo), ("box4", box4), ("cov", cov), ("qrel", g["qrel"]), ("box_c", g["box_c"].astype(np.float32))):
                    per[kk].append(v)
                img_index.append(k)
            rows_all.append(rows)
        rows_all = np.concatenate(rows_all)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items()}
        out.update(side=self.S, x=torch.stack(xs), fvalid=torch.from_numpy(np.stack(fvalid)),
                   img_index=torch.as_tensor(img_index, dtype=torch.long), rows=torch.from_numpy(rows_all),
                   lv=torch.from_numpy(self.lv[rows_all]), Y=self.Y[rows_all],
                   group=torch.as_tensor(np.concatenate([np.full(len(self.groups[gi]), k) for k, gi in enumerate(idx)]), dtype=torch.long))
        return out


def check(n_img=40):
    """(1) at 336/14, full_r2_geo == common_rr.roi_geo (the R2 full-canvas convention of the earlier Full probes);
    (2) mode 'dec' R2 geometry == BBImageDS 'c0' geometry (C0 oracle probe) at 384/16; (3) decoder query coordinates qrel == the C0 R2 rel coordinates."""
    import src.readout.common as RC
    from src.ivc.bbdata import BBImageDS
    RR = RC.RR
    dtr, Ytr = RC.load_data(splits=("train",))["train"]
    groups = C.image_groups(dtr)
    n = 0
    for gi in range(n_img):
        for r in groups[gi]:
            W, H = (int(v) for v in dtr[["W", "H"]].values[r])
            bx = tuple(int(v) for v in dtr[["l", "t", "r", "b"]].values[r])
            g = roi_geom(W, H, bx, 336, 14)
            geo, box4, cov = full_r2_geo(g, 336, 14)
            from data.preprocessing import Letterbox  # noqa: E402  (on sys.path via src.readout.common)
            geo0, box40, cov0 = RR.roi_geo(Letterbox(W, H), bx)
            assert np.allclose(geo, geo0, rtol=1e-6, atol=1e-5) and np.abs(box4 - box40).max() < 1e-6 and np.abs(cov - cov0).max() < 1e-6, (gi, r)
            n += 1
    idx = list(range(n_img))
    a, b = PacoDS(dtr, Ytr, False, "dec")[idx], BBImageDS(dtr, Ytr, False, "c0", 384, 16)[idx]
    for k in ("geo", "box4", "cov", "rows", "lv", "Y", "group"):
        assert torch.equal(a[k], b[k]), k
    assert torch.allclose(a["qrel"], a["geo"][..., :2], atol=1e-5)          # query (i, j) <-> C0 crop cell (i, j)
    print(f"g1x data check OK: {n} ROIs full_r2_geo == roi_geo @336/14; dec geometry == BBImageDS c0 @384/16; qrel == C0 rel")


if __name__ == "__main__":
    check()
