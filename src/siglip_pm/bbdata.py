"""Established BBImageDS ROI/flip/letterbox protocol with SigLIP2 normalization; geometry unchanged."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.readout.common import C
from src.siglip_pm.data import geometry

G = 24


def crop_geo(cl, S, P):
    X0, Y0, bw, bh = cl.ox, cl.oy, cl.nw, cl.nh
    ci, cj = np.divmod(np.arange(G * G), G)
    cx, cy = (cj + 0.5) * P, (ci + 0.5) * P
    cov = cl.coverage()
    geo = np.stack([(cx - X0) / bw, (cy - Y0) / bh, cov, np.full(G * G, math.log(bw / P)), np.full(G * G, math.log(bh / P))], 1)
    return geo.astype(np.float32), np.array([X0 / S, Y0 / S, bw / S, bh / S], np.float32), cov


def full_geo(full, W, H, ltrb, S, P):
    """Context geometry of every full-canvas cell for one ROI (R2 rule on the full canvas): rel = cell centre relative
    to the box mapped to the full canvas, cov = cell / box overlap fraction, log(bw/P), log(bh/P)."""
    l, t, r, b = ltrb
    X0, X1 = full.ox + l * full.nw / W, full.ox + r * full.nw / W
    Y0, Y1 = full.oy + t * full.nh / H, full.oy + b * full.nh / H
    bw, bh = max(X1 - X0, 1e-3), max(Y1 - Y0, 1e-3)
    e = np.arange(G + 1) * P
    cx = np.clip(np.minimum(e[1:], X1) - np.maximum(e[:-1], X0), 0, None) / P
    cy = np.clip(np.minimum(e[1:], Y1) - np.maximum(e[:-1], Y0), 0, None) / P
    ci, cj = np.divmod(np.arange(G * G), G)
    geo = np.stack([((cj + 0.5) * P - X0) / bw, ((ci + 0.5) * P - Y0) / bh, (cy[:, None] * cx[None, :]).reshape(-1),
                    np.full(G * G, math.log(bw / P)), np.full(G * G, math.log(bh / P))], 1)
    return geo.astype(np.float32)


class BBImageDS(Dataset):
    """mode 'c0ctx' (encoder FT, src.ivc.ft / src.ivc.g1): the 'c0' crops + one full canvas per image: full, roi_img
    (ROI -> image), bil_grid (R, 576, 2), fgeo (R, 576, 5) context geometry, fvalid (B_img, 576) full-canvas content cells.
    mode 'fullroi' (G1 probe with the predictor): the same keys without the crops (x = full canvases, img_index = roi_img)."""

    def __init__(self, df, Y, flip, mode, S, P):
        assert mode in ("c0", "bil", "c0ctx", "fullroi")
        self.groups = C.image_groups(df)
        self.p, self.box = df.path.values, df[["l", "t", "r", "b"]].values.astype(int)
        self.WH = df[["W", "H"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.Y, self.flip, self.mode, self.S, self.P = torch.as_tensor(Y), flip, mode, S, P
        self.n_roi = np.array([len(g) for g in self.groups])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        xs, per, img_index, rows_all, grids = [], {k: [] for k in ("cov", "geo", "box4")}, [], [], []
        fulls, roi_img, fgeo, fvalid = [], [], [], []
        for k, gi in enumerate(idx):
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[rows[0]]), self.p[rows[0]]
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            if f:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            full_added = False
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                crop, full, grid = geometry(W, H, bx, self.S, self.P)
                if self.mode == "fullroi":
                    if not full_added:
                        fulls.append(full.canvas(img))
                        fvalid.append(full.coverage() > 0)
                        full_added = True
                    roi_img.append(len(fulls) - 1)
                    img_index.append(len(fulls) - 1)
                    fgeo.append(full_geo(full, W, H, bx, self.S, self.P))
                    grids.append(grid)
                elif self.mode in ("c0", "c0ctx"):
                    xs.append(crop.canvas(img.crop(bx)))
                    img_index.append(len(xs) - 1)
                    if self.mode == "c0ctx":
                        grids.append(grid)
                        if not full_added:
                            fulls.append(full.canvas(img))
                            fvalid.append(full.coverage() > 0)
                            full_added = True
                        roi_img.append(len(fulls) - 1)
                        fgeo.append(full_geo(full, W, H, bx, self.S, self.P))
                else:
                    if not full_added:
                        xs.append(full.canvas(img))
                        full_added = True
                    img_index.append(len(xs) - 1)
                    grids.append(grid)
                geo, box4, cov = crop_geo(crop, self.S, self.P)
                per["geo"].append(geo)
                per["box4"].append(box4)
                per["cov"].append(cov)
            rows_all.append(rows)
        rows_all = np.concatenate(rows_all)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items()}
        if self.mode == "fullroi":
            xs = fulls
        out.update(side=self.S, x=torch.stack(xs), img_index=torch.as_tensor(img_index, dtype=torch.long),
                   rows=torch.from_numpy(rows_all), lv=torch.from_numpy(self.lv[rows_all]), Y=self.Y[rows_all],
                   group=torch.as_tensor(np.concatenate([np.full(len(self.groups[gi]), k) for k, gi in enumerate(idx)]), dtype=torch.long))
        if self.mode == "bil":
            out["bil_grid"] = torch.from_numpy(np.stack(grids))
        if self.mode in ("c0ctx", "fullroi"):
            out["bil_grid"] = torch.from_numpy(np.stack(grids))
            out.update(full=torch.stack(fulls), roi_img=torch.as_tensor(roi_img, dtype=torch.long),
                       fgeo=torch.from_numpy(np.stack(fgeo)), fvalid=torch.from_numpy(np.stack(fvalid)))
        return out


def bil_sample(tok, b):
    """tok (B_img, 576, D) full-image tokens -> (R, 576, D) on the crop grid (= src.ivc.common.sample_roi 'bilinear')."""
    maps = tok.transpose(1, 2).reshape(tok.shape[0], -1, G, G)[b["img_index"]]
    out = F.grid_sample(maps.float(), b["bil_grid"].view(-1, G, G, 2), mode="bilinear", padding_mode="border", align_corners=False)
    return out.flatten(2).transpose(1, 2)


def check(n_img=60):
    """At 336/14 the batches must equal the DINOv2 IVC datasets (no flip): c0 vs RC.PacoImageDS('c0'), bil grid/geo vs
    MSDImageDS (geo_c / box4_c / cov_c / geom bil_grid / full canvas)."""
    import src.msd.common as MS
    import src.readout.common as RC
    dtr, Ytr = RC.load_data(splits=("train",))["train"]
    idx = list(range(n_img))
    a, b = BBImageDS(dtr, Ytr, False, "c0", 336, 14)[idx], RC.PacoImageDS(dtr, Ytr, flip=False, mode="c0")[idx]
    for k in ("x", "geo", "box4", "cov", "img_index", "rows", "lv", "Y", "group"):
        assert torch.equal(a[k], b[k]), f"c0 {k}"
    a, m = BBImageDS(dtr, Ytr, False, "bil", 336, 14)[idx], MS.MSDImageDS(dtr, Ytr, flip=False, want_crops=False)[idx]
    assert torch.equal(a["x"], m["full"]), "bil full canvas"
    assert torch.equal(a["geo"], m["geo_c"]) and torch.equal(a["box4"], m["box4_c"]) and torch.equal(a["cov"], m["cov_c"]), "bil geo"
    assert (a["bil_grid"].view(-1, 2) - m["geom"]["bil_grid"].view(-1, 2)).abs().max() < 1e-5, "bil grid"
    assert torch.equal(a["img_index"], m["geom"]["img_index"]), "bil img_index"
    print(f"bbdata check OK on {n_img} train images ({len(a['rows'])} ROIs): c0 == PacoImageDS, bil == MSDImageDS")


if __name__ == "__main__":
    check()
