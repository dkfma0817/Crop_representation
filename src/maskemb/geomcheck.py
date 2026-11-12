"""Geometry checks (CPU, no model): box -> 336 canvas -> 24 x 24 grid for the Full and C0 inputs, with figures.
  (1) src.g1x.data.check(): full_r2_geo at 336/14 == the original R2 full-canvas rule (common_rr.roi_geo / Letterbox)
  (2) on official val ROIs: the PacoDS 'full' item (what R2 reads) vs an independent recomputation of box_c / cov from
      the raw annotation bbox (float COCO box -> outward integer box -> letterbox scale/offset); C0: BBImageDS 'c0'
      cov == crop-content coverage, padding cells cov == 0; canvases decode back to the image (CLIP mean / std)
  (3) figures sanity/geom_<ann>.png: full canvas with the GT box, the ROI cells (cov > 0) and the C0 canvas with its
      valid cells, for small / medium / large objects and parts.
  python -m src.maskemb.geomcheck
"""
import json
import math
import os

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.maskemb import common as MC  # noqa: E402

DATA = "output/2026-11-07/paco_oracle_attr/data"


def denorm(x):
    m, s = torch.tensor(MC.CLIP_MEAN).view(3, 1, 1), torch.tensor(MC.CLIP_STD).view(3, 1, 1)
    return (x * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    CP = MC.set_clip_norm()
    from src.g1x.data import PacoDS, check as g1x_check
    from src.ivc.bbdata import BBImageDS
    out = {}
    g1x_check(40)
    out["full_r2_geo_equals_roi_geo_336_14"] = True
    M = pd.read_parquet(os.path.join(DATA, "manifest_val.parquet"))
    rng = np.random.default_rng(0)
    imgs = rng.choice(M.image_id.unique(), 60, replace=False)
    sub = M[M.image_id.isin(imgs)].reset_index(drop=True)
    Y = np.zeros((len(sub), 59), np.int8)
    full = PacoDS(sub, Y, False, "full", MC.S, MC.P)
    c0 = BBImageDS(sub, Y, False, "c0", MC.S, MC.P)
    errs, c0err, n = [], [], 0
    keep = []
    for gi in range(len(full)):
        bf, bc = full[[gi]], c0[[gi]]
        for j, r in enumerate(bf["rows"].tolist()):
            row = sub.iloc[r]
            W, H = int(row.W), int(row.H)
            x, y, w, h = float(row.bx), float(row.by), float(row.bw), float(row.bh)
            l, t = max(0, min(W - 1, math.floor(x))), max(0, min(H - 1, math.floor(y)))
            rr, bb = max(l + 1, min(W, math.ceil(x + w))), max(t + 1, min(H, math.ceil(y + h)))
            assert (l, t, rr, bb) == (int(row.l), int(row.t), int(row.r), int(row.b))
            s = MC.S / max(W, H)
            nw, nh = min(MC.S, max(1, round(W * s))), min(MC.S, max(1, round(H * s)))
            ox, oy = (MC.S - nw) // 2, (MC.S - nh) // 2
            X0, X1, Y0, Y1 = ox + l * nw / W, ox + rr * nw / W, oy + t * nh / H, oy + bb * nh / H
            e = np.arange(MC.G + 1) * MC.P
            cx = np.clip(np.minimum(e[1:], X1) - np.maximum(e[:-1], X0), 0, None) / MC.P
            cy = np.clip(np.minimum(e[1:], Y1) - np.maximum(e[:-1], Y0), 0, None) / MC.P
            cov = (cy[:, None] * cx[None, :]).reshape(-1)
            errs.append(max(np.abs(bf["box_c"][j].numpy() - [X0, Y0, X1, Y1]).max(), np.abs(bf["cov"][j].numpy() - cov).max()))
            k = bc["rows"].tolist().index(r)
            cw, ch = rr - l, bb - t
            cs = MC.S / max(cw, ch)
            cnw, cnh = min(MC.S, max(1, round(cw * cs))), min(MC.S, max(1, round(ch * cs)))
            cox, coy = (MC.S - cnw) // 2, (MC.S - cnh) // 2
            ccx = np.clip(np.minimum(e[1:], cox + cnw) - np.maximum(e[:-1], cox), 0, None) / MC.P
            ccy = np.clip(np.minimum(e[1:], coy + cnh) - np.maximum(e[:-1], coy), 0, None) / MC.P
            ccov = (ccy[:, None] * ccx[None, :]).reshape(-1)
            xc = bc["x"][k]
            padmask = torch.ones(MC.S, MC.S, dtype=torch.bool)
            padmask[coy:coy + cnh, cox:cox + cnw] = False
            c0err.append(max(np.abs(bc["cov"][k].numpy() - ccov).max(), float(xc[:, padmask].abs().max()) if padmask.any() else 0.0))
            n += 1
            keep.append(dict(r=r, gi=gi, j=j, k=k, area=w * h, level=row.level, n_full=int((cov > 0).sum()), n_c0=int((ccov > 0).sum())))
    out["val_rois_checked"] = n
    out["full_boxc_cov_max_abs_err"] = float(max(errs))
    out["c0_cov_and_padding_max_abs_err"] = float(max(c0err))
    assert out["full_boxc_cov_max_abs_err"] < 1e-4 and out["c0_cov_and_padding_max_abs_err"] < 1e-5
    # figures: 6 ROIs spanning size / level
    K = pd.DataFrame(keep)
    pick = []
    for lvl in ("object", "part"):
        for lo, hi in ((0, 32 ** 2), (32 ** 2, 96 ** 2), (96 ** 2, 1e12)):
            c = K[(K.level == lvl) & (K.area >= lo) & (K.area < hi)]
            if len(c):
                pick.append(c.iloc[len(c) // 2])
    fig, axs = plt.subplots(len(pick), 2, figsize=(10, 5 * len(pick)))
    for ax, p in zip(axs, pick):
        bf, bc = full[[int(p.gi)]], c0[[int(p.gi)]]
        row = sub.iloc[int(p.r)]
        for a_, x, cov, box, ttl in ((ax[0], bf["x"][0], bf["cov"][int(p.j)], bf["box_c"][int(p.j)], f"Full 336: ann {row.ann_id} {row.level}, box {row.bw:.0f}x{row.bh:.0f}px, ROI cells {int(p.n_full)}"),
                                     (ax[1], bc["x"][int(p.k)], bc["cov"][int(p.k)], None, f"C0 crop 336: valid cells {int(p.n_c0)}")):
            a_.imshow(denorm(x), extent=(0, MC.S, MC.S, 0))
            cv = cov.numpy().reshape(MC.G, MC.G)
            for i in range(MC.G):
                for jj in range(MC.G):
                    if cv[i, jj] > 0:
                        a_.add_patch(Rectangle((jj * MC.P, i * MC.P), MC.P, MC.P, fill=True, alpha=0.15 + 0.35 * min(1, cv[i, jj]), color="tab:cyan", lw=0))
            for g in range(MC.G + 1):
                a_.axhline(g * MC.P, color="w", lw=0.2)
                a_.axvline(g * MC.P, color="w", lw=0.2)
            if box is not None:
                X0, Y0, X1, Y1 = box.tolist()
                a_.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, color="red", lw=1.5))
            a_.set_title(ttl, fontsize=8)
            a_.set_xlim(0, MC.S)
            a_.set_ylim(MC.S, 0)
    plt.tight_layout()
    fn = MC.run_path("sanity", "geom_examples.png")
    plt.savefig(fn, dpi=110)
    out["figure"] = fn
    json.dump(out, open(MC.run_path("sanity", "geomcheck.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
