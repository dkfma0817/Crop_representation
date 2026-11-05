"""ROI coordinate alignment check: does output cell (i, j) of the bilinear / bicubic crop-grid read sit at the same
original-image location as C0 cell (i, j)?  PACO VAL only.

1. Analytic (all ROIs): C0 cell centre -> original px, recomputed here with an INDEPENDENT formula
       x_orig = l + ((j + 0.5) * 14 - ox_c) * (r - l) / nw_c
   (crop letterbox of the integer box), then -> full 336 canvas X = ox_f + x_orig * nw_f / W -> grid_sample coordinate
   X / 168 - 1 (align_corners=False: full cell k has its centre at (k + 0.5) * 14 px <-> (2k + 1) / 24 - 1).
   Compared with roi_geometry()['bil_grid'] that the sampler actually uses. Also: share of valid cells whose sample point
   lies outside the full-grid centre hull [7, 329] px (border clamp active) and outside the image content.
2. Synthetic coordinate map: feature channels = full-canvas X / Y of every cell centre; sampled with the real call
   (bilinear and bicubic). Inside the hull both must return the target coordinate (linear functions are reproduced).
3. Feature shift scan (GPU, fixed sample): bicubic read with the sampling grid shifted by (dx, dy) in {-14, -7, 0, 7, 14}
   full-canvas px; mean valid-token cosine to C0. Correct alignment -> maximum at (0, 0).
4. Figures for the fixed alignment ROIs (configs/ivc/sample_rois.json, chosen with seed 0 before any feature was computed).

  python -m src.ivc.align --out output/2026-11-05/ivc/align [--no_gpu]
"""
import argparse
import json
import os

import numpy as np
import torch

import src.readout.common as RC
from src.ivc.common import EXCL, G, sample_roi
from src.msd.common import Letterbox, roi_geometry

S, P = 336, 14


def independent_grid(W, H, ltrb):
    """-> (576, 2) original px of every C0 cell centre and the (576, 2) grid_sample coordinate, from scratch."""
    l, t, r, b = ltrb
    w, h = r - l, b - t
    s = S / max(w, h)
    nwc, nhc = min(S, max(1, int(round(w * s)))), min(S, max(1, int(round(h * s))))
    oxc, oyc = (S - nwc) // 2, (S - nhc) // 2
    sf = S / max(W, H)
    nwf, nhf = min(S, max(1, int(round(W * sf)))), min(S, max(1, int(round(H * sf))))
    oxf, oyf = (S - nwf) // 2, (S - nhf) // 2
    i, j = np.divmod(np.arange(G * G), G)
    xo = l + ((j + 0.5) * P - oxc) * w / nwc
    yo = t + ((i + 0.5) * P - oyc) * h / nhc
    X, Y = oxf + xo * nwf / W, oyf + yo * nhf / H
    return np.stack([xo, yo], 1), np.stack([X / (S / 2) - 1, Y / (S / 2) - 1], 1), (oxf, oyf, nwf, nhf)


def analytic(dva):
    errs, clamp, outside, n_valid = [], 0, 0, 0
    worst = (0.0, None)
    for W, H, l, t, r, b in dva[["W", "H", "l", "t", "r", "b"]].values.astype(int):
        g = roi_geometry(W, H, (l, t, r, b))
        _, grid, (oxf, oyf, nwf, nhf) = independent_grid(W, H, (l, t, r, b))
        cov = g["crop"].coverage()
        v = cov > 0
        e = np.abs(grid - g["bil_grid"].astype(np.float64))[v] * (S / 2)        # canvas px
        errs.append(e.max())
        if e.max() > worst[0]:
            worst = (float(e.max()), (int(W), int(H), int(l), int(t), int(r), int(b)))
        Xp = (g["bil_grid"][v] + 1) * (S / 2)
        clamp += int(((Xp < P / 2) | (Xp > S - P / 2)).any(1).sum())
        outside += int(((Xp[:, 0] < oxf) | (Xp[:, 0] > oxf + nwf) | (Xp[:, 1] < oyf) | (Xp[:, 1] > oyf + nhf)).sum())
        n_valid += int(v.sum())
    errs = np.array(errs)
    return dict(n_roi=len(errs), max_err_canvas_px=float(errs.max()), mean_roi_max_err_canvas_px=float(errs.mean()),
                worst=worst, n_valid_cells=n_valid, frac_valid_cells_border_clamped=clamp / n_valid,
                frac_valid_cells_outside_image_content=outside / n_valid,
                note="error = |independent formula - roi_geometry bil_grid| in full-canvas px (float32 storage of the grid)")


def synthetic(dva, n=2000):
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(dva), min(n, len(dva)), replace=False))
    k = np.arange(G) * P + P / 2
    fmap = torch.zeros(1, 2, G, G, dtype=torch.float64)
    fmap[0, 0] = torch.as_tensor(k)[None, :].expand(G, G)
    fmap[0, 1] = torch.as_tensor(k)[:, None].expand(G, G)
    tok = fmap.flatten(2).transpose(1, 2).float()                                # (1, 576, 2)
    out = {}
    for mode in ("bilinear", "bicubic"):
        e_in, e_border = [], []
        for r in rows:
            W, H, l, t, rr, b = dva[["W", "H", "l", "t", "r", "b"]].values[r].astype(int)
            g = roi_geometry(W, H, (l, t, rr, b))
            geom = dict(img_index=torch.zeros(1, dtype=torch.long), bil_grid=torch.as_tensor(g["bil_grid"])[None])
            got = sample_roi(tok, geom, mode)[0].double().numpy()                    # (576, 2) sampled canvas coords
            tgt = (g["bil_grid"].astype(np.float64) + 1) * (S / 2)
            v = g["crop"].coverage() > 0
            hull = ((tgt >= P / 2) & (tgt <= S - P / 2)).all(1)
            e = np.abs(got - tgt).max(1)
            e_in.append(e[v & hull])
            e_border.append(e[v & ~hull])
        ei, eb = np.concatenate(e_in), np.concatenate(e_border)
        out[mode] = dict(n_cells_in_hull=len(ei), max_err_in_hull_px=float(ei.max()), n_cells_border=len(eb),
                         mean_err_border_px=float(eb.mean()) if len(eb) else 0.0, max_err_border_px=float(eb.max()) if len(eb) else 0.0)
    out["note"] = "sampled value of a map whose channels are the full-canvas X / Y of each cell centre vs the target point; border = target within 7 px of the canvas edge (clamped)"
    return out


@torch.no_grad()
def shift_scan(dva, Yva, rows, dev):
    import src.msd.common as MS
    vit = RC.load_vit().to(dev)
    ds = MS.MSDImageDS(dva, Yva, flip=False, want_crops=True)
    groups = RC.C.image_groups(dva)
    img_of = np.empty(len(dva), int)
    for gi, g in enumerate(groups):
        img_of[g] = gi
    shifts = [-14, -7, 0, 7, 14]
    acc = np.zeros((len(rows), len(shifts), len(shifts)))
    for k0 in range(0, len(rows), 64):
        rr = rows[k0:k0 + 64]
        imgs = sorted(set(img_of[rr].tolist()))
        b = MS.to_dev(ds[imgs], dev)
        full = RC.encode_layers(vit, b["full"], (12,))[12]
        brow = b["rows"].cpu().numpy()
        pos = np.array([int(np.flatnonzero(brow == r)[0]) for r in rr])
        crops = b["crops"][pos]
        c0 = torch.cat([RC.encode_layers(vit, crops[i:i + 64], (12,))[12] for i in range(0, len(crops), 64)])
        geom = {"img_index": b["geom"]["img_index"][pos], "bil_grid": b["geom"]["bil_grid"][pos]}
        cov = b["cov_c"][pos].float()
        w = cov / cov.sum(1, keepdim=True)
        for a_, dy in enumerate(shifts):
            for c_, dx in enumerate(shifts):
                gsh = geom["bil_grid"] + torch.tensor([dx, dy], device=dev, dtype=torch.float32) / (S / 2)
                f = sample_roi(full, {"img_index": geom["img_index"], "bil_grid": gsh}, "bicubic")
                cs = torch.nn.functional.cosine_similarity(f, c0, dim=-1)
                acc[k0:k0 + len(rr), a_, c_] = (w * cs).sum(1).cpu().numpy()
    return shifts, acc


def figures(dva, rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib.patches import Rectangle
    for k, r in enumerate(rows):
        d = dva.iloc[r]
        W, H, l, t, rr, b = (int(d[c]) for c in ("W", "H", "l", "t", "r", "b"))
        img = Image.open(d.path).convert("RGB")
        g = roi_geometry(W, H, (l, t, rr, b))
        cl, fl = g["crop"], g["full"]
        cov = cl.coverage()
        v = cov > 0
        xo, _, _ = independent_grid(W, H, (l, t, rr, b))
        fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
        ax[0].imshow(img)
        ax[0].add_patch(Rectangle((l, t), rr - l, b - t, fill=False, ec="yellow", lw=1.5))
        ax[0].scatter(xo[v, 0], xo[v, 1], s=2, c="lime")
        ax[0].scatter(xo[~v, 0], xo[~v, 1], s=2, c="red")
        pad = max(rr - l, b - t) * 0.6
        ax[0].set_xlim(max(0, l - pad), min(W, rr + pad))
        ax[0].set_ylim(min(H, b + pad), max(0, t - pad))
        ax[0].set_title(f"original + GT box, C0 cell centres\n(green valid / red padding) ann {d.ann_id} {d.level} n_in {d.n_in}", fontsize=9)
        x = RC.to_tensor_canvas(img.crop((l, t, rr, b)), cl)
        mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        ax[1].imshow((x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        for e in range(0, S + 1, P):
            ax[1].axhline(e - 0.5, c="w", lw=0.2)
            ax[1].axvline(e - 0.5, c="w", lw=0.2)
        ax[1].add_patch(Rectangle((cl.ox - 0.5, cl.oy - 0.5), cl.nw, cl.nh, fill=False, ec="yellow", lw=1))
        ax[1].set_title(f"C0 input crop (336 letterbox, content {cl.nw}x{cl.nh} at ({cl.ox},{cl.oy}))", fontsize=9)
        xf = RC.to_tensor_canvas(img, fl)
        ax[2].imshow((xf * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        Xp = (g["bil_grid"] + 1) * (S / 2)
        ax[2].scatter(Xp[v, 0] - 0.5, Xp[v, 1] - 0.5, s=1.5, c="lime")
        ax[2].scatter(Xp[~v, 0] - 0.5, Xp[~v, 1] - 0.5, s=1.5, c="red")
        for e in range(0, S + 1, P):
            ax[2].axhline(e - 0.5, c="w", lw=0.15)
            ax[2].axvline(e - 0.5, c="w", lw=0.15)
        X0, Y0 = fl.to_canvas(l, t)
        X1, Y1 = fl.to_canvas(rr, b)
        ax[2].add_patch(Rectangle((X0 - 0.5, Y0 - 0.5), X1 - X0, Y1 - Y0, fill=False, ec="yellow", lw=1))
        m = 1.2 * max(X1 - X0, Y1 - Y0) + 20
        cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
        ax[2].set_xlim(cx - m, cx + m)
        ax[2].set_ylim(cy + m, cy - m)
        ax[2].set_title("F336 canvas (24x24 full grid) + bilinear/bicubic sample points", fontsize=9)
        im = ax[3].imshow(cov.reshape(G, G), vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(im, ax=ax[3], fraction=0.046)
        ax[3].set_title("C0 letterbox coverage = valid-token mask\n(cov>0 main mask; cov=1 structure mask)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"align_{k:02d}_ann{d.ann_id}.png"), dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_gpu", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dva, Yva = RC.load_data(EXCL, splits=("val",))["val"]
    smp = json.load(open("configs/ivc/sample_rois.json"))
    rows_fig = [int(np.flatnonzero(dva.ann_id.values == x)[0]) for x in smp["align_ann_ids"]]
    rep = dict(analytic=analytic(dva), synthetic=synthetic(dva))
    print(json.dumps(rep, indent=1), flush=True)
    figures(dva, rows_fig, a.out)
    if not a.no_gpu:
        rows = np.array([int(np.flatnonzero(dva.ann_id.values == x)[0]) for x in smp["shift_scan_ann_ids"]])
        shifts, acc = shift_scan(dva, Yva, rows, torch.device("cuda"))
        np.savez(os.path.join(a.out, "shift_scan.npz"), rows=rows, shifts=shifts, cos=acc)
        mean = acc.mean(0)
        am = np.array([np.unravel_index(np.argmax(x), x.shape) for x in acc])
        at0 = float(((am[:, 0] == 2) & (am[:, 1] == 2)).mean())
        # sub-px peak of the mean surface: parabola through the 3 central shifts per axis
        def peak(v):
            d = v[0] - 2 * v[1] + v[2]
            return float(0.5 * (v[0] - v[2]) / d * 7) if d != 0 else 0.0
        rep["shift_scan"] = dict(n_roi=len(rows), shifts_px=shifts, mean_cos=mean.round(4).tolist(),
                                 frac_roi_argmax_at_0=at0, mean_peak_offset_px=dict(dx=peak(mean[2, 1:4]), dy=peak(mean[1:4, 2])),
                                 note="bicubic read, grid shifted by (dx, dy) full-canvas px (14 px = 1 full cell); value = coverage-weighted mean token cosine to C0")
        print(json.dumps(rep["shift_scan"], indent=1), flush=True)
    json.dump(rep, open(os.path.join(a.out, "alignment.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
