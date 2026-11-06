"""Alignment checks for the RAE diagnosis (PACO val). -> <out>/alignment_checks/{alignment.json, align_*.png}

1. Feature grid (all ROIs): C0 cell centre -> original px -> full 256 canvas, recomputed with an independent formula, vs the
   roi_geometry bil_grid used by bilinear / bicubic sampling (16 x 16 RAE grid, 16 px cells on the 256 canvas).
2. Synthetic coordinate map (2,000 ROIs): channels = full-canvas X / Y of each RAE cell centre, sampled by the real calls.
3. RGB resampling (B / B_in): synthetic full canvas whose pixel values are its own X / Y coordinates, resampled with
   run.resample_full_to_crop; expected value per crop-canvas content pixel from an independent formula.
4. Pixel registration (B_in vs GT crop canvas, fixed 600 ROIs): L1 of B_in when its sampling map is shifted by
   (dx, dy) in {-2, -1, 0, 1, 2} crop-canvas px; correct alignment -> minimum at (0, 0).
5. Feature registration (GPU, same 600 ROIs): token cosine of the bicubic read to the C0 latent T with the grid shifted by
   {-16, -8, 0, 8, 16} full-canvas px (16 px = one RAE cell); correct alignment -> maximum at (0, 0).
6. Figures for the fixed alignment ROIs (configs/raediag/sample_rois.json, reused from the IVC list).
"""
import argparse
import json
import os

import numpy as np
import torch

import src.readout.common as RD
import src.rae.common as RC
from src.raediag.run import EXCL, ValImages, bicubic, resample_full_to_crop

S, P, G = RC.S, RC.PATCH, RC.G


def indep_grid(W, H, ltrb):
    l, t, r, b = ltrb
    w, h = r - l, b - t
    sc = S / max(w, h)
    nwc, nhc = min(S, max(1, int(round(w * sc)))), min(S, max(1, int(round(h * sc))))
    oxc, oyc = (S - nwc) // 2, (S - nhc) // 2
    sf = S / max(W, H)
    nwf, nhf = min(S, max(1, int(round(W * sf)))), min(S, max(1, int(round(H * sf))))
    oxf, oyf = (S - nwf) // 2, (S - nhf) // 2
    i, j = np.divmod(np.arange(G * G), G)
    xo = l + ((j + 0.5) * P - oxc) * w / nwc
    yo = t + ((i + 0.5) * P - oyc) * h / nhc
    X, Y = oxf + xo * nwf / W, oyf + yo * nhf / H
    return np.stack([X / (S / 2) - 1, Y / (S / 2) - 1], 1), (oxc, oyc, nwc, nhc, oxf, oyf, nwf, nhf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_gpu", action="store_true")
    a = ap.parse_args()
    O = os.path.join(a.out, "alignment_checks")
    os.makedirs(O, exist_ok=True)
    dva, Yva = RD.load_data(EXCL, splits=("val",))["val"]
    smp = json.load(open("configs/raediag/sample_rois.json"))
    rep = {}
    # 1
    errs, clamp, nv = [], 0, 0
    for W, H, l, t, r, b in dva[["W", "H", "l", "t", "r", "b"]].values.astype(int):
        g = RC.roi_geometry(W, H, (l, t, r, b))
        gi, _ = indep_grid(W, H, (l, t, r, b))
        v = g["valid_w"] > 0
        errs.append(np.abs(gi - g["bil_grid"].astype(np.float64))[v].max() * (S / 2))
        Xp = (g["bil_grid"][v] + 1) * (S / 2)
        clamp += int(((Xp < P / 2) | (Xp > S - P / 2)).any(1).sum())
        nv += int(v.sum())
    rep["feature_grid"] = dict(n_roi=len(errs), max_err_px=float(np.max(errs)), frac_valid_cells_border_clamped=clamp / nv,
                               note="|independent formula - bil_grid| on the 256 canvas; clamped = sample point within 8 px of the canvas edge")
    # 2
    k = np.arange(G) * P + P / 2
    tok = torch.zeros(1, G * G, 2)
    tok[0, :, 0] = torch.as_tensor(np.tile(k, G), dtype=torch.float32)
    tok[0, :, 1] = torch.as_tensor(np.repeat(k, G), dtype=torch.float32)
    rows = np.sort(np.random.default_rng(0).choice(len(dva), 2000, replace=False))
    syn = {}
    for nm, fn in (("bilinear", RC.bilinear), ("bicubic", bicubic)):
        e_in, e_b = [], []
        for r in rows:
            W, H, l, t, rr, bb = dva[["W", "H", "l", "t", "r", "b"]].values[r].astype(int)
            g = RC.roi_geometry(W, H, (l, t, rr, bb))
            geom = dict(img_index=torch.zeros(1, dtype=torch.long), bil_grid=torch.as_tensor(g["bil_grid"])[None])
            got = fn(tok, geom)[0].double().numpy()
            tgt = (g["bil_grid"].astype(np.float64) + 1) * (S / 2)
            v = g["valid_w"] > 0
            hull = ((tgt >= P / 2) & (tgt <= S - P / 2)).all(1)
            d = np.minimum(tgt, S - tgt).min(1)
            e = np.abs(got - tgt).max(1)
            e_in.append(e[v & (d >= 1.5 * P)])
            e_b.append(e[v & hull & (d < 1.5 * P)])
        ei, eb = np.concatenate(e_in), np.concatenate(e_b)
        syn[nm] = dict(interior_max_px=float(ei.max()), interior_mean_px=float(ei.mean()), near_edge_max_px=float(eb.max()) if len(eb) else 0.0)
    syn["note"] = "interior = >= 1.5 cells from the canvas edge; bicubic (A = -0.75) does not reproduce linear ramps (kernel property, grid identical)"
    rep["feature_synthetic"] = syn
    # 3
    ds = ValImages(dva)
    gl = [int(np.flatnonzero([dva.ann_id[x] in set(smp["shift_scan_ann_ids"]) for x in grp]).size > 0) for grp in ds.groups]
    sel_img = [i for i, f in enumerate(gl) if f]
    xx = torch.arange(S, dtype=torch.float32) + 0.5
    coord = torch.stack([xx[None].expand(S, S), xx[:, None].expand(S, S), torch.zeros(S, S)])[None]
    e3, l1s = [], {}
    shifts = [-2, -1, 0, 1, 2]
    for ci in range(0, len(sel_img), 8):
        b = ds[sel_img[ci:ci + 8]]
        keep = np.array([dva.ann_id[int(r)] in set(smp["shift_scan_ann_ids"]) for r in b["rows"].numpy()])
        ii = b["geom"]["img_index"]
        got = resample_full_to_crop(coord.expand(len(b["full_u8"]), -1, -1, -1), ii, b["affine"])
        m = RC.content_mask(b["rects"], "cpu")
        for k in np.flatnonzero(keep):
            r = int(b["rows"][k])
            d = dva.loc[r]
            _, (oxc, oyc, nwc, nhc, oxf, oyf, nwf, nhf) = indep_grid(int(d.W), int(d.H), (int(d.l), int(d.t), int(d.r), int(d.b)))
            xo = int(d.l) + (xx - oxc) * (int(d.r) - int(d.l)) / nwc
            yo = int(d.t) + (xx - oyc) * (int(d.b) - int(d.t)) / nhc
            Xe, Ye = oxf + xo * nwf / int(d.W), oyf + yo * nhf / int(d.H)
            mk = m[k, 0] > 0
            inner = mk & (Xe[None].expand(S, S) > 0.5) & (Xe[None].expand(S, S) < S - 0.5) & (Ye[:, None].expand(S, S) > 0.5) & (Ye[:, None].expand(S, S) < S - 0.5)
            ex = (got[k, 0] - Xe[None].expand(S, S)).abs()[inner]
            ey = (got[k, 1] - Ye[:, None].expand(S, S)).abs()[inner]
            if len(ex):
                e3.append(float(torch.maximum(ex.max(), ey.max())))
        gt = b["crops"].float() / 255
        full = b["full_u8"].float() / 255
        for dy in shifts:
            for dx in shifts:
                aff = b["affine"].clone()
                aff[:, 1] += dx * aff[:, 0]
                aff[:, 3] += dy * aff[:, 2]
                y = resample_full_to_crop(full, ii, aff)
                l1 = ((y - gt).abs() * m).sum((1, 2, 3)) / (3 * m.sum((1, 2, 3)))
                l1s.setdefault((dy, dx), []).extend(l1[keep].tolist())
    L = np.array([[np.mean(l1s[(dy, dx)]) for dx in shifts] for dy in shifts])
    rep["rgb_resampling"] = dict(n_roi=len(e3), max_err_px=float(np.max(e3)), note="synthetic coordinate canvas through resample_full_to_crop vs independent formula, interior content pixels")
    rep["pixel_registration_B_in"] = dict(n_roi=len(l1s[(0, 0)]), shifts_crop_px=shifts, mean_l1=L.round(5).tolist(),
                                          argmin=[shifts[int(i)] for i in np.unravel_index(np.argmin(L), L.shape)],
                                          note="B_in sampling map shifted by (dy, dx) crop-canvas px; L1 to the GT crop canvas on content pixels")
    # 5
    if not a.no_gpu:
        dev = "cuda"
        RC.set_fp32()
        rae = RC.FrozenRAE(dev, decoder=False)
        fs = [-16, -8, 0, 8, 16]
        acc = {}
        for ci in range(0, len(sel_img), 8):
            b = ds[sel_img[ci:ci + 8]]
            keep = torch.as_tensor([dva.ann_id[int(r)] in set(smp["shift_scan_ann_ids"]) for r in b["rows"].numpy()])
            g = RC.to_device(b["geom"], dev)
            Zf, T = rae.encode(b["full_u8"]), rae.encode(b["crops"])
            w = g["valid_w"] / g["valid_w"].sum(1, keepdim=True)
            for dy in fs:
                for dx in fs:
                    gg = dict(g, bil_grid=g["bil_grid"] + torch.tensor([dx, dy], device=dev, dtype=torch.float32) / (S / 2))
                    cs = torch.nn.functional.cosine_similarity(bicubic(Zf, gg), T, dim=-1)
                    acc.setdefault((dy, dx), []).extend(((w * cs).sum(1)[keep.to(dev)]).tolist())
        M = np.array([[np.mean(acc[(dy, dx)]) for dx in fs] for dy in fs])
        rep["feature_registration"] = dict(n_roi=len(acc[(0, 0)]), shifts_full_px=fs, mean_cos=M.round(4).tolist(),
                                           argmax=[fs[int(i)] for i in np.unravel_index(np.argmax(M), M.shape)])
    json.dump(rep, open(os.path.join(O, "alignment.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    figures(dva, ds, smp["align_ann_ids"], O)


def figures(dva, ds, ann_ids, O):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    img_of = {int(r): gi for gi, grp in enumerate(ds.groups) for r in grp}
    for n, aid in enumerate(ann_ids):
        r = int(np.flatnonzero(dva.ann_id.values == aid)[0])
        r = int(dva.index[r])
        b = ds[[img_of[r]]]
        k = int(np.flatnonzero(b["rows"].numpy() == r)[0])
        d = dva.loc[r]
        W, H, l, t, rr, bb = (int(d[c]) for c in ("W", "H", "l", "t", "r", "b"))
        g = RC.roi_geometry(W, H, (l, t, rr, bb))
        v = g["valid_w"] > 0
        full = b["full_u8"][0].permute(1, 2, 0).numpy()
        crop = b["crops"][k].permute(1, 2, 0).numpy()
        bin_ = resample_full_to_crop(b["full_u8"].float() / 255, b["geom"]["img_index"][k:k + 1], b["affine"][k:k + 1])[0]
        m = RC.content_mask(b["rects"][k:k + 1], "cpu")[0, 0].numpy()
        fig, ax = plt.subplots(1, 5, figsize=(25, 5.4))
        img = Image.open(d.path).convert("RGB")
        ax[0].imshow(img)
        ax[0].add_patch(Rectangle((l, t), rr - l, bb - t, fill=False, ec="yellow", lw=1.5))
        pad = max(rr - l, bb - t) * 0.6
        ax[0].set_xlim(max(0, l - pad), min(W, rr + pad))
        ax[0].set_ylim(min(H, bb + pad), max(0, t - pad))
        ax[0].set_title(f"original + GT box  ann {aid} {d.level} n_in(F336) {d.n_in} / RAE {g['n_in']}", fontsize=9)
        ax[1].imshow(crop)
        for e in range(0, S + 1, P):
            ax[1].axhline(e - 0.5, c="w", lw=0.25)
            ax[1].axvline(e - 0.5, c="w", lw=0.25)
        ax[1].set_title("A input: GT crop canvas (256 letterbox, 16x16 RAE cells)", fontsize=9)
        ax[2].imshow(full)
        Xp = (g["bil_grid"] + 1) * (S / 2)
        ax[2].scatter(Xp[v, 0] - 0.5, Xp[v, 1] - 0.5, s=3, c="lime")
        ax[2].scatter(Xp[~v, 0] - 0.5, Xp[~v, 1] - 0.5, s=3, c="red")
        X0, Y0 = g["full"].to_canvas(l, t)
        X1, Y1 = g["full"].to_canvas(rr, bb)
        ax[2].add_patch(Rectangle((X0 - 0.5, Y0 - 0.5), X1 - X0, Y1 - Y0, fill=False, ec="yellow", lw=1))
        for e in range(0, S + 1, P):
            ax[2].axhline(e - 0.5, c="w", lw=0.2)
            ax[2].axvline(e - 0.5, c="w", lw=0.2)
        mm = 1.2 * max(X1 - X0, Y1 - Y0) + 24
        cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
        ax[2].set_xlim(cx - mm, cx + mm)
        ax[2].set_ylim(cy + mm, cy - mm)
        ax[2].set_title("full 256 canvas (RAE grid) + C/D sample points (green valid / red padding)", fontsize=9)
        ax[3].imshow(bin_.permute(1, 2, 0).clamp(0, 1).numpy())
        ax[3].set_title("B_in: full INPUT canvas resampled onto the crop canvas", fontsize=9)
        ax[4].imshow(m, vmin=0, vmax=1)
        ax[4].set_title("content (valid-pixel) mask; cells = valid_w > 0 for latents", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(O, f"align_{n:02d}_ann{aid}.png"), dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    main()
