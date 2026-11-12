"""CPU sanity check of the coordinate chain used for M_r: crop-canvas cell centre -> original pixel -> Full384 -> feature cell -> grid_sample coordinate.

  1. Each aux_grid entry is recomputed from the letterbox parameters and compared with geometry().
  2. Photometric test: a smooth two-channel coordinate image (R = x/W, G = y/H) is run through the real crop and full canvases. Sampling the full canvas on aux_grid must give
     the colour of the matching crop cell centre.
  3. Coverage statistics of the 24 x 24 grid on training ROIs (padding cells, partly covered cells, partly covered cells whose centre lies in padding).
  4. Figures: original + ROI + traced points, Full384, Crop384 with cell grid and padding, and a table of the traced coordinates.

  python -m src.pdit.geometry_check --out <run>/geometry
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.config import COCO_TRAIN, STREAM  # noqa: E402
from src.msd.pretrain import eval_exclusions  # noqa: E402
from src.viz.c0_pca import LB, MEAN_T, STD_T, geometry  # noqa: E402

S, P, G = 384, 16, 24
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]


def recompute(W, H, bx, crop, full):
    l, t, r, b = bx
    i, j = np.divmod(np.arange(G * G), G)
    xc, yc = (j + .5) * P, (i + .5) * P
    xo, yo = l + (xc - crop.ox) * (r - l) / crop.nw, t + (yc - crop.oy) * (b - t) / crop.nh
    Xf, Yf = full.ox + xo * full.nw / W, full.oy + yo * full.nh / H
    return dict(xc=xc, yc=yc, xo=xo, yo=yo, Xf=Xf, Yf=Yf, fx=Xf / P, fy=Yf / P, gx=Xf / (S / 2) - 1, gy=Yf / (S / 2) - 1)


def to_rgb(x):
    return (x * STD_T + MEAN_T).clamp(0, 1)


def photometric(W, H, bx, grid, cov):
    xs, ys = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5)
    a = np.stack([xs / W, ys / H, np.zeros_like(xs)], -1)
    img = Image.fromarray((a * 255).round().astype(np.uint8))
    crop, full = LB(bx[2] - bx[0], bx[3] - bx[1], S, P), LB(W, H, S, P)
    xc, xf = to_rgb(crop.canvas(img.crop(bx)))[None], to_rgb(full.canvas(img))[None]
    sampled = F.grid_sample(xf, torch.from_numpy(grid).reshape(1, G, G, 2), mode="bilinear", padding_mode="border", align_corners=False)
    centre = F.avg_pool2d(xc[:, :, 7:9 + 368, 7:9 + 368], 2, stride=P)   # 2 x 2 pixels around each cell centre = value at the centre
    mask = torch.from_numpy(cov.reshape(G, G) >= 1).float()[None, None]
    # Negative control: the grid obtained by ignoring the crop letterbox (ROI stretched over the whole canvas). The test must reject it for non-square ROIs.
    l, t, r, b = bx
    i, j = np.divmod(np.arange(G * G), G)
    xo, yo = l + (j + .5) * P * (r - l) / S, t + (i + .5) * P * (b - t) / S
    naive = np.stack([(full.ox + xo * full.nw / W) / (S / 2) - 1, (full.oy + yo * full.nh / H) / (S / 2) - 1], 1).astype(np.float32)
    wrong = F.grid_sample(xf, torch.from_numpy(naive).reshape(1, G, G, 2), mode="bilinear", padding_mode="border", align_corners=False)
    return ((sampled[:, :2] - centre[:, :2]).abs() * mask).max().item(), ((wrong[:, :2] - centre[:, :2]).abs() * mask).max().item()


def boxes(n):
    ex = set(eval_exclusions()["raev2_stream"])
    shared = "output/2026-11-08/hcf_pilot/shared"
    ids, obj, nb = np.load(f"{shared}/image_ids.npy"), np.load(f"{shared}/object_boxes.npy"), np.load(f"{shared}/nested_boxes.npy")
    pos = {int(i): k for k, i in enumerate(ids)}
    stream = np.load(STREAM)
    out = []
    for step in range(len(stream)):
        for c, iid in enumerate(stream[step]):
            iid = int(iid)
            if iid in ex:
                continue
            for b in list(obj[pos[iid]]) + [x for x in nb[step, c] if x[2] > x[0]]:
                out.append((iid, tuple(int(v) for v in b)))
                if len(out) >= n:
                    return out
    return out


def sample_cells(cov):
    valid = np.argwhere(cov.reshape(G, G) >= 1)
    r0, c0, r1, c1 = valid[:, 0].min(), valid[:, 1].min(), valid[:, 0].max(), valid[:, 1].max()
    want = [(r0, c0), (r0, c1), ((r0 + r1) // 2, (c0 + c1) // 2), (r1, c0), (r1, c1)]
    return [(int(r), int(c)) for r, c in want]


def figure(out, idx, iid, bx, trace, cells):
    img = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
    W, H = img.size
    crop, full = LB(bx[2] - bx[0], bx[3] - bx[1], S, P), LB(W, H, S, P)
    xc, xf = to_rgb(crop.canvas(img.crop(bx))).permute(1, 2, 0).numpy(), to_rgb(full.canvas(img)).permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="white")
    ax[0].imshow(img)
    ax[0].add_patch(mpatches.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec="yellow", lw=2))
    ax[0].set_title(f"Original {W}x{H}, ROI {bx}")
    ax[1].imshow(xf)
    ax[1].add_patch(mpatches.Rectangle((full.ox + bx[0] * full.nw / W, full.oy + bx[1] * full.nh / H), (bx[2] - bx[0]) * full.nw / W, (bx[3] - bx[1]) * full.nh / H, fill=False, ec="yellow", lw=2))
    ax[1].set_title("Full384 canvas (feature grid 24x24 / 48x48)")
    ax[2].imshow(xc)
    ax[2].add_patch(mpatches.Rectangle((crop.ox, crop.oy), crop.nw, crop.nh, fill=False, ec="yellow", lw=2))
    for k in range(0, S + 1, P):
        ax[2].axhline(k - .5, color="white", lw=.25, alpha=.6)
        ax[2].axvline(k - .5, color="white", lw=.25, alpha=.6)
    cov = crop.coverage().reshape(G, G)
    pad = np.ma.masked_where(cov > 0, np.ones_like(cov))
    ax[2].imshow(pad, extent=(0, S, S, 0), cmap="gray", alpha=.55, vmin=0, vmax=1, interpolation="nearest")
    ax[2].set_title(f"Crop384 (nw={crop.nw}, nh={crop.nh}), padding grey")
    for c, (r, q) in enumerate(cells):
        k = r * G + q
        ax[0].plot(trace["xo"][k], trace["yo"][k], "o", color=COLORS[c], ms=8, mec="white")
        ax[1].plot(trace["Xf"][k], trace["Yf"][k], "o", color=COLORS[c], ms=8, mec="white")
        ax[2].plot(trace["xc"][k], trace["yc"][k], "o", color=COLORS[c], ms=8, mec="white")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / f"roi_{idx}.png", dpi=110, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-stats", type=int, default=3000)
    ap.add_argument("--n-figs", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = boxes(a.n_stats)
    max_grid = max_photo = 0.
    naive_err = []
    photo_rows = 0
    pad, partial, centre_out, n = 0, 0, 0, 0
    traced = []
    for idx, (iid, bx) in enumerate(rows):
        with Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")) as img:
            W, H = img.size
        crop, full, grid = geometry(W, H, bx, S, P)
        tr = recompute(W, H, bx, crop, full)
        max_grid = max(max_grid, float(np.abs(grid[:, 0] - tr["gx"]).max()), float(np.abs(grid[:, 1] - tr["gy"]).max()))
        cov = crop.coverage()
        c = cov.reshape(G, G)
        pad += int((c == 0).sum())
        part = (c > 0) & (c < 1)
        partial += int(part.sum())
        inside = (tr["xc"] >= crop.ox) & (tr["xc"] <= crop.ox + crop.nw) & (tr["yc"] >= crop.oy) & (tr["yc"] <= crop.oy + crop.nh)
        centre_out += int((part & ~inside.reshape(G, G)).sum())
        n += G * G
        if idx < 300:
            err, bad = photometric(W, H, bx, grid, cov)
            max_photo = max(max_photo, err)
            naive_err.append((bad, abs((bx[2] - bx[0]) / (bx[3] - bx[1]) - 1)))
            photo_rows += 1
        if idx < a.n_figs:
            cells = sample_cells(cov)
            figure(out, idx, iid, bx, tr, cells)
            traced.append(dict(image_id=iid, box=list(bx), W=W, H=H, crop=dict(ox=crop.ox, oy=crop.oy, nw=crop.nw, nh=crop.nh), full=dict(ox=full.ox, oy=full.oy, nw=full.nw, nh=full.nh),
                               cells=[dict(row=r, col=q, crop_xy=[float(tr["xc"][r * G + q]), float(tr["yc"][r * G + q])], original_xy=[float(tr["xo"][r * G + q]), float(tr["yo"][r * G + q])], full384_xy=[float(tr["Xf"][r * G + q]), float(tr["Yf"][r * G + q])],
                                           feature24_xy=[float(tr["fx"][r * G + q]), float(tr["fy"][r * G + q])], grid_xy=[float(grid[r * G + q, 0]), float(grid[r * G + q, 1])]) for r, q in cells]))
    summary = dict(n_rois=len(rows), max_abs_grid_error=max_grid, photometric_max_error_01_scale=max_photo, photometric_rois=photo_rows, photometric_note="full-coverage cells only; error includes BICUBIC resample and integer letterbox rounding",
                   fraction_padding_cells=pad / n, fraction_partial_cells=partial / n, fraction_partial_with_centre_in_padding=centre_out / n, pass_grid=max_grid < 1e-3, pass_photometric=max_photo < .02,
                   negative_control_median_error_nonsquare=float(np.median([e for e, d in naive_err if d > .2])), negative_control_rejected_fraction_nonsquare=float(np.mean([e > .02 for e, d in naive_err if d > .2])))
    (out / "trace.json").write_text(json.dumps(traced, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    assert summary["pass_grid"] and summary["pass_photometric"], summary


if __name__ == "__main__":
    main()
