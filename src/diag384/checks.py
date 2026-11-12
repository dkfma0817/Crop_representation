"""D1 CPU checks and RGB analysis. No GPU; no model.

  geometry  bit-level checks of the C1 path on real Val images -> <out>/geometry.json, sample crops -> <out>/samples/
  rgb       every Val and Test ROI: C0 vs C1 final crop RGB (uint8 -> [0,1], crop content rectangle only, padding excluded) -> <out>/rgb_rois.parquet, <out>/rgb_table.csv
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.diag384.bbdata import box_384, c1_rgb, full_rgb, to_canvas
from src.viz.c0_pca import LB, MEAN_T, STD_T, geometry

S, P, DATA = 384, 16, "output/2026-11-07/paco_oracle_attr/data"
BINS = [("tiny_lt4_tokens", 0, 32 ** 2), ("small_4to16_tokens", 32 ** 2, 64 ** 2), ("medium_16to64_tokens", 64 ** 2, 128 ** 2), ("large_ge64_tokens", 128 ** 2, 1e12)]    # = src.siglip_pm.size_diagnostic.BINS (that module needs the paco_eval env)


def size_bin(bw, bh, W, H):
    a = bw * bh * (S / max(W, H)) ** 2                # the official ROI-size rule (src.g1up.support.size)
    return next(n for n, lo, hi in BINS if lo <= a <= hi)


def canvas_equal(M, n_img):
    """Checks on real images; every assert stops the queue before any GPU work."""
    r, groups = dict(rois=0), list(M.groupby("image_id"))[:n_img]
    max_grid_error, max_inv_error = 0.0, 0.0
    for _, g in groups:
        img = Image.open(g.path.iloc[0]).convert("RGB")
        W, H = img.size
        rgb384 = full_rgb(img, LB(W, H, S, P))
        for row in g.itertuples(index=False):
            bx = (int(row.l), int(row.t), int(row.r), int(row.b))
            crop, full, grid = geometry(W, H, bx, S, P)
            assert full.nw == rgb384.size[0] and full.nh == rgb384.size[1]
            assert torch.equal(to_canvas(rgb384, full), full.canvas(img)), "Full384 RGB is not the array LB.canvas normalises"
            c1, c0 = to_canvas(c1_rgb(rgb384, full, W, H, bx, crop), crop), crop.canvas(img.crop(bx))
            pad = torch.ones(S, S, dtype=torch.bool)
            pad[crop.oy:crop.oy + crop.nh, crop.ox:crop.ox + crop.nw] = False
            assert c1[:, pad].eq(0).all() and c0[:, pad].eq(0).all(), "padding is not 0 in normalised space"
            rgb = (c1[:, ~pad] * STD_T.view(3, 1) + MEAN_T.view(3, 1)).numpy() * 255        # one inverse must land on integers: normalised exactly once
            max_inv_error = max(max_inv_error, float(np.abs(rgb - rgb.round()).max()))
            assert rgb.min() > -1e-3 and rgb.max() < 255 + 1e-3
            x0, y0, x1, y1 = box_384(full, W, H, bx)
            assert 0 <= x0 < x1 <= full.nw and 0 <= y0 < y1 <= full.nh
            i, j = np.divmod(np.arange(576), 24)                                             # C1 cell centres -> Full384 canvas = the Full384 ROI-read grid
            X = full.ox + x0 + ((j + .5) * P - crop.ox) * (x1 - x0) / crop.nw
            Y = full.oy + y0 + ((i + .5) * P - crop.oy) * (y1 - y0) / crop.nh
            max_grid_error = max(max_grid_error, float(np.abs(np.stack([X, Y], 1) / (S / 2) - 1 - grid).max()))
            r["rois"] += 1
    assert max_inv_error < 1e-3 and max_grid_error < 1e-4, (max_inv_error, max_grid_error)
    r.update(images=len(groups), max_inverse_normalisation_error_uint8=max_inv_error, max_grid_error=max_grid_error, full384_is_student_canvas=True, padding_zero_in_c0_and_c1=True)
    return r


def identity(M, n_img):
    """If the image is already at Full384 the box is integral and C1 must equal C0 bit for bit."""
    n, groups = 0, list(M.groupby("image_id"))[:n_img]
    for _, g in groups:
        src = Image.open(g.path.iloc[0]).convert("RGB")
        k = S / max(src.size)
        W, H = max(1, round(src.size[0] * k)), max(1, round(src.size[1] * k))
        img = src.resize((W, H), resample=Image.BICUBIC)
        full = LB(W, H, S, P)
        assert (full.nw, full.nh) == (W, H)
        rgb384 = full_rgb(img, full)
        rows = g.head(4)
        for row in rows.itertuples(index=False):
            bx = (int(row.l * k), int(row.t * k), max(int(row.l * k) + 1, min(W, int(np.ceil(row.r * k)))), max(int(row.t * k) + 1, min(H, int(np.ceil(row.b * k)))))
            crop = LB(bx[2] - bx[0], bx[3] - bx[1], S, P)
            assert torch.equal(to_canvas(c1_rgb(rgb384, full, W, H, bx, crop), crop), crop.canvas(img.crop(bx))), bx
            n += 1
    return dict(identity_rois=n, identity_bit_exact=True)


def constant():
    """A solid image must stay exactly solid: padding or a normalised value would show up as another colour."""
    W, H = 640, 427
    img = Image.new("RGB", (W, H), (255, 0, 0))
    full = LB(W, H, S, P)
    rgb384 = full_rgb(img, full)
    for bx in [(0, 0, 3, 3), (0, 0, 640, 427), (637, 424, 640, 427), (100, 200, 101, 201), (320, 0, 323, 427)]:
        crop = LB(bx[2] - bx[0], bx[3] - bx[1], S, P)
        a = np.asarray(c1_rgb(rgb384, full, W, H, bx, crop))
        assert (a == (255, 0, 0)).all(), bx
    return dict(constant_image_exact=True)


def samples(M, out, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    M = M.copy()
    M["bin"] = [size_bin(a, b, c, d) for a, b, c, d in zip(M.bw, M.bh, M.W, M.H)]
    rng, rows = np.random.default_rng(seed), []
    for level, name in [(lv, n) for lv in ("object", "part") for n, _, _ in BINS]:
        s = M[(M.level == level) & (M.bin == name)]
        for i in rng.choice(len(s), min(3, len(s)), replace=False):
            rows.append(s.iloc[int(i)])
    canvas = Image.new("RGB", (3 * 240 + 40, len(rows) * 250 + 10), (255, 255, 255))
    meta = []
    for k, row in enumerate(rows):
        img = Image.open(row.path).convert("RGB")
        W, H = img.size
        bx = (int(row.l), int(row.t), int(row.r), int(row.b))
        full, crop = LB(W, H, S, P), LB(bx[2] - bx[0], bx[3] - bx[1], S, P)
        c1 = c1_rgb(full_rgb(img, full), full, W, H, bx, crop)
        c0 = img.crop(bx).resize((crop.nw, crop.nh), resample=Image.BICUBIC)
        a, b = np.asarray(c0, np.float32), np.asarray(c1, np.float32)
        diff = Image.fromarray(np.uint8(np.clip(np.abs(a - b) * 4, 0, 255)))
        for j, panel in enumerate((c0, c1, diff)):
            p = Image.new("RGB", (S, S), (124, 116, 104))     # the normalised-0 padding colour, for display
            p.paste(panel, (crop.ox, crop.oy))
            p.save(out / f"{k:02d}_{row.level}_{row.bin}_{['c0', 'c1', 'diff4x'][j]}.png")
            canvas.paste(p.resize((240, 240), Image.NEAREST), (10 + j * 250, 5 + k * 250))
        meta.append(dict(k=k, ann_id=int(row.ann_id), level=row.level, size_bin=row.bin, full384_w=(bx[2] - bx[0]) * full.nw / W, full384_h=(bx[3] - bx[1]) * full.nh / H, mse=float(np.mean(((a - b) / 255) ** 2))))
    canvas.save(out / "grid_c0_c1_diff4x.png")
    (out / "samples.json").write_text(json.dumps(meta, indent=1))


def one_image(group):
    rows = []
    path, items = group
    img = Image.open(path).convert("RGB")
    W, H = img.size
    full = LB(W, H, S, P)
    rgb384 = full_rgb(img, full)
    for ann_id, split, level, bw, bh, l, t, r, b in items:
        bx = (int(l), int(t), int(r), int(b))
        crop = LB(bx[2] - bx[0], bx[3] - bx[1], S, P)
        x0, y0, x1, y1 = box_384(full, W, H, bx)
        a = np.asarray(img.crop(bx).resize((crop.nw, crop.nh), resample=Image.BICUBIC), np.float64) / 255
        c = np.asarray(c1_rgb(rgb384, full, W, H, bx, crop), np.float64) / 255
        assert a.shape == c.shape
        mse = float(np.mean((a - c) ** 2))
        rows.append(dict(ann_id=int(ann_id), split=split, level=level, size_bin=size_bin(bw, bh, W, H), w384=x1 - x0, h384=y1 - y0, crop_nw=crop.nw, crop_nh=crop.nh, mse=mse, psnr=float(10 * np.log10(1 / mse)) if mse > 0 else np.inf, round_collapse=bool(round(x1) <= round(x0) or round(y1) <= round(y0))))
    return rows


def rgb(out, procs):
    groups = []
    for split in ("val", "test"):
        M = pd.read_parquet(f"{DATA}/manifest_{split}.parquet")
        for path, g in M.groupby("path"):
            groups.append((path, list(zip(g.ann_id, g.split, g.level, g.bw, g.bh, g.l, g.t, g.r, g.b))))
    with Pool(procs) as pool:
        df = pd.DataFrame([row for rows in pool.imap_unordered(one_image, groups, chunksize=64) for row in rows])
    df.to_parquet(out / "rgb_rois.parquet")
    bins = [n for n, _, _ in BINS] + ["all"]
    table = []
    for (split, level), g in df.groupby(["split", "level"]):
        for name in bins:
            s = g if name == "all" else g[g.size_bin == name]
            finite = s.psnr[np.isfinite(s.psnr)]
            table.append(dict(split=split, level=level, size_bin=name, rois=len(s), mse_mean=s.mse.mean(), psnr_of_mean_mse=float(10 * np.log10(1 / s.mse.mean())), psnr_mean=finite.mean(), psnr_median=s.psnr.median(), full384_w_mean=s.w384.mean(), full384_h_mean=s.h384.mean(), lt1px=int(((s.w384 < 1) | (s.h384 < 1)).sum()), lt2px=int(((s.w384 < 2) | (s.h384 < 2)).sum()), round_collapse=int(s.round_collapse.sum()), mse_zero=int((s.mse == 0).sum())))
    pd.DataFrame(table).to_csv(out / "rgb_table.csv", index=False)
    print(pd.DataFrame(table).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("geometry", "rgb"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if a.cmd == "geometry":
        M = pd.read_parquet(f"{DATA}/manifest_val.parquet")
        result = dict(rois=canvas_equal(M, a.n), identity=identity(M, 80), constant=constant(), status="PASS")
        samples(M, out / "samples")
        (out / "geometry.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result), flush=True)
    else:
        rgb(out, a.procs)


if __name__ == "__main__":
    main()
