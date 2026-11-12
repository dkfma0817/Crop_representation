"""Combined report: PACO de-duplicated R2 (paco/paco_dedup.json) + ADE20K linear segmentation (ade/), image-level paired
bootstrap of the dataset mIoU (resample val images, re-sum per-class intersection / union, B = 1000, seed 0).

  python -m src.g1eval.report --run output/2026-11-07/g1_leakcheck_ade20k
"""
import argparse
import json
import os

import numpy as np

from src.g1in.common import MODELS

NAME = {"original": "Original DINOv3", "g1pool": "G1-pool", "g1cross": "G1-cross Encoder Only"}


def miou_rows(I, U):
    i, u = I.sum(-2), U.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(u > 0, i / np.maximum(u, 1), np.nan)
    return 100 * np.nanmean(iou, axis=-1)


def f(s):
    return f"{s['point']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]{'*' if s['lo'] > 0 or s['hi'] < 0 else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = a.run
    z = np.load(os.path.join(R, "ade", "val_inter_union.npz"))
    I = {m: z[f"I_{m}"].astype(np.int64) for m in MODELS}
    U = {m: z[f"U_{m}"].astype(np.int64) for m in MODELS}
    n = len(I["original"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, (1000, n))
    pt = {m: float(miou_rows(I[m], U[m])) for m in MODELS}
    bs = {}
    for m in MODELS:                                         # same resampled image sets for every model
        bs[m] = np.array([miou_rows(I[m][k], U[m][k]) for k in idx])
    ci = {}
    for x, y in (("g1pool", "original"), ("g1cross", "original"), ("g1cross", "g1pool")):
        d = bs[x] - bs[y]
        ci[f"{x}-{y}"] = dict(point=pt[x] - pt[y], lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
    done = json.load(open(os.path.join(R, "ade", "done.json")))
    cfg = json.load(open(os.path.join(R, "ade", "config.json")))
    # per-class IoU change summary
    ci_cls = {}
    for m in ("g1pool", "g1cross"):
        io = lambda q: I[q].sum(0) / np.maximum(U[q].sum(0), 1)  # noqa: E731
        dlt = 100 * (io(m) - io("original"))
        ci_cls[m] = dict(n_classes_down=int((dlt < 0).sum()), n_classes_up=int((dlt > 0).sum()), median=float(np.median(dlt)),
                         worst5=[(int(c), round(float(dlt[c]), 2)) for c in np.argsort(dlt)[:5]])
    paco = json.load(open(os.path.join(R, "paco", "paco_dedup.json")))
    out = dict(ade=dict(mIoU=pt, ci=ci, per_class=ci_cls, n_val=n, train=done, cfg=cfg["cfg"]), paco=paco)
    json.dump(out, open(os.path.join(R, "results.json"), "w"), indent=1, default=float)
    L = ["## Table B: ADE20K linear segmentation (val 2,000 images, mIoU at original resolution)", "",
         "| Method | mIoU | Δ vs Frozen (95% CI) |", "|---|---:|---:|"]
    for m in MODELS:
        L.append(f"| {NAME[m]} | {pt[m]:.2f} | {'–' if m == 'original' else f(ci[m + '-original'])} |")
    L += ["", f"G1-cross − G1-pool: {f(ci['g1cross-g1pool'])}", "",
          "Per-class IoU change vs Original: " + "; ".join(f"{NAME[m]}: {v['n_classes_down']} classes down / {v['n_classes_up']} up, median {v['median']:+.2f}" for m, v in ci_cls.items()),
          "", f"train {done['train_time_s']} s, eval {done['eval_time_s']} s, iters {done['iters']}; encoder unchanged: {done['unchanged']}"]
    open(os.path.join(R, "results.md"), "w").write(open(os.path.join(R, "paco", "paco_dedup.md")).read() + "\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
