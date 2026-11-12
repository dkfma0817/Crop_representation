"""CPU checks: build tables / exclusion, ROI alignment on the geometry, world-size invariance of the step plan.
  python -m src.crop_aware_pretrain.checks tables   (once)
  python -m src.crop_aware_pretrain.checks geometry
  python -m src.crop_aware_pretrain.checks plan
"""
import json
import os
import sys

import numpy as np

from src.crop_aware_pretrain import config as CF
from src.crop_aware_pretrain import data as DT
from src.g1x.data import roi_geom


def check_geometry(n=2000, K=3, S=384, P=16, seed=0):
    """Coarse cell (a, b) on the full canvas and on the crop canvas must be the SAME rectangle of the original image, and the area weights must sum
    to the rectangle area in cells on both grids (letterbox padding excluded)."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        W, H = int(rng.integers(200, 641)), int(rng.integers(150, 641))
        l = int(rng.integers(0, W - 8))
        t = int(rng.integers(0, H - 8))
        r = int(rng.integers(l + 8, W + 1))
        b = int(rng.integers(t + 8, H + 1))
        g = roi_geom(W, H, (l, t, r, b), S, P)
        ws, wt = DT.coarse_weights(g, K, S, P)
        X0, Y0, X1, Y1 = (float(x) for x in g["box_c"])
        c, fl = g["crop_lb"], g["full_lb"]
        xa, ya, xb, yb = DT.coarse_rects(X0, Y0, X1, Y1, K)
        ca, cc, cb, cd = DT.coarse_rects(c.ox, c.oy, c.ox + c.nw, c.oy + c.nh, K)
        ox0 = (xa - fl.ox) * W / fl.nw
        oy0 = (ya - fl.oy) * H / fl.nh
        oxc = l + (ca - c.ox) * (r - l) / c.nw
        oyc = t + (cc - c.oy) * (b - t) / c.nh
        assert np.abs(ox0 - oxc).max() < 1e-6 and np.abs(oy0 - oyc).max() < 1e-6, "coarse cell is not the same original-image rectangle"
        assert np.allclose(ws.sum(1), (xb - xa) * (yb - ya) / P ** 2, atol=1e-4)
        assert np.allclose(wt.sum(1), (cb - ca) * (cd - cc) / P ** 2, atol=1e-4)
        assert np.isclose(ws.sum(), g["wfull"].sum(), atol=1e-3), "coarse cells must partition the pooled-ROI weight"
    print(f"geometry OK: {n} random boxes, K={K}: full-canvas and crop-canvas coarse cells = same original-image rectangle; weights = exact areas; padding excluded")


def check_plan(cfg, steps=(0, 1, 7, 6000, 123456)):
    """The images and ROIs of a step are identical for every world size, the shards partition them, and normalisers do not depend on the world."""
    p = DT.StepPlan(os.path.join(cfg["run_dir"], "data", "roi_table.parquet"), cfg)
    for g in steps:
        items, info = p.step_items(g)
        items2, info2 = p.step_items(g)
        assert info == info2 and all(a["iid"] == b["iid"] and np.array_equal(a["rois"], b["rois"]) for a, b in zip(items, items2)), "plan is not deterministic"
        for world, accum in ((1, 2), (2, 1)):
            got = [x["iid"] for m in range(accum) for r in range(world) for x in p.shard(items, world, accum, r, m)]
            assert got == [x["iid"] for x in items], (world, accum)
        bins = [b for x in items for b in x["rois"]]
        assert info["n_roi"] == len(bins) and info["n_img"] == p.B
    print(f"plan OK: {p.n} images, {p.spe} steps/epoch, steps {steps} identical and split identically for world 1 (accum 2) and world 2 (accum 1)")


def roi_stats(cfg, n=2000):
    """Sampling balance check: distribution of the ROI size bins actually drawn vs the raw annotation distribution."""
    p = DT.StepPlan(os.path.join(cfg["run_dir"], "data", "roi_table.parquet"), cfg)
    import pandas as pd
    t = pd.read_parquet(os.path.join(cfg["run_dir"], "data", "roi_table.parquet"))
    raw = np.bincount(t.bin.values, minlength=3) / len(t)
    drawn = np.zeros(3)
    sv = [0, 0]
    for g in range(n):
        items, _ = p.step_items(g)
        for x in items:
            bins = [int(np.digitize((b[2] - b[0]) * (b[3] - b[1]), cfg["data"]["area_bins_px2"], right=False)) for b in x["rois"]]
            drawn += np.bincount(bins, minlength=3)
            sv[0] += int(x["sv"].sum()); sv[1] += len(x["sv"])
    drawn = drawn / drawn.sum()
    rep = dict(raw_bin_fraction=raw.round(3).tolist(), drawn_bin_fraction_by_box_area=drawn.round(3).tolist(), spatial_valid_fraction=round(sv[0] / sv[1], 3), steps_checked=n)
    print(json.dumps(rep))
    return rep


def main():
    cfg = CF.load()
    cmd = sys.argv[1]
    if cmd == "tables":
        DT.build_roi_table(cfg, CF.dirs(cfg)["data"], workers=int(os.environ.get("WORKERS", 24)))
    elif cmd == "geometry":
        check_geometry()
    elif cmd == "plan":
        check_plan(cfg)
        print(DT.verify_exclusion(CF.dirs(cfg)["data"]))
    elif cmd == "rois":
        roi_stats(cfg)
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
