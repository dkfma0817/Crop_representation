"""Decoder cost: parameters, analytic FLOPs per ROI, measured latency (CUDA events, bf16, batch = 1 ROI, median of 30 after 10 warm-ups),
peak memory, and the per-image cost of the frozen encoder at 384 / 768 for reference. Measured on a shared GPU (other jobs running):
latencies are indicative only.

  python -m src.tsplit.efficiency --run <run> --out <run>/efficiency.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.tsplit.features import frozen_vit, student, teacher, up2
from src.tsplit.model import G, TokenSplit, flops_per_roi, param_report, rect_index, split_flops_per_image

DATA = "output/2026-11-07/paco_oracle_attr/data"


def timed(fn, reps=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    run = Path(a.run)
    vit = frozen_vit(dev)
    x = torch.randn(1, 3, 384, 384, device=dev)
    x768 = torch.randn(1, 3, 768, 768, device=dev)
    res = dict(params=param_report(), note="shared GPU; latency indicative only; batch 1 image / 1 ROI; bf16 autocast")
    res["encoder_ms_per_image"] = dict(dinov3_384=timed(lambda: student(vit, x)), dinov3_768=timed(lambda: teacher(vit, x768)))
    fin, ear = student(vit, x)
    res["b0_up2_ms_per_image"] = timed(lambda: up2(fin))
    M = pd.read_parquet(f"{DATA}/manifest_val.parquet")
    lat = {}
    for k in ("S1", "S2", "S3"):
        m = TokenSplit(k).to(dev).eval()
        m.load_state_dict(torch.load(run / "train" / f"{k}.pt", map_location=dev, weights_only=False)["model"])

        def cm():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return m.child_map(fin, ear)
        lat[k] = dict(split_ms_per_image=timed(cm), split_gflops_per_image=split_flops_per_image(k) / 1e9, per_roi={})
        cmap = cm()
        if k == "S1":
            continue
        for side in (1, 2, 3, 4, 6, 8, 12, 16, 24):
            r = torch.tensor([[0, 0, side, side]], device=dev)
            idx, cy, cx, ok = rect_index(r)
            img = torch.zeros(1, dtype=torch.long, device=dev)

            def rf():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return m.refine(cmap, img, idx, cy, cx, ok)
            torch.cuda.reset_peak_memory_stats()
            t = timed(rf)
            lat[k]["per_roi"][f"{side}x{side}"] = dict(children=4 * side * side, ms=t, gflops=flops_per_roi(k, 4 * side * side) / 1e9, peak_mib=torch.cuda.max_memory_allocated() / 2 ** 20)
    res["decoder"] = lat
    # ROI-size distribution of the PACO Val read rectangles (approximate: box footprint + 1 parent margin), for weighting
    s = 384 / np.maximum(M.W.values, M.H.values)
    side = np.maximum(np.ceil((M.r - M.l).values * s / 16) + 1, np.ceil((M.b - M.t).values * s / 16) + 1)
    res["paco_val_roi_parent_side_approx"] = dict(median=float(np.median(side)), p90=float(np.percentile(side, 90)), frac_le6=float(np.mean(side <= 6)))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
