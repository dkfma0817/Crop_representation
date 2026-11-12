"""FA inference cost, split into Full-image DINOv3 384 / fine stem + memory (per image) / ROI decoder (per ROI, by size) / R2 reader
(per ROI). CUDA events, bf16, median of 30 after 10 warm-ups, batch 1 image / 1 ROI (R2: a 128-ROI batch divided by 128).
Shared GPU -> indicative. Analytic multiply-add FLOPs (x2) for the trainable parts.

  python -m src.finedec.efficiency --ckpt <model.pt> --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.finedec.feats import student_pre
from src.finedec.model import D_FEAT, D_LOCAL, DEPTH, MLP, WIDTH, FineAware
from src.readout.models import Reader
from src.tsplit.features import frozen_vit
from src.tsplit.model import rect_index


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


def flops_image():
    ov = 2 * 48 * 48 * 768 * 3 * 16 * 16
    loc = 2 * 48 * 48 * D_LOCAL * 3 * 8 * 8
    fuse = 2 * 2304 * (D_FEAT + D_LOCAL) * WIDTH
    mem = 2 * 576 * D_FEAT * WIDTH + DEPTH * 2 * 576 * WIDTH * 2 * WIDTH
    return dict(frozen_overlap_patch_embed=ov, local_stem=loc, fuse=fuse, memory_kv=mem, total=ov + loc + fuse + mem)


def flops_roi(N, P):
    d = WIDTH
    blk = 2 * N * (3 * d * d + d * d) + 2 * 2 * N * N * d + 2 * N * (d * d + d * d) + 2 * 2 * N * 576 * d + 2 * N * 2 * MLP * d * d
    heads = 2 * P * (d * d + d * D_FEAT) + 2 * N * (d * d + d * D_FEAT)
    return DEPTH * blk + heads


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    vit = frozen_vit(dev)
    m = FineAware(vit).to(dev).eval()
    m.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    x = torch.randn(1, 3, 384, 384, device=dev)
    final, xpre = student_pre(vit, x)
    res = dict(note="shared GPU, bf16, batch 1; indicative only", params=dict(trainable=sum(p.numel() for p in m.parameters() if p.requires_grad), frozen_extra=sum(p.numel() for p in m.parameters() if not p.requires_grad)))
    res["dinov3_384_ms_per_image"] = timed(lambda: student_pre(vit, x))

    def side():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m.image_side(x, final)
    res["fine_stem_and_memory_ms_per_image"] = timed(side)
    res["fine_stem_and_memory_gflops_per_image"] = {k: v / 1e9 for k, v in flops_image().items()}
    fine, kvs = side()
    img = torch.zeros(1, dtype=torch.long, device=dev)
    per = {}
    for s in (1, 2, 3, 4, 6, 8, 12, 16, 24):
        r = torch.tensor([[0, 0, s, s]], device=dev)
        idx, cy, cx, ok = rect_index(r)

        def dec():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return m.decode(fine, kvs, xpre, img, idx, cy, cx, ok, ok, r[:, 0], r[:, 1], r[:, 3])
        per[f"{s}x{s}"] = dict(children=4 * s * s, ms=timed(dec), gflops=flops_roi(4 * s * s, s * s) / 1e9)
    res["roi_decoder"] = per
    reader = Reader("r2", layers=(12,)).to(dev).eval()
    R = 128
    b = dict(geo=torch.randn(R, 576, 5, device=dev), box4=torch.rand(R, 4, device=dev), cov=torch.ones(R, 576, device=dev), lv=torch.zeros(R, dtype=torch.long, device=dev),
             img_index=torch.arange(R, device=dev), side=384)
    f = {12: torch.randn(R, 576, 768, device=dev)}

    def rd():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return reader(f, b)
    res["r2_ms_per_roi"] = timed(rd) / R
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
