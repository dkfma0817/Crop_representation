"""P1 / P2 inference cost (GPU 0, shared with other work -> indicative). CUDA events, bf16, median of 20 after 5 warm-ups.
  original frozen DINOv3 384 full image (with the blocks[7] hook) | DINOv3 768 native full image (F768 reference)
  block-8 state storage (bytes per image) | P1 / P2 decoder per ROI for K = 1, 4, 9, 36, 144, 576 parents (batch 1)
  P2 for 10 ROIs of 3 x 3 parents on one image (batched) | analytic FLOPs (x2 multiply-adds)

  python -m src.fte.efficiency --p1 <ckpt> --p2 <ckpt> --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.fte.models import P1, P2
from src.fte.train import student_h8
from src.tsplit.features import frozen_vit, teacher


def timed(fn, reps=20, warm=5):
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


def vit_block_flops(N, D=768):
    return 2 * N * (4 * D * D + 8 * D * D) + 2 * 2 * N * N * D


def p1_flops(K, W=384, D=768, depth=6):
    blk = 2 * K * (4 * W * W + 8 * W * W) + 2 * 2 * K * K * W
    return 2 * K * D * W + depth * blk + 2 * K * W * 4 * D + 2 * 4 * K * D * D


def p2_flops(K):
    N = 5 + 576 + 4 * K
    return 2 * K * 768 * 4 * 768 + 4 * vit_block_flops(N)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    vit = frozen_vit(dev)
    p1, p2 = P1(vit).to(dev).eval(), P2(vit).to(dev).eval()
    p1.load_state_dict(torch.load(a.p1, map_location=dev, weights_only=False)["model"])
    p2.load_state_dict(torch.load(a.p2, map_location=dev, weights_only=False)["model"])
    x, x768 = torch.randn(1, 3, 384, 384, device=dev), torch.randn(1, 3, 768, 768, device=dev)
    final, h8 = student_h8(vit, x)
    res = dict(note="shared GPU 0, bf16, batch 1 unless stated; indicative",
               dinov3_384_ms=timed(lambda: student_h8(vit, x)), dinov3_768_native_ms=timed(lambda: teacher(vit, x768)),
               dinov3_384_gflops=12 * vit_block_flops(581) / 1e9, dinov3_768_gflops=12 * vit_block_flops(2309) / 1e9,
               block8_state_bytes_per_image=int(h8.numel() * h8.element_size()), params=dict(P1=sum(p.numel() for p in p1.parameters() if p.requires_grad), P2=sum(p.numel() for p in p2.parameters() if p.requires_grad)))
    img = torch.zeros(1, dtype=torch.long, device=dev)
    per = {}
    for side in (1, 2, 3, 6, 12, 24):
        r = torch.tensor([[0, 0, side, side]], device=dev)

        def f1():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return p1(final, img, r)

        def f2():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return p2(h8, img, r)
        K = side * side
        per[f"{side}x{side}"] = dict(K=K, P1_ms=timed(f1), P1_gflops=p1_flops(K) / 1e9, P2_ms=timed(f2), P2_gflops=p2_flops(K) / 1e9, P2_tokens=5 + 576 + 4 * K)
    res["per_roi"] = per
    r10 = torch.tensor([[2 * k % 21, (3 * k) % 21, 3, 3] for k in range(10)], device=dev)
    i10 = torch.zeros(10, dtype=torch.long, device=dev)

    def f10a():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return p1(final, i10, r10)

    def f10b():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return p2(h8, i10, r10)
    res["ten_rois_3x3_one_image"] = dict(P1_ms=timed(f10a), P2_ms=timed(f10b), P1_gflops=10 * p1_flops(9) / 1e9, P2_gflops=10 * p2_flops(9) / 1e9)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
