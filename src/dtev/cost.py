"""Inference cost of the direct dense-teacher read vs F768 re-encoding (GPU; bf16 autocast).

FLOPs (2 x MAC of matmul / conv / SDPA, the torch flop-counter convention; LayerNorm, softmax, GELU not counted):
  masked, coarse cached (the 384 forward that F384 already runs is reused: per-layer K/V of 581 tokens), per ROI with n fine tokens:
      fine patch embed  n * 768 * 3 * p_T^2 * 2
      12 blocks         24 n d^2 (qkv 6, proj 2, mlp 16)  +  4 n (581 + n) d   (QK^T and AV over 581 cached + n own keys; the
                        implementation passes all 581 coarse keys with a key mask, so masked-out keys are counted)
  full (unrestricted single ROI), nothing can be cached (coarse rows attend to the fine tokens), per ROI with N = 581 + n tokens:
      coarse patch embed 576 * 768 * 768 * 2  +  fine patch embed  +  12 x (24 N d^2 + 4 N^2 d)
  F768: one forward of the native 768 canvas (2,309 tokens) per image, measured with the flop counter.
The analytic formula is validated against torch.utils.flop_counter on real fine_masked / fine_full calls.
PACO Test ROIs (all 131,899 GT boxes, no flip): ROI rectangle = box parents + 1 margin; added GFLOPs per ROI by the same box-size bins as
the size AP (bbox area at the 384 scale; tiny < 4 tokens, small 4-16, medium 16-64, large >= 64) and per image (sum over its ROIs).
Latency: batch = 1 ROI (masked: coarse K/V cached), CUDA events, median of 30 after 5 warm-up.

  python -m src.dtev.cost --out <json>
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.dtev.teacher import DT
from src.tsplit.features import frozen_vit
from src.tsplit.official_infer import DATA
from src.viz.c0_pca import LB

D, L, NC = 768, 12, 581
BINS = [("tiny_lt4_tokens", 0, 32 ** 2), ("small_4to16_tokens", 32 ** 2, 64 ** 2), ("medium_16to64_tokens", 64 ** 2, 128 ** 2), ("large_ge64_tokens", 128 ** 2, 1e12)]


def gf_masked(n, pt):
    return (n * D * 3 * pt * pt * 2 + L * (24 * n * D * D + 4 * n * (NC + n) * D)) / 1e9


def gf_full(n, pt):
    N = NC + n
    return (576 * D * D * 2 + n * D * 3 * pt * pt * 2 + L * (24 * N * D * D + 4 * N * N * D)) / 1e9


def counted(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 1e9


def timeit(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    vit = frozen_vit(dev)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    x = torch.randn(1, 3, 384, 384, device=dev)
    x768 = torch.randn(1, 3, 768, 768, device=dev)
    out = dict(setup="H200 (GPU 1), bf16 autocast; FLOPs = 2 x MAC (torch flop counter convention); latency batch 1, median of 30")

    def f384():
        with ac():
            vit.forward_features(x)

    def f768():
        with ac():
            vit.forward_features(x768)
    out["DINOv3_384_per_image"] = dict(gflops=counted(f384), ms=timeit(f384))
    out["F768_reencode_per_image"] = dict(gflops=counted(f768), ms=timeit(f768))
    out["C0_crop_reencode_per_ROI"] = dict(out["DINOv3_384_per_image"], note="one 384 forward per ROI")
    sides = (3, 5, 8, 14, 24)                           # parent rectangle side incl. margin (1-parent box -> 3; full image -> 24)
    for pt in (8, 4):
        dt = DT(vit, pt).to(dev)
        with ac():
            final, kv = dt.coarse(x)
            h0, _ = dt.embed0(x)
            femb = dt.fine_embed(x)
        rimg = torch.zeros(1, dtype=torch.long, device=dev)
        for s in sides:
            pr = torch.tensor([[0, 0, s, s]], device=dev)
            n = (s * dt.r) ** 2

            def fm():
                with ac():
                    dt.fine_masked(dt.fine_embed(x), kv, pr, rimg)

            def ff():
                with ac():
                    dt.fine_full(dt.embed0(x)[0], dt.fine_embed(x), pr, rimg)
            out[f"pt{pt}_{s}x{s}parents_{n}fine"] = dict(
                masked_coarse_cached=dict(gflops_analytic=gf_masked(n, pt), gflops_counted=counted(fm), ms=timeit(fm)),
                full_single_roi=dict(gflops_analytic=gf_full(n, pt), gflops_counted=counted(ff), ms=timeit(ff)))
            print(pt, s, json.dumps(out[f"pt{pt}_{s}x{s}parents_{n}fine"]), flush=True)
    # PACO Test ROI distribution
    M = pd.read_parquet(f"{DATA}/manifest_test.parquet")
    sc = 384 / np.maximum(M.W.values, M.H.values)
    area = M.bw.values * M.bh.values * sc ** 2
    lbs = {(w, h): LB(w, h, 384, 16) for w, h in set(zip(M.W.values.tolist(), M.H.values.tolist()))}
    ox = np.array([lbs[(w, h)].ox for w, h in zip(M.W, M.H)], float)
    oy = np.array([lbs[(w, h)].oy for w, h in zip(M.W, M.H)], float)
    nw = np.array([lbs[(w, h)].nw for w, h in zip(M.W, M.H)], float)
    nh = np.array([lbs[(w, h)].nh for w, h in zip(M.W, M.H)], float)
    boxc = torch.tensor(np.stack([ox + M.l.values * nw / M.W.values, oy + M.t.values * nh / M.H.values,
                                  ox + M.r.values * nw / M.W.values, oy + M.b.values * nh / M.H.values], 1), dtype=torch.float64)
    pr = DT.parent_rects(boxc, 1)
    npar = (pr[:, 2] * pr[:, 3]).numpy()
    dist = dict(n_roi=len(M), n_images=int(M.image_id.nunique()), parents_mean=float(npar.mean()), parents_median=float(np.median(npar)))
    f768 = out["F768_reencode_per_image"]["gflops"]
    f384 = out["DINOv3_384_per_image"]["gflops"]
    for pt in (8, 4):
        n = npar * (16 // pt) ** 2
        gm, gfu = gf_masked(n.astype(np.float64), pt), gf_full(n.astype(np.float64), pt)
        per = {}
        for name, lo, hi in BINS + [("all", 0, 1e12)]:
            m = (area >= lo) & (area < hi) if name != "all" else np.ones(len(M), bool)
            if name == "large_ge64_tokens":
                m = area >= lo
            per[name] = dict(n_roi=int(m.sum()), fine_tokens_mean=float(n[m].mean()), parents_mean=float(npar[m].mean()),
                             masked_added_gflops_per_roi=float(gm[m].mean()), full_gflops_per_roi=float(gfu[m].mean()),
                             masked_over_F768=float(gm[m].mean() / f768), full_over_F768=float(gfu[m].mean() / f768))
        img = pd.DataFrame(dict(image_id=M.image_id.values, gm=gm, gfu=gfu)).groupby("image_id").sum()
        dist[f"pt{pt}"] = dict(by_size=per, per_image=dict(
            rois_per_image_mean=float(len(M) / M.image_id.nunique()),
            masked_added_gflops_mean=float(img.gm.mean()), masked_added_gflops_median=float(img.gm.median()),
            masked_total_incl_384_mean=float(img.gm.mean() + f384), full_total_mean=float(img.gfu.mean()),
            F768_per_image=f768, masked_total_over_F768=float((img.gm.mean() + f384) / f768), full_total_over_F768=float(img.gfu.mean() / f768)))
    out["paco_test"] = dist
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(dist, indent=1), flush=True)


if __name__ == "__main__":
    main()
