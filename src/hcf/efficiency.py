"""Encode GFLOPs (MACs from module hooks: Linear, Conv2d, LayerNorm, attention matmul), stored-feature memory, and latency on one otherwise idle GPU.
Latency: bf16, batch of one 384 x 384 image, torch.cuda.synchronize, 10 warm-up + 50 measured repeats (median). Readout = ROIAlign + fuse + R2; k ROIs from one real PACO val image.
GT Crop = k crop encodes + R2. Reader weights = the val-selected epochs.   python -m src.hcf.efficiency --run <run>
"""
import argparse
import json
import statistics as st
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.hcf.model import SCALARS, FineBlock, WindowAttn, load_hcf, load_vit
from src.hcf.probe import roi_align
from src.hcf.infer import C0, load_reader
from src.ivc.bbdata import BBImageDS
import src.readout.common as RC

KS = (1, 5, 10, 20)


def flops(m, x):
    n = [0.0]
    h = []

    def lin(mod, i, o):
        n[0] += i[0].numel() / i[0].shape[-1] * mod.in_features * mod.out_features

    def conv(mod, i, o):
        n[0] += o.numel() * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]

    def ln(mod, i, o):
        n[0] += 2 * i[0].numel()

    def eva(mod, i, o):
        B, N, C = i[0].shape
        n[0] += B * N * N * C

    def win(mod, i, o):
        B, N, C = i[0].shape
        n[0] += B * N * N * C
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            h.append(mod.register_forward_hook(lin))
        elif isinstance(mod, nn.Conv2d):
            h.append(mod.register_forward_hook(conv))
        elif isinstance(mod, nn.LayerNorm):
            h.append(mod.register_forward_hook(ln))
        elif isinstance(mod, WindowAttn):
            h.append(mod.register_forward_hook(win))
        elif type(mod).__name__ == "EvaAttention":
            h.append(mod.register_forward_hook(eva))
    with torch.no_grad():
        m(x) if not hasattr(m, "forward_all") else m.forward_all(x)
    for k in h:
        k.remove()
    return n[0] / 1e9


def timed(fn, reps=60, warm=10):
    ts = []
    for i in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if i >= warm:
            ts.append(1e3 * (time.perf_counter() - t))
    return st.median(ts)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    run, sel = a.run, json.load(open(f"{a.run}/selection.json"))
    M = pd.read_parquet("output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet")
    iid = M.groupby("image_id").size().loc[lambda s: s >= 20].index[0]
    df = M[M.image_id == iid].iloc[:20].reset_index(drop=True)
    Y = np.zeros((len(df), 59), np.int8)
    bil = RC.to_dev(BBImageDS(df, Y, False, "bil", 384, 16)[[0]], dev)
    c0 = RC.to_dev(BBImageDS(df, Y, False, "c0", 384, 16)[[0]], dev)
    x = bil["x"]
    res = []

    def slice_b(b, k, c0mode=False):
        o = {kk: (v[:k] if torch.is_tensor(v) and kk in ("geo", "box4", "cov", "lv", "bil_grid", "img_index", "rows") else v) for kk, v in b.items()}
        return o

    orig = load_vit(dev).eval()
    for name, model, rn, rl, enc in (("Original", orig, "orig_r2", 12, "vit"), ("M", load_hcf(f"{run}/M/checkpoints/ema_last.pt", dev)[0], "M_r2Q", "Q", "hcf"),
                                     ("P", load_hcf(f"{run}/P/checkpoints/ema_last.pt", dev)[0], "P_r2fused", "Q", "hcf")):
        job = {"orig_r2": ("probe_orig", "r2"), "M_r2Q": ("probe_M", "r2Q"), "P_r2fused": ("probe_P", "r2fused")}[rn]
        r = load_reader(f"{run}/{job[0]}/{job[1]}/ep{sel[rn]['epoch']}.pt", [rl], dev)

        def enc_fn():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.forward_features(x) if enc == "vit" else model.forward_all(x)
        te = timed(enc_fn)
        o = enc_fn()
        row = dict(method=name, gflops=flops(model, x), stored_scalars=SCALARS.get(name if name != "Original" else "original"), encode_ms=te)
        row["feature_MB_fp32"], row["feature_MB_fp16"] = row["stored_scalars"] * 4 / 2 ** 20, row["stored_scalars"] * 2 / 2 ** 20
        for k in KS:
            b = slice_b(bil, k)
            b["img_index"] = torch.zeros(k, dtype=torch.long, device=dev)

            def read():
                bb = dict(b)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if enc == "vit":
                        A = roi_align(o[:, 5:].float(), 1, bb)
                        f = {12: A}
                    else:
                        A = roi_align(o["coarse"].float(), 1, bb)
                        mem = {m: roi_align(v.float(), o["grids"][m], bb) for m, v in o["mem"].items()}
                        f = {"Q": model.fuse(A, mem).float()}
                    bb["img_index"] = torch.arange(k, device=dev)
                    return r(f, bb)["logits"]
            tr = timed(read)
            row[f"readout_ms_k{k}"], row[f"readout_per_roi_ms_k{k}"], row[f"total_ms_k{k}"] = tr, tr / k, te + tr
        res.append(row)
    rc = load_reader(f"{C0}/r2/ep{sel['c0_r2']['epoch']}.pt", [12], dev)
    row = dict(method="GT Crop", gflops=None, stored_scalars=0, encode_ms=None, feature_MB_fp32=0.0, feature_MB_fp16=0.0)
    gf1 = flops(orig, x)
    row["gflops_per_roi"] = gf1
    for k in KS:
        b = slice_b(c0, k)
        X = c0["x"][:k]

        def full():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = orig.forward_features(X)[:, 5:].float()
                bb = dict(b)
                bb["img_index"] = torch.arange(k, device=dev)
                return rc({12: t}, bb)["logits"]
        t = timed(full)
        row[f"total_ms_k{k}"], row[f"readout_per_roi_ms_k{k}"] = t, t / k
    res.append(row)
    x336 = torch.randn(1, 3, 336, 336, device=dev)
    res.append(dict(method="F336 (Original @336, encode only)", gflops=flops(orig, x336), stored_scalars=21 * 21 * 768, encode_ms=timed(lambda: orig.forward_features(x336))))
    pd.DataFrame(res).to_csv(f"{run}/efficiency.csv", index=False)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
