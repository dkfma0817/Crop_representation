"""Encoder precision choice (smoke step, BEFORE any probe training; one GPU). Pre-registered rule:
  reference = fp32 with TF32 disabled. Candidates in order: 'tf32' (fp32 activations, TF32 matmul / conv), then 'fp32'.
  choose tf32 iff for BOTH encoders the per-token cosine to the reference has min >= 0.999 over the check tokens;
  otherwise fp32. bf16 / fp16 (autocast) are measured for the record only (CLIP ViT-L massive activations).
Also records encoder throughput (images / s, batch 32) per precision. Same choice for all three conditions.
  CUDA_VISIBLE_DEVICES=<gpu> python -m src.maskemb.prec_check --n_img 64
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.maskemb import common as MC

DATA = "output/2026-11-07/paco_oracle_attr/data"


def run(vit, x, mode):
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    torch.backends.cudnn.allow_tf32 = mode == "tf32"
    with torch.no_grad():
        if mode in ("bf16", "fp16"):
            return MC.tokens(vit, x, amp=True, dtype=torch.bfloat16 if mode == "bf16" else torch.float16, chunk=32)
        return MC.tokens(vit, x, amp=False, chunk=32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_img", type=int, default=64)
    a = ap.parse_args()
    CP = MC.set_clip_norm()
    M = pd.read_parquet(os.path.join(DATA, "manifest_val.parquet"))
    rng = np.random.default_rng(1)
    paths = M.drop_duplicates("image_id").sample(a.n_img, random_state=1).path.tolist()
    xs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        xs.append(CP.LB(im.size[0], im.size[1], MC.S, MC.P).canvas(im))
    x = torch.stack(xs).cuda()
    out = dict(n_images=a.n_img, gpu=torch.cuda.get_device_name(), modes={})
    for enc in MC.ENCODERS:
        vit, _ = MC.build_encoder(enc, "cuda")
        ref = run(vit, x, "fp32")
        out["modes"][enc] = {}
        for mode in ("tf32", "fp16", "bf16", "fp32"):
            t = run(vit, x, mode)
            c = torch.nn.functional.cosine_similarity(t, ref, dim=-1).flatten().cpu()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                run(vit, x[:32], mode)
            torch.cuda.synchronize()
            out["modes"][enc][mode] = dict(cos_min=float(c.min()), cos_p001=float(c.quantile(0.001)), cos_mean=float(c.mean()),
                                           frac_below_0999=float((c < 0.999).float().mean()), img_per_s=round(96 / (time.time() - t0), 1))
        del vit
        torch.cuda.empty_cache()
    ok = all(out["modes"][e]["tf32"]["cos_min"] >= 0.999 for e in MC.ENCODERS)
    out["rule"] = "tf32 iff min token cos to fp32 (TF32 off) >= 0.999 for both encoders, else fp32"
    out["choice"] = "tf32" if ok else "fp32"
    json.dump(out, open(MC.run_path("sanity", "precision.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
