"""Diagnostic (CPU, no training): how much do the G1 encoders move the final-LN patch tokens on ADE20K val?
First N val images (sorted), shorter side 384 + centre crop 384, fp32. Reports per-token L2 norm and per-token cosine to
Original, and the same for the per-image token-mean-removed features (spatial pattern).

  python -m src.g1eval.patch_stats --out output/2026-11-07/g1_leakcheck_ade20k/ade/patch_stats.json --n 64
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.g1eval.ade import ROOT, norm
from src.g1in.common import MODELS, load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(8)
    d = os.path.join(ROOT, "images", "validation")
    files = sorted(os.listdir(d))[: a.n]
    xs = []
    for f in files:
        im = Image.open(os.path.join(d, f)).convert("RGB")
        s = 384 / min(im.size)
        im = im.resize((max(384, round(im.size[0] * s)), max(384, round(im.size[1] * s))), Image.BILINEAR)
        L, T = (im.size[0] - 384) // 2, (im.size[1] - 384) // 2
        xs.append(norm(im.crop((L, T, L + 384, T + 384))))
    x = torch.stack(xs)
    tok = {}
    for m in MODELS:
        e = load_encoder(m, "cpu")
        with torch.no_grad():
            tok[m] = torch.cat([e.forward_features(x[i:i + 8])[:, 5:] for i in range(0, len(x), 8)]).float()
        del e
    o = tok["original"]
    oc = o - o.mean(1, keepdim=True)
    out = dict(n_images=len(files), norm={m: float(t.norm(dim=-1).mean()) for m, t in tok.items()})
    for m in ("g1pool", "g1cross"):
        t = tok[m]
        tc = t - t.mean(1, keepdim=True)
        out[f"tokcos_{m}"] = float(F.cosine_similarity(t, o, dim=-1).mean())
        out[f"centered_tokcos_{m}"] = float(F.cosine_similarity(tc, oc, dim=-1).mean())
        out[f"tokcos_p10_{m}"] = float(np.percentile(F.cosine_similarity(t, o, dim=-1).numpy(), 10))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
