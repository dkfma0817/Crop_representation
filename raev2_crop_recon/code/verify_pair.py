"""Step 1: is (dinov2-vit-b encoder, dinov2b-k1 decoder.pt, stats.pt) a compatible official pair?

ImageNet val images (5,000 extracted; first N sorted), official sample.py loading (centre square crop -> 256 BICUBIC -> [0,1]).
  * shapes / registers / LN affine of the official encoder
  * reconstruction PSNR / LPIPS for encoder variants: official default (registers, LN without affine), [woreg], [norm]
    -> the matching pair must be clearly best
  * empirical per-position mean / var of raw tokens vs stats.pt
  * decoder(raw) == official RAE.decode(normalise(raw)); BF16-autocast vs fp32 decoder output
  python code/verify_pair.py  -> results/verify_pair.json, results/viz/verify_pair_imagenet.png
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rae_common as RC  # noqa: E402

sys.path.insert(0, "/workspace/Areum/research/2026-10-26/rae_crop_pilot/_pylib")
import lpips  # noqa: E402

N = int(os.environ.get("NV", 256))
dev = "cuda"
RC.set_fp32()


def load_sample(path, res=256):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((res, res), Image.BICUBIC)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)


files = sorted(glob.glob("/dataset/_imagenet_val_extracted_5000/*.jpg"))[:N]
X = torch.stack([load_sample(f) for f in files])            # uint8 (N,3,256,256)
lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
dec = RC.official_decoder(dev)
mean, var = RC.load_stats(dev)
out = {"n_images": N, "decoder_params": sum(p.numel() for p in dec.parameters())}


def recon(enc, bf16=True, bs=32):
    rec, toks = [], []
    for i in range(0, N, bs):
        x = X[i:i + bs].to(dev).float()
        with torch.no_grad():
            z = enc(x).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                y = dec.unpatchify(dec(z, drop_cls_token=False).logits).float()
        rec.append(y.clamp(0, 1).cpu())
        toks.append(z.cpu())
    return torch.cat(rec), torch.cat(toks)


def score(rec):
    y = X.float() / 255
    mse = ((rec - y) ** 2).mean((1, 2, 3))
    psnr = (10 * torch.log10(1 / mse)).mean().item()
    with torch.no_grad():
        l = torch.cat([lp(rec[i:i + 32].to(dev) * 2 - 1, y[i:i + 32].to(dev) * 2 - 1).flatten().cpu() for i in range(0, N, 32)])
    return dict(psnr=psnr, lpips=l.mean().item(), l1=(rec - y).abs().mean().item())


variants = {}
for name in ("dinov2-vit-b", "dinov2-vit-b[woreg]", "dinov2-vit-b[norm]"):
    enc = RC.official_encoder(dev, name)
    m = enc.model
    info = dict(num_register_tokens=int(getattr(m, "num_register_tokens", 0)), norm_affine=bool(m.norm.elementwise_affine),
                pos_embed=list(m.pos_embed.shape), patch=enc.patch_size, embed_dim=enc.embed_dim)
    rec, toks = recon(enc)
    info.update(tokens_shape=list(toks.shape), **score(rec))
    if name == "dinov2-vit-b":
        info["bf16_vs_fp32"] = None
        rec32, _ = recon(enc, bf16=False)
        info["fp32"] = score(rec32)
        info["bf16_vs_fp32_maxabs"] = float((rec - rec32).abs().max())
        info["bf16_vs_fp32_meanabs"] = float((rec - rec32).abs().mean())
        # official normalise / decode round trip (RAE.encode / RAE.decode formulas)
        z = toks[:8].to(dev)
        zn = (z.transpose(1, 2).reshape(8, 768, 16, 16) - mean) / torch.sqrt(var + 1e-5)
        zr = (zn * torch.sqrt(var + 1e-5) + mean).view(8, 768, 256).transpose(1, 2)
        info["normalise_roundtrip_maxabs"] = float((zr - z).abs().max())
        # empirical stats (per position) vs stats.pt
        T = toks.transpose(1, 2).reshape(N, 768, 16, 16)
        em, ev = T.mean(0).to(dev), T.var(0).to(dev)
        info["stats_mean_corr"] = float(torch.corrcoef(torch.stack([em.flatten(), mean.flatten()]))[0, 1])
        info["stats_mean_absdiff_over_sd"] = float(((em - mean).abs() / var.sqrt()).mean())
        info["stats_var_ratio_median"] = float((ev / var).median())
        info["normalised_latent_mean_abs"] = float(((T.to(dev) - mean) / torch.sqrt(var + 1e-5)).mean(0).abs().mean())
        info["normalised_latent_var_mean"] = float(((T.to(dev) - mean) / torch.sqrt(var + 1e-5)).var(0).mean())
        info["raw_token_norm_mean"] = float(toks.norm(dim=-1).mean())
        info["raw_token_absmax"] = float(toks.abs().max())
        rec_def = rec
    variants[name] = info
    print(name, json.dumps(info), flush=True)
    del enc
    torch.cuda.empty_cache()
out["variants"] = variants
os.makedirs(os.path.join(RC.RES, "viz"), exist_ok=True)
json.dump(out, open(os.path.join(RC.RES, "verify_pair.json"), "w"), indent=1)
k = 8
row = lambda t: np.concatenate([np.asarray(t[i].permute(1, 2, 0) * (255 if t.dtype != torch.uint8 else 1)).astype(np.uint8) for i in range(k)], 1)
Image.fromarray(np.concatenate([row(X), row(rec_def)], 0)).save(os.path.join(RC.RES, "viz", "verify_pair_imagenet.png"))
print("DONE")
