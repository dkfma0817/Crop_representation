"""Shared pieces: RGB reconstruction head, data, metrics (RAE-style crop pilot).

Head (identical for every input representation and backbone; ~13M params):
  input  (B,576,768) dense crop feature -> per-channel z-score with the TRAINING representation's statistics
         (valid cells of head_train) -> cells with zero content coverage (letterbox pad) set to 0
  Linear 768->384 + learned 24x24 position embedding -> 4 pre-LN Transformer blocks (384, 6 heads, MLP x4)
  -> 384x24x24 -> [ResBlock, nearest x2, conv] x3 (384->192->96->48 ch; 24->48->96->192 px) -> ResBlock -> conv3x3 -> 3
  -> sigmoid = RGB in [0,1] at 192x192
Loss: masked RGB L1 (content pixels only) + 1.0 * LPIPS-VGG on the composite pred*mask + gt*(1-mask) (inputs in [-1,1]).
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import RAE_PILOT as PILOT  # earlier project (not on this server)
CACHE = os.path.join(PILOT, "_cache")
sys.path.insert(0, os.path.join(PILOT, "_pylib"))
REPS = ("crop", "bilinear", "R00", "L25", "L50")


class Block(nn.Module):
    def __init__(self, d=384, h=6):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        y = self.n1(x)
        x = x + self.att(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
                               nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.f(x)


class RGBHead(nn.Module):
    def __init__(self, d_in=768, d=384, depth=4):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, 576, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[Block(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        chans = [d, 192, 96, 48]
        up = []
        for a, b in zip(chans[:-1], chans[1:]):
            up += [Res(a), nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(a, b, 3, padding=1)]
        self.up = nn.Sequential(*up, Res(48), nn.GroupNorm(8, 48), nn.SiLU(), nn.Conv2d(48, 3, 3, padding=1))

    def set_stats(self, mu, sd):
        self.mu.copy_(mu)
        self.sd.copy_(sd)

    def forward(self, f, vw):
        x = (f.float() - self.mu) / self.sd
        x = x * (vw > 0)[..., None]
        x = self.norm(self.blocks(self.inp(x) + self.pos))
        x = x.transpose(1, 2).reshape(x.shape[0], -1, 24, 24)
        return torch.sigmoid(self.up(x))


def channel_stats(X, vw, n=3000):
    """Per-channel mean / std over valid cells of the first n rows (deterministic)."""
    x = torch.as_tensor(np.asarray(X[:n]), dtype=torch.float32)
    m = torch.as_tensor(vw[:n] > 0)
    v = x[m]
    return v.mean(0), v.std(0) + 1e-6


def load_split(bb, split, rep):
    return np.load(os.path.join(CACHE, bb, f"{split}_{rep}.npy"), mmap_mode="r")


def load_rgb(split):
    return np.load(os.path.join(CACHE, f"rgb_{split}.npy"), mmap_mode="r"), np.load(os.path.join(CACHE, f"mask_{split}.npy"), mmap_mode="r")


def to_img(rgb_u8, mask, dev):
    y = torch.as_tensor(np.asarray(rgb_u8), device=dev).permute(0, 3, 1, 2).float() / 255.0
    m = torch.as_tensor(np.asarray(mask), device=dev)[:, None].float()
    return y, m


def masked_l1(p, y, m):
    return ((p - y).abs() * m).sum((1, 2, 3)) / (3 * m.sum((1, 2, 3)))


def composite(p, y, m):
    return p * m + y * (1 - m)


# ------------------------------------------------------------------------------------------------ metrics
def rgb_to_lab(x):
    """x (B,3,H,W) sRGB in [0,1] -> CIELAB (D65)."""
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
                     device=x.device)
    xyz = torch.einsum("ij,bjhw->bihw", M, lin) / torch.tensor([0.95047, 1.0, 1.08883], device=x.device)[None, :, None, None]
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-12) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def grad_energy(x, m):
    """Mean |finite-difference gradient| of luminance over pixel pairs fully inside the content mask."""
    g = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    mm = m[:, 0]
    dx, mx = (g[:, :, 1:] - g[:, :, :-1]).abs(), mm[:, :, 1:] * mm[:, :, :-1]
    dy, my = (g[:, 1:] - g[:, :-1]).abs(), mm[:, 1:] * mm[:, :-1]
    return ((dx * mx).sum((1, 2)) + (dy * my).sum((1, 2))) / (mx.sum((1, 2)) + my.sum((1, 2))).clamp_min(1)


def hp_corr(p, y, m, k=7):
    """Pearson correlation of high-pass luminance (x - box blur k) between pred and GT over content pixels (eroded by k//2):
    high only if the predicted fine detail is where the GT detail is (hallucinated texture -> energy without correlation)."""
    lum = lambda x: (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    hp = lambda x: lum(x) - F.avg_pool2d(lum(x), k, 1, k // 2, count_include_pad=False)
    mm = (F.avg_pool2d(m, k, 1, k // 2) > 0.999).float()
    a, b = hp(p), hp(y)
    n = mm.sum((1, 2, 3)).clamp_min(1)
    a = a - (a * mm).sum((1, 2, 3), keepdim=True) / n[:, None, None, None]
    b = b - (b * mm).sum((1, 2, 3), keepdim=True) / n[:, None, None, None]
    return (a * b * mm).sum((1, 2, 3)) / ((a * a * mm).sum((1, 2, 3)).sqrt() * (b * b * mm).sum((1, 2, 3)).sqrt()).clamp_min(1e-8)


def metrics(p, y, m, lp):
    """Per-ROI metrics on content pixels. p, y in [0,1]."""
    l1 = masked_l1(p, y, m)
    mse = ((p - y) ** 2 * m).sum((1, 2, 3)) / (3 * m.sum((1, 2, 3)))
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    lpv = lp(composite(p, y, m) * 2 - 1, y * 2 - 1).flatten()
    Lp, Ly = rgb_to_lab(p), rgb_to_lab(y)
    w = m.sum((1, 2, 3))
    mean_p, mean_y = (Lp * m).sum((2, 3)) / w[:, None], (Ly * m).sum((2, 3)) / w[:, None]
    de_mean = (mean_p - mean_y).norm(dim=1)                     # CIE76 of the mean colour
    de_pix = ((Lp - Ly).norm(dim=1, keepdim=True) * m).sum((1, 2, 3)) / w   # mean per-pixel CIE76
    pool = lambda z: F.avg_pool2d(z * m, 24) / F.avg_pool2d(m, 24).clamp_min(1e-6)   # 8x8 regional colour
    mw = (F.avg_pool2d(m, 24) > 0.5).float()
    de_8 = ((pool(Lp) - pool(Ly)).norm(dim=1, keepdim=True) * mw).sum((1, 2, 3)) / mw.sum((1, 2, 3)).clamp_min(1)
    chroma_p, chroma_y = mean_p[:, 1:].norm(dim=1), mean_y[:, 1:].norm(dim=1)
    ge_p, ge_y = grad_energy(p, m), grad_energy(y, m)
    hp_c = hp_corr(p, y, m)
    return dict(hp_corr=hp_c, l1=l1, psnr=psnr, lpips=lpv, de_mean=de_mean, de_pix=de_pix, de_8x8=de_8,
                chroma_pred=chroma_p, chroma_gt=chroma_y, grad_pred=ge_p, grad_gt=ge_y, hf_ratio=ge_p / ge_y.clamp_min(1e-6))


def cosine_lr(step, total, warm, base):
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
