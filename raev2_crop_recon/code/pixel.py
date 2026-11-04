"""Pixel / feature loss terms and metrics shared by sanity, calibration, training and evaluation.

Pixel loss (B, C): on the 256 crop-letterbox canvas of the ORIGINAL GT crop (u8_canvas, BICUBIC), content pixels only:
  L1  = masked mean |D(Z_pred) - GT|                        (D(Z) unclamped: no hard clamp in the training path)
  LPIPS-VGG(composite(D(Z_pred) inside content, GT outside), GT), inputs mapped to [-1, 1]
Feature loss (A, C) = losses_mlr.loss_terms (MSE_w + 1.0 (1 - cos)_w on raw tokens, w = teacher content coverage).
Metrics: rcp.metrics definitions (l1, psnr, lpips, dE_mean, dE_8x8, hp_corr, hf_ratio ...) on clamped outputs, with the
8x8 regional pooling scaled to 256 px (pool 32 instead of 24 at 192).
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcp  # noqa: E402
import lpips  # noqa: E402
from losses_mlr import dense_cos, dense_mse  # noqa: E402


def load_lpips(dev):
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    return lp


def pix_terms(y, gt, m, lp):
    """Per-ROI (L1, LPIPS); gradient flows into y. LPIPS under BF16 autocast (as visual_recon_loss)."""
    l1 = rcp.masked_l1(y, gt, m)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lpv = lp(rcp.composite(y, gt, m) * 2 - 1, gt * 2 - 1).flatten().float()
    return l1, lpv


def feat_mse_cos(pred, target, w):
    return dense_mse(pred, target, w), dense_cos(pred, target, w)


def svar_ratio(pred, target, w):
    """Per ROI: mean over channels of the variance across content cells (w > 0), pred / teacher."""
    m = (w > 0).float()[..., None]
    n = m.sum(1).clamp_min(2)

    def v(x):
        mu = (x * m).sum(1) / n
        return (((x - mu[:, None]) ** 2) * m).sum(1).div(n - 1).mean(-1)
    return v(pred.float()) / v(target.float()).clamp_min(1e-8)


@torch.no_grad()
def rgb_metrics(p, y, m, lp):
    l1 = rcp.masked_l1(p, y, m)
    mse = ((p - y) ** 2 * m).sum((1, 2, 3)) / (3 * m.sum((1, 2, 3)))
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    lpv = torch.cat([lp(rcp.composite(p[i:i + 64], y[i:i + 64], m[i:i + 64]) * 2 - 1, y[i:i + 64] * 2 - 1).flatten() for i in range(0, len(p), 64)])
    Lp, Ly = rcp.rgb_to_lab(p), rcp.rgb_to_lab(y)
    w = m.sum((1, 2, 3))
    mean_p, mean_y = (Lp * m).sum((2, 3)) / w[:, None], (Ly * m).sum((2, 3)) / w[:, None]
    de_mean = (mean_p - mean_y).norm(dim=1)
    k = p.shape[-1] // 8
    pool = lambda z: F.avg_pool2d(z * m, k) / F.avg_pool2d(m, k).clamp_min(1e-6)
    mw = (F.avg_pool2d(m, k) > 0.5).float()
    de_8 = ((pool(Lp) - pool(Ly)).norm(dim=1, keepdim=True) * mw).sum((1, 2, 3)) / mw.sum((1, 2, 3)).clamp_min(1)
    ge_p, ge_y = rcp.grad_energy(p, m), rcp.grad_energy(y, m)
    return dict(l1=l1, psnr=psnr, lpips=lpv, de_mean=de_mean, de_8x8=de_8, hp_corr=rcp.hp_corr(p, y, m), hf_ratio=ge_p / ge_y.clamp_min(1e-6))


def save_grid(path, rows, m, labels):
    """rows: list of (n,3,H,W) in [0,1]; padding shown grey-dimmed; one row per representation."""
    out = []
    for r in rows:
        r = r.detach().float().cpu()
        mm = m.cpu()
        r = r * mm + (1 - mm) * (0.5 + 0.3 * r)
        out.append(np.concatenate([(x.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8) for x in r], 1))
    img = Image.fromarray(np.concatenate(out, 0))
    img.save(path)
    with open(os.path.splitext(path)[0] + "_rows.txt", "w") as f:
        f.write("\n".join(labels) + "\n")
