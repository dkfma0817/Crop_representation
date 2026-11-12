"""PixelDiT conditioning variants for the P1/P2 pilot. Baseline blocks are imported unchanged (src.siglip_pixel.crop_objectives is a byte copy of the DINOv3 heads, sha f7c4ff75).

  B   CropPixelDiT, memory = coverage-weighted 8x8 pool of the 24x24 crop-canvas ROI tokens (64 tokens)      the existing hybrid
  P1  CropPixelDiT, memory = the 24x24 crop-canvas ROI tokens (576 tokens); the head class itself is unchanged
  P2  SpatialCondPixelDiT: no ROI cross-attention; per-token adaLN from c_j = CondMLP(M_j, e_t); subpatch path unchanged
"""
import math

import torch
from torch import nn

from src.hcf.model import sincos_2d
from src.siglip_pixel.crop_objectives import CompactPixelBlock, CropPixelDiT, compact_pixels, patchify, pooled_grid, uncompact_pixels

GRID = 24


class SpatialBlock(nn.Module):
    """ConditionalBlock minus the ROI cross-attention. Scale, shift and gate are per token; last layer zero-initialised like the baseline adaLN."""

    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.n1, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, c):
        s1, c1, g1, s2, c2, g2 = self.mod(c).chunk(6, -1)
        a = self.n1(x) * (1 + c1) + s1
        x = x + g1.sigmoid() * self.self_attn(a, a, a, need_weights=False)[0]
        a = self.n3(x) * (1 + c2) + s2
        return x + g2.sigmoid() * self.ffn(a)


class SpatialCondPixelDiT(nn.Module):
    """Patch DiT with position-wise ROI conditioning -> the baseline compact pixel transformer -> RGB velocity."""

    def __init__(self, dim=384, depth=2):
        super().__init__()
        self.dim = dim
        self.memory = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim))
        self.patch = nn.Linear(768, dim)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([SpatialBlock(dim) for _ in range(depth)])
        self.pixel_embed = nn.Linear(48, 64)
        self.pixel = CompactPixelBlock(dim, 64)
        self.output = nn.Linear(64, 48)

    def forward(self, noisy, t, memory, cov):
        assert memory.shape[1] == GRID * GRID, memory.shape
        valid = cov > 0
        assert valid.any(1).all()
        # M_j = Project(Q_j) + PE_j; padding cells carry no ROI evidence, so they get a zero vector rather than features sampled outside the crop.
        m = (self.memory(memory) + sincos_2d(GRID, GRID, self.dim, memory.device).reshape(1, GRID * GRID, self.dim)) * valid[:, :, None].to(memory.dtype)
        freq = torch.exp(-math.log(10000) * torch.arange(self.dim // 2, device=t.device) / (self.dim // 2 - 1))
        angle = t[:, None] * freq[None] * 1000
        time = self.time(torch.cat([angle.cos(), angle.sin()], -1))
        c = self.cond(torch.cat([m, time[:, None].expand(-1, GRID * GRID, -1).to(m.dtype)], -1))
        x = self.patch(patchify(noisy)) + sincos_2d(GRID, GRID, self.dim, noisy.device).reshape(1, GRID * GRID, self.dim)
        for block in self.blocks:
            x = block(x, c)
        p = self.pixel_embed(compact_pixels(noisy)) + sincos_2d(4, 4, 64, noisy.device).reshape(1, 1, 16, 64)
        return uncompact_pixels(self.output(self.pixel(p, x)))


def build_head(condition):
    return SpatialCondPixelDiT() if condition == "P2" else CropPixelDiT()


def memory_for(condition, features, cov):
    """The only baseline-vs-P1 change: B pools 24x24 -> 8x8 with coverage weights, P1/P2 keep the 576 tokens."""
    if condition == "B":
        return pooled_grid(features, cov, 8)
    return features, cov
