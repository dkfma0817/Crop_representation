"""P2 head: the PixelDiT-style head of crop_objectives.py with position-wise AdaLN in place of ROI cross-attention.

P1 needs no new class: CropPixelDiT already takes any square memory, so P1 feeds it 576 tokens instead of 64.
P2 keeps the patch embedding, timestep embedding, block count, compact pixel block and output of CropPixelDiT.
Only the conditioning of the patch blocks differs:
  m_j = (LayerNorm + Linear 768 -> 384)(M_j) + sincos position      padding cells (cov == 0): learned null token + position
  c_j = CondMLP(m_j, e_t)                                           one vector per 24 x 24 position
  shift, scale and gate of self-attention and FFN come from c_j     last Linear zero-initialised, as in ConditionalBlock
The ROI cross-attention (and its two LayerNorms) is removed.
"""
import math

import torch
from torch import nn

from src.hcf.model import sincos_2d
from src.pdit.crop_objectives import CompactPixelBlock, compact_pixels, patchify, uncompact_pixels


class SpatialBlock(nn.Module):
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


class SpatialPixelDiT(nn.Module):
    def __init__(self, dim=384, depth=2):
        super().__init__()
        self.dim = dim
        self.memory = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim))
        self.patch = nn.Linear(768, dim)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.null = nn.Parameter(torch.zeros(1, 1, dim))
        self.cond = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([SpatialBlock(dim) for _ in range(depth)])
        self.pixel_embed = nn.Linear(48, 64)
        self.pixel = CompactPixelBlock(dim, 64)
        self.output = nn.Linear(64, 48)
        self.last_c = None

    def condition(self, memory, cov, time):
        assert memory.shape[1] == 576, memory.shape
        valid = cov > 0
        assert valid.any(1).all()
        m = torch.where(valid[..., None], self.memory(memory), self.null.expand(memory.shape[0], 576, -1))
        m = m + sincos_2d(24, 24, self.dim, memory.device).reshape(1, 576, self.dim)
        return self.cond(torch.cat([m, time[:, None].expand(-1, 576, -1)], -1))

    def forward(self, noisy, t, memory, cov):
        freq = torch.exp(-math.log(10000) * torch.arange(self.dim // 2, device=t.device) / (self.dim // 2 - 1))
        angle = t[:, None] * freq[None] * 1000
        time = self.time(torch.cat([angle.cos(), angle.sin()], -1))
        c = self.condition(memory, cov, time)
        self.last_c = c.detach()
        x = self.patch(patchify(noisy)) + sincos_2d(24, 24, self.dim, noisy.device).reshape(1, 576, self.dim)
        for block in self.blocks:
            x = block(x, c)
        p = self.pixel_embed(compact_pixels(noisy)) + sincos_2d(4, 4, 64, noisy.device).reshape(1, 1, 16, 64)
        return uncompact_pixels(self.output(self.pixel(p, x)))
