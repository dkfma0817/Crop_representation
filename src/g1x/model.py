"""G1-cross ROI cross-attention decoder (2026-11-06 pilot). No feature interpolation anywhere: the 24 x 24 ROI output is
produced by 576 position-specific queries that READ the full-image tokens with attention.

Inputs (per ROI r; canvas S = 384, patch P = 16, G = 24):
  full_tok (B_img, 576, 768)  final-norm patch tokens of the full-image encoder (CLS / registers dropped)
  roi_img  (R,)               ROI -> image index
  box_c    (R, 4)             ROI box on the full canvas, pixels [X0, Y0, X1, Y1]
  qrel     (R, 576, 2)        ROI-relative coordinate of every query cell = C0 crop-grid cell centre relative to the
                              crop content rect ((cx - ox) / nw, (cy - oy) / nh): query (i, j) <-> teacher crop token (i, j)
                              (content cells in [0, 1]; letterbox padding cells fall outside and are masked by the loss)
  fvalid   (B_img, 576)       full-canvas cells with image content (letterbox padding keys are masked; NO ROI mask)

Query  q_rq = MLP_q([Fourier(qrel), Fourier(abs canvas xy of the query point), log(bw/P), log(bh/P), aspect])
              + E_grid[q]   (learned 576 x 768 embedding, std 0.02)
       -> every ROI gets queries that depend on its position and size; the 576 queries differ from each other.
Key/Value  kv_rk = LN(tok_k + MLP_k([Fourier(abs xy of key cell), Fourier(squash(ROI-relative xy of key cell)),
              overlap(cell, box), inside flag, log(bw/P), log(bh/P)]))   -> all 576 tokens, inside AND outside the box.
Blocks  2 x [q = q + MHA(LN q, kv) ; q = q + FFN(LN q)]  (8 heads, d 768, FFN x4, pre-LN, SDPA)
Output  Z = Linear(LN q)  (R, 576, 768)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D, G = 768, 24


def fourier(x, nf, max_log2):
    """x (..., k) -> (..., k * (2 nf + 1)): [x, sin(2^f pi x), cos(2^f pi x)] for f in linspace(0, max_log2, nf)."""
    fr = (2.0 ** torch.linspace(0, max_log2, nf, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * fr
    return torch.cat([x, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)


def squash(x):
    return torch.sign(x) * torch.log1p(x.abs())


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.GELU(), nn.Linear(o, o))


class XBlock(nn.Module):
    def __init__(self, d=D, heads=8, ratio=4):
        super().__init__()
        self.h = heads
        self.ln_q, self.ln_kv, self.ln_m = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ratio * d), nn.GELU(), nn.Linear(ratio * d, d))

    def forward(self, q, kv, mask):
        """q (n, Q, d), kv (n, T, d), mask (n, T) bool True = attend."""
        n, Q, d = q.shape
        T = kv.shape[1]
        hd = d // self.h
        Qh = self.q(self.ln_q(q)).view(n, Q, self.h, hd).transpose(1, 2)
        K, V = self.kv(self.ln_kv(kv)).view(n, T, 2, self.h, hd).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(Qh, K, V, attn_mask=mask[:, None, None, :])
        q = q + self.o(att.transpose(1, 2).reshape(n, Q, d))
        return q + self.mlp(self.ln_m(q))


class ROIDecoder(nn.Module):
    NF_Q, NF_K = 16, 8

    def __init__(self, S=384, P=16, depth=2, heads=8):
        super().__init__()
        self.S, self.P = S, P
        dq = 2 * (2 * self.NF_Q + 1) * 2 + 3
        dk = 2 * (2 * self.NF_K + 1) * 2 + 4
        self.q_mlp, self.k_mlp = mlp(dq, D), mlp(dk, D)
        self.q_grid = nn.Parameter(torch.zeros(1, G * G, D))
        nn.init.trunc_normal_(self.q_grid, std=0.02)
        self.ln_k = nn.LayerNorm(D)
        self.blocks = nn.ModuleList([XBlock(D, heads) for _ in range(depth)])
        self.ln_out, self.out = nn.LayerNorm(D), nn.Linear(D, D)
        ci, cj = torch.div(torch.arange(G * G), G, rounding_mode="floor"), torch.arange(G * G) % G
        self.register_buffer("cell_c", torch.stack([(cj + 0.5) * P, (ci + 0.5) * P], -1).float(), persistent=False)  # (576, 2) px
        e = torch.arange(G + 1).float() * P
        self.register_buffer("edges", e, persistent=False)

    def queries(self, box_c, qrel):
        X0, Y0, X1, Y1 = box_c.unbind(-1)
        bw, bh = (X1 - X0).clamp_min(1e-3), (Y1 - Y0).clamp_min(1e-3)
        ax = (X0[:, None] + qrel[..., 0] * bw[:, None]) / self.S
        ay = (Y0[:, None] + qrel[..., 1] * bh[:, None]) / self.S
        size = torch.stack([torch.log(bw / self.P), torch.log(bh / self.P), torch.log(bw / bh)], -1)[:, None].expand(-1, qrel.shape[1], -1)
        f = torch.cat([fourier(qrel, self.NF_Q, 6.0), fourier(torch.stack([ax, ay], -1), self.NF_Q, 6.0), size], -1)
        return self.q_mlp(f) + self.q_grid

    def key_geo(self, box_c):
        """-> (R, 576, dk) geometry of every full-canvas key cell relative to the ROI box, + overlap (R, 576)."""
        X0, Y0, X1, Y1 = box_c.unbind(-1)
        bw, bh = (X1 - X0).clamp_min(1e-3), (Y1 - Y0).clamp_min(1e-3)
        c = self.cell_c
        absxy = (c / self.S)[None].expand(len(box_c), -1, -1)
        rx = (c[None, :, 0] - (X0 + X1)[:, None] / 2) / bw[:, None]
        ry = (c[None, :, 1] - (Y0 + Y1)[:, None] / 2) / bh[:, None]
        e = self.edges
        ox = (torch.minimum(e[None, 1:], X1[:, None]) - torch.maximum(e[None, :-1], X0[:, None])).clamp_min(0) / self.P   # (R, 24)
        oy = (torch.minimum(e[None, 1:], Y1[:, None]) - torch.maximum(e[None, :-1], Y0[:, None])).clamp_min(0) / self.P
        ov = (oy[:, :, None] * ox[:, None, :]).flatten(1)                                                               # (R, 576)
        size = torch.stack([torch.log(bw / self.P), torch.log(bh / self.P)], -1)[:, None].expand(-1, G * G, -1)
        f = torch.cat([fourier(absxy, self.NF_K, 5.0), fourier(squash(torch.stack([rx, ry], -1)), self.NF_K, 3.0),
                       ov[..., None], (ov > 0).float()[..., None], size], -1)
        return f, ov

    def forward(self, full_tok, roi_img, box_c, qrel, fvalid, chunk=64):
        outs = []
        for i in range(0, len(roi_img), chunk):
            sl = slice(i, i + chunk)
            bc, ri = box_c[sl].float(), roi_img[sl]
            q = self.queries(bc, qrel[sl].float())
            kg, _ = self.key_geo(bc)
            kv = self.ln_k(full_tok[ri] + self.k_mlp(kg))
            m = fvalid[ri].bool()
            for blk in self.blocks:
                q = blk(q, kv, m)
            outs.append(self.out(self.ln_out(q)))
        return torch.cat(outs)
