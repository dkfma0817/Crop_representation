"""RAEv2 port (2026-11-04): this file = 2026-10-26/multilevel_relation_decoder/code/models_mlr.py with ONE change, the import of
G, PATCH, S from rae_common (16x16 output grid of the RAE latent instead of 24x24). Every layer, init and the forward are unchanged.

Decoder = existing coordinate-conditioned MLP body (gen_decoders/code/models_g.py: CoordMLP, copied verbatim below so
that it runs on either backbone) + a zero-initialised bottleneck adapter on the full-image ROI(+ring) tokens:

  single-level (R00, R01):  F = F_final + Adapter(F_final)
  multi-level  (R10, R11):  F = F_final + Adapter(F_middle)
  Adapter = LN(768) -> Linear(768, 192) -> GELU -> Linear(192, 768)   (last Linear zero-init, 297,408 params)
  out     = CoordMLP(F, geometry)                                      (unchanged body, 9,863,424 params)

Same structure, parameter count and output dimension in both conditions; at step 0 F = F_final, so the output equals the
CoordMLP body output exactly (checked in sanity_mlr.py against the existing gen_decoders mlp_s0 checkpoint).
Geometry helpers are copies of gen_decoders/code/gcommon.py with G, PATCH, S taken from the active backbone (both 24 x 24;
S / 2 / PATCH = 12 for 336/14 and 384/16).
"""
import math

import torch
import torch.nn as nn

from src.rae.common import G, PATCH, S   # RAEv2 port: S=256, PATCH=16, G=16 (only change)

KMIN, KMAX = -3, 7          # models/previous_decoder.py Fourier
ADAPTER_HID = 192


class Fourier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("f", (2.0 ** torch.arange(KMIN, KMAX + 1, dtype=torch.float32)) * math.pi)
        self.nf = KMAX - KMIN + 1

    def forward(self, p):
        x = p[..., None] * self.f
        return torch.cat([x.sin(), x.cos()], -1).flatten(-2)


def roi_inside_mask(b):
    return b["inside"] & ~b["pad"]


def roi_summary(F_r, b):
    m = roi_inside_mask(b).float()[..., None]
    return (F_r * m).sum(1) / m.sum(1).clamp_min(1)


def box_vector(b):
    tok, ins = b["tok"], roi_inside_mask(b)
    i, j = tok // G, tok % G
    big = 10 ** 4
    j0, j1 = torch.where(ins, j, big).amin(1), torch.where(ins, j, -1).amax(1)
    i0, i1 = torch.where(ins, i, big).amin(1), torch.where(ins, i, -1).amax(1)
    cov = b["valid_w"].float().view(-1, G, G)
    total = cov.sum((1, 2))
    wide = cov.sum(2).amax(-1) >= cov.sum(1).amax(-1)
    short = total / G
    A = torch.where(wide, torch.full_like(short, G), short)
    B = torch.where(wide, short, torch.full_like(short, G))
    pw, ph = b["coords"][:, 0, 2], b["coords"][:, 0, 3]
    bw, bh = (A / G) / pw, (B / G) / ph
    return torch.stack([j0 / G, i0 / G, (j1 + 1) / G, (i1 + 1) / G, bw.clamp_min(1e-3).log(), bh.clamp_min(1e-3).log(),
                        (A / B).log()], -1).float()


def query_full_xy(b):
    g = b["bil_grid"].float()
    return (g + 1) * (S / 2) / PATCH - 0.5


def token_full_xy(b):
    tok = b["tok"]
    return torch.stack([tok % G, tok // G], -1).float()


class CoordMLP(nn.Module):
    """Verbatim copy of gen_decoders/code/models_g.py CoordMLP (state_dict compatible)."""
    name = "mlp"

    def __init__(self, d=768, k=4, hid=1536):
        super().__init__()
        self.k = k
        self.four = Fourier()
        nf = 2 * self.four.nf
        self.tok_ln = nn.LayerNorm(d)
        self.sum_ln = nn.LayerNorm(d)
        self.l_local = nn.Linear(k * d + k * 2 * nf, hid)
        self.l_glob = nn.Linear(d + 7, hid)
        self.l_pos = nn.Linear(4 * nf, hid)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, d))
        self.register_buffer("mean_T", torch.zeros(d))
        self.register_buffer("std_T", torch.ones(d))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def set_stats(self, mean, std):
        self.mean_T.copy_(mean)
        self.std_T.copy_(std)

    def neighbours(self, b):
        u = query_full_xy(b)
        tp = token_full_xy(b)
        d2 = (tp[:, None] - u[:, :, None]).pow(2).sum(-1).masked_fill(b["pad"][:, None], float("inf"))
        idx = d2.topk(self.k, dim=-1, largest=False).indices
        rel = torch.gather(tp[:, None].expand(-1, u.shape[1], -1, -1), 2, idx[..., None].expand(-1, -1, -1, 2)) - u[:, :, None]
        return idx, rel, u

    def forward(self, F_r, b, P):
        R, N, D = F_r.shape
        idx, rel, u = self.neighbours(b)
        f = self.tok_ln(F_r)
        flat = (idx + torch.arange(R, device=idx.device)[:, None, None] * N).reshape(-1)
        loc = f.reshape(R * N, D)[flat].view(R, idx.shape[1], self.k, D)
        loc = torch.cat([loc.flatten(2), self.four(rel).flatten(2)], -1)
        glob = torch.cat([self.sum_ln(roi_summary(F_r, b)), box_vector(b)], -1)
        pos = torch.cat([self.four(P)[None].expand(R, -1, -1), self.four(u / G)], -1)
        h = self.l_local(loc) + self.l_glob(glob)[:, None] + self.l_pos(pos)
        return self.mlp(h).float() * self.std_T + self.mean_T


class Adapter(nn.Module):
    def __init__(self, d=768, hid=ADAPTER_HID):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, d)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.ln(x))))


class MLRDecoder(nn.Module):
    def __init__(self, source="final"):
        super().__init__()
        assert source in ("final", "middle")
        self.source = source
        self.adapter = Adapter()
        self.body = CoordMLP()

    def forward(self, F_fin_r, F_mid_r, b, P):
        src = F_fin_r if self.source == "final" else F_mid_r
        F = F_fin_r + self.adapter(src).masked_fill(b["pad"][..., None], 0.0)
        return self.body(F, b, P)


def count(m):
    return sum(p.numel() for p in m.parameters())
