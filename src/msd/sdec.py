"""S-series spatial ROI decoders (S1 hierarchical vs flat, S3 two-way vs one-way). All output Z (R, 576, 768) in
block-12 space on the 24 x 24 C0 crop grid, take the SAME inputs as H1 (frozen block 4/8/12 maps, fused
F = F12 + A4(F4) + A8(F8) with zero-init adapters A = models_mlr.Adapter) and have the SAME aux heads as H1.

Shared pieces (identical in every condition, built in the same order with seed 0):
  in_adapt A4/A8, in_proj LN(768) -> Linear(768, d), positional Fourier embeddings (models_mlr.Fourier) for grid
  positions (2-d) and cache-token coords (4-d: crop-frame x, y, cell w, h), output head
  Z = bilinear(F12) + sd12 * Linear_out(LN(h)) with Linear_out ZERO-init  (step 0 == the Bilinear baseline),
  aux heads Linear_d((Z - mu12) / sd12), d = 4, 8 (training-only).
Grid features at resolution r: the fused map sampled bilinearly at the r x r crop-cell centres (the 24 x 24 bil_grid is
affine in the cell centre, so its bilinear resize to r x r gives the exact r x r source points).

A0 flat  : 4 blocks [self-attn, cross-attn to the ROI+ring cache tokens, FFN x4], all on the 24 x 24 grid
A1 hier  : the same 4 blocks on 4x4 -> 8x8 -> 16x16 -> 24x24; between stages the state is bilinearly upsampled and the
           stage-r grid features (+ pos) are added (A0 adds them once, at its start). Same params, less compute.
B0 one-way: K = 16 box-conditioned queries, working map W (24 x 24, from grid features) and cache tokens; 3 stages.
           query: += CA(q, cache) ; += CA(q, W) ; += FFN ; and (compute/param control) += SA(q) ; += FFN2(q).  W fixed.
B1 two-way: same, but the two control modules act on the map: W += CA(W, q) ; W += FFN_w(W).  Same param count.
Both B: final readout W_out = W + CA_out(W, q) + FFN_out -> output head (B0 reads from its unchanged W0).
The frozen encoder maps ("cache") are never modified (checked: checksum before / after forward + backward).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.msd.common import MM, D, G, gather_roi, query_coords_t
from models.roi_reader import bilinear  # noqa: E402


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, kv, allow=None):
        R, Lq, d = x.shape
        Lk = kv.shape[1]
        q = self.q(x).view(R, Lq, self.h, -1).transpose(1, 2)
        k = self.k(kv).view(R, Lk, self.h, -1).transpose(1, 2)
        v = self.v(kv).view(R, Lk, self.h, -1).transpose(1, 2)
        m = None if allow is None else allow[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o(y.transpose(1, 2).reshape(R, Lq, d))


def ffn(d, r):
    return nn.Sequential(nn.Linear(d, r * d), nn.GELU(), nn.Linear(r * d, d))


class Block(nn.Module):
    """pre-LN: x += SA(x) ; x += CA(x, cache) ; x += FFN(x)."""

    def __init__(self, d, heads, r):
        super().__init__()
        self.n1, self.n2, self.n3, self.nk = (nn.LayerNorm(d) for _ in range(4))
        self.sa, self.ca, self.ff = Attn(d, heads), Attn(d, heads), ffn(d, r)

    def forward(self, x, kv, allow):
        h = self.n1(x)
        x = x + self.sa(h, h)
        x = x + self.ca(self.n2(x), self.nk(kv), allow)
        return x + self.ff(self.n3(x))


class SDecBase(nn.Module):
    arch = "base"

    def __init__(self, d=384, heads=6):
        super().__init__()
        self.d = d
        self.in_adapt = nn.ModuleDict({"4": MM.Adapter(), "8": MM.Adapter()})
        self.in_proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.four = MM.Fourier()
        nf = 2 * self.four.nf
        self.pos_grid = nn.Linear(2 * nf, d)
        self.pos_tok = nn.Linear(4 * nf, d)
        self.out_ln, self.out = nn.LayerNorm(d), nn.Linear(d, D)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.aux = nn.ModuleDict({"4": nn.Linear(D, D), "8": nn.Linear(D, D)})
        self.register_buffer("mu12", torch.zeros(D))
        self.register_buffer("sd12", torch.ones(D))
        self.register_buffer("P", query_coords_t())

    def set_stats(self, mu, sd):
        self.mu12.copy_(mu)
        self.sd12.copy_(sd)

    # ---------------------------------------------------------------- shared inputs
    def fused(self, feats):
        return feats[12] + self.in_adapt["4"](feats[4]) + self.in_adapt["8"](feats[8])

    def grid_feat(self, fmap, g, r):
        """fused full map (n_img,576,768) sampled at the r x r crop-cell centres -> (R, r*r, 768)."""
        bg = g["bil_grid"].float().view(-1, G, G, 2).permute(0, 3, 1, 2)
        if r != G:
            bg = F.interpolate(bg, size=(r, r), mode="bilinear", align_corners=False)
        maps = fmap.transpose(1, 2).reshape(fmap.shape[0], -1, G, G)[g["img_index"]]
        out = F.grid_sample(maps.float(), bg.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=False)
        return out.flatten(2).transpose(1, 2)

    def grid_pos(self, r, dev):
        c = (torch.arange(r, device=dev, dtype=torch.float32) + 0.5) / r
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        return self.pos_grid(self.four(torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)))

    def cache_tokens(self, fmap, g):
        kv = self.in_proj(gather_roi(fmap, g)) + self.pos_tok(self.four(g["coords"].float()))
        return kv, ~g["pad"]

    def head(self, h, feats, g):
        base = bilinear(feats[12], g)                                  # (R, 576, 768) Bilinear baseline
        return base + self.out(self.out_ln(h)).float() * self.sd12

    def aux_pred(self, Zn):
        return {d: self.aux[str(d)](Zn) for d in (4, 8)}

    def trunk_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("aux.")]


class FlatDec(SDecBase):
    arch = "flat"

    def __init__(self, d=384, heads=6, nblk=4, r=4, res=(24, 24, 24, 24)):
        super().__init__(d, heads)
        self.res = res
        self.blocks = nn.ModuleList([Block(d, heads, r) for _ in range(nblk)])

    def forward(self, feats, g, return_stages=False):
        fmap = self.fused(feats)
        kv, allow = self.cache_tokens(fmap, g)
        x, stages, prev = None, [], None
        for i, (blk, r) in enumerate(zip(self.blocks, self.res)):
            inp = self.in_proj(self.grid_feat(fmap, g, r)) + self.grid_pos(r, kv.device)
            if x is None:
                x = inp
            elif r != prev:                          # hierarchical step: upsample state + add the finer grid input
                R = x.shape[0]
                x = F.interpolate(x.transpose(1, 2).reshape(R, self.d, prev, prev), size=(r, r), mode="bilinear",
                                  align_corners=False).flatten(2).transpose(1, 2) + inp
            x = blk(x, kv, allow)
            prev = r
            if return_stages:
                stages.append(self._to24(x, r))
        Z = self.head(x if prev == G else self._to24(x, prev), feats, g)
        return (Z, [self.head(s, feats, g) for s in stages]) if return_stages else Z

    def _to24(self, x, r):
        if r == G:
            return x
        R = x.shape[0]
        return F.interpolate(x.transpose(1, 2).reshape(R, self.d, r, r), size=(G, G), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)


class HierDec(FlatDec):
    arch = "hier"

    def __init__(self, d=384, heads=6):
        super().__init__(d, heads, nblk=4, r=4, res=(4, 8, 16, 24))


class WayDec(SDecBase):
    """S3. two_way=False -> B0 (queries only), True -> B1 (queries and working map)."""

    def __init__(self, two_way, d=384, heads=6, K=16, stages=3, r=2):
        super().__init__(d, heads)
        self.two_way, self.K = two_way, K
        self.arch = "twoway" if two_way else "oneway"
        self.query = nn.Parameter(torch.zeros(1, K, d))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.box_mlp = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.st = nn.ModuleList()
        for _ in range(stages):
            self.st.append(nn.ModuleDict(dict(
                nq1=nn.LayerNorm(d), nk=nn.LayerNorm(d), ca_cache=Attn(d, heads),
                nq2=nn.LayerNorm(d), nw=nn.LayerNorm(d), ca_map=Attn(d, heads),
                nq3=nn.LayerNorm(d), ff=ffn(d, r),
                # control pair: B0 -> SA(q) + FFN2(q) on the queries; B1 -> CA(W <- q) + FFN_w(W) on the map
                nx1=nn.LayerNorm(d), nx1k=nn.LayerNorm(d), extra_attn=Attn(d, heads), nx2=nn.LayerNorm(d), extra_ff=ffn(d, r))))
        self.rn_w, self.rn_q, self.r_ca, self.rn_f, self.r_ff = nn.LayerNorm(d), nn.LayerNorm(d), Attn(d, heads), nn.LayerNorm(d), ffn(d, r)

    @staticmethod
    def box_vec(g):
        """ROI box on the full canvas from the crop-grid source points of the valid cells (+ log size)."""
        bg = g["bil_grid"].float()
        v = (g["valid_w"] > 0)[..., None]
        lo = torch.where(v, bg, torch.full_like(bg, 9.0)).amin(1)
        hi = torch.where(v, bg, torch.full_like(bg, -9.0)).amax(1)
        wh = (hi - lo).clamp(min=1e-3)
        return torch.cat([lo, wh, wh.log()], -1)

    def forward(self, feats, g, return_stages=False):
        fmap = self.fused(feats)
        kv, allow = self.cache_tokens(fmap, g)
        W = self.in_proj(self.grid_feat(fmap, g, G)) + self.grid_pos(G, kv.device)
        q = self.query.expand(W.shape[0], -1, -1) + self.box_mlp(self.box_vec(g))[:, None]
        stages = [W] if return_stages else []          # stage0 = initial working map W0
        for s in self.st:
            q = q + s["ca_cache"](s["nq1"](q), s["nk"](kv), allow)
            q = q + s["ca_map"](s["nq2"](q), s["nw"](W))
            q = q + s["ff"](s["nq3"](q))
            if self.two_way:
                W = W + s["extra_attn"](s["nx1"](W), s["nx1k"](q))
                W = W + s["extra_ff"](s["nx2"](W))
            else:
                h = s["nx1"](q)
                q = q + s["extra_attn"](h, s["nx1k"](q))
                q = q + s["extra_ff"](s["nx2"](q))
            if return_stages:
                stages.append(W)
        Wo = W + self.r_ca(self.rn_w(W), self.rn_q(q))
        Wo = Wo + self.r_ff(self.rn_f(Wo))
        Z = self.head(Wo, feats, g)
        return (Z, stages) if return_stages else Z


def build(arch):
    if arch == "flat":
        return FlatDec()
    if arch == "hier":
        return HierDec()
    if arch == "oneway":
        return WayDec(two_way=False)
    if arch == "twoway":
        return WayDec(two_way=True)
    raise ValueError(arch)
