"""Readers on frozen DINOv2 tokens. Every reader = (layer fusion) + R2-style geometry-conditioned cross-attention.

Shared with the earlier R2 (paco_region_reader common_rr.AttnReader, unchanged pieces): geo = [rel_x, rel_y, cov,
log(bw/14), log(bh/14)], geo_mlp / box_mlp (5|4 -> 768 -> 768), learned query + box embedding, kv = LN(token + geo_mlp(geo)),
CrossBlock (8 heads, MLP x2, pre-LN), LN -> level heads Linear(768 -> 59) x 2. Token count T is free (576 at 336, 2304 at 672).

  r2      : 2 cross blocks, mask cov > 0. layers=[12] -> earlier "F336 R2"; layers=ML_LAYERS -> queue BASELINE ml_r2
            (per-layer LN on block 4/8/12 outputs, concat -> Linear(3*768 -> 768) per token)
  E08 (all final-output loss only, every step's prediction recorded):
    rep   : ONE cross block applied T times with shared weights (+ step embedding)      (T=1: read once; T=3: read 3x)
    ffn   : the same block reads ONCE, then its own FFN (ln_m + mlp, shared) is applied T-1 more times without
            re-reading the features = same parameters and extra compute as rep T, no repeated reading
  E09:
    ads   : r2 + K query-conditioned sampling points per block (box coords, expanded box [-0.25, 1.25]^2), bilinear
            sampling of the fused map, points initialised exactly on the fixed grid
    fix   : identical, points fixed on the grid (offset predictor absent; 2 x Linear(768, 2K) = 0.11M params fewer)
  E11 (context = tokens outside the ROI box, cov == 0):
    concat: ONE stream, 2 blocks over [local tokens (cov > 0) ; all 576 tokens] + token-type embedding  (= simple join)
    gate  : local stream (2 blocks, cov > 0) and context stream (2 blocks, all tokens) read SEPARATELY, per-channel
            sigmoid gate g(local, context, box size) -> g * local + (1 - g) * context
    ctx   : one stream over all tokens (ablation, = earlier R3 on the same features)
    ctx_mode at eval: 'remove' (context part restricted to ROI tokens) / 'swap' (outside-ROI tokens taken from another
    image of the batch at the same positions; ROI tokens kept)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.readout.common import D, ML_LAYERS, PATCH, RR


class Fuse(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = tuple(layers)
        if len(self.layers) > 1:
            self.ln = nn.ModuleList([nn.LayerNorm(D) for _ in self.layers])
            self.proj = nn.Linear(D * len(self.layers), D)

    def forward(self, feats):
        if len(self.layers) == 1:
            return feats[self.layers[0]]
        return self.proj(torch.cat([ln(feats[l]) for ln, l in zip(self.ln, self.layers)], -1))


def grid_points(k):
    n = int(round(math.sqrt(k)))
    assert n * n == k
    c = (torch.arange(n) + 0.5) / n
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)          # (k, 2) box coords in (0, 1)


def ctx_tokens(tok_img, b, ctx_mode):
    """Per-ROI token sets: own image's tokens; for 'swap' the outside-ROI tokens come from the next image in the batch."""
    tok = tok_img[b["img_index"]]
    if ctx_mode == "swap":
        other = tok_img[(b["img_index"] + 1) % tok_img.shape[0]]
        tok = torch.where((b["cov"] > 0)[..., None], tok, other)
    return tok


class Reader(nn.Module):
    def __init__(self, kind, layers=ML_LAYERS, T=2, K=36, heads=8, mlp_ratio=2):
        super().__init__()
        self.kind, self.T, self.K = kind, T, K
        self.fuse = Fuse(layers)
        self.geo_mlp = nn.Sequential(nn.Linear(5, D), nn.GELU(), nn.Linear(D, D))
        self.box_mlp = nn.Sequential(nn.Linear(4, D), nn.GELU(), nn.Linear(D, D))
        self.query = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.ln_kv = nn.LayerNorm(D)
        nb = 1 if kind in ("rep", "ffn") else 2
        self.blocks = nn.ModuleList([RR.CrossBlock(heads, mlp_ratio) for _ in range(nb)])
        if kind in ("rep", "ffn"):
            self.it_emb = nn.Parameter(torch.zeros(T, 1, D))
            nn.init.trunc_normal_(self.it_emb, std=0.02)
        if kind == "gate":
            self.cblocks = nn.ModuleList([RR.CrossBlock(heads, mlp_ratio) for _ in range(2)])
            self.gate = nn.Sequential(nn.Linear(2 * D + 4, D), nn.GELU(), nn.Linear(D, D))
            nn.init.zeros_(self.gate[2].weight)
            nn.init.zeros_(self.gate[2].bias)                       # g = 0.5 at init
        if kind == "concat":
            self.type_emb = nn.Parameter(torch.zeros(2, D))
            nn.init.trunc_normal_(self.type_emb, std=0.02)
        if kind in ("ads", "fix"):
            self.register_buffer("grid0", grid_points(K))
            if kind == "ads":
                self.off = nn.ModuleList([nn.Linear(D, 2 * K) for _ in range(2)])
                for m in self.off:
                    nn.init.zeros_(m.weight)
                    with torch.no_grad():
                        m.bias.copy_(torch.atanh((self.grid0 - 0.5) / 0.75).reshape(-1))
        self.ln_out, self.heads = nn.LayerNorm(D), RR.Heads()

    def _kv(self, tok, geo):
        return self.ln_kv(tok + self.geo_mlp(geo))

    def _q0(self, b):
        return self.query.expand(len(b["geo"]), -1, -1) + self.box_mlp(b["box4"])[:, None]

    def _sample(self, tok_img, b, pts):
        """pts (n, K, 2) box coords -> sampled kv (n, K, D) from the fused map (bilinear, zero outside the canvas)."""
        n, K, _ = pts.shape
        side = b["side"]
        Gc = side // PATCH
        X0, Y0, bw, bh = (b["box4"][:, i:i + 1] * side for i in range(4))
        X = X0 + pts[..., 0] * bw
        Y = Y0 + pts[..., 1] * bh
        grid = torch.stack([X / (side / 2) - 1, Y / (side / 2) - 1], -1)[:, None]          # (n,1,K,2)
        fmap = tok_img.transpose(1, 2).reshape(tok_img.shape[0], D, Gc, Gc)
        smp = F.grid_sample(fmap[b["img_index"]], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        smp = smp[:, :, 0].transpose(1, 2)
        inside = ((pts >= 0) & (pts <= 1)).all(-1).float()
        geo = torch.stack([pts[..., 0], pts[..., 1], inside, torch.log(bw / PATCH).expand(n, K), torch.log(bh / PATCH).expand(n, K)], -1)
        return self._kv(smp, geo)

    def _run(self, blocks, q, kv, mask):
        for blk in blocks:
            q, _ = blk(q, kv, mask)
        return q

    def forward(self, feats, b, ctx_mode="normal"):
        """-> dict(logits (n,59), extra={name: (n, ...)}). Loss is always on `logits` only."""
        tok_img = self.fuse(feats)
        lv, mask = b["lv"], b["cov"] > 0
        geo = b["geo"]
        q = self._q0(b)
        out = dict(extra={})
        hd = lambda z: self.heads(self.ln_out(z[:, 0]), lv)  # noqa: E731
        if self.kind == "r2":
            q = self._run(self.blocks, q, self._kv(tok_img[b["img_index"]], geo), mask)
        elif self.kind in ("rep", "ffn"):
            kv = self._kv(tok_img[b["img_index"]], geo)
            blk, its = self.blocks[0], []
            for t in range(self.T):
                if t == 0 or self.kind == "rep":
                    q, _ = blk(q + self.it_emb[t], kv, mask)
                else:                                                   # same FFN, no new read of the features
                    z = q + self.it_emb[t]
                    q = z + blk.mlp(blk.ln_m(z))
                its.append(hd(q))
            out["extra"]["step_logits"] = torch.stack(its, 1)           # (n,T,59)
            out["logits"] = its[-1]
            return out
        elif self.kind in ("ads", "fix"):
            kv = self._kv(tok_img[b["img_index"]], geo)
            n = len(kv)
            pts = self.grid0.expand(n, -1, -1)
            m = torch.cat([mask, torch.ones(n, self.K, dtype=torch.bool, device=mask.device)], 1)
            for i, blk in enumerate(self.blocks):
                if self.kind == "ads":
                    pts = 0.5 + 0.75 * torch.tanh(self.off[i](q[:, 0])).view(n, self.K, 2)
                q, _ = blk(q, torch.cat([kv, self._sample(tok_img, b, pts)], 1), m)
                out["extra"][f"points_b{i}"] = pts
        elif self.kind in ("concat", "gate", "ctx"):
            kv_own = self._kv(tok_img[b["img_index"]], geo)
            kv_c = self._kv(ctx_tokens(tok_img, b, ctx_mode), geo) if ctx_mode == "swap" else kv_own
            cmask = mask if ctx_mode == "remove" else None
            if self.kind == "ctx":
                q = self._run(self.blocks, q, kv_c, cmask)
            elif self.kind == "concat":
                kv = torch.cat([kv_own + self.type_emb[0], kv_c + self.type_emb[1]], 1)
                cm = mask if cmask is None else cmask
                m = torch.cat([mask, torch.ones_like(mask) if cmask is None else cm], 1)
                q = self._run(self.blocks, q, kv, m)
            else:
                ql = self._run(self.blocks, q, kv_own, mask)
                qc = self._run(self.cblocks, q, kv_c, cmask)
                box = b["box4"]
                sz = torch.stack([torch.log(box[:, 2] * b["side"] / PATCH), torch.log(box[:, 3] * b["side"] / PATCH), box[:, 2], box[:, 3]], 1)
                g = torch.sigmoid(self.gate(torch.cat([ql[:, 0], qc[:, 0], sz], 1)))
                q = (g * ql[:, 0] + (1 - g) * qc[:, 0])[:, None]
                out["extra"]["gate"] = g.mean(1, keepdim=True)
        else:
            raise ValueError(self.kind)
        out["logits"] = hd(q)
        return out


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def param_groups(reader, lr, head_lr, wd):
    return RR.reader_param_groups(reader, lr, head_lr, wd)
