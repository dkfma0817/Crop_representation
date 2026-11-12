"""RSP masked fine-feature predictor (trainable; the DINOv3 encoder stays frozen and outside this module).

Tokens (design doc rsp_architecture.html, 'Masking과 위치', 'Attention mask'):
  L0 context  : 581 = Linear(768 -> d)(frozen DINOv3 384 final tokens: CLS + 4 registers + 576 patches). Patches: RoPE at the coarse
                cell centre + size emb (16 px). Prefix: learned prefix embedding, no RoPE (as in DINOv3). Shared by every ROI of the image.
  L1 evidence : 64 = Conv(3 -> d, 8, stride 8) of the ROI RGB (original image, resized to 64 x 64) + size emb (evidence cell size);
                RoPE at each evidence cell's true centre in the full-image canvas.            [RSP-Masked only; absent in noPix]
  L1 z-token  : 1 = MLP(cx, cy, w, h of the ROI box / 384) + learned z type; no RoPE.
  L2 queries  : K = shared learnable mask token + size emb (p_T = 4 px) + z emb; RoPE at the fine cell centre to be predicted.
Attention (allow(i->j) = valid(j) & lv(j) <= lv(i) & (lv(j) < 2 | i = j) & (crop(j) = crop(i) | lv(j) = 0)):
  context sees context; L1 sees context + own ROI's L1; a query sees context + own ROI's L1 + itself (no query-query attention).
  Implemented level by level (context once per image, L1 once per ROI, queries in any chunking) — exactly equal to the dense masked
  sequence (ref_forward; src.rsp.checks).
Output: LayerNorm -> Linear(d -> 768) = the prediction in the teacher-NORMALISED space ((T - mu) / sigma, mu / sigma fixed from prep).
Fine teacher features never enter this module.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rsp.common import EV_PATCH, EV_SIDE, P, PREFIX, PT, S, Rope2D, apply_rope, coarse_centres

D, DEPTH, HEADS, MLP = 192, 12, 3, 4
LV_CTX, LV_L1, LV_Q = 0, 1, 2


def level_mask(lv, crop, valid):
    """Design-doc formula. lv, crop (N,) long, valid (N,) bool -> allow (N, N) bool, rows = query side i, cols = key side j."""
    i_lv, j_lv = lv[:, None], lv[None, :]
    eye = torch.eye(len(lv), dtype=torch.bool, device=lv.device)
    return valid[None, :] & (j_lv <= i_lv) & ((j_lv < LV_Q) | eye) & ((crop[None, :] == crop[:, None]) | (j_lv == LV_CTX))


class Block(nn.Module):
    def __init__(self, d=D, heads=HEADS, mlp=MLP):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def qkv_heads(self, x, rope):
        R, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(R, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        return apply_rope(q, rope).type_as(v), apply_rope(k, rope).type_as(v), v

    def finish(self, x, a):
        R, N, d = x.shape
        x = x + self.proj(a.transpose(1, 2).reshape(R, N, d))
        return x + self.mlp(self.n2(x))


class RSPPredictor(nn.Module):
    def __init__(self, pix=True, d=D, depth=DEPTH, heads=HEADS, mlp=MLP, rope_min=0.05, rope_max=8.0):
        super().__init__()
        self.pix, self.d = pix, d
        self.ctx_in = nn.Linear(768, d)
        self.prefix_emb = nn.Parameter(torch.zeros(PREFIX, d))
        self.size_mlp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.z_mlp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.z_type = nn.Parameter(torch.zeros(d))
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, heads, mlp) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 768)
        self.apply(self._init)
        for p_ in (self.prefix_emb, self.z_type, self.mask_token):
            nn.init.trunc_normal_(p_, std=.02)
        # evidence path created LAST so RSP-Masked and noPix get bit-identical initial values for every shared module (same seed)
        self.ev_embed = nn.Conv2d(3, d, EV_PATCH, EV_PATCH) if pix else None
        self.rope = Rope2D(d // heads, rope_min, rope_max)
        self.rope_cfg = dict(min_period=rope_min, max_period=rope_max, coords="2 * centre_px / 384 - 1, (y, x)")

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def groups(self):
        g = dict(ctx_in=list(self.ctx_in.parameters()) + [self.prefix_emb], size_z=list(self.size_mlp.parameters()) + list(self.z_mlp.parameters()) + [self.z_type],
                 mask_token=[self.mask_token], blocks=list(self.blocks.parameters()), out=list(self.out_norm.parameters()) + list(self.out.parameters()))
        if self.pix:
            g["evidence"] = list(self.ev_embed.parameters())
        return g

    def size_emb(self, wh_px):
        return self.size_mlp(torch.log(wh_px.float().clamp_min(1e-3) / P))

    # ---------------------------------------------------------------- L0 context (once per image)
    def context(self, feats):
        """feats (B, 581, 768) frozen DINOv3 384 final tokens -> per-block (k, v) of the context tokens (B, h, 581, dh)."""
        B = feats.shape[0]
        dev = feats.device
        h = self.ctx_in(feats)
        size16 = self.size_emb(torch.full((1, 2), float(P), device=dev))
        h = torch.cat([h[:, :PREFIX] + self.prefix_emb, h[:, PREFIX:] + size16], 1)
        cen = torch.cat([torch.zeros(PREFIX, 2, device=dev), coarse_centres(dev)], 0)
        pos = torch.cat([torch.zeros(PREFIX, device=dev), torch.ones(len(cen) - PREFIX, device=dev)])
        rope = self.rope(cen, pos)[None, None]
        cache = []
        for li, blk in enumerate(self.blocks):
            q, k, v = blk.qkv_heads(h, rope)
            cache.append((k, v))
            if li + 1 < len(self.blocks):                                    # the last block's context output is never read
                h = blk.finish(h, F.scaled_dot_product_attention(q, k, v))
        return cache

    # ---------------------------------------------------------------- L1 evidence + z (once per ROI)
    def l1_tokens(self, box, ev):
        """box (R, 4) ltrb canvas px; ev (R, 3, 64, 64) or None -> tokens (R, n1, d), centres (R, n1, 2), positional (R, n1), z emb (R, d)."""
        R, dev = len(box), box.device
        l, t, r, b = box.float().unbind(1)
        bw, bh = (r - l).clamp_min(1e-3), (b - t).clamp_min(1e-3)
        z = self.z_mlp(torch.stack([(l + r) / 2 / S, (t + b) / 2 / S, bw / S, bh / S], 1))
        toks, cens, poss = [], [], []
        if self.pix:
            assert ev is not None and ev.shape[-2:] == (EV_SIDE, EV_SIDE)
            g = EV_SIDE // EV_PATCH
            e = self.ev_embed(ev).flatten(2).transpose(1, 2)                 # (R, 64, d)
            e = e + self.size_emb(torch.stack([bw / g, bh / g], 1))[:, None]
            i, j = torch.meshgrid(torch.arange(g, device=dev), torch.arange(g, device=dev), indexing="ij")
            cy = t[:, None] + (i.flatten()[None].float() + .5) * (bh / g)[:, None]
            cx = l[:, None] + (j.flatten()[None].float() + .5) * (bw / g)[:, None]
            toks.append(e)
            cens.append(torch.stack([cy, cx], -1))
            poss.append(torch.ones(R, g * g, device=dev))
        else:
            assert ev is None, "noPix has no ROI RGB input path"
        toks.append((z + self.z_type)[:, None])
        cens.append(torch.zeros(R, 1, 2, device=dev))
        poss.append(torch.zeros(R, 1, device=dev))
        return torch.cat(toks, 1), torch.cat(cens, 1), torch.cat(poss, 1), z

    def level1(self, cache, rimg, box, ev):
        """-> per-block (k, v) of the L1 tokens (R, h, n1, dh), z emb (R, d). L1 attends to context(image) + own L1."""
        h, cen, pos, z = self.l1_tokens(box, ev)
        rope = self.rope(cen, pos)[:, None]
        l1 = []
        for li, (blk, (kc, vc)) in enumerate(zip(self.blocks, cache)):
            q, k, v = blk.qkv_heads(h, rope)
            l1.append((k, v))
            if li + 1 < len(self.blocks):
                a = F.scaled_dot_product_attention(q, torch.cat([kc[rimg], k], 2), torch.cat([vc[rimg], v], 2))
                h = blk.finish(h, a)
        return l1, z

    # ---------------------------------------------------------------- L2 queries (independent)
    def queries(self, cache, l1, rimg, z, cen):
        """cen (R, K, 2) canvas-px centres of the fine cells to predict -> (R, K, 768) normalised prediction.
        Query i attends to context(image of its ROI) + its ROI's L1 + itself only."""
        R, K, _ = cen.shape
        dev = cen.device
        size4 = self.size_emb(torch.full((1, 2), float(PT), device=dev))
        h = (self.mask_token + size4)[None] + z[:, None]
        h = h.expand(R, K, self.d).contiguous()
        rope = self.rope(cen)[:, None]
        for blk, (kc, vc), (kl, vl) in zip(self.blocks, cache, l1):
            q, k, v = blk.qkv_heads(h, rope)
            Ks = torch.cat([kc[rimg], kl], 2)
            Vs = torch.cat([vc[rimg], vl], 2)
            sc = 1.0 / math.sqrt(q.shape[-1])
            logit = torch.cat([torch.matmul(q, Ks.transpose(-1, -2)), (q * k).sum(-1, keepdim=True)], -1) * sc
            a = logit.float().softmax(-1).to(q.dtype)
            o = torch.matmul(a[..., :-1], Vs) + a[..., -1:] * v
            h = blk.finish(h, o)
        return self.out(self.out_norm(h))

    def forward(self, feats, rimg, box, ev, cen):
        cache = self.context(feats)
        l1, z = self.level1(cache, rimg, box, ev if self.pix else None)
        return self.queries(cache, l1, rimg, z, cen)

    # ---------------------------------------------------------------- literal reference (verification only)
    def ref_forward(self, feats1, box, ev, cen, qvalid=None):
        """ONE image: feats1 (1, 581, 768); M ROIs box (M, 4), ev (M, 3, 64, 64) | None, cen (M, K, 2), qvalid (M, K) bool.
        Builds the full sequence [ctx 581 | (L1_m, Q_m) for m] with the design-doc mask (level_mask) and runs dense masked attention
        with the same weights. -> (M, K, 768), mask, lv, crop."""
        dev = feats1.device
        M, K, _ = cen.shape
        qvalid = torch.ones(M, K, dtype=torch.bool, device=dev) if qvalid is None else qvalid
        h0 = self.ctx_in(feats1)[0]
        size16 = self.size_emb(torch.full((1, 2), float(P), device=dev))
        h0 = torch.cat([h0[:PREFIX] + self.prefix_emb, h0[PREFIX:] + size16], 0)
        c0 = torch.cat([torch.zeros(PREFIX, 2, device=dev), coarse_centres(dev)], 0)
        p0 = torch.cat([torch.zeros(PREFIX, device=dev), torch.ones(len(c0) - PREFIX, device=dev)])
        toks, cens, poss, lvs, crops, vals = [h0], [c0], [p0], [torch.full((len(h0),), LV_CTX)], [torch.full((len(h0),), -1)], [torch.ones(len(h0), dtype=torch.bool)]
        t1, c1, p1, z = self.l1_tokens(box, ev if self.pix else None)
        size4 = self.size_emb(torch.full((1, 2), float(PT), device=dev))
        for m in range(M):
            toks += [t1[m], (self.mask_token + size4) + z[m:m + 1].expand(K, -1)]
            cens += [c1[m], cen[m]]
            poss += [p1[m], torch.ones(K, device=dev)]
            lvs += [torch.full((len(t1[m]),), LV_L1), torch.full((K,), LV_Q)]
            crops += [torch.full((len(t1[m]) + K,), m)]
            vals += [torch.ones(len(t1[m]), dtype=torch.bool), qvalid[m].cpu()]
        h = torch.cat(toks)[None]
        rope = self.rope(torch.cat(cens), torch.cat(poss))[None, None]
        lv, crop, valid = torch.cat(lvs).to(dev), torch.cat(crops).to(dev), torch.cat(vals).to(dev)
        mask = level_mask(lv, crop, valid)
        for blk in self.blocks:
            q, k, v = blk.qkv_heads(h, rope)
            h = blk.finish(h, F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None]))
        out = self.out(self.out_norm(h))[0]
        qsel = (lv == LV_Q)
        return out[qsel].view(M, K, -1), mask, lv, crop


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
