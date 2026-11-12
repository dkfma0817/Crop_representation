"""Dense teacher used DIRECTLY as the ROI feature extractor (no predictor, no training of any feature model).

Frozen DINOv3-B/16 (timm vit_base_patch16_dinov3.lvd1689m), RGB384 letterbox canvas. Besides the 576 coarse 16 px tokens, each ROI's
area is tokenised at p_T px (p_T = 8: x2, the F768 token density; p_T = 4: the RSP training teacher) with the PI-resized frozen kernel
w_pT = U^T w (U = bilinear p_T -> 16 upsampling of a patch; src.rsp.common.pi_resize_up, exact PI identity), bias copied.
RoPE: DINOv3's own RoPE at every token's canvas centre in one full-image frame (src.rsp.common.dino_rope); prefix (CLS + 4 reg) none.

ROI rectangle (parent units): the parent cells the GT box covers + 1 parent cell of margin on every side, clamped to the 24 x 24 grid.
All fine cells of the rectangle are tokens (letterbox padding cells included, exactly like the coarse stream, which also tokenises
padding cells).

attn = 'masked'  (= the RSP teacher contract): prefix + coarse rows attend to prefix + coarse only (= plain student forward, cached
                  once per image: per-layer post-RoPE K/V); the fine rows of ROI r attend to prefix, coarse cells OUTSIDE ROI r's
                  rectangle, and ROI r's own fine tokens. ROIs never see each other -> any number of ROIs batched.
attn = 'full'    (unrestricted single-ROI): ONE sequence [prefix 5, coarse 576, fine of ROI r] per ROI with NO attention restriction
                  (coarse sees fine, fine sees every coarse cell incl. the ones it replaces). A separate forward per ROI (ROIs batched
                  only as independent sequences with key padding).
Output: the ROI's fine tokens after the frozen final LayerNorm, (n, 768) row-major over the fine rectangle.
"""
import torch
import torch.nn.functional as F

from src.rsp.common import G, P, PREFIX, apply_rope, coarse_centres, dino_rope, pi_resize_up

S = 384


class DT(torch.nn.Module):
    def __init__(self, vit, pt):
        super().__init__()
        assert P % pt == 0
        self.vit, self.pt, self.gf, self.r = vit, pt, S // pt, P // pt
        w = vit.patch_embed.proj.weight.detach().float()
        wp, U = pi_resize_up(w, pt, "bilinear")
        self.register_buffer("wp", wp.contiguous(), persistent=False)
        self.register_buffer("bp", vit.patch_embed.proj.bias.detach().float().clone(), persistent=False)
        self.register_buffer("U", U.float(), persistent=False)
        self.heads = vit.blocks[0].attn.num_heads

    # ------------------------------------------------------------------ geometry
    def rect_index(self, frects, n_max=None):
        """frects (R, 4) long [fy, fx, h, w] fine rects -> flat idx, uy, ux, ok (R, N), row-major."""
        fy0, fx0, h, w = frects.unbind(1)
        n = h * w
        N = int(n.max()) if n_max is None else n_max
        t = torch.arange(N, device=frects.device)[None]
        ww = w[:, None].clamp_min(1)
        uy = fy0[:, None] + torch.div(t, ww, rounding_mode="floor")
        ux = fx0[:, None] + t % ww
        ok = t < n[:, None]
        uy, ux = torch.where(ok, uy, 0).clamp(0, self.gf - 1), torch.where(ok, ux, 0).clamp(0, self.gf - 1)
        return uy * self.gf + ux, uy, ux, ok

    def centres(self, uy, ux):
        return torch.stack([(uy.float() + .5) * self.pt, (ux.float() + .5) * self.pt], -1)

    @staticmethod
    def parent_rects(boxc, margin=1):
        """boxc (R, 4) ltrb canvas px -> (R, 4) long [py, px, H, W]: parent cells the box covers (+ margin), clamped."""
        l, t, r, b = boxc.unbind(1)
        px0, py0 = torch.floor(l / P), torch.floor(t / P)
        px1, py1 = torch.ceil(r / P) - 1, torch.ceil(b / P) - 1
        px1, py1 = torch.maximum(px1, px0), torch.maximum(py1, py0)
        px0, py0 = (px0 - margin).clamp(0, G - 1), (py0 - margin).clamp(0, G - 1)
        px1, py1 = (px1 + margin).clamp(0, G - 1), (py1 + margin).clamp(0, G - 1)
        return torch.stack([py0, px0, py1 - py0 + 1, px1 - px0 + 1], 1).long()

    def fine_rects(self, prects):
        return prects * self.r

    @staticmethod
    def coarse_allowed(prects):
        """(R, 4) parent rects -> (R, 576) bool: coarse cell NOT inside the rectangle."""
        py, px, H, W = prects.unbind(1)
        i = torch.arange(G, device=prects.device)
        iny = (i[None] >= py[:, None]) & (i[None] < (py + H)[:, None])
        inx = (i[None] >= px[:, None]) & (i[None] < (px + W)[:, None])
        return ~(iny[:, :, None] & inx[:, None, :]).reshape(-1, G * G)

    # ------------------------------------------------------------------ pieces
    def _qkv(self, blk, y, rope, prefix=PREFIX):
        B, N, C = y.shape
        q, k, v = blk.attn.qkv(y).reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        if prefix:
            q = torch.cat([q[:, :, :prefix], apply_rope(q[:, :, prefix:], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :prefix], apply_rope(k[:, :, prefix:], rope)], 2).type_as(v)
        else:
            q, k = apply_rope(q, rope).type_as(v), apply_rope(k, rope).type_as(v)
        return q, k, v

    def _block(self, blk, h, q, K, V, mask):
        a = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        h = h + blk.gamma_1 * blk.attn.proj(a.transpose(1, 2).reshape(h.shape))
        return h + blk.gamma_2 * blk.mlp(blk.norm2(h))

    def fine_embed(self, x):
        """(B, 3, 384, 384) -> (B, gf * gf, 768) PI-resized p_T x p_T patch embedding of the whole canvas."""
        return F.conv2d(x, self.wp, self.bp, stride=self.pt).flatten(2).transpose(1, 2)

    def embed0(self, x):
        """timm's own input of block 0: prefix + coarse patch embedding (B, 581, 768), coarse rope (576, 2 dh)."""
        h, rope = self.vit._pos_embed(self.vit.patch_embed(x))
        return self.vit.norm_pre(h), rope

    @torch.no_grad()
    def coarse(self, x):
        """-> final (B, 581, 768) (= forward_features), kv: per block (k, v) post-RoPE (B, h, 581, dh)."""
        h, rope = self.embed0(x)
        kv = []
        for blk in self.vit.blocks:
            q, k, v = self._qkv(blk, blk.norm1(h), rope)
            kv.append((k, v))
            h = self._block(blk, h, q, k, v, None)
        return self.vit.norm(h), kv

    @torch.no_grad()
    def fine_masked(self, femb, kv, prects, rimg):
        """Masked fine streams of R ROIs. -> T (R, N, 768) post-norm, ok (R, N)."""
        idx, uy, ux, ok = self.rect_index(self.fine_rects(prects))
        h = femb[rimg[:, None], idx]
        rope = dino_rope(self.vit, self.centres(uy, ux).view(-1, 2)).view(*uy.shape, -1)[:, None]
        R = len(prects)
        km = torch.cat([torch.ones(R, PREFIX, dtype=torch.bool, device=h.device), self.coarse_allowed(prects), ok], 1)[:, None, None, :]
        for blk, (kc, vc) in zip(self.vit.blocks, kv):
            q, k, v = self._qkv(blk, blk.norm1(h), rope, prefix=0)
            h = self._block(blk, h, q, torch.cat([kc[rimg].to(k.dtype), k], 2), torch.cat([vc[rimg].to(v.dtype), v], 2), km)
        return self.vit.norm(h), ok

    @torch.no_grad()
    def fine_full(self, h0, femb, prects, rimg):
        """Unrestricted single-ROI sequences [prefix, coarse, fine_r] (one independent sequence per ROI, key padding only).
        h0 (B, 581, 768) = embed0. -> T (R, N, 768) post-norm fine part, ok (R, N)."""
        idx, uy, ux, ok = self.rect_index(self.fine_rects(prects))
        R, N = idx.shape
        h = torch.cat([h0[rimg], femb[rimg[:, None], idx]], 1)
        rc = dino_rope(self.vit, coarse_centres(h.device))[None].expand(R, -1, -1)
        rf = dino_rope(self.vit, self.centres(uy, ux).view(-1, 2)).view(R, N, -1)
        rope = torch.cat([rc, rf], 1)[:, None]
        pad = ok.all()
        km = None if pad else torch.cat([torch.ones(R, PREFIX + G * G, dtype=torch.bool, device=h.device), ok], 1)[:, None, None, :]
        for blk in self.vit.blocks:
            q, k, v = self._qkv(blk, blk.norm1(h), rope)
            h = self._block(blk, h, q, k, v, km)
        return self.vit.norm(h)[:, PREFIX + G * G:], ok

    # ------------------------------------------------------------------ literal references (verification only)
    @torch.no_grad()
    def ref_masked(self, x1, prects):
        """ONE sequence [prefix, coarse, fine_1 .. fine_M] through timm's own blocks with the explicit (Nt x Nt) mask."""
        dev = x1.device
        hc, _ = self.embed0(x1)
        femb = self.fine_embed(x1)[0]
        parts, ropes, owner, cok = [], [dino_rope(self.vit, coarse_centres(dev))], [], []
        for m in range(len(prects)):
            idx, uy, ux, ok = self.rect_index(self.fine_rects(prects[m:m + 1]))
            parts.append(femb[idx[0]])
            ropes.append(dino_rope(self.vit, self.centres(uy[0], ux[0])))
            owner.append(torch.full((idx.shape[1],), m, device=dev))
            cok.append(self.coarse_allowed(prects[m:m + 1])[0])
        h = torch.cat([hc[0]] + parts, 0)[None]
        rope = torch.cat(ropes, 0)
        Nc = PREFIX + G * G
        owner = torch.cat([torch.full((Nc,), -1, device=dev)] + owner)
        Nt = len(owner)
        mask = torch.zeros(Nt, Nt, dtype=torch.bool, device=dev)
        mask[:Nc, :Nc] = True
        for m in range(len(prects)):
            col = torch.zeros(Nt, dtype=torch.bool, device=dev)
            col[:PREFIX] = True
            col[PREFIX:Nc] = cok[m]
            col[Nc:] = owner[Nc:] == m
            mask[owner == m] = col
        for blk in self.vit.blocks:
            h = blk(h, rope=rope, attn_mask=mask[None, None])
        h = self.vit.norm(h)[0]
        outs, s = [], Nc
        for p in parts:
            outs.append(h[s:s + len(p)])
            s += len(p)
        return h[:Nc], outs

    @torch.no_grad()
    def ref_full(self, x1, prect):
        """ONE unrestricted sequence [prefix, coarse, fine] through timm's own blocks, no mask."""
        dev = x1.device
        hc, _ = self.embed0(x1)
        idx, uy, ux, _ = self.rect_index(self.fine_rects(prect[None]))
        h = torch.cat([hc[0], self.fine_embed(x1)[0][idx[0]]], 0)[None]
        rope = torch.cat([dino_rope(self.vit, coarse_centres(dev)), dino_rope(self.vit, self.centres(uy[0], ux[0]))], 0)
        for blk in self.vit.blocks:
            h = blk(h, rope=rope)
        return self.vit.norm(h)[0, PREFIX + G * G:]


def chunks_by_budget(n, budget):
    """n list of per-ROI token counts -> list of index lists (sorted by size) with max_n * count <= budget."""
    order = sorted(range(len(n)), key=lambda r: n[r])
    out, i = [], 0
    while i < len(order):
        j = i + 1
        while j < len(order) and n[order[j]] * (j + 1 - i) <= budget:
            j += 1
        out.append(order[i:j])
        i = j
    return out
