"""Frozen dense teacher (design doc 2안): the SAME RGB384 canvas through the SAME frozen DINOv3-B/16, but every ROI's area is
additionally tokenised at p_T = 4 px with the PI-resized 16 -> 4 patch-embed kernel w_hat = U^T w (U = bilinear 4 -> 16 upsampling;
FlexiViT's PI identity in its exact regime: the token of a 4 x 4 canvas patch equals DINOv3's token of that patch zoomed 4x; frozen,
derived from the original weights; no new parameters). The literal FlexiViT down-LS form pinv(B^T) w is method 'down_bilinear'. Output: the ROI's fine tokens after the frozen final LayerNorm = target T_crop.

Attention contract (masked teacher, several ROIs per forward):
  * prefix (CLS + 4 registers) and the 576 coarse 16-px tokens attend to prefix + coarse only  -> they never see fine tokens, so the
    coarse stream IS the plain student forward (F) at every layer.
  * the fine tokens of ROI r attend to: prefix, the coarse tokens OUTSIDE ROI r's parent cells (the cells the fine tokens replace:
    'crop 밖 16px · crop 안 p_T px'), and the valid fine tokens of ROI r. Never other ROIs. Letterbox-invalid fine tokens are not keys.
  * every patch token gets DINOv3 RoPE at its own centre in the shared full-image coordinate system (src.rsp.common); prefix tokens
    get none (as in DINOv3).

Implementations
  DenseTeacher.coarse / .fine  : efficient. Coarse stream once per image (keeps per-layer post-RoPE K, V); fine streams of any
                                 number of ROIs batched, keys = gathered coarse K/V (+ key mask) + own fine K/V.  Used for training.
  ref_forward                  : literal. ONE sequence [prefix, coarse, fine_1 .. fine_M] through timm's own blocks with an explicit
                                 (Nt x Nt) boolean attention mask. M = 1 is the single-ROI teacher. Used only for verification.
"""
import torch
import torch.nn.functional as F

from src.rsp.common import G, GF, P, PREFIX, PT, R_FP, apply_rope, coarse_centres, dino_rope, fine_rect_index, fine_valid, pi_resize, pi_resize_up


class DenseTeacher(torch.nn.Module):
    def __init__(self, vit, method="bilinear"):
        super().__init__()
        self.vit = vit
        w = vit.patch_embed.proj.weight.detach().float()
        w4, B = pi_resize_up(w, PT, method) if method == "bilinear" else pi_resize(w, PT, method.replace("down_", ""))
        self.register_buffer("w4", w4.contiguous(), persistent=False)
        self.register_buffer("b4", vit.patch_embed.proj.bias.detach().float().clone(), persistent=False)
        self.register_buffer("B", B.float(), persistent=False)
        self.method = method
        self.heads = vit.blocks[0].attn.num_heads

    def coarse_rope(self, dev):
        return dino_rope(self.vit, coarse_centres(dev))

    # ------------------------------------------------------------------ efficient path
    @torch.no_grad()
    def coarse(self, x):
        """x (B, 3, 384, 384) -> final (B, 581, 768) post-norm (prefix + patches), kv: list over blocks of (k, v) (B, h, 581, dh) post-RoPE.
        Numerically the plain timm forward (checked); run under the caller's autocast."""
        vit = self.vit
        h, rope = vit._pos_embed(vit.patch_embed(x))
        h = vit.norm_pre(h)
        kv = []
        for blk in vit.blocks:
            y = blk.norm1(h)
            q, k, v = self._qkv(blk, y, rope)
            kv.append((k, v))
            a = F.scaled_dot_product_attention(q, k, v)
            h = h + blk.gamma_1 * blk.attn.proj(a.transpose(1, 2).reshape(h.shape))
            h = h + blk.gamma_2 * blk.mlp(blk.norm2(h))
        return vit.norm(h), kv

    def _qkv(self, blk, y, rope, prefix=PREFIX):
        B, N, C = y.shape
        q, k, v = blk.attn.qkv(y).reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        if prefix:
            q = torch.cat([q[:, :, :prefix], apply_rope(q[:, :, prefix:], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :prefix], apply_rope(k[:, :, prefix:], rope)], 2).type_as(v)
        else:
            q, k = apply_rope(q, rope).type_as(v), apply_rope(k, rope).type_as(v)
        return q, k, v

    def fine_embed(self, x):
        """(B, 3, 384, 384) -> (B, 9216, 768) PI-resized 4 x 4 patch embedding of the whole canvas (fine flat index u * 96 + v)."""
        e = F.conv2d(x, self.w4, self.b4, stride=PT)                          # autocast handles bf16 like timm's patch_embed conv
        return e.flatten(2).transpose(1, 2)

    @staticmethod
    def coarse_allowed(frects):
        """(R, 4) fine rects -> (R, 576) bool: coarse cell NOT inside the parent cells the fine rectangle covers."""
        fy, fx, h, w = frects.unbind(1)
        py0, px0 = torch.div(fy, R_FP, rounding_mode="floor"), torch.div(fx, R_FP, rounding_mode="floor")
        py1, px1 = torch.div(fy + h - 1, R_FP, rounding_mode="floor"), torch.div(fx + w - 1, R_FP, rounding_mode="floor")
        i = torch.arange(G, device=frects.device)
        iny = (i[None] >= py0[:, None]) & (i[None] <= py1[:, None])
        inx = (i[None] >= px0[:, None]) & (i[None] <= px1[:, None])
        return ~(iny[:, :, None] & inx[:, None, :]).reshape(-1, G * G)

    @torch.no_grad()
    def fine(self, femb, kv, frects, rimg, content_roi, n_max=None):
        """Fine streams of R ROIs in one batch. femb (B, 9216, 768) from fine_embed; kv from coarse (per image); frects (R, 4) long
        [fy, fx, h, w]; rimg (R,) ROI -> image; content_roi (R, 4) letterbox content of the ROI's image.
        -> T (R, N, 768) post-norm fine tokens (row-major over the rectangle), valid (R, N) bool, idx (R, N) flat fine index."""
        vit = self.vit
        idx, uy, ux, ok = fine_rect_index(frects, n_max)
        valid = ok & fine_valid(content_roi, uy, ux)
        h = femb[rimg[:, None], idx]
        cen = torch.stack([(uy.float() + .5) * PT, (ux.float() + .5) * PT], -1)
        rope = dino_rope(vit, cen.view(-1, 2)).view(*cen.shape[:2], -1)[:, None]          # (R, 1, N, 2 dh)
        R = len(frects)
        km = torch.cat([torch.ones(R, PREFIX, dtype=torch.bool, device=h.device), self.coarse_allowed(frects), valid], 1)[:, None, None, :]
        for blk, (kc, vc) in zip(vit.blocks, kv):
            y = blk.norm1(h)
            q, k, v = self._qkv(blk, y, rope, prefix=0)
            K = torch.cat([kc[rimg].to(k.dtype), k], 2)
            V = torch.cat([vc[rimg].to(v.dtype), v], 2)
            a = F.scaled_dot_product_attention(q, K, V, attn_mask=km)
            h = h + blk.gamma_1 * blk.attn.proj(a.transpose(1, 2).reshape(h.shape))
            h = h + blk.gamma_2 * blk.mlp(blk.norm2(h))
        return vit.norm(h), valid, idx

    @torch.no_grad()
    def targets(self, x, frects, rimg, content_img, budget=60000):
        """Convenience: coarse + chunked fine for all ROIs (ROIs sorted by size; padded tokens per chunk <= budget).
        -> list over ROIs of (T (n, 768) float, valid (n,), idx (n,)) and the coarse final (B, 581, 768)."""
        final, kv = self.coarse(x)
        femb = self.fine_embed(x)
        n = (frects[:, 2] * frects[:, 3]).tolist()
        order = sorted(range(len(n)), key=lambda r: n[r])
        out = [None] * len(n)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and n[order[j]] * (j + 1 - i) <= budget:
                j += 1
            sel = torch.tensor(order[i:j], device=x.device)
            T, valid, idx = self.fine(femb, kv, frects[sel], rimg[sel], content_img[rimg[sel]])
            for k, r in enumerate(sel.tolist()):
                out[r] = (T[k, :n[r]].float(), valid[k, :n[r]], idx[k, :n[r]])
            i = j
        return out, final

    # ------------------------------------------------------------------ literal reference (verification only)
    @torch.no_grad()
    def ref_forward(self, x1, frects, content1):
        """x1 (1, 3, 384, 384); frects (M, 4); content1 (4,). One sequence [prefix 5, coarse 576, fine_1 .. fine_M] through the timm
        blocks themselves (blk(x, rope=, attn_mask=)) with the explicit mask. -> prefix+coarse final (581, 768), list of fine finals."""
        vit = self.vit
        dev = x1.device
        hc, _ = vit._pos_embed(vit.patch_embed(x1))
        hc = vit.norm_pre(hc)
        femb = self.fine_embed(x1)[0]
        parts, ropes, owner, valids, coarse_ok = [], [dino_rope(vit, coarse_centres(dev))], [], [], []
        for m in range(len(frects)):
            idx, uy, ux, ok = fine_rect_index(frects[m:m + 1])
            v = (ok & fine_valid(content1[None], uy, ux))[0]
            parts.append(femb[idx[0]])
            ropes.append(dino_rope(vit, torch.stack([(uy[0].float() + .5) * PT, (ux[0].float() + .5) * PT], -1)))
            owner.append(torch.full((idx.shape[1],), m, device=dev))
            valids.append(v)
            coarse_ok.append(self.coarse_allowed(frects[m:m + 1])[0])
        h = torch.cat([hc[0]] + parts, 0)[None]
        rope = torch.cat(ropes, 0)
        Nc = PREFIX + G * G
        owner = torch.cat([torch.full((Nc,), -1, device=dev)] + owner)
        fval = torch.cat(valids)
        Nt = len(owner)
        mask = torch.zeros(Nt, Nt, dtype=torch.bool, device=dev)
        mask[:Nc, :Nc] = True                                                # prefix / coarse rows: prefix + coarse only
        for m in range(len(frects)):
            rows = (owner == m).nonzero().squeeze(1)
            colok = torch.zeros(Nt, dtype=torch.bool, device=dev)
            colok[:PREFIX] = True
            colok[PREFIX:Nc] = coarse_ok[m]
            colok[Nc:] = (owner[Nc:] == m) & fval
            mask[rows] = colok
        for blk in vit.blocks:
            h = blk(h, rope=rope, attn_mask=mask[None, None])
        h = vit.norm(h)[0]
        outs, s = [], Nc
        for p in parts:
            outs.append(h[s:s + len(p)])
            s += len(p)
        return h[:Nc], outs, mask, owner
