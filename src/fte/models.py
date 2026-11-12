"""P1 (Small ViT + late split) and P2 (DINOv3 block-8 state -> early split -> copied pretrained blocks 9-12), x2 per axis.

P1 = the professor's ExpansionDecoder (dinov3_token_expansion_final.py) with scale 2: final patch tokens of the K selected parents
     (detached) -> Linear(768, 384) + continuous 2D Fourier position of the parent centre (reference position_encoding) -> 6 pre-LN
     TransformerEncoderLayer(384, 6 heads, FFN 1536, GELU, dropout 0) with key padding mask -> LayerNorm -> split Linear(384, 4 x 768)
     -> Z (R, K, 4, 768) -> native projection Linear(768, 768) identity-initialised -> frozen copy of the DINOv3 final patch norm -> Y.
     SIGReg latent = Z (after the split, before the native projection).
P2 = for every ROI: the original frozen forward's blocks[7] output h (581 x 768, raw residual stream, prefix included) ->
     children of each selected parent p: child_k = h_p + W_k h_p + b_k (k = 0..3, W_k / b_k zero-initialised) -> sequence
     [5 prefix] + [576 coarse tokens, the selected parents MASKED OUT as keys (= removed)] + [4K children] (+ padding, masked) ->
     deepcopy(blocks[8:12]) (trainable, pretrained init) with mixed-resolution RoPE (coarse: 24 x 24 coords, children: their 48 x 48
     coords, full-image) -> children Z (before norm) -> frozen copy of the final norm -> Y. SIGReg latent = Z.
     Masked-out parents / padding still pass through the blocks as queries but are never keys, so no other token sees them.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fte.common import G, child_coords, child_ids, parent_coords, rect_parents, rope_from_coords

D = 768
PREFIX = 5


def fourier(yx, width):
    """Reference position_encoding (continuous 2D Fourier)."""
    freq = 2 * math.pi * torch.logspace(0, math.log10(16), width // 4, device=yx.device)
    phase = yx.float().unsqueeze(-1) * freq
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)


class P1(nn.Module):
    def __init__(self, vit, width=384, depth=6, heads=6, s=2):
        super().__init__()
        self.s, self.width = s, width
        self.in_proj = nn.Linear(D, width)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(depth)])
        self.coarse_norm = nn.LayerNorm(width)
        self.split = nn.Linear(width, s * s * D)
        self.native_proj = nn.Linear(D, D)
        nn.init.eye_(self.native_proj.weight)
        nn.init.zeros_(self.native_proj.bias)
        self.output_norm = copy.deepcopy(vit.norm).eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.output_norm.eval()
        return self

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def groups(self):
        return dict(in_proj=list(self.in_proj.parameters()), blocks=list(self.blocks.parameters()) + list(self.coarse_norm.parameters()),
                    split=list(self.split.parameters()), native_proj=list(self.native_proj.parameters()))

    def latent_params(self):
        """Parameters on the path to Z (reached by both the prediction loss and SIGReg)."""
        return list(self.in_proj.parameters()) + list(self.blocks.parameters()) + list(self.coarse_norm.parameters()) + list(self.split.parameters())

    def forward(self, final, img, rects):
        """final (B, 576, 768) post-norm patch tokens; img (R,); rects (R, 4) -> Y, Z (R, K, 4, 768) fp32 / native, pid, pok."""
        pid, pok = rect_parents(rects)
        x = final[img[:, None], pid].detach()
        x = x.masked_fill(~pok[..., None], 0)
        pyx = parent_coords(pid // G, pid % G).masked_fill(~pok[..., None], 0)
        h = self.in_proj(x) + fourier(pyx, self.width).to(x.dtype)
        for blk in self.blocks:
            h = blk(h, src_key_padding_mask=~pok)
        z = self.split(self.coarse_norm(h)).reshape(*pid.shape, self.s * self.s, D)
        y = self.native_proj(z)
        with torch.autocast("cuda", enabled=False):
            Y = self.output_norm(y.float())
        return Y, z, pid, pok


class P2(nn.Module):
    def __init__(self, vit, start=8, s=2):
        super().__init__()
        self.s, self.start = s, start
        self.blocks = copy.deepcopy(vit.blocks[start:])
        for p in self.blocks.parameters():
            p.requires_grad_(True)
        self.split = nn.Linear(D, s * s * D)
        nn.init.zeros_(self.split.weight)
        nn.init.zeros_(self.split.bias)
        self.norm = copy.deepcopy(vit.norm).eval().requires_grad_(False)
        self.rope = vit.rope                       # periods buffer only (no parameters); not modified
        self.blocks.eval()                          # drop_path / dropout are Identity / 0 in this checkpoint; kept in eval for determinism

    def train(self, mode=True):
        super().train(mode)
        self.norm.eval()
        return self

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def groups(self):
        return {**{f"block{self.start + 1 + k}": list(b.parameters()) for k, b in enumerate(self.blocks)}, "split": list(self.split.parameters())}

    def latent_params(self):
        return list(self.blocks.parameters()) + list(self.split.parameters())

    def coarse_rope(self, dev):
        i = torch.arange(G * G, device=dev)
        return parent_coords(i // G, i % G)

    def forward(self, h8, img, rects, return_coarse=False):
        """h8 (B, 581, 768) blocks[7] output of the original forward; img (R,); rects (R, 4).
        -> Y (R, K, 4, 768) fp32 post frozen norm, Z (R, K, 4, 768) pre-norm, pid, pok."""
        s2 = self.s * self.s
        pid, pok = rect_parents(rects)
        R, K = pid.shape
        hp = h8[img[:, None], PREFIX + pid]                                     # (R, K, 768)
        corr = self.split(hp).view(R, K, s2, D)
        child = (hp[:, :, None, :] + corr).reshape(R, K * s2, D)
        cid, ci, cj = child_ids(pid, self.s)
        x = torch.cat([h8[img], child.to(h8.dtype)], 1)                        # [prefix 5][coarse 576][children 4K]
        sel = torch.zeros(R, G * G, dtype=torch.bool, device=h8.device)
        sel.scatter_(1, pid, pok)                                             # selected parents (removed as keys)
        key = torch.cat([torch.ones(R, PREFIX, dtype=torch.bool, device=h8.device), ~sel, pok[..., None].expand(-1, -1, s2).reshape(R, K * s2)], 1)
        coords = torch.cat([self.coarse_rope(h8.device)[None].expand(R, -1, -1), child_coords(ci, cj, self.s).reshape(R, K * s2, 2)], 1)
        rope = rope_from_coords(self.rope, coords)[:, None]                   # (R, 1, 576 + 4K, 2 * head_dim)
        mask = key[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, rope=rope, attn_mask=mask)                               # fp32 rope, as in timm forward_features
        Z = x[:, PREFIX + G * G:].reshape(R, K, s2, D)
        with torch.autocast("cuda", enabled=False):
            Y = self.norm(Z.float())
        if return_coarse:
            return Y, Z, pid, pok, x
        return Y, Z, pid, pok

    @torch.no_grad()
    def reference_path(self, h8):
        """Check 1: no children, regular 24 x 24 RoPE from rope_from_coords -> copied blocks -> frozen norm (patch tokens)."""
        B = len(h8)
        rope = rope_from_coords(self.rope, self.coarse_rope(h8.device))
        x = h8
        for blk in self.blocks:
            x = blk(x, rope=rope)
        return self.norm(x)[:, PREFIX:]
