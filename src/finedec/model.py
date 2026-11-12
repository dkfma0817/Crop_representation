"""Fine-aware Global Decoder (FA) — one architecture shared by FA-A (mean/residual KD) and FA-B (patch CE + SIGReg).

Per image (computed once, reused by every ROI):
  global  : frozen DINOv3-B/16 on RGB384 -> final patch tokens G (576 x 768, after the final norm) and X_pre (576 x 768, the input of
            the final norm). 5 prefix tokens (CLS + 4 registers) are dropped.
  fine    : (1) frozen overlapping patch embedding = Conv(3 -> 768, k16, s8, p4) with the DINOv3 patch_embed weight/bias copied
            (the backbone's own patch_embed is untouched); (2) trainable local stem Conv(3 -> 192, k8, s8, p0). Both 48 x 48, cell centre
            (4 + 8u, 4 + 8v) on the 384 canvas. Concat (960) -> Linear(960, 384) = fine queries F (2304 x 384).
  memory  : Linear(768, 384)(G) + 2D sin-cos of the parent position -> M (576 x 384).
Per ROI (parent rectangle; all its children together):
  q = F[children] + pos(child); 2 blocks [masked child self-attention -> cross-attention to ALL 576 memory tokens of the image -> FFN],
  width 384, 6 heads, FFN x4, pre-norm; Z = LN(q)   (Z = SIGReg latent of FA-B)
  DeltaMean[p]  = head_mean(mean of Z over the 4 children of parent p)
  DeltaRes[c]   = head_res(Z[c]) - mean over the 4 children of its parent    (sums to 0 per parent)
  Y[c] = FrozenNorm(X_pre[p] + DeltaMean[p] + DeltaRes[c])  (fp32; FrozenNorm = copy of the DINOv3 final LayerNorm, no grad to its params)
Only the last Linear of each correction head is zero-initialised -> before training Y[c] = G[p] exactly (4 identical children).
Position code (same coordinate system as S2/S3, src.tsplit.model.child_pos): D/4 parent row, D/4 parent col, D/4 sub-row, D/4 sub-col;
memory tokens use sub = 0.5 (the parent centre), so query and memory share the parent coordinates.
The output depends on the ROI rectangle (self-attention over the ROI's children) -> it is NOT a query-independent shared map.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tsplit.model import G, G2, sincos

D_FEAT, WIDTH, DEPTH, HEADS, MLP, D_LOCAL = 768, 384, 2, 6, 4, 192


def pos2d(py, px, sy, sx, dim=WIDTH):
    q = dim // 4
    return torch.cat([sincos(py.float(), q), sincos(px.float(), q), sincos(sy.float(), q), sincos(sx.float(), q)], -1)


def child_pos(cy, cx):
    return pos2d(cy // 2, cx // 2, cy % 2, cx % 2)


def parent_pos(dev):
    i = torch.arange(G * G, device=dev)
    h = torch.full_like(i, 0).float() + 0.5
    return pos2d(i // G, i % G, h, h)


class XBlock(nn.Module):
    def __init__(self, d=WIDTH, heads=HEADS, mlp=MLP):
        super().__init__()
        self.h = heads
        self.n1, self.n2, self.n3, self.nm = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.q, self.kv, self.cproj = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def kv_mem(self, mem):
        """mem (B, 576, d) -> k, v (B, heads, 576, d / heads), computed once per image."""
        B, M, d = mem.shape
        k, v = self.kv(self.nm(mem)).view(B, M, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, key_ok, k_img, v_img, img):
        R, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(R, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_ok[:, None, None, :])
        x = x + self.proj(a.transpose(1, 2).reshape(R, N, d))
        q = self.q(self.n2(x)).view(R, N, self.h, d // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k_img[img], v_img[img])                      # all 576 memory tokens of the ROI's image
        x = x + self.cproj(a.transpose(1, 2).reshape(R, N, d))
        return x + self.mlp(self.n3(x))


def head():
    h = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, D_FEAT))
    nn.init.zeros_(h[2].weight)
    nn.init.zeros_(h[2].bias)
    return h


class FineAware(nn.Module):
    def __init__(self, vit):
        super().__init__()
        pe = vit.patch_embed.proj
        self.ov = nn.Conv2d(3, D_FEAT, 16, stride=8, padding=4)
        with torch.no_grad():
            self.ov.weight.copy_(pe.weight)
            self.ov.bias.copy_(pe.bias)
        self.norm = copy.deepcopy(vit.norm)
        for p in list(self.ov.parameters()) + list(self.norm.parameters()):
            p.requires_grad_(False)
        self.local = nn.Conv2d(3, D_LOCAL, 8, stride=8, padding=0)
        self.fuse = nn.Linear(D_FEAT + D_LOCAL, WIDTH)
        self.mem = nn.Linear(D_FEAT, WIDTH)
        self.blocks = nn.ModuleList([XBlock() for _ in range(DEPTH)])
        self.ln = nn.LayerNorm(WIDTH)
        self.head_mean, self.head_res = head(), head()

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def groups(self):
        return dict(local_stem=list(self.local.parameters()), fuse=list(self.fuse.parameters()), memory=list(self.mem.parameters()),
                    blocks=list(self.blocks.parameters()) + list(self.ln.parameters()), head_mean=list(self.head_mean.parameters()), head_res=list(self.head_res.parameters()))

    def image_side(self, x, final):
        """x (B, 3, 384, 384) canvas, final (B, 576, 768) -> fine (B, 2304, 384), per-block memory (k, v)."""
        f = torch.cat([self.ov(x), self.local(x)], 1).flatten(2).transpose(1, 2)            # (B, 2304, 960)
        fine = self.fuse(f)
        mem = self.mem(final) + parent_pos(x.device).to(final.dtype)
        return fine, [blk.kv_mem(mem) for blk in self.blocks]

    def decode(self, fine, kvs, xpre, img, idx, cy, cx, ok, key_ok, py, px, W):
        """One pass over every child of each ROI rectangle. idx/cy/cx/ok/key_ok (R, N) from rect_index; py/px/W (R,) rectangle origin
        and width in parents. Returns Y (R, N, 768) fp32 post-norm, Z (R, N, 384), DeltaMean per child (R, N, 768), DeltaRes (R, N, 768)."""
        R, N = idx.shape
        q = fine[img[:, None], idx] + child_pos(cy, cx).to(fine.dtype)
        for blk, (k, v) in zip(self.blocks, kvs):
            q = blk(q, key_ok, k, v, img)
        Z = self.ln(q)
        lp = (cy // 2 - py[:, None]) * W[:, None] + (cx // 2 - px[:, None])                   # local parent index
        lp = torch.where(ok, lp, torch.zeros_like(lp))
        P = int(lp.max()) + 1
        okf = ok.float()[..., None]
        zs = torch.zeros(R, P, WIDTH, device=Z.device, dtype=torch.float32).scatter_add_(1, lp[..., None].expand(-1, -1, WIDTH), Z.float() * okf)
        cnt = torch.zeros(R, P, 1, device=Z.device).scatter_add_(1, lp[..., None], okf)
        zmean = (zs / cnt.clamp_min(1)).to(Z.dtype)
        dmean = self.head_mean(zmean).float()                                                  # (R, P, 768)
        dres = self.head_res(Z).float() * okf
        rs = torch.zeros(R, P, D_FEAT, device=Z.device).scatter_add_(1, lp[..., None].expand(-1, -1, D_FEAT), dres)
        dres = (dres - (rs / cnt.clamp_min(1)).gather(1, lp[..., None].expand(-1, -1, D_FEAT))) * okf
        dmean_c = dmean.gather(1, lp[..., None].expand(-1, -1, D_FEAT))
        par = (cy // 2) * G + cx // 2
        base = xpre[img[:, None], par].float()
        with torch.autocast("cuda", enabled=False):
            Y = self.norm(base + dmean_c + dres)
        return Y, Z, dmean_c, dres


def n_params(m, trainable=True):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)
