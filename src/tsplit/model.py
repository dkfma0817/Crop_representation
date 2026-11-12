"""Learnable sub-patch token splitting (x2) on frozen DINOv3-B/16 384 features.

Layout: parent grid 24 x 24 (384 / 16), child grid 48 x 48. Parent (i, j) owns children (2i + a, 2j + b), a, b in {0, 1};
child flat index = cy * 48 + cx. Split output channel block k = 2a + b holds child (a, b).

  S1  child = Linear_k(final parent)                              (no neighbours, no transformer)
  S2  y = Linear_k(final parent); h = W_in y + pos(child); 4 pre-norm blocks over ALL children of the ROI (key mask = ROI
      padding + letterbox-invalid children); out = y + W_out LN(h)   (W_out zero-init -> S2 starts exactly at S1's init)
  S3  = S2 with the parent input [final, LN(patch-embed)] (concat, 1536) -> the same split / transformer sizes
Split init: every child = its parent (identity blocks; S3's early columns zero), so S1/S2/S3 all start at nearest copy.
pos(child) = fixed 2D sin-cos: D/4 parent row, D/4 parent col (global 0..23), D/4 sub-row, D/4 sub-col (0 / 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D_FEAT, G, G2 = 768, 24, 48
WIDTH, DEPTH, HEADS, MLP = 384, 4, 6, 4


def sincos(pos, dim):
    """pos (...,) float -> (..., dim): [sin(pos w_k), cos(pos w_k)], w_k = 10000^(-k / (dim / 2))."""
    w = 1.0 / 10000 ** (torch.arange(dim // 2, device=pos.device, dtype=torch.float32) / (dim // 2))
    a = pos.float()[..., None] * w
    return torch.cat([a.sin(), a.cos()], -1)


def child_pos(cy, cx, dim=WIDTH):
    q = dim // 4
    return torch.cat([sincos(cy // 2, q), sincos(cx // 2, q), sincos(cy % 2, q), sincos(cx % 2, q)], -1)


def rect_index(rects, n_max=None):
    """rects (R, 4) long [py, px, H, W] parent rectangles -> child flat idx (R, N), cy, cx (R, N), pad_ok (R, N) bool.
    Children are listed row-major over the (2H) x (2W) child rectangle."""
    py, px, H, W = rects.unbind(1)
    n = 4 * H * W
    N = int(n.max()) if n_max is None else n_max
    t = torch.arange(N, device=rects.device)[None]
    w2 = (2 * W)[:, None]
    cy = 2 * py[:, None] + torch.div(t, w2, rounding_mode="floor")
    cx = 2 * px[:, None] + t % w2
    ok = t < n[:, None]
    cy, cx = torch.where(ok, cy, 0).clamp(0, G2 - 1), torch.where(ok, cx, 0).clamp(0, G2 - 1)
    return cy * G2 + cx, cy, cx, ok


class Split(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.lin = nn.Linear(din, 4 * D_FEAT)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.zero_()
            for k in range(4):
                self.lin.weight[k * D_FEAT:(k + 1) * D_FEAT, :D_FEAT].copy_(torch.eye(D_FEAT))

    def forward(self, p):
        """p (B, 576, din) -> (B, 2304, 768) children in 48 x 48 row-major order."""
        B = p.shape[0]
        y = self.lin(p).view(B, G, G, 2, 2, D_FEAT).permute(0, 1, 3, 2, 4, 5)
        return y.reshape(B, G2 * G2, D_FEAT)


class Block(nn.Module):
    def __init__(self, d=WIDTH, heads=HEADS, mlp=MLP):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x, key_ok):
        R, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(R, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_ok[:, None, None, :])
        x = x + self.proj(a.transpose(1, 2).reshape(R, N, d))
        return x + self.mlp(self.n2(x))


class TokenSplit(nn.Module):
    def __init__(self, kind):
        super().__init__()
        assert kind in ("S1", "S2", "S3")
        self.kind = kind
        self.early_ln = nn.LayerNorm(D_FEAT) if kind == "S3" else None
        self.split = Split(2 * D_FEAT if kind == "S3" else D_FEAT)
        if kind != "S1":
            self.inp = nn.Linear(D_FEAT, WIDTH)
            self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
            self.ln = nn.LayerNorm(WIDTH)
            self.head = nn.Linear(WIDTH, D_FEAT)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @property
    def refines(self):
        return self.kind != "S1"

    def child_map(self, final, early=None):
        x = final if self.kind != "S3" else torch.cat([final, self.early_ln(early)], -1)
        return self.split(x)

    def refine(self, cmap, img, idx, cy, cx, key_ok):
        """cmap (B, 2304, 768); img (R,) image of each ROI; idx/cy/cx/key_ok (R, N) -> (R, N, 768) refined children."""
        y = cmap[img[:, None], idx]
        if not self.refines:
            return y
        h = self.inp(y) + child_pos(cy, cx).to(y.dtype)
        for blk in self.blocks:
            h = blk(h, key_ok)
        return y + self.head(self.ln(h))

    def groups(self):
        """Named parameter groups for gradient logging (S3 split weight split into final / early columns)."""
        g = dict(split=[self.split.lin.weight, self.split.lin.bias])
        if self.kind == "S3":
            g["early_ln"] = list(self.early_ln.parameters())
        if self.refines:
            g["inp"] = list(self.inp.parameters())
            g["blocks"] = list(self.blocks.parameters())
            g["head"] = list(self.ln.parameters()) + list(self.head.parameters())
        return g


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def param_report():
    out = {}
    for k in ("S1", "S2", "S3"):
        m = TokenSplit(k)
        g = {n: sum(p.numel() for p in v) for n, v in m.groups().items()}
        out[k] = dict(total=n_params(m), **g)
    out["S3_fusion_extra_vs_S2"] = out["S3"]["total"] - out["S2"]["total"]
    out["note"] = "S3 extra = early columns of the split Linear (768 x 3072) + early LayerNorm (1536); transformer width/depth/output identical to S2"
    return out


def flops_per_roi(kind, n_children):
    """Analytic multiply-adds x2 for the per-ROI part (transformer) + per-image split share is reported separately."""
    if kind == "S1":
        return 0
    N, d = n_children, WIDTH
    per_block = 2 * N * (3 * d * d + d * d + 2 * MLP * d * d) + 2 * 2 * N * N * d
    return 2 * N * D_FEAT * d + DEPTH * per_block + 2 * N * d * D_FEAT


def split_flops_per_image(kind):
    din = 2 * D_FEAT if kind == "S3" else D_FEAT
    return 2 * G * G * din * 4 * D_FEAT


if __name__ == "__main__":
    import json
    print(json.dumps(param_report(), indent=1))
