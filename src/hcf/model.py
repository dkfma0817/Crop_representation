"""Hierarchical Coarse-Fine pilot (2026-11-08): the two shared-memory encoders. Both wrap the same timm DINOv3-B/16 ViT.

  P  CoarseFine   : coarse ViT (24x24x768) + fine path (real 8x8-px patch embedding -> 48x48x384, 4 window/shifted-window
                    blocks); after every 3 coarse blocks: one fine block, then a coarse <-> fine residual exchange.
                    memory = {coarse 24x24x768, fine 48x48x384}.   Q = A + s_d W_d(B)    (W_d: Linear 384 -> 768)
  M  MultiLayer   : the plain ViT; memory = L4, L8 (raw block outputs -> LN) and L12 (final-LN tokens), 3 x 24x24x768.
                    Q = L12 + s4 W4(LN(L4)) + s8 W8(LN(L8))                       (W: Linear 768 -> 768)
Stored scalars are equal: 24*24*768 + 48*48*384 = 3*24*24*768 = 1,327,104.

No ROI enters encode: `forward_all(x)` is ROI-free; `fuse(A, mem)` is linear and acts on anything aligned (pooled vectors or
tokens sampled on an ROI grid). The readout has no transformer, no query, no decoder.

Fine-path rules fixed for this pilot (one setting, not searched):
  fine patch embed Conv2d(3, 384, 8, 8), 2-D sin-cos absolute position (no parameters), 6 heads, MLP x4, window 8 fine tokens
  (= 64 px), shift 0 in fine blocks 1, 3 and 4 in blocks 2, 4 (cyclic shift + Swin attention mask), non-square input padded
  bottom/right to a window multiple (ADE only; PACO / ImageNet are 384 x 384 -> 48 x 48 = 6 x 6 windows, no padding).
Interaction (simultaneous, computed from pre-update copies):
  c <- c + s_f2c L_f2c(avgpool_2x2(f))     f <- f + s_c2f L_c2f(broadcast_2x2(c))        patch tokens only; CLS + registers untouched
Residual scales start small and non-zero (s0, see build): zero init would leave the fine path without gradient.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D, FD, HEADS, WS, FP = 768, 384, 6, 8, 8
ML = (4, 8, 12)
SCALARS = dict(P=24 * 24 * 768 + 48 * 48 * 384, M=3 * 24 * 24 * 768, original=24 * 24 * 768)


def sincos_2d(h, w, dim, device):
    assert dim % 4 == 0
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    om = torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4 - 1)
    om = 1.0 / 10000 ** om
    e = []
    for v in (ys.reshape(-1), xs.reshape(-1)):
        o = v[:, None] * om[None]
        e += [o.sin(), o.cos()]
    return torch.cat(e, 1).reshape(h, w, dim)


class WindowAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.qkv, self.proj = heads, nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask[:, None])
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class FineBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift, mlp_ratio=4):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1, self.attn, self.norm2 = nn.LayerNorm(dim), WindowAttn(dim, heads), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self._mask = {}

    def _attn_mask(self, H, W, dev):
        key = (H, W, str(dev))
        if key not in self._mask:
            ws, s = self.ws, self.shift
            cnt, r = torch.zeros(H, W, device=dev, dtype=torch.long), 0
            for hs in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                for wsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                    cnt[hs, wsl] = r
                    r += 1
            win = cnt.reshape(H // ws, ws, W // ws, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
            self._mask[key] = win[:, None] == win[:, :, None]            # True = attend (same sub-region after the cyclic shift)
        return self._mask[key]

    def forward(self, x):
        B, H, W, C = x.shape
        ws, s = self.ws, self.shift
        y = self.norm1(x)
        pb, pr = (ws - H % ws) % ws, (ws - W % ws) % ws
        if pb or pr:
            y = F.pad(y.permute(0, 3, 1, 2), (0, pr, 0, pb)).permute(0, 2, 3, 1)
        Hp, Wp = H + pb, W + pr
        if s:
            y = y.roll((-s, -s), (1, 2))
        win = y.reshape(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        win = self.attn(win, self._attn_mask(Hp, Wp, x.device).repeat(B, 1, 1) if s else None)
        y = win.reshape(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if s:
            y = y.roll((s, s), (1, 2))
        y = y[:, :H, :W]
        x = x + y
        return x + self.mlp(self.norm2(x))


class Interaction(nn.Module):
    def __init__(self, cd, fd, s0):
        super().__init__()
        self.f2c, self.c2f = nn.Linear(fd, cd), nn.Linear(cd, fd)
        self.s_f2c, self.s_c2f = nn.Parameter(torch.tensor(float(s0))), nn.Parameter(torch.tensor(float(s0)))

    def forward(self, c, f):
        """c (B,H,W,D), f (B,2H,2W,FD) -> updated, ratios ||res|| / ||target|| (mean token norm, detached)"""
        fp = F.avg_pool2d(f.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        cu = c.repeat_interleave(2, 1).repeat_interleave(2, 2)
        rc, rf = self.s_f2c * self.f2c(fp), self.s_c2f * self.c2f(cu)
        with torch.no_grad():
            d = dict(c=float(rc.norm(dim=-1).mean() / c.norm(dim=-1).mean().clamp_min(1e-8)), f=float(rf.norm(dim=-1).mean() / f.norm(dim=-1).mean().clamp_min(1e-8)))
        return c + rc, f + rf, d


def _prefix(vit, x):
    """patch embed + prefix tokens + RoPE (= Eva.forward_features up to the blocks). -> tokens (B, 5 + N, D), rope, (H, W)"""
    h = vit.patch_embed(x)
    H, W = h.shape[1], h.shape[2]
    t, rope = vit._pos_embed(h)
    return vit.norm_pre(t), rope, H, W


def _block(vit, i, t, rope):
    assert not getattr(vit, "rope_mixed", False)
    return vit.blocks[i](t, rope=rope)


class CoarseFine(nn.Module):
    kind = "P"

    def __init__(self, vit, s0=0.1):
        super().__init__()
        self.vit, self.npt = vit, vit.num_prefix_tokens
        self.fine_embed = nn.Conv2d(3, FD, FP, FP)
        self.fine_blocks = nn.ModuleList([FineBlock(FD, HEADS, WS, 0 if i % 2 == 0 else WS // 2) for i in range(4)])
        self.inter = nn.ModuleList([Interaction(D, FD, s0) for _ in range(4)])
        self.fine_norm, self.Wd = nn.LayerNorm(FD), nn.Linear(FD, D)
        self.s_d = nn.Parameter(torch.tensor(float(s0)))
        self.diag = {}

    @property
    def num_prefix_tokens(self):
        return self.npt

    @property
    def cls_token(self):
        return self.vit.cls_token

    @property
    def reg_token(self):
        return self.vit.reg_token

    def forward_head(self, t, pre_logits=False):
        return self.vit.forward_head(t, pre_logits=pre_logits)

    def forward_all(self, x, collect=False):
        v, npt = self.vit, self.npt
        t, rope, H, W = _prefix(v, x)
        f = self.fine_embed(x).permute(0, 2, 3, 1)
        f = f + sincos_2d(2 * H, 2 * W, FD, x.device)[None]
        d = {}
        for st in range(4):
            for i in range(3 * st, 3 * st + 3):
                t = _block(v, i, t, rope)
            f = self.fine_blocks[st](f)
            p, c = t[:, :npt], t[:, npt:].reshape(len(x), H, W, D)
            c, f, r = self.inter[st](c, f)
            t = torch.cat([p, c.reshape(len(x), H * W, D)], 1)
            if collect:
                d.update({f"stage{st + 1}_res_c": r["c"], f"stage{st + 1}_res_f": r["f"]})
        t = v.norm(t)
        f = self.fine_norm(f)
        if collect:
            self.diag = d
        return dict(cls=t[:, 0], coarse=t[:, npt:], H=H, W=W, mem=dict(fine=f.reshape(len(x), 4 * H * W, FD)), grids=dict(fine=2), reg=t[:, 1:npt])

    def fuse(self, A, mem):
        return A + self.s_d * self.Wd(mem["fine"])

    def forward_features(self, x):
        o = self.forward_all(x)
        return torch.cat([o["cls"][:, None], o["reg"], o["coarse"]], 1)

    def new_modules(self):
        return [self.fine_embed, self.fine_blocks, self.inter, self.fine_norm, self.Wd]

    def new_named(self):
        for n, p in self.named_parameters():
            if not n.startswith("vit."):
                yield n, p


class MultiLayer(nn.Module):
    kind = "M"

    def __init__(self, vit, s0=0.1, layers=ML):
        super().__init__()
        self.vit, self.npt, self.layers = vit, vit.num_prefix_tokens, tuple(layers)
        assert self.layers[-1] == len(vit.blocks) == 12
        self.ln = nn.ModuleDict({str(l): nn.LayerNorm(D) for l in self.layers[:-1]})
        self.W = nn.ModuleDict({str(l): nn.Linear(D, D) for l in self.layers[:-1]})
        self.s = nn.ParameterDict({str(l): nn.Parameter(torch.tensor(float(s0))) for l in self.layers[:-1]})
        self.diag = {}

    @property
    def num_prefix_tokens(self):
        return self.npt

    @property
    def cls_token(self):
        return self.vit.cls_token

    @property
    def reg_token(self):
        return self.vit.reg_token

    def forward_head(self, t, pre_logits=False):
        return self.vit.forward_head(t, pre_logits=pre_logits)

    def forward_all(self, x, collect=False):
        v, npt = self.vit, self.npt
        t, rope, H, W = _prefix(v, x)
        raw = {}
        for i in range(len(v.blocks)):
            t = _block(v, i, t, rope)
            if i + 1 in self.layers[:-1]:
                raw[i + 1] = t[:, npt:]
        t = v.norm(t)
        mem = {f"l{l}": self.ln[str(l)](raw[l]) for l in raw}
        return dict(cls=t[:, 0], coarse=t[:, npt:], H=H, W=W, mem=mem, grids={k: 1 for k in mem}, raw=raw, reg=t[:, 1:npt])

    def fuse(self, A, mem):
        q = A
        for l in self.layers[:-1]:
            q = q + self.s[str(l)] * self.W[str(l)](mem[f"l{l}"])
        return q

    def forward_features(self, x):
        o = self.forward_all(x)
        return torch.cat([o["cls"][:, None], o["reg"], o["coarse"]], 1)

    def new_modules(self):
        return [self.ln, self.W, self.s]

    def new_named(self):
        for n, p in self.named_parameters():
            if not n.startswith("vit."):
                yield n, p


def load_vit(dev, ckpt=None):
    """timm DINOv3-B/16 @384 (dynamic size). ckpt: G1 run checkpoint path (key 'enc' = the G1 EMA/student encoder state dict)."""
    import timm
    vit = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, img_size=384, dynamic_img_size=True)
    if ckpt is not None:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        vit.load_state_dict(ck["enc"], strict=True)
    return vit.to(dev)


def build_student(kind, dev, ckpt, s0):
    vit = load_vit(dev, ckpt)
    for p in vit.parameters():
        p.requires_grad_(True)
    torch.manual_seed(7)                                              # new-module init seed (fixed; calibration = training init)
    m = (CoarseFine(vit, s0) if kind == "P" else MultiLayer(vit, s0)).to(dev)
    return m


def checksum(params):
    return float(sum(p.double().sum() for p in params))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load_hcf(path, dev):
    """Saved checkpoint -> frozen eval model (P / M; EMA weights unless the checkpoint is a student)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vit = load_vit(dev)
    m = (CoarseFine(vit, ck["s0"]) if ck["cond"] == "P" else MultiLayer(vit, ck["s0"])).to(dev)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck
