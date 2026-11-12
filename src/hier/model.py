"""H0/H1 Shared Hierarchical Patch-Pixel encoder: DINOv3 P384 (src.hcf.model.CoarseFine, unchanged) + one ROI-free Detail branch.

  detail (B, 3, 640, 640) -> Conv2d stride 8 -> 80 x 80 x 192 + fixed 2-D sin-cos -> 2 non-overlapping 8 x 8 window blocks (FineBlock, shift 0)
  -> LayerNorm -> bilinear antialiased resample to the P384 fine grid 48 x 48 -> memory 'detail'
  Q = A + s_d W_d(fine) + alpha Proj(detail); alpha learnable, init 0.1.     ROI boxes only choose where the finished maps are read.

The Detail branch is run once per image; coarse tokens, fine tokens and the global (CLS / patch mean) outputs are P384's.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hcf.model import CoarseFine, FineBlock, D, load_vit, sincos_2d

DD, DHEADS, DWS, DSTRIDE, DCANVAS, DGRID, FGRID = 192, 3, 8, 8, 640, 80, 48


def align_to_fine(z):
    """(B, C, 80, 80) -> (B, C, 48, 48), same physical extent (640 canvas = 5/3 x the 384 canvas)."""
    with torch.autocast("cuda", enabled=False):
        return F.interpolate(z.float(), size=(FGRID, FGRID), mode="bilinear", align_corners=False, antialias=True)


class Hier(CoarseFine):
    kind = "H"

    def __init__(self, vit, s0=0.1, alpha0=0.1):
        super().__init__(vit, s0)
        self.detail_embed = nn.Conv2d(3, DD, DSTRIDE, DSTRIDE)
        self.detail_blocks = nn.ModuleList([FineBlock(DD, DHEADS, DWS, 0) for _ in range(2)])
        self.detail_norm = nn.LayerNorm(DD)
        self.detail_proj = nn.Linear(DD, D)
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))
        self.detail_off = False                    # inference-only ablation: Q = A + s_d W_d(fine)

    def detail_tokens(self, d):
        assert d.shape[-2:] == (DCANVAS, DCANVAS), d.shape
        t = self.detail_embed(d).permute(0, 2, 3, 1)
        t = t + sincos_2d(DGRID, DGRID, DD, d.device)[None]
        for block in self.detail_blocks:
            t = block(t)
        z = align_to_fine(self.detail_norm(t).permute(0, 3, 1, 2))
        return z.flatten(2).transpose(1, 2)

    def forward_all(self, x, detail, collect=False):
        o = super().forward_all(x, collect=collect)
        o["mem"]["detail"] = self.detail_tokens(detail)
        o["grids"]["detail"] = 2
        return o

    def fuse_base(self, A, mem):
        return A + self.s_d * self.Wd(mem["fine"])

    def detail_residual(self, mem):
        return self.alpha * self.detail_proj(mem["detail"])

    def fuse(self, A, mem):
        q = self.fuse_base(A, mem)
        return q if self.detail_off else q + self.detail_residual(mem)

    def detail_named(self):
        for n, p in self.named_parameters():
            if n.startswith(("detail_", "alpha")):
                yield n, p


def load_hier(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["cond"] in ("H0", "H1")
    m = Hier(load_vit(dev), ck["s0"]).to(dev)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck
