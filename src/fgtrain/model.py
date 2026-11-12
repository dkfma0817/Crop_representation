"""One-way, RGB-free auxiliary decoder. Reuse G1's geometry and cross-attention blocks."""
import torch
from torch import nn
from src.g1x.model import ROIDecoder, XBlock, mlp


class FineDecoder(ROIDecoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.S, self.P = 384, 16
        d = 256
        self.input = nn.Linear(768, d)
        self.q_mlp = mlp(2 * (2 * self.NF_Q + 1) * 2 + 3, d)
        self.k_mlp = mlp(2 * (2 * self.NF_K + 1) * 2 + 4, d)
        self.q_grid = nn.Parameter(torch.empty(1, 48 * 48, d))
        nn.init.trunc_normal_(self.q_grid, std=0.02)
        self.ln_k = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([XBlock(d, 8) for _ in range(2)])
        self.ln_out, self.out = nn.LayerNorm(d), nn.Linear(d, 768)
        yy, xx = torch.meshgrid(torch.arange(24), torch.arange(24), indexing='ij')
        self.register_buffer('cell_c', torch.stack([(xx.flatten() + .5) * 16, (yy.flatten() + .5) * 16], -1), persistent=False)
        self.register_buffer('edges', torch.arange(25).float() * 16, persistent=False)

    def forward(self, full_tok, roi_img, box_c, qrel, fvalid):
        # No detach, no RGB, no return path to encoder.forward_features.
        return super().forward(self.input(full_tok), roi_img, box_c, qrel, fvalid, chunk=16)


class PixelDecoder(nn.Module):
    """B2 only: each patch feature predicts its own 16x16 RGB pixels, without RGB input."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 3 * 16 * 16))

    def forward(self, patches):
        return self.net(patches)
