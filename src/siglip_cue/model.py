"""M2: the SigLIP2 P8 CoarseFine plus a light convolutional Detail path that sees the undistorted image once.

    Native RGB (<= 640 per side, no upscaling) -> conv stem, stride 8 -> resampled onto the P8 48x48 fine grid -> zero-initialized 1x1 projection
    -> added to the final fine tokens.

No ROI enters the model: the fine map stays 48x48x384, so the stored representation, the ROI read (read_q / R2 grid sampling) and fuse() are the
P8 ones. The Detail path runs once per image, independent of how many ROIs are read later. M1 is P8 unchanged (src.siglip_pm.model.CoarseFine).
"""
import torch
from torch import nn
import torch.nn.functional as F

from src.siglip_pm.model import MODEL_ID, CoarseFine, load_vit

NATIVE = 640
FINE_GRID = 48
FD = 384
DETAIL_DIM = 128
DETAIL_SEED = 700


def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 4, 2, 1), nn.GroupNorm(8, c_out), nn.GELU())


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.GroupNorm(8, dim), nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1), nn.GroupNorm(8, dim))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class DetailEncoder(nn.Module):
    """Stride 8. Kernel-4 / stride-2 / pad-1 downsampling keeps feature cell k centred at input coordinate (k + .5) * stride."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(block(3, 32), block(32, 64), block(64, DETAIL_DIM), ResBlock(DETAIL_DIM), ResBlock(DETAIL_DIM))
        self.proj = nn.Conv2d(DETAIL_DIM, FD, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.images_seen = 0

    @staticmethod
    def grid(geo, n=2 * FINE_GRID):
        """Sample points (n x n) on the 384 canvas, in normalized coordinates of the padded native canvas, and their inside-image mask."""
        centre = (torch.arange(n, device=geo.device, dtype=torch.float32) + .5) * 384 / n
        W, H, nw, nh, ox, oy = geo.unbind(1)
        x = (centre[None] - ox[:, None]) * W[:, None] / nw[:, None]
        y = (centre[None] - oy[:, None]) * H[:, None] / nh[:, None]
        grid = torch.stack([x[:, None, :].expand(-1, n, -1), y[:, :, None].expand(-1, -1, n)], -1)
        valid = ((grid[..., 0] >= 0) & (grid[..., 0] <= W[:, None, None]) & (grid[..., 1] >= 0) & (grid[..., 1] <= H[:, None, None])).float()
        return grid * 2 / NATIVE - 1, valid[:, None]

    def forward(self, native, geo):
        self.images_seen += len(native)
        z = self.stem(native)
        grid, valid = self.grid(geo)
        sampled = F.grid_sample(z, grid.to(z.dtype), mode='bilinear', padding_mode='zeros', align_corners=False)
        fine = F.avg_pool2d(sampled, 2)
        coverage = F.avg_pool2d(valid.to(z.dtype), 2)
        out = self.proj(fine) * coverage
        return out.flatten(2).transpose(1, 2)


class CoarseFineDetail(CoarseFine):
    def __init__(self, vit, s0=.1):
        super().__init__(vit, s0)
        with torch.random.fork_rng():
            torch.manual_seed(DETAIL_SEED)
            self.detail = DetailEncoder()
        self.pending = None
        self.pos = 0

    def set_pending(self, native, geo):
        self.pending, self.pos = (native, geo), 0

    def clear_pending(self):
        self.pending, self.pos = None, 0

    def forward_all(self, x, collect=False, native=None, geo=None):
        if native is None:
            assert self.pending is not None, 'M2 needs the native-resolution image'
            native = self.pending[0][self.pos:self.pos + len(x)]
            geo = self.pending[1][self.pos:self.pos + len(x)]
            self.pos += len(x)
        assert len(native) == len(x) == len(geo)
        o = super().forward_all(x, collect)
        fine = o['mem']['fine']
        with torch.autocast('cuda', enabled=False):
            residual = self.detail(native.float(), geo.float())
        o['mem'] = dict(fine=fine + residual.to(fine.dtype))
        return o


def build_m2(dev, s0=.1):
    torch.manual_seed(0)
    vit = load_vit(dev)
    torch.manual_seed(0)
    return CoarseFineDetail(vit, s0).to(dev)


def load_m2(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['backbone'] == MODEL_ID and ck['cond'] == 'M2'
    m = build_m2(dev, ck.get('s0', .1))
    m.load_state_dict(ck['model'], strict=True)
    m.eval().requires_grad_(False)
    return m, ck


def from_initial(path, dev):
    """The P8 start state with the Detail path added (proj zero-initialized: the output equals P8 at step 0)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['cond'] == 'P' and ck['step'] == 0 and ck['backbone'] == MODEL_ID
    m = build_m2(dev, ck.get('s0', .1))
    missing, unexpected = m.load_state_dict(ck['model'], strict=False)
    assert not unexpected and all(k.startswith('detail.') for k in missing) and missing, (missing, unexpected)
    return m, ck
