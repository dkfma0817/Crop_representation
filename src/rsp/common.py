"""RSP (ROI Sub-patch Predictor, masked fine-feature prediction) — shared geometry, PI-resize and RoPE helpers.

Coordinate contract (every token of student, predictor and dense teacher):
  * canvas = the RGB384 letterbox canvas (src.viz.c0_pca.LB(W, H, 384, 16)); a token's position = the CENTRE of the canvas
    area it covers, in canvas px (y, x).
  * RoPE coordinate = 2 * centre / 384 - 1 in [-1, 1], (y, x) order ('ij'), identical to timm make_coords_dinov3('separate')
    for the 24 x 24 coarse grid (checked in src.rsp.checks).
  * coarse cell (i, j): centre ((i + .5) 16, (j + .5) 16); fine cell (u, v) at p_T = 4 px: centre ((u + .5) 4, (v + .5) 4).
    Parent (i, j) owns fine cells (4i + a, 4j + b), a, b in 0..3. Fine grid side GF = 96, flat index u * 96 + v.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

S, P, PT = 384, 16, 4
G = S // P              # 24 coarse cells per side
GF = S // PT            # 96 fine cells per side
R_FP = P // PT          # 4 fine cells per parent side
PREFIX = 5              # DINOv3 CLS + 4 registers (checked against the checkpoint)
D_FEAT = 768
EV_SIDE, EV_PATCH = 64, 8   # ROI evidence: RGB 64 x 64 -> Conv 8 x 8 stride 8 -> 8 x 8 = 64 tokens
N_EV = (EV_SIDE // EV_PATCH) ** 2
K_TRAIN = 256


def to_rope_coord(c):
    """canvas px centre -> [-1, 1]."""
    return 2.0 * c / S - 1.0


def coarse_centres(dev=None):
    i, j = torch.meshgrid(torch.arange(G, device=dev), torch.arange(G, device=dev), indexing="ij")
    return torch.stack([(i.flatten() + .5) * P, (j.flatten() + .5) * P], -1).float()       # (576, 2) (y, x) px


def fine_rect_index(frects, n_max=None):
    """frects (R, 4) long [fy, fx, h, w] fine-cell rectangles -> flat fine idx (R, N), fy, fx (R, N), ok (R, N); row-major."""
    fy0, fx0, h, w = frects.unbind(1)
    n = h * w
    N = int(n.max()) if n_max is None else n_max
    t = torch.arange(N, device=frects.device)[None]
    ww = w[:, None].clamp_min(1)
    uy = fy0[:, None] + torch.div(t, ww, rounding_mode="floor")
    ux = fx0[:, None] + t % ww
    ok = t < n[:, None]
    uy, ux = torch.where(ok, uy, 0).clamp(0, GF - 1), torch.where(ok, ux, 0).clamp(0, GF - 1)
    return uy * GF + ux, uy, ux, ok


def fine_valid(content, uy, ux):
    """content (R, 4) [x0, y0, x1, y1] canvas px (letterbox content of the ROI's image); fine cells (uy, ux) -> bool overlaps content."""
    x0, y0, x1, y1 = (content[:, k:k + 1] for k in range(4))
    cx0, cy0 = ux.float() * PT, uy.float() * PT
    ox = (torch.minimum(cx0 + PT, x1) - torch.maximum(cx0, x0)).clamp_min(0)
    oy = (torch.minimum(cy0 + PT, y1) - torch.maximum(cy0, y0)).clamp_min(0)
    return (ox * oy) > 0


# ---------------------------------------------------------------- PI-resize (FlexiViT, Beyer et al. 2023, Sec. 3.4)
def resize_matrix(p_from, p_to, method="bilinear"):
    """Linear map B (p_to^2, p_from^2) of the image resize p_from x p_from -> p_to x p_to used by FlexiViT
    (jax.image.resize 'bilinear', antialias on downsampling == torch F.interpolate(bilinear, antialias=True)).
    Built by resizing the p_from^2 basis images."""
    eye = torch.eye(p_from * p_from, dtype=torch.float64).view(-1, 1, p_from, p_from)
    if method == "area":
        out = F.adaptive_avg_pool2d(eye, p_to)
    else:
        out = F.interpolate(eye, size=(p_to, p_to), mode=method, antialias=True, align_corners=False)
    return out.view(p_from * p_from, p_to * p_to).T.contiguous()            # B: x_to = B x_from


def pi_resize(w, p_to, method="bilinear"):
    """FlexiViT PI-resize, literal downsampling form: w (C_out, C_in, p, p) -> w_hat (C_out, C_in, p_to, p_to) = pinv(B^T) w with
    B = resize p -> p_to. <B x, w_hat> = <x, w> holds for x in the row space of B; least-squares (isotropic-x assumption) otherwise.
    Kept for verification only (on real canvases its tokens differ from the 4x-zoomed tokens by ~59 %, src.rsp.checks)."""
    co, ci, p, _ = w.shape
    B = resize_matrix(p, p_to, method)                                       # (p_to^2, p^2)
    Pm = torch.linalg.pinv(B.T)                                              # (p_to^2, p^2)
    wh = w.detach().double().cpu().reshape(co * ci, p * p) @ Pm.T             # (co*ci, p_to^2)
    return wh.reshape(co, ci, p_to, p_to).to(device=w.device, dtype=w.dtype), B


def upsample_matrix(p_small, p_big, method="bilinear"):
    """U (p_big^2, p_small^2): x_big = U x_small, the 4x upsampling of a p_small patch (F.interpolate, align_corners False, the patch's
    own border clamped)."""
    eye = torch.eye(p_small * p_small, dtype=torch.float64).view(-1, 1, p_small, p_small)
    out = F.interpolate(eye, size=(p_big, p_big), mode=method, align_corners=False)
    return out.view(p_small * p_small, p_big * p_big).T.contiguous()


def pi_resize_up(w, p_to, method="bilinear"):
    """PI-resize in its exact (upsampling) regime, used by the dense teacher: the p_to x p_to canvas patch x_small is what a 16 x 16
    patch of the p/p_to-times zoomed canvas, U x_small, would contain. FlexiViT's identity <x_big, w> = <x_small, w_hat> with
    x_big = U x_small is solved EXACTLY for every x_small by w_hat = U^T w (bias unchanged). Frozen, derived from the original kernel."""
    co, ci, p, _ = w.shape
    U = upsample_matrix(p_to, p, method)                                     # (p^2, p_to^2)
    wh = w.detach().double().cpu().reshape(co * ci, p * p) @ U               # (co*ci, p_to^2)
    return wh.reshape(co, ci, p_to, p_to).to(device=w.device, dtype=w.dtype), U


# ---------------------------------------------------------------- RoPE helpers
def dino_rope(vit, centres_px):
    """centres (N, 2) (y, x) canvas px -> DINOv3 rope embedding (N, 2 * head_dim) [sin, cos] via the model's own period buffer."""
    sin, cos = vit.rope._get_pos_embed_from_coords(to_rope_coord(centres_px.float()))
    return torch.cat([sin, cos], -1)


def rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


def apply_rope(x, emb):
    """x (..., N, dh); emb (..., N, 2 dh) [sin, cos] broadcastable (rotate-half layout, = timm apply_rot_embed_cat(half=True))."""
    sin, cos = emb.chunk(2, -1)
    return x * cos + rot_half(x) * sin


class Rope2D:
    """Predictor RoPE: continuous (y, x) in [-1, 1], head_dim dh, dh / 4 log-spaced periods in [min_p, max_p] per axis
    (DINOv3 layout: angles (y-freqs, x-freqs) tiled twice, rotate-half). Non-positional tokens get the identity (sin 0, cos 1)."""

    def __init__(self, dh, min_p=0.05, max_p=8.0):
        self.dh = dh
        n = dh // 4
        self.periods = min_p * (max_p / min_p) ** torch.linspace(0, 1, n, dtype=torch.float64)

    def __call__(self, centres_px, positional=None):
        c = to_rope_coord(centres_px.double())                               # (..., 2)
        ang = 2 * math.pi * c[..., None] / self.periods.to(c.device)        # (..., 2, n)
        ang = ang.flatten(-2)                                                # (..., dh / 2)
        if positional is not None:
            ang = ang * positional[..., None].double()
        ang = ang.repeat(*([1] * (ang.dim() - 1)), 2)                        # tile -> (..., dh)
        return torch.cat([ang.sin(), ang.cos()], -1).float()


def box_canvas_from_rect(rects, content):
    """parent rects (R, 4) [py, px, H, W] and content (R, 4) [x0, y0, x1, y1] -> ROI box (R, 4) ltrb canvas px = rect ∩ content."""
    py, px, H, W = (rects[:, k].float() for k in range(4))
    l, t, r, b = px * P, py * P, (px + W) * P, (py + H) * P
    return torch.stack([torch.maximum(l, content[:, 0]), torch.maximum(t, content[:, 1]), torch.minimum(r, content[:, 2]), torch.minimum(b, content[:, 3])], 1)


def read_frects(bil_grid, cov):
    """(R, 576, 2) grid_sample coords on the 384 canvas, (R, 576) coverage -> (R, 4) long [fy, fx, h, w] fine rectangle covering every
    bilinear corner (align_corners=False) of every valid (cov > 0) read point on the 96 x 96 fine grid."""
    u = ((bil_grid.float() + 1) * GF / 2 - 0.5).clamp(0, GF - 1)            # (..., 2) = (x, y) fine coordinates; border padding clamps here
    ok = (cov > 0)[..., None]
    big = torch.tensor(1e9, device=u.device)
    lo = torch.where(ok, u, big).amin(1).floor().clamp(0, GF - 1).long()
    hi = (torch.where(ok, u, -big).amax(1).floor() + 1).clamp(0, GF - 1).long()
    none = ~(cov > 0).any(1)
    if none.any():                                                           # no valid read point: 1 x 1 cell at the grid centre (never read)
        lo[none], hi[none] = GF // 2, GF // 2
    return torch.stack([lo[:, 1], lo[:, 0], hi[:, 1] - lo[:, 1] + 1, hi[:, 0] - lo[:, 0] + 1], 1)


def sample_local(maps, frects, grid):
    """maps (R, D, h, w) local fine maps of each rectangle (ZERO-padded to the chunk max) -> (R, 576, D) bilinear reads at grid
    (= grid_sample on the full 96 x 96 map, align_corners=False, for every point whose corners lie in the rectangle)."""
    R, D, hm, wm = maps.shape
    u = ((grid.float() + 1) * GF / 2 - 0.5).clamp(0, GF - 1)                 # fine coords (x, y), clamped like padding_mode='border'
    lx = u[..., 0] - frects[:, 1:2].float()
    ly = u[..., 1] - frects[:, 0:1].float()
    gx = (lx + .5) / wm * 2 - 1
    gy = (ly + .5) / hm * 2 - 1
    g = torch.stack([gx, gy], -1).view(R, 24, 24, 2)
    return F.grid_sample(maps.float(), g, mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)


def np_rng(*keys):
    return np.random.default_rng(list(keys))
