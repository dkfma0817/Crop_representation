"""Shared pieces of the P1 / P2 fine-token expansion pilot (x2 per axis, 384 -> 768 grid).

Layout (scale-general, used with scale = 2): parent (i, j) on the 24 x 24 grid owns children (s*i + a, s*j + b), a, b in [0, s);
child slot k = a * s + b (parent-major, child row-major = the professor's fine_layout); fine id = (s*i + a) * (s*24) + (s*j + b).
Coordinates (full image, never re-normalised inside a ROI): parent y = 2 (i + 0.5) / 24 - 1; child y = 2 (s*i + a + 0.5) / (s*24) - 1
(DINOv3 'separate' normalisation; identical to timm make_coords_dinov3 on the regular grids).
ROI read (P1 and P2 identical): the S3 read rectangle (src.tsplit.features.read_rects: every bilinear corner of every valid read point
inside) -> decoder children of the rectangle form a LOCAL (sH x sW) map -> the C0 crop-grid read points (continuous GT-box geometry,
bil_grid) are shifted into local coordinates -> grid_sample(bilinear, border, align_corners=False) -> 24 x 24 R2 tokens.
Nothing outside the rectangle is used for a valid read point; no teacher feature, zero or F384 value is inserted.
"""
import math

import torch
import torch.nn.functional as F

G = 24


_TAB = {}


def _table(n, dev):
    """timm make_coords_dinov3(n, n) (computed exactly as timm does, on CPU) -> device; row-major (i, j) -> (y, x) in [-1, 1]."""
    key = (n, str(dev))
    if key not in _TAB:
        from timm.layers.pos_embed_sincos import make_coords_dinov3
        _TAB[key] = make_coords_dinov3(n, n).to(dev)
    return _TAB[key]


def parent_coords(i, j):
    """2 (i + 0.5) / 24 - 1, looked up from the timm 24 x 24 table (bit-identical to the original RoPE coordinates)."""
    return _table(G, i.device)[i * G + j]


def child_coords(ci, cj, s=2):
    """2 (ci + 0.5) / (s * 24) - 1, looked up from the timm (s * 24)^2 table (the same formula; regular grid of the fine positions)."""
    return _table(s * G, ci.device)[ci * (s * G) + cj]


def rope_from_coords(rope, coords):
    """Arbitrary-coordinate version of timm RotaryEmbeddingDinoV3._get_pos_embed_from_coords (same periods buffer, same half layout,
    no train-time augmentation). coords (..., N, 2) in [-1, 1] -> (..., N, 2 * head_dim) = cat(sin, cos)."""
    periods = rope.periods
    ang = 2 * math.pi * coords.to(periods.dtype)[..., None] / periods          # (..., N, 2, dim/4)
    ang = ang.flatten(-2)                                                      # (..., N, dim/2)
    ang = ang.tile(2) if rope.rotate_half else ang.repeat_interleave(2, dim=-1)
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


def rect_parents(rects, Kmax=None):
    """rects (R, 4) [py, px, H, W] -> parent ids (R, K) row-major over the rectangle, valid (R, K)."""
    py, px, H, W = rects.unbind(1)
    K = H * W
    Km = int(K.max()) if Kmax is None else Kmax
    t = torch.arange(Km, device=rects.device)[None]
    ok = t < K[:, None]
    i = py[:, None] + torch.div(t, W[:, None], rounding_mode="floor")
    j = px[:, None] + t % W[:, None]
    pid = torch.where(ok, i * G + j, torch.zeros_like(i))
    return pid, ok


def child_ids(pid, s=2):
    """pid (R, K) -> fine ids (R, K, s*s), child row / col (R, K, s*s)."""
    i, j = pid // G, pid % G
    a = torch.arange(s, device=pid.device)
    ci = (s * i[..., None, None] + a[:, None]).expand(*pid.shape, s, s).reshape(*pid.shape, s * s)
    cj = (s * j[..., None, None] + a[None, :]).expand(*pid.shape, s, s).reshape(*pid.shape, s * s)
    return ci * (s * G) + cj, ci, cj


def local_read_loop(Y, rects, grid, s=2):
    """Y (R, K, s*s, D) decoder children of each rectangle (K = H*W row-major, padded) -> (R, 576, D) R2 tokens.
    Local map (R, D, sH, sW) built per ROI from its own children only; read points shifted into local coordinates."""
    R, K, C, D = Y.shape
    py, px, H, W = rects.unbind(1)
    Hm, Wm = int(H.max()), int(W.max())
    out = []
    for r in range(R):                         # rectangles differ in shape; per-ROI map (no padding values ever sampled)
        h, w = int(H[r]), int(W[r])
        m = Y[r, :h * w].view(h, w, s, s, D).permute(4, 0, 2, 1, 3).reshape(D, s * h, s * w)
        u = (grid[r].float() + 1) * (s * G) / 2 - 0.5                           # (576, 2) child-grid coords (x, y)
        ux = u[:, 0] - s * px[r].float()
        uy = u[:, 1] - s * py[r].float()
        gl = torch.stack([(ux + 0.5) / (s * w) * 2 - 1, (uy + 0.5) / (s * h) * 2 - 1], -1)
        out.append(F.grid_sample(m[None].float(), gl.view(1, 24, 24, 2), mode="bilinear", padding_mode="border", align_corners=False)[0].flatten(1).T)
    return torch.stack(out)


def local_read(Y, rects, grid, s=2):
    """Vectorised local_read_loop: each ROI's local (s*h x s*w) map is extended to a common (s*Hmax x s*Wmax) canvas by edge replication
    (= what grid_sample 'border' does at the per-ROI map edge), read points normalised by the canvas size -> identical samples to the
    per-ROI map at every point (checked against local_read_loop); no value outside the ROI's own children is ever used."""
    R, K, C, D = Y.shape
    py, px, H, W = rects.unbind(1)
    Hm, Wm = int(H.max()), int(W.max())
    yy = torch.arange(s * Hm, device=Y.device)[None, :, None]
    xx = torch.arange(s * Wm, device=Y.device)[None, None, :]
    yc = torch.minimum(yy, (s * H - 1)[:, None, None])
    xc = torch.minimum(xx, (s * W - 1)[:, None, None])
    slot = torch.div(yc, s, rounding_mode="floor") * W[:, None, None] + torch.div(xc, s, rounding_mode="floor")      # (R, sHm, sWm)
    sub = (yc % s) * s + (xc % s)
    flat = (slot * C + sub).reshape(R, -1)
    m = Y.float().reshape(R, K * C, D).gather(1, flat[..., None].expand(-1, -1, D)).view(R, s * Hm, s * Wm, D)
    u = (grid.float() + 1) * (s * G) / 2 - 0.5
    ux = u[..., 0] - s * px[:, None].float()
    uy = u[..., 1] - s * py[:, None].float()
    gl = torch.stack([(ux + 0.5) / (s * Wm) * 2 - 1, (uy + 0.5) / (s * Hm) * 2 - 1], -1)
    return F.grid_sample(m.permute(0, 3, 1, 2), gl.view(R, 24, 24, 2), mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)
