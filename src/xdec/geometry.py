"""Shared ROI geometry for the decoder-architecture pilot (E1-E7, R00). ONE coordinate chain for everything:

  C0 teacher cell centre (crop canvas px)  --C0 letterbox inverse-->  original image px
  --F336 full letterbox-->  full canvas px  --/14 - 0.5-->  continuous full-feature grid coordinate

The chain itself is the unchanged `roi_geometry` (dinov2_oracle_region_decoder data.preprocessing, also used by R00,
MSD, S-series and IVC): its `bil_grid` is the bicubic / bilinear sampling grid, `valid_w` the C0 content coverage,
`inside` the ROI-token selection. This module only ADDS one per-ROI vector `roi_frame` (8 floats) from which the
ROI-relative coordinates of queries and memory tokens are computed on the GPU:

  roi_frame = [L, T, R, B, ox_c, oy_c, nw_c, nh_c]
      L, T, R, B   : the integer GT box (l, t, r, b) mapped to the F336 full canvas  (X = ox_f + x * sx_f)
      ox_c .. nh_c : content rectangle of the C0 crop letterbox (crop canvas px)

  ROI-relative coordinate (u, v): 0 = left / top ROI edge, 1 = right / bottom edge, NOT clamped outside [0, 1].
      query cell (i, j)  : u = ((j + 0.5) * 14 - ox_c) / nw_c          (C0 padding cells -> u < 0 or u > 1)
      full token (i, j)  : u = ((j + 0.5) * 14 - L) / (R - L)
  Both are the same affine chain: original x = l + u (r - l)  (crop: sx_c = nw_c / w exactly; full: X = L + u (R - L)),
  so query and memory share one coordinate system (checked in alignment_test: X_query from u == X from bil_grid).
  Cell size in ROI units: query (14 / nw_c, 14 / nh_c); full token (14 / (R - L), 14 / (B - T)).

Install with `install()` BEFORE the datasets are built (patches src.msd.common.roi_geometry / GEOM_KEYS /
collate_geoms so that MSDImageDS and CocoStreamDS carry `roi_frame`; nothing else of the batch changes).
"""
import numpy as np
import torch
import torch.nn.functional as F

import src.msd.common as MS

G, PATCH, S = 24, 14, 336
_ORIG = {}


def roi_frame(g):
    l, t, r, b = g["ltrb"]
    full, crop = g["full"], g["crop"]
    L, T = full.to_canvas(l, t)
    R, B = full.to_canvas(r, b)
    return np.array([L, T, R, B, crop.ox, crop.oy, crop.nw, crop.nh], np.float64)


def install():
    """Idempotent monkey-patch of src.msd.common: roi_geometry() also returns 'roi_frame'; collate adds it (float64 -> float32
    after subtraction is NOT done here: we keep float64 on CPU and cast in the model after forming differences)."""
    if _ORIG:
        return
    _ORIG["rg"], _ORIG["cg"], _ORIG["keys"] = MS.roi_geometry, MS.collate_geoms, MS.GEOM_KEYS

    def rg(W, H, ltrb):
        g = _ORIG["rg"](W, H, ltrb)
        g["roi_frame"] = roi_frame(g)
        return g

    def cg(geoms, img_index):
        out = _ORIG["cg"](geoms, img_index)
        out["roi_frame"] = torch.from_numpy(np.stack([g["roi_frame"] for g in geoms]))
        return out

    MS.roi_geometry, MS.collate_geoms = rg, cg
    MS.GEOM_KEYS = tuple(_ORIG["keys"]) + ("roi_frame",)


def _cell_centres(dev):
    c = (torch.arange(G, device=dev, dtype=torch.float64) + 0.5) * PATCH
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)                                         # (576,) canvas px, row-major


def query_uv(g):
    """-> uv (R, 576, 2), size (R, 2): ROI-relative coordinates of the 24x24 C0 output cells (float32)."""
    f = g["roi_frame"].double()
    xx, yy = _cell_centres(f.device)
    u = (xx[None] - f[:, 4:5]) / f[:, 6:7]
    v = (yy[None] - f[:, 5:6]) / f[:, 7:8]
    size = torch.stack([PATCH / f[:, 6], PATCH / f[:, 7]], -1)
    return torch.stack([u, v], -1).float(), size.float()


def full_uv(g):
    """-> uv (R, 576, 2), size (R, 2): ROI-relative coordinates of all 576 full-image tokens (float32)."""
    f = g["roi_frame"].double()
    xx, yy = _cell_centres(f.device)
    w, h = f[:, 2] - f[:, 0], f[:, 3] - f[:, 1]
    u = (xx[None] - f[:, 0:1]) / w[:, None]
    v = (yy[None] - f[:, 1:2]) / h[:, None]
    size = torch.stack([PATCH / w, PATCH / h], -1)
    return torch.stack([u, v], -1).float(), size.float()


def query_canvas_from_uv(g):
    """Full-canvas px of the query cells recomputed from (u, v) and the ROI box: X = L + u (R - L). (alignment test)"""
    f = g["roi_frame"].double()
    uv, _ = query_uv(g)
    uv = uv.double()
    X = f[:, 0:1] + uv[..., 0] * (f[:, 2:3] - f[:, 0:1])
    Y = f[:, 1:2] + uv[..., 1] * (f[:, 3:4] - f[:, 1:2])
    return torch.stack([X, Y], -1)


def bicubic_roi(full_tokens, g):
    """THE bicubic ROI tensor B used by E1/E4/E5/E6 and the Bicubic-only baseline: exactly the 11-05 IVC call
    src.ivc.common.sample_roi(feats[12], geom, 'bicubic') = grid_sample(fp32, bil_grid, mode='bicubic',
    padding_mode='border', align_corners=False), no clamping of feature values. Always run in fp32 (autocast off)."""
    from src.ivc.common import sample_roi
    with torch.autocast("cuda", enabled=False):
        return sample_roi(full_tokens.float(), g, "bicubic")


# ------------------------------------------------------------------------------------------------ masks
def roi_token_mask(g):
    """(R, n_ctx) bool: ROI-token selection (roi_geometry 'inside' = every full-grid cell that overlaps the GT box on the
    F336 canvas, floor / ceil rule) AND not batch padding. Always >= 1 token per ROI (roi_geometry raises otherwise)."""
    return g["inside"] & ~g["pad"]


def valid_query_mask(g):
    """(R, 576) bool: C0 valid-coverage mask (crop letterbox content coverage > 0)."""
    return g["valid_w"] > 0
