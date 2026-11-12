"""Decoder study on CLIP-L G1-cross (2026-11-07 night): feature paths that turn ONE frozen full-image encoding into a
24 x 24 crop-grid token map for a ROI, aligned cell-for-cell with the C0 crop teacher.

Crop-grid <-> full-canvas alignment (no square warping):
  the C0 teacher sees the original-pixel GT crop letterboxed into the 336 canvas (src.viz.c0_pca.LB: aspect kept,
  content rect (ox, oy, nw, nh), padding outside). Crop-grid cell (i, j) has centre ((j + .5) P, (i + .5) P); its position
  relative to the content rect is qrel = ((cx - ox) / nw, (cy - oy) / nh) (src.g1x.data.roi_geom, the decoder query
  coordinate). The same scene point on the FULL canvas is (X0 + qrel_x * bw, Y0 + qrel_y * bh) with (X0, Y0, X1, Y1) the
  ROI box on the full canvas. Because both canvases keep the aspect ratio, this is one affine map per axis (checked
  numerically in src.decstudy.checks). Cells whose centre falls into the crop letterbox padding (qrel outside [0, 1]) are
  sampled at the clamped box edge; they carry crop coverage 0 and are excluded from every loss term and from R2 (cov = 0).

  bilinear_roi : F (full tokens, cell centres at (k + .5) P) sampled bilinearly at those points
                 (grid_sample, align_corners=False, border padding), fp32, no autocast.
  BilinearOnly : B0 path, Z = B (no parameters).
  ResidualDecoder : D1 / L1 / L2 path. = src.g1clip.model.ROIDecoderD (same q_mlp / q_grid / key geometry / k_mlp / ln_k /
                 2 cross blocks / ln_out / out, created under the same seed) with
                   query  q0 = LN_qin(B + q_mlp(position features)) + q_grid       (new: ln_qin; B enters the query)
                   output Delta = out(ln_out(q))  with `out` ZERO-initialised (weight and bias)
                   Z = B + Delta   (residual scale fixed at 1)
                 keys / values unchanged: LN(full token + MLP(key geometry)) over ALL full-canvas tokens (context kept).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.g1x.model import G, fourier
from src.g1clip.model import ROIDecoderD


def roi_points(box_c, qrel):
    """-> (R, 576, 2) full-canvas pixel coordinates of every crop-grid cell centre (qrel clamped to the content rect)."""
    u = qrel.float().clamp(0.0, 1.0)
    X0, Y0, X1, Y1 = box_c.float().unbind(-1)
    x = X0[:, None] + u[..., 0] * (X1 - X0)[:, None]
    y = Y0[:, None] + u[..., 1] * (Y1 - Y0)[:, None]
    return torch.stack([x, y], -1)


def bilinear_roi(full_tok, roi_img, box_c, qrel, S, chunk=64):
    """full_tok (B_img, 576, D) -> (R, 576, D) fp32 bilinear samples on the C0 crop grid."""
    with torch.autocast(full_tok.device.type, enabled=False):
        Bi, N, D = full_tok.shape
        fmap = full_tok.float().transpose(1, 2).reshape(Bi, D, G, G)
        pts = roi_points(box_c, qrel)
        grid = (pts / S) * 2 - 1                              # align_corners=False: -1 / +1 = outer canvas edges
        out = []
        for i in range(0, len(roi_img), chunk):
            sl = slice(i, i + chunk)
            g = grid[sl].view(-1, G, G, 2)
            o = F.grid_sample(fmap[roi_img[sl]], g, mode="bilinear", padding_mode="border", align_corners=False)
            out.append(o.flatten(2).transpose(1, 2))
        return torch.cat(out)


class BilinearOnly(nn.Module):
    """B0: decoder-free path with the decoder call signature (used by the probe / inference code unchanged)."""

    def __init__(self, S):
        super().__init__()
        self.S = S

    def forward(self, full_tok, roi_img, box_c, qrel, fvalid, chunk=64):
        return bilinear_roi(full_tok, roi_img, box_c, qrel, self.S, chunk)


class ResidualDecoder(ROIDecoderD):
    def __init__(self, S, P, D, depth=2, heads=8):
        super().__init__(S, P, D, depth, heads)
        self.ln_qin = nn.LayerNorm(D)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def pos_feat(self, box_c, qrel):
        X0, Y0, X1, Y1 = box_c.unbind(-1)
        bw, bh = (X1 - X0).clamp_min(1e-3), (Y1 - Y0).clamp_min(1e-3)
        ax = (X0[:, None] + qrel[..., 0] * bw[:, None]) / self.S
        ay = (Y0[:, None] + qrel[..., 1] * bh[:, None]) / self.S
        size = torch.stack([torch.log(bw / self.P), torch.log(bh / self.P), torch.log(bw / bh)], -1)[:, None].expand(-1, qrel.shape[1], -1)
        return torch.cat([fourier(qrel, self.NF_Q, 6.0), fourier(torch.stack([ax, ay], -1), self.NF_Q, 6.0), size], -1)

    def parts(self, full_tok, roi_img, box_c, qrel, fvalid, chunk=64):
        """-> (B, Delta), both fp32 (R, 576, D). The decoder body runs under the caller's autocast (bf16 = original)."""
        Bf = bilinear_roi(full_tok, roi_img, box_c, qrel, self.S, chunk)
        dl = []
        for i in range(0, len(roi_img), chunk):
            sl = slice(i, i + chunk)
            bc, ri = box_c[sl].float(), roi_img[sl]
            q = self.ln_qin(Bf[sl] + self.q_mlp(self.pos_feat(bc, qrel[sl].float()))) + self.q_grid
            kg, _ = self.key_geo(bc)
            kv = self.ln_k(full_tok[ri] + self.k_mlp(kg))
            m = fvalid[ri].bool()
            for blk in self.blocks:
                q = blk(q, kv, m)
            dl.append(self.out(self.ln_out(q)).float())
        return Bf, torch.cat(dl)

    def forward(self, full_tok, roi_img, box_c, qrel, fvalid, chunk=64):
        Bf, d = self.parts(full_tok, roi_img, box_c, qrel, fvalid, chunk)
        return Bf + d


def build_residual(S, P, D, seed=7):
    torch.manual_seed(seed)                                   # = the G1-cross decoder init seed
    return ResidualDecoder(S, P, D)
