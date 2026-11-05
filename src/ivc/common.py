"""Interpolation (bilinear / bicubic) ROI features vs C0 GT-crop features: shared pieces.

Everything that decides comparability is reused unchanged:
  * full-image encoder / C0 crop letterbox / crop-grid geometry : src.msd.common (MSDImageDS, Student) = data.preprocessing
  * bilinear ROI sampling                                         : models.roi_reader.bilinear (grid_sample, align_corners=False,
                                                                    border) on roi_geometry()['bil_grid']
  * R2 reader, training loop, M0 / M2, bootstrap                  : src.readout.* (11-05 MSD probe protocol)
New here:
  * bicubic ROI sampling = the SAME grid / settings, only mode='bicubic'
  * R0 reader (paco_region_reader common_rr.R0: coverage-weighted ROI mean -> per-level z-score -> Linear heads) on the
    crop grid (weights = C0 crop letterbox coverage, identical for the three conditions)
  * per-ROI feature-difference metrics (all fp32)
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.readout.common import D, RR

G = 24
T = G * G
EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
DBINS = (("D1-4", 1, 4), ("D5-16", 5, 16), ("D17+", 17, 10 ** 9))


# ------------------------------------------------------------------------------------------------ ROI sampling
def sample_roi(full_tokens, geom, mode):
    """full_tokens (B_img, 576, D) -> (R, 576, D) on the C0 crop grid. mode 'bilinear' == models.roi_reader.bilinear;
    'bicubic' = identical call with mode='bicubic' (PyTorch cubic convolution, A = -0.75, 4x4 support, border clamp)."""
    maps = full_tokens.transpose(1, 2).reshape(full_tokens.shape[0], -1, G, G)[geom["img_index"]]
    out = F.grid_sample(maps.float(), geom["bil_grid"].view(-1, G, G, 2), mode=mode, padding_mode="border", align_corners=False)
    return out.flatten(2).transpose(1, 2)


# ------------------------------------------------------------------------------------------------ R0
class R0(nn.Module):
    """paco_region_reader common_rr.R0 on a token grid: coverage-weighted ROI mean (weights = b['cov'], here the C0
    crop letterbox coverage) -> per-level z-score (stats of THIS condition's train features) -> Linear heads (59 x 2).
    Same call signature as src.readout.models.Reader."""

    def __init__(self):
        super().__init__()
        self.kind = "r0"

        class _F:                       # train.py reads reader.fuse.layers
            layers = (12,)
        self.fuse = _F()
        self.register_buffer("mu", torch.zeros(2, D))
        self.register_buffer("sd", torch.ones(2, D))
        self.heads = RR.Heads()

    def set_stats(self, mu, sd):
        self.mu.copy_(mu)
        self.sd.copy_(sd)

    @staticmethod
    def pooled(tok, cov):
        w = cov.float()
        return (tok.float() * w[..., None]).sum(1) / w.sum(1, keepdim=True)

    def forward(self, feats, b, ctx_mode="normal"):
        m = self.pooled(feats[12][b["img_index"]], b["cov"])
        lv = b["lv"]
        z = (m - self.mu[lv]) / self.sd[lv]
        return dict(logits=self.heads(z, lv), extra={})


# ------------------------------------------------------------------------------------------------ feature metrics
def dct_mat(n, dev):
    """Orthonormal DCT-II matrix (n, n): X_k = sum_i M[k, i] x_i."""
    i = torch.arange(n, device=dev, dtype=torch.float64)
    k = i[:, None]
    M = torch.cos(math.pi * (i[None] + 0.5) * k / n) * math.sqrt(2.0 / n)
    M[0] /= math.sqrt(2.0)
    return M.float()


BANDS = (("low", 0.0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("high", 2 / 3, 1.01))


def band_energy(x, h, w):
    """x (h*w, D) fp32 tokens of the fully-covered rectangle (row-major) -> {band: energy} of the 2-D DCT-II per
    channel, DC term (0, 0) excluded. Band of coefficient (u, v): rho = max(u / h, v / w) (0 <= rho < 1); low < 1/3 <=
    mid < 2/3 <= high. Energy = sum of squared coefficients over channels / (h * w * D) (orthonormal -> Parseval)."""
    dev = x.device
    X = x.view(h, w, -1)
    Ch, Cw = dct_mat(h, dev), dct_mat(w, dev)
    Z = torch.einsum("uh,hwd,vw->uvd", Ch, X, Cw)
    e = Z.pow(2).sum(-1)                                             # (h, w)
    u = torch.arange(h, device=dev)[:, None].float() / h
    v = torch.arange(w, device=dev)[None].float() / w
    rho = torch.maximum(u.expand(h, w), v.expand(h, w))
    nz = torch.ones(h, w, dtype=torch.bool, device=dev)
    nz[0, 0] = False
    out = {}
    for nm, lo, hi in BANDS:
        m = nz & (rho >= lo) & (rho < hi)
        out[nm] = float(e[m].sum()) / (h * w * x.shape[1])
    return out


def full_rect(cov):
    """cov (576,) -> (i0, i1, j0, j1) of the fully covered cells (cov == 1, a rectangle for a letterbox) or None."""
    c = cov.view(G, G) >= 1 - 1e-6
    rows, cols = torch.nonzero(c.any(1)).flatten(), torch.nonzero(c.any(0)).flatten()
    if not len(rows) or not len(cols):
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def erank(x):
    """Effective rank (Roy & Vetterli 2007) of the centred token matrix x (n, D): exp(H(p)), p = s^2 / sum s^2."""
    xc = x - x.mean(0, keepdim=True)
    try:
        ev = torch.linalg.eigvalsh((xc @ xc.T).double()).clamp_min(0)
    except torch.linalg.LinAlgError:                     # near-degenerate token set: same quantity via CPU float64 SVD
        ev = torch.linalg.svdvals(xc.double().cpu()).pow(2).to(x.device)
    p = ev / ev.sum().clamp_min(1e-30)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def cosmat(x):
    xn = F.normalize(x, dim=-1)
    return xn @ xn.T


def roi_metrics(a, c, cov):
    """One ROI. a = interpolated (576, D), c = C0 (576, D) fp32 on the same crop grid, cov (576,) C0 letterbox coverage.
    Main mask M = cov > 0 with weights w = cov (= R0 pooling, = R2 attention mask); structure metrics on the fully
    covered rectangle M1 (cov == 1). Returns a flat dict (a-only / c-only / pair metrics)."""
    w = cov.float()
    M = w > 0
    wm = w[M] / w[M].sum()
    am, cm = a[M], c[M]
    # token level (weighted over M)
    tcos = F.cosine_similarity(am, cm, dim=-1)
    tse = (am - cm).pow(2).mean(-1)                                   # per-token MSE (per channel)
    pa, pc = (wm[:, None] * am).sum(0), (wm[:, None] * cm).sum(0)       # pooled = coverage-weighted mean (= R0 input)
    e_tot = float((wm * tse).sum())
    e_mean = float((pa - pc).pow(2).mean())
    ra, rc = am - pa, cm - pc
    e_cent = float((wm * (ra - rc).pow(2).mean(-1)).sum())              # e_tot = e_mean + e_cent (exact)
    sv_a, sv_c = float((wm * ra.pow(2).mean(-1)).sum()), float((wm * rc.pow(2).mean(-1)).sum())
    ccos = F.cosine_similarity(ra, rc, dim=-1)                          # centred token cosine
    out = dict(n_M=int(M.sum()), tok_cos=float((wm * tcos).sum()), tok_cos_unw=float(tcos.mean()),
               tok_cos_p10=float(torch.quantile(tcos, 0.10)), tok_cos_p50=float(torch.quantile(tcos, 0.5)),
               tok_mse=e_tot, pool_cos=float(F.cosine_similarity(pa, pc, dim=0)), pool_l2=float((pa - pc).norm()),
               e_mean=e_mean, e_cent=e_cent, frac_cent=e_cent / max(e_tot, 1e-30), cent_cos=float((wm * ccos).sum()),
               sv_a=sv_a, sv_c=sv_c, sv_ratio=sv_a / max(sv_c, 1e-30),
               norm_a=float((wm * am.norm(dim=-1)).sum()), norm_c=float((wm * cm.norm(dim=-1)).sum()))
    # all 576 cells (incl. padding) and padding only (unweighted)
    tc_all = F.cosine_similarity(a, c, dim=-1)
    out.update(tok_cos_all=float(tc_all.mean()), tok_mse_all=float((a - c).pow(2).mean()),
               tok_cos_pad=float(tc_all[~M].mean()) if (~M).any() else float("nan"))
    # structure on M1
    r = full_rect(cov)
    nanf = float("nan")
    if r is None or (r[1] - r[0]) * (r[3] - r[2]) < 4:
        out.update(n_M1=0 if r is None else (r[1] - r[0]) * (r[3] - r[2]), ss_diff=nanf, ss_mean_a=nanf, ss_mean_c=nanf,
                   erank_a=nanf, erank_c=nanf, adj_a=nanf, adj_c=nanf,
                   **{f"dct_{k}_{nm}": nanf for k in ("a", "c", "d") for nm, _, _ in BANDS})
        return out
    i0, i1, j0, j1 = r
    h, wd = i1 - i0, j1 - j0
    A = a.view(G, G, -1)[i0:i1, j0:j1].reshape(h * wd, -1)
    Cc = c.view(G, G, -1)[i0:i1, j0:j1].reshape(h * wd, -1)
    Sa, Sc = cosmat(A), cosmat(Cc)
    off = ~torch.eye(h * wd, dtype=torch.bool, device=a.device)
    out.update(n_M1=h * wd, ss_diff=float((Sa - Sc)[off].abs().mean()), ss_rms=float((Sa - Sc)[off].pow(2).mean().sqrt()),
               ss_mean_a=float(Sa[off].mean()), ss_mean_c=float(Sc[off].mean()), erank_a=erank(A), erank_c=erank(Cc))
    adj = lambda X: torch.cat([F.cosine_similarity(X[:, 1:], X[:, :-1], dim=-1).flatten(),  # noqa: E731
                               F.cosine_similarity(X[1:], X[:-1], dim=-1).flatten()])
    Ag, Cg = A.view(h, wd, -1), Cc.view(h, wd, -1)
    out.update(adj_a=float(adj(Ag).mean()) if h * wd > 1 else nanf, adj_c=float(adj(Cg).mean()) if h * wd > 1 else nanf)
    for k, X in (("a", A), ("c", Cc), ("d", A - Cc)):
        for nm, v in band_energy(X, h, wd).items():
            out[f"dct_{k}_{nm}"] = v
    return out
