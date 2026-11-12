"""Losses and feature diagnostics for the decoder study. Z (student, R x 576 x D), T (C0 crop teacher, detached),
cc (R x 576) = crop content coverage of every crop-grid cell (0 = letterbox padding -> excluded everywhere).

L_original = lam_detail * L_detail   (the decoder's whole objective in G1-cross: L_roi / L_global only reach the encoder)
  L_detail = mean_ROI [sum_q cc_q (1 - cos(Z_q, T_q)) / sum_q cc_q] + mean_ROI [1 - cos(pool_cc(Z), pool_cc(T))]
L_cov (L1): per ROI on the VALID tokens (cc > 0, N of them; ROIs with N < MIN_TOK are skipped), tokens L2-normalised
  (the distillation is cosine-based, so token scale is left free; the covariance compares directions), centred over the
  valid tokens, C = Xc^T Xc / max(N - 1, 1) (D x D, fp32), L = ||C_Z - C_T||_F^2 / (||C_T||_F^2 + eps), mean over ROIs.
L_rel (L2): ROI vectors p = pool_cc(tokens) (the SAME pooling as the pooled term of L_detail) for every ROI of the
  optimizer step (64); S = cosine matrix, diagonal excluded; per row KL(softmax(S_T / tau) || softmax(S_Z / tau)),
  mean over rows. No labels, no forced negatives: the target distribution is whatever the teacher says.
"""
import torch
import torch.nn.functional as F

MIN_TOK = 4
TAU = 0.1
EPS = 1e-8


def pool(x, w):
    w = w.float()
    return (x.float() * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)


def cosd(a, b):
    return 1 - F.cosine_similarity(a.float(), b.float(), dim=-1)


def detail_terms(Z, T, cc):
    """-> per-ROI (token term, pooled term) = src.g1clip.distill.terms (dec part)."""
    cc = cc.float()
    return (cosd(Z, T) * cc).sum(1) / cc.sum(1), cosd(pool(Z, cc), pool(T, cc))


def _cov(X, m):
    """X (R, N, D) fp32, m (R, N) {0,1} -> (R, D, D) covariance of the L2-normalised valid tokens."""
    Xn = F.normalize(X.float(), dim=-1) * m[..., None]
    n = m.sum(1)
    mu = Xn.sum(1) / n.clamp_min(1)[:, None]
    Xc = (Xn - mu[:, None]) * m[..., None]
    return torch.bmm(Xc.transpose(1, 2), Xc) / (n - 1).clamp_min(1)[:, None, None]


def cov_terms(Z, T, cc, chunk=16):
    """-> (per-ROI normalised covariance error, per-ROI valid flag, per-ROI covariance cosine similarity). fp32, no autocast."""
    with torch.autocast(Z.device.type, enabled=False):
        m = (cc > 0).float()
        ok = m.sum(1) >= MIN_TOK
        errs, sims = [], []
        for i in range(0, len(Z), chunk):
            sl = slice(i, i + chunk)
            Cz, Ct = _cov(Z[sl], m[sl]), _cov(T[sl].detach(), m[sl])
            d = (Cz - Ct).pow(2).sum((1, 2))
            nt = Ct.pow(2).sum((1, 2))
            errs.append(d / (nt + EPS))
            sims.append((Cz * Ct).sum((1, 2)) / (Cz.pow(2).sum((1, 2)).sqrt() * nt.sqrt() + EPS))
        return torch.cat(errs), ok, torch.cat(sims)


def rel_matrix(p):
    pn = F.normalize(p.float(), dim=-1)
    return pn @ pn.t()


def rel_kl_rows(pz_rows, pt_rows, pz_all, pt_all, row_index, tau=TAU):
    """Rows = this process's ROIs (global indices row_index), columns = all ROIs of the step.
    -> per-row KL(softmax(S_T/tau) || softmax(S_Z/tau)) with self-similarity excluded."""
    with torch.autocast(pz_rows.device.type, enabled=False):
        sz = F.normalize(pz_rows.float(), dim=-1) @ F.normalize(pz_all.float(), dim=-1).t()
        st = F.normalize(pt_rows.float(), dim=-1) @ F.normalize(pt_all.float(), dim=-1).t()
        mask = torch.zeros_like(sz, dtype=torch.bool)
        mask[torch.arange(len(row_index), device=sz.device), row_index] = True
        neg = torch.finfo(sz.dtype).min
        lz = F.log_softmax((sz / tau).masked_fill(mask, neg), -1)
        lt = F.log_softmax((st / tau).masked_fill(mask, neg), -1)
        pt = lt.exp()
        return (pt * (lt - lz)).masked_fill(mask, 0.0).sum(-1)


def rel_kl(pz, pt, tau=TAU):
    return rel_kl_rows(pz, pt.detach(), pz, pt.detach(), torch.arange(len(pz), device=pz.device), tau)


def sv(x, w):
    """ROI spatial token variance (cc-weighted, mean over channels) = src.g1clip.distill.sv."""
    w = w.float()
    mu = pool(x, w)
    return ((x.float() - mu[:, None]).pow(2).mean(-1) * w).sum(1) / w.sum(1).clamp_min(1e-12)


def relation_preservation(pz, pt):
    """Pearson r between the off-diagonal entries of the student / teacher ROI cosine matrices (one batch)."""
    Sz, St = rel_matrix(pz), rel_matrix(pt)
    off = ~torch.eye(len(pz), dtype=torch.bool, device=pz.device)
    a, b = Sz[off], St[off]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + EPS))
