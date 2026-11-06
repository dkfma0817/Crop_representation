"""Centered Direction Loss (ROI-level, fp32).

Per ROI r with coverage weights w_i (= valid_w, the C0 letterbox content coverage used by L_old; padding cells w = 0):
  mu_S = sum_i w_i S_i / sum_i w_i ,  mu_T likewise ;  S_c = S - mu_S ,  T_c = T - mu_T   (each its own spatial mean)
  c_dir = sum_i w_i <S_c,i , T_c,i>  /  ( sqrt(sum_i w_i ||S_c,i||^2) * sqrt(sum_i w_i ||T_c,i||^2) + eps )
        = cosine of the two whole-ROI centred patterns (all positions x 768 channels as ONE vector, rows scaled by sqrt(w_i)),
          i.e. w enters linearly in every sum (never squared).
  L_dir = mean over VALID ROIs of (1 - c_dir).   Valid = sum_i w_i > 0 and teacher centred energy
          e_T = sum_i w_i ||T_c,i||^2 / (768 sum_i w_i) > thr  (thr fixed before training from the calibration batches).
  Excluded ROIs are excluded from L_dir only (L_old unchanged). Student norm never excludes a ROI.
Teacher is detached inside the function (no gradient can reach it).
"""
import torch

D = 768
EPS = 1e-8


def centered_dir(S, T, w):
    """S, T (R, N, D), w (R, N) -> c (R,), e_T (R,) mean centred teacher energy per channel, w_sum (R,). fp32."""
    S, T, w = S.float(), T.detach().float(), w.float()
    ws = w.sum(1)
    wsc = ws.clamp_min(1e-12)[:, None]
    mS = (S * w[..., None]).sum(1) / wsc
    mT = (T * w[..., None]).sum(1) / wsc
    Sc, Tc = S - mS[:, None], T - mT[:, None]
    wv = w[..., None]
    num = (wv * Sc * Tc).sum((1, 2))
    eS = (wv * Sc * Sc).sum((1, 2))
    eT = (wv * Tc * Tc).sum((1, 2))
    c = num / (torch.sqrt(eS.clamp_min(1e-20)) * torch.sqrt(eT.clamp_min(1e-20)) + EPS)
    return c, eT / (wsc[:, 0] * S.shape[-1]), ws


def dir_terms(S, T, w, thr):
    """-> (sum over valid ROIs of (1 - c), n_valid, c (R,), valid (R,) bool, e_T (R,))."""
    c, eT, ws = centered_dir(S, T, w)
    valid = (ws > 0) & (eT > thr)
    l = torch.where(valid, 1 - c, torch.zeros_like(c))
    return l.sum(), valid.sum(), c, valid, eT


def centered_dir_flatten_ref(S, T, w):
    """Independent reference: flatten sqrt(w)-scaled centred patterns and take a plain cosine (for the checks)."""
    out = []
    for s, t, ww in zip(S.double(), T.double(), w.double()):
        m = ww.sum()
        sc = s - (ww[:, None] * s).sum(0) / m
        tc = t - (ww[:, None] * t).sum(0) / m
        a, b = (ww.sqrt()[:, None] * sc).reshape(-1), (ww.sqrt()[:, None] * tc).reshape(-1)
        out.append(torch.dot(a, b) / (a.norm() * b.norm()))
    return torch.stack(out)
