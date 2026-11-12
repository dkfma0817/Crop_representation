"""S3-Distill losses: child-level teacher-bank KL + ROI-level local relation KL (LightlyTrain DistillationV3-style, region-level variant).

All of it runs with autocast DISABLED and TF32 DISABLED (true FP32); L2 normalisation is applied only inside these losses (the decoder output and the
R2 input keep the S3-L2 convention: raw final-norm-scale features).

  bank  : P_T = softmax(n(T) Q^T / tau_b), log P_S = log_softmax(n(S) Q^T / tau_b) over the 8,192 bank rows;
          KL(P_T || P_S) elementwise (reduction none) -> SUM over the bank axis -> MEAN over the ROI's valid children -> mean over ROIs.
  local : A = n(X) n(X)^T over the ROI's children; invalid / padding keys get a large finite negative logit before the softmax
          (exp underflows to exactly 0 -> no 0 x inf); KL(P_T,i || P_S,i) per query -> SUM over keys -> MEAN over the ROI's valid
          queries -> mean over ROIs. Diagonal included (as in the original). ROIs without a valid child are skipped.
"""
from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F

NEG = -1.0e4          # masked logit; |valid logits| <= 1 / 0.07 = 14.3, so exp(NEG - max) == 0 in fp32


@contextmanager
def fp32():
    """autocast off AND TF32 off (true FP32 matmuls) inside the loss; the previous TF32 setting is restored afterwards
    (the decoder forward keeps the S3-L2 setting: bf16 autocast, TF32 allowed)."""
    prev, prec = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
        torch.set_float32_matmul_precision(prec)


def _roi_mean(per_item, roi_of_item, n_roi):
    """per_item (n,), roi index (n,) -> (mean over ROIs that own at least one item, per-ROI values, mask)."""
    s = torch.zeros(n_roi, device=per_item.device, dtype=per_item.dtype).index_add_(0, roi_of_item, per_item)
    c = torch.zeros(n_roi, device=per_item.device, dtype=per_item.dtype).index_add_(0, roi_of_item, torch.ones_like(per_item))
    ok = c > 0
    per_roi = s[ok] / c[ok]
    return per_roi.mean(), per_roi, ok


def bank_kl(S, T, valid, Q, tau=0.07, chunk=8192):
    """S, T (R, N, D) any dtype (S keeps its graph); valid (R, N) bool; Q (K, D) L2-normalised fp32 bank (no grad).
    Returns (loss, stats). Softmax over the FULL bank for every query; only the query axis is chunked."""
    with fp32():
        R = S.shape[0]
        roi = torch.arange(R, device=S.device)[:, None].expand_as(valid)[valid]
        s, t = F.normalize(S.float()[valid], dim=-1), F.normalize(T.float()[valid], dim=-1)
        kls, hs, ht = [], [], []
        for i in range(0, len(s), chunk):
            ls = F.log_softmax(s[i:i + chunk] @ Q.T / tau, dim=-1)
            with torch.no_grad():
                lt = F.log_softmax(t[i:i + chunk] @ Q.T / tau, dim=-1)
            kls.append(F.kl_div(ls, lt, reduction="none", log_target=True).sum(-1))
            with torch.no_grad():
                hs.append(-(ls.exp() * ls).sum(-1))
                ht.append(-(lt.exp() * lt).sum(-1))
        kl = torch.cat(kls)
        loss, _, _ = _roi_mean(kl, roi, R)
        logK = math.log(Q.shape[0])
        hs, ht = torch.cat(hs), torch.cat(ht)
        stats = dict(bank_H_student=float(hs.mean()), bank_H_teacher=float(ht.mean()), bank_Hnorm_student=float(hs.mean()) / logK, bank_Hnorm_teacher=float(ht.mean()) / logK)
    return loss, stats


def local_kl(S, T, valid, tau=0.07, one_by_one=None):
    """S, T (R, N, D); valid (R, N) bool (padding and letterbox-invalid children False). Returns (loss, stats).
    one_by_one (R,) bool: ROIs that are 1 x 1 parent rectangles (logged separately)."""
    with fp32():
        R, N, _ = S.shape
        s, t = F.normalize(S.float(), dim=-1), F.normalize(T.float(), dim=-1)
        key = valid[:, None, :]
        As = (s @ s.transpose(1, 2) / tau).masked_fill(~key, NEG)
        with torch.no_grad():
            At = (t @ t.transpose(1, 2) / tau).masked_fill(~key, NEG)
            lt = F.log_softmax(At, dim=-1)
        ls = F.log_softmax(As, dim=-1)
        kl = torch.where(key, F.kl_div(ls, lt, reduction="none", log_target=True), torch.zeros((), device=S.device)).sum(-1)    # (R, N)
        roi = torch.arange(R, device=S.device)[:, None].expand_as(valid)[valid]
        loss, _, _ = _roi_mean(kl[valid], roi, R)
        with torch.no_grad():
            stats = {}
            n = valid.sum(1).float()
            eye = torch.eye(N, device=S.device, dtype=torch.bool)[None]
            for who, lp in (("student", ls), ("teacher", lt)):
                p = lp.exp() * key
                H = -(torch.where(key, p * lp, torch.zeros((), device=S.device))).sum(-1)                         # (R, N)
                diag = (p * eye).sum(-1)                                                                        # P_ii
                ok = n > 0
                d_roi = (diag * valid).sum(1)[ok] / n[ok]                                                       # sum(P_ii) / N
                H_roi = (H * valid).sum(1)[ok] / n[ok]
                Hn_roi = torch.where(n[ok] > 1, H_roi / torch.log(n[ok].clamp_min(2)), torch.full_like(H_roi, float("nan")))
                stats[f"local_diag_{who}"] = float(d_roi.mean())
                stats[f"local_H_{who}"] = float(H_roi.mean())
                stats[f"local_Hnorm_{who}"] = float(torch.nanmean(Hn_roi))
                if one_by_one is not None and one_by_one[ok].any():
                    m = one_by_one[ok]
                    stats[f"local_diag_{who}_1x1"] = float(d_roi[m].mean())
                    stats[f"local_H_{who}_1x1"] = float(H_roi[m].mean())
                    stats[f"local_Hnorm_{who}_1x1"] = float(torch.nanmean(Hn_roi[m]))
            stats["local_uniform_diag"] = float((1 / n[n > 0]).mean())
            if one_by_one is not None and (one_by_one & (n > 0)).any():
                stats["local_uniform_diag_1x1"] = float((1 / n[one_by_one & (n > 0)]).mean())
                kl1 = (kl * valid).sum(1)[one_by_one & (n > 0)] / n[one_by_one & (n > 0)]
                stats["local_kl_1x1"] = float(kl1.mean())
    return loss, stats


# ------------------------------------------------------------------------------------------------ reference (tests only)
def ref_bank(S, T, valid, Q, tau):
    out = []
    for r in range(S.shape[0]):
        m = valid[r]
        if not m.any():
            continue
        s, t = F.normalize(S[r][m].double(), dim=-1), F.normalize(T[r][m].double(), dim=-1)
        pt = torch.softmax(t @ Q.double().T / tau, -1)
        ls = torch.log_softmax(s @ Q.double().T / tau, -1)
        out.append((pt * (pt.log() - ls)).sum(-1).mean())
    return torch.stack(out).mean()


def ref_local(S, T, valid, tau):
    out = []
    for r in range(S.shape[0]):
        m = valid[r]
        if not m.any():
            continue
        s, t = F.normalize(S[r][m].double(), dim=-1), F.normalize(T[r][m].double(), dim=-1)
        pt = torch.softmax(t @ t.T / tau, -1)
        ls = torch.log_softmax(s @ s.T / tau, -1)
        out.append((pt * (pt.log() - ls)).sum(-1).mean())
    return torch.stack(out).mean()
