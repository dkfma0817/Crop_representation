"""Verbatim copy of dinov2_oracle_region_decoder/models/losses.py (also used for SigLIP2, whose project has no cosine loss).
Training loss (fixed, docs/protocol.md section 3).

Features are the raw frozen DINOv2 final-LN patch tokens; no further normalisation, whitening or centring is
applied to student or teacher. Per ROI, with w_p = teacher-crop content coverage of cell p (padding cells w = 0):
  L_mse = sum_p w_p ||pred_p - T_p||^2 / (768 sum_p w_p)          (magnitude + direction, raw units)
  L_cos = sum_p w_p (1 - cos(pred_p, T_p)) / sum_p w_p             (direction only, scale-invariant)
  L     = mean_ROI (L_mse + LAMBDA * L_cos), LAMBDA = 1, all in fp32.
Because MSE is computed on un-normalised tokens, the two terms are not the same objective (L2-normalised MSE
would be 2 L_cos up to a constant); L_cos alone cannot fix norms, L_mse alone is dominated by high-norm channels.
"""
import torch.nn.functional as F

LAMBDA = 1.0


def dense_mse(pred, target, w):
    d = (pred.float() - target.float()).pow(2).mean(-1)
    w = w.float()
    return (d * w).sum(-1) / w.sum(-1)


def dense_cos(pred, target, w):
    c = F.cosine_similarity(pred.float(), target.float(), dim=-1, eps=1e-6)
    w = w.float()
    return (c * w).sum(-1) / w.sum(-1)


def loss_terms(pred, target, w):
    """-> (loss scalar, per-ROI mse, per-ROI cos)."""
    m, c = dense_mse(pred, target, w), dense_cos(pred, target, w)
    return (m + LAMBDA * (1 - c)).mean(), m, c
