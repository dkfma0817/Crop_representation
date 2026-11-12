"""L = lambda_crop * L_crop + lambda_preserve * L_preserve. Global normalisers come from the plan, so the sum over micro steps and ranks
does not depend on how the batch is split. Final-LN tokens (fp32) go straight into the cosines; teacher outputs are detached.

L_crop     = w_pool * L_pool + w_spatial * L_spatial
  L_pool    per ROI: 1 - cos(area-weighted pooled student tokens over the box, content-coverage pooled crop-teacher tokens)
  L_spatial per ROI: mean over the K x K coarse cells of 1 - cos(pooled student cell, pooled teacher cell); the two cells cover the same
            rectangle of the original image. ROIs too small for the grid (< spatial_min_cells token per coarse cell) are skipped.
L_preserve = (w_cls * CLS + w_mean * mean patch + w_out * per-token outside all ROIs of the image) / sum w, vs the frozen encoder on the same image.
Normalised-feature MSE would be exactly 2 * (1 - cos), so it is not added as a separate term.
"""
import torch
import torch.nn.functional as F


def pool(x, w):
    w = w.float()
    return (x * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)


def cosd(a, b):
    return 1 - F.cosine_similarity(a, b, dim=-1)


def coarse_pool(x, w):
    """x (R, 576, D), w (R, K2, 576) -> (R, K2, D): area-weighted mean per coarse cell."""
    return torch.bmm(w, x) / w.sum(-1, keepdim=True).clamp_min(1e-12)


def compute(b, ftok, fcls, T, otok, ocls, lc):
    ri, fv = b["roi_img"], b["fvalid"].float()
    F_roi = ftok[ri]
    tp = pool(T, b["ccov"])
    pool_per = cosd(pool(F_roi, b["wfull"]), tp)
    Sc, Tc = coarse_pool(F_roi, b["ws"]), coarse_pool(T, b["wt"])
    spatial_per = cosd(Sc, Tc).mean(1)
    sv = b["sv"].float()
    cover = torch.zeros(fv.shape, device=fv.device, dtype=torch.float32)
    cover.index_add_(0, ri, (b["wfull"] > 0).float())
    outside = ((cover == 0) & (fv > 0)).float()
    tc = cosd(ftok, otok)
    tok_per = (tc * outside).sum(1) / outside.sum(1).clamp_min(1.0)
    cls_per = cosd(fcls, ocls)
    mean_per = cosd(pool(ftok, fv), pool(otok, fv))
    n_roi, n_sp, n_img = float(b["n_roi"]), max(float(b["n_sp"]), 1.0), float(b["n_img"])
    l_pool, l_spatial = pool_per.sum() / n_roi, (spatial_per * sv).sum() / n_sp
    l_cls, l_mean, l_tok = cls_per.sum() / n_img, mean_per.sum() / n_img, tok_per.sum() / n_img
    c, pr = lc["crop"], lc["preserve"]
    W = pr["cls"] + pr["mean_patch"] + pr["token_outside"]
    l_crop = c["pool"] * l_pool + c["spatial"] * l_spatial
    l_pres = (pr["cls"] * l_cls + pr["mean_patch"] * l_mean + pr["token_outside"] * l_tok) / W
    loss = lc["lambda_crop"] * l_crop + lc["lambda_preserve"] * l_pres
    with torch.no_grad():
        stats = dict(roi_pool_cos=float((1 - pool_per).sum()), spatial_cos=float(((1 - spatial_per) * sv).sum()),
                     cls_cos=float((1 - cls_per).sum()), mean_cos=float((1 - mean_per).sum()), out_tok_cos=float((1 - tok_per).sum()),
                     student_norm=float(F_roi.norm(dim=-1).mean()), teacher_norm=float(T.norm(dim=-1).mean()))
    parts = dict(loss=loss, l_pool=l_pool, l_spatial=l_spatial, l_cls=l_cls, l_mean=l_mean, l_tok=l_tok, l_crop=l_crop, l_preserve=l_pres)
    return parts, stats
