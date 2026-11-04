"""Shared training stream (category-balanced, identical for R00/R10/R01/R11 of one backbone) and the relation loss.

Stream (build_stream): each step = 16 images x their 4 pre-sampled ROIs (the existing coco_rois.parquet, 3 GT + 1
random per image, unchanged) = 64 ROIs, as in the existing recipe. Images are chosen category-balanced so that every
batch contains same-category GT ROIs: 4 distinct COCO categories per step, drawn with probability proportional to
sqrt(#pool train images that contain a GT ROI of that category); for each, 4 distinct images containing such a ROI.
The whole (steps x 16) index list is generated once from a fixed seed and saved; all four runs read the same file.

Relation loss (only GT ROIs; random boxes have cat = -1 and are excluded):
  e(x)    = L2-normalise((pooled(x) - mean_T) / std_T)       pooled = coverage-weighted mean over valid teacher cells
            (same pooling for teacher and student; mean_T / std_T = the train teacher channel stats of the backbone)
  anchor i valid  <=>  >= 2 OTHER GT ROIs of the same category in the batch (self excluded)
  p_T(j | i) = softmax_j( e(T_i) . e(T_j) / tau )   over j in C_i = same category, j != i      (teacher, no grad)
  p_S(j | i) = softmax_j( e(S_i) . e(T_j) / tau )   student anchor vs the same teacher candidates
  L_rel  = mean over valid anchors of KL(p_T || p_S)      (0 if the batch has no valid anchor)
"""
import numpy as np
import torch
import torch.nn.functional as F

N_CAT, N_PER_CAT = 4, 4


def build_stream(rois_train, steps, seed):
    groups = sorted(rois_train.image_id.unique())
    pos = {im: k for k, im in enumerate(groups)}
    gt = rois_train[(rois_train.kind == "gt") & (rois_train.cat_id >= 0)]
    cat_imgs = {c: np.array(sorted({pos[i] for i in g.image_id}), np.int64) for c, g in gt.groupby("cat_id")}
    cats = np.array(sorted(c for c, v in cat_imgs.items() if len(v) >= N_PER_CAT))
    w = np.sqrt([len(cat_imgs[c]) for c in cats])
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    out = np.zeros((steps, N_CAT * N_PER_CAT), np.int64)
    for s in range(steps):
        chosen, used = rng.choice(cats, size=N_CAT, replace=False, p=w), set()
        row = []
        for c in chosen:
            cand = np.setdiff1d(cat_imgs[c], np.fromiter(used, np.int64, len(used)), assume_unique=False)
            pick = rng.choice(cand, size=N_PER_CAT, replace=False)
            used.update(pick.tolist())
            row += pick.tolist()
        out[s] = row
    info = dict(n_groups=len(groups), n_cats=len(cats), cat_probs={int(c): float(p) for c, p in zip(cats, w)},
                n_unique_images=int(len(np.unique(out))), mean_image_repeats=float(out.size / len(np.unique(out))))
    return out, info


def embed(x, w, mu, sd):
    p = (x.float() * w.float()[..., None]).sum(1) / w.float().sum(1, keepdim=True)
    return F.normalize((p - mu) / sd, dim=-1)


def relation_mask(cats):
    """cats (R,) long, -1 = not a GT ROI -> (R,R) bool candidate mask, (R,) bool valid anchors."""
    R = cats.shape[0]
    gt = cats >= 0
    M = (cats[:, None] == cats[None, :]) & gt[:, None] & gt[None, :] & ~torch.eye(R, dtype=torch.bool, device=cats.device)
    valid = M.sum(1) >= 2
    return M, valid


def relation_loss(student, teacher, w, cats, mu, sd, tau, return_stats=False):
    M, valid = relation_mask(cats)
    et = embed(teacher, w, mu, sd).detach()
    es = embed(student, w, mu, sd)
    if not bool(valid.any()):
        z = student.sum() * 0.0
        return (z, dict(n_valid=0)) if return_stats else z
    neg = torch.finfo(torch.float32).min
    lt = (et @ et.T / tau).masked_fill(~M, neg)
    ls = (es @ et.T / tau).masked_fill(~M, neg)
    pt = F.softmax(lt, -1)
    lpt, lps = F.log_softmax(lt, -1), F.log_softmax(ls, -1)
    kl = (pt * (lpt - lps)).masked_fill(~M, 0.0).sum(-1)
    loss = kl[valid].mean()
    if not return_stats:
        return loss
    k = M.sum(1).float()
    ent = -(pt * lpt).masked_fill(~M, 0.0).sum(-1)
    st = dict(n_valid=int(valid.sum()), n_gt=int((cats >= 0).sum()), mean_cands=float(k[valid].mean()),
              norm_entropy=float((ent[valid] / k[valid].log()).mean()),
              teacher_sim_mean=float((et @ et.T)[M].mean()), teacher_sim_std=float((et @ et.T)[M].std()),
              pt_max_mean=float(pt.masked_fill(~M, 0).amax(-1)[valid].mean()))
    return loss, st
