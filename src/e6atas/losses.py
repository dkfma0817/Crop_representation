"""ATAS-inspired auxiliary losses for the E6 decoder (2026-11-06 pilot). NOT an ATAS reproduction: ATAS distils a CLIP
model into itself (global CLS-attention-pooled region vectors, local-to-local patch relations); here the teacher is the
frozen DINOv2 GT-crop (C0) dense feature map and the student is the E6 decoder output on the same 24x24 crop grid.
Teacher tensors are always detached. All fp32.

Patch Relation (ATAS local-to-local idea), per ROI:
  positions: up to K = 128 valid cells (C0 valid coverage w > 0; letterbox padding never used), sampled with a dedicated
  torch.Generator (seed fixed per training step and rank, identical across conditions; the global RNG is never touched);
  the SAME indices for student and teacher (same crop-grid cell = same original-image location).
  S_hat, T_hat = channel-wise L2-normalised features; sim_S = S_hat S_hat^T, sim_T = T_hat T_hat^T (K x K);
  L_rel,r = sum_{i != j} w_i w_j (sim_S,ij - sim_T,ij)^2 / sum_{i != j} w_i w_j   (pair weight = product of the coverages);
  L_rel = mean over ROIs with >= 2 valid sampled cells.

Region Contrastive (ATAS global-to-local idea, dense-teacher variant):
  region vector = L2-normalised coverage-weighted mean over valid cells (student: decoder output; teacher: C0 map; no
  projection head). InfoNCE with temperature tau = 0.07 (fixed, CLIP / MoCo default; not tuned):
  L_con,r = -log softmax_j(s_r . t_j / tau)[pos(r)], candidates j = every teacher region vector of the global step
  (all ranks), positive = the same ROI. Excluded from the negatives (false-negative rule, fixed before training):
    (a) same image and box IoU >= 0.5 (duplicate / heavily overlapping ROI of the same object), or
    (b) teacher-teacher cosine >= 0.95 (practically identical crop content).
  A ROI with no remaining negative is excluded from L_con (counted). PACO labels are never used.
"""
import torch
import torch.nn.functional as F

K_REL = 128
TAU = 0.07
IOU_FN = 0.5
TCOS_FN = 0.95


# ------------------------------------------------------------------------------------------------ patch relation
def sample_idx(valid, gen, K=K_REL):
    """valid (R, N) bool -> idx (R, K) long, ok (R, K) bool (False = fewer than K valid cells, slot unused)."""
    u = torch.rand(valid.shape, generator=gen, device=valid.device).masked_fill(~valid, 2.0)
    idx = u.argsort(1)[:, :K]
    ok = torch.gather(u, 1, idx) < 2.0
    return idx, ok


def relation_terms(S, T, w, idx, ok):
    """-> (sum of per-ROI errors over ROIs with >= 2 cells, n such ROIs, per-ROI error (R,))."""
    D = S.shape[-1]
    S, T, w = S.float(), T.detach().float(), w.float()
    Sg = torch.gather(S, 1, idx[..., None].expand(-1, -1, D))
    Tg = torch.gather(T, 1, idx[..., None].expand(-1, -1, D))
    wg = torch.gather(w, 1, idx) * ok.float()
    Sn, Tn = F.normalize(Sg, dim=-1, eps=1e-6), F.normalize(Tg, dim=-1, eps=1e-6)
    simS, simT = Sn @ Sn.transpose(1, 2), Tn @ Tn.transpose(1, 2)
    k = idx.shape[1]
    pw = wg[:, :, None] * wg[:, None, :] * (1 - torch.eye(k, device=S.device))[None]
    den = pw.sum((1, 2))
    err = ((simS - simT).pow(2) * pw).sum((1, 2)) / den.clamp_min(1e-12)
    use = den > 0
    return torch.where(use, err, torch.zeros_like(err)).sum(), use.sum(), err


@torch.no_grad()
def relation_error_full(S, T, w):
    """Evaluation metric (no sampling): the same weighted pair MSE over ALL valid cells. -> (R,)"""
    S, T, w = F.normalize(S.float(), dim=-1, eps=1e-6), F.normalize(T.float(), dim=-1, eps=1e-6), w.float()
    pw = w[:, :, None] * w[:, None, :] * (1 - torch.eye(S.shape[1], device=S.device))[None]
    return ((S @ S.transpose(1, 2) - T @ T.transpose(1, 2)).pow(2) * pw).sum((1, 2)) / pw.sum((1, 2)).clamp_min(1e-12)


# ------------------------------------------------------------------------------------------------ region contrastive
def region_vec(X, w):
    w = w.float()
    return F.normalize((X.float() * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12), dim=-1, eps=1e-6)


def box_iou(a, b):
    """a (R, 4), b (N, 4) ltrb -> (R, N)."""
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).clamp_min(0).prod(-1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter).clamp_min(1e-12)


def negative_mask(t_loc, key_loc, box_loc, pos, t_all, key_all, box_all, valid_all):
    """-> neg_ok (R, N) bool, stats dict. pos (R,) index of each local ROI in the global list."""
    same = key_loc[:, None] == key_all[None]
    fn_iou = same & (box_iou(box_loc, box_all) >= IOU_FN)
    fn_cos = (t_loc @ t_all.T) >= TCOS_FN
    ok = valid_all[None].expand(len(t_loc), -1).clone()
    ok &= ~(fn_iou | fn_cos)
    ok[torch.arange(len(t_loc), device=ok.device), pos] = False
    isp = torch.zeros_like(ok)
    isp[torch.arange(len(t_loc), device=ok.device), pos] = True
    cand = valid_all[None] & ~isp
    stats = dict(n_cand=float(cand.sum()), n_excl_iou=float((fn_iou & cand).sum()), n_excl_cos=float((fn_cos & cand & ~fn_iou).sum()),
                 n_neg=float(ok.sum()))
    return ok, stats


def contrastive_terms(s_loc, t_all, pos, neg_ok, tau=TAU):
    """s_loc (R, D) normalised student vectors (grad), t_all (N, D) normalised teacher vectors (detached).
    -> (sum CE over ROIs with >= 1 negative, n such ROIs, per-ROI CE)."""
    logits = (s_loc.float() @ t_all.detach().float().T) / tau
    allow = neg_ok.clone()
    allow[torch.arange(len(pos), device=pos.device), pos] = True
    logits = logits.masked_fill(~allow, float("-inf"))
    ce = F.cross_entropy(logits, pos, reduction="none")
    use = neg_ok.any(1)
    return torch.where(use, ce, torch.zeros_like(ce)).sum(), use.sum(), ce
