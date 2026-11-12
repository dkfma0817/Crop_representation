"""Numerical checks of the ATAS-inspired losses (CPU) -> <out>/loss_checks.json"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from src.e6atas.losses import contrastive_terms, negative_mask, region_vec, relation_error_full, relation_terms, sample_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    R, N, D = 6, 576, 768
    T = torch.randn(R, N, D)
    w = torch.rand(R, N)
    w[:, :200] = 0
    w[3, 220:] = 0                         # ROI 3: only 20 valid cells (< K)
    w[4] = 0
    w[4, 300] = 1.0                        # ROI 4: a single valid cell -> excluded from L_rel
    valid = w > 0
    res = {}
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    i1, o1 = sample_idx(valid, g1)
    i2, o2 = sample_idx(valid, g2)
    rng_before = torch.random.get_rng_state()
    i3, _ = sample_idx(valid, torch.Generator().manual_seed(5))
    res["sampling_deterministic_same_seed"] = bool(torch.equal(i1, i2) and torch.equal(o1, o2))
    res["sampling_uses_no_global_rng"] = bool(torch.equal(rng_before, torch.random.get_rng_state()))
    res["sampled_cells_all_valid"] = bool(torch.gather(valid, 1, i1)[o1].all())
    res["n_ok_per_roi"] = o1.sum(1).tolist()
    s, n, err = relation_terms(T.clone(), T, w, i1, o1)
    res["relation_identical_features_error_max"] = float(err.max())
    s, n, err = relation_terms(T * 3.0 + 0.0, T, w, i1, o1)
    res["relation_positive_rescale_error_max"] = float(err.max())
    Sx = T.clone(); Sx[:, :200] = torch.randn(R, 200, D) * 100
    s, n, err = relation_terms(Sx, T, w, i1, o1)
    res["relation_padding_cells_ignored_error_max"] = float(err.max())
    res["relation_n_valid_rois"] = int(n)
    Sg = torch.randn(R, N, D, requires_grad=True)
    Tg = T.clone().requires_grad_(True)
    s, n, err = relation_terms(Sg, Tg, w, i1, o1)
    (s / n).backward()
    res["relation_grad_finite"] = bool(torch.isfinite(Sg.grad).all())
    res["relation_teacher_grad_none"] = Tg.grad is None
    res["relation_grad_on_padding_zero"] = float(Sg.grad[:, :200].abs().max())
    full = relation_error_full(T, T, w)
    res["relation_full_identical"] = float(full[:4].max())
    # contrastive
    t = region_vec(T, w.clamp_min(0))
    keys = torch.tensor([0, 0, 1, 2, 3, 4])
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10], [0, 0, 5, 5], [5, 5, 9, 9], [2, 2, 8, 8]], dtype=torch.float)
    pos = torch.arange(R)
    allv = torch.ones(R, dtype=torch.bool)
    neg_ok, st = negative_mask(t, keys, boxes, pos, t, keys, boxes, allv)
    res["fn_same_image_iou_excluded_pair01"] = bool(not neg_ok[0, 1] and not neg_ok[1, 0])
    res["diff_image_same_box_kept_pair02"] = bool(neg_ok[0, 2])
    res["positive_never_negative"] = bool(not neg_ok.diagonal().any())
    res["neg_stats"] = st
    T2 = T.clone(); T2[5] = T2[2] * 1.001
    w2 = w.clone(); w2[5] = w2[2]                                   # same crop content and coverage, different image
    t2 = region_vec(T2, w2)
    ok2, st2 = negative_mask(t2, keys, boxes, pos, t2, keys, boxes, allv)
    res["fn_near_identical_crop_excluded_pair25"] = bool(not ok2[2, 5] and not ok2[5, 2])
    sv = t.clone().requires_grad_(True)
    tt = t.clone().requires_grad_(True)
    l, n, ce = contrastive_terms(sv, tt, pos, neg_ok)
    (l / n).backward()
    res["contrastive_perfect_student_ce"] = ce.tolist()
    res["contrastive_teacher_grad_none"] = tt.grad is None
    res["contrastive_grad_finite"] = bool(torch.isfinite(sv.grad).all())
    only = torch.zeros(R, R, dtype=torch.bool)                       # no negatives at all -> excluded, finite
    l0, n0, ce0 = contrastive_terms(t, t, pos, only)
    res["contrastive_no_negative_excluded"] = bool(int(n0) == 0 and torch.isfinite(l0))
    ok = (res["sampling_deterministic_same_seed"] and res["sampling_uses_no_global_rng"] and res["sampled_cells_all_valid"]
          and res["relation_identical_features_error_max"] < 1e-10 and res["relation_positive_rescale_error_max"] < 1e-10
          and res["relation_padding_cells_ignored_error_max"] < 1e-10 and res["relation_n_valid_rois"] == 5 and res["relation_grad_finite"]
          and res["relation_teacher_grad_none"] and res["relation_grad_on_padding_zero"] == 0 and res["relation_full_identical"] < 1e-10
          and res["fn_same_image_iou_excluded_pair01"] and res["diff_image_same_box_kept_pair02"] and res["positive_never_negative"]
          and res["fn_near_identical_crop_excluded_pair25"] and res["contrastive_teacher_grad_none"] and res["contrastive_grad_finite"]
          and res["contrastive_no_negative_excluded"])
    res["ALL_PASS"] = bool(ok)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "loss_checks.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
