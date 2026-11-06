"""Numerical checks of the Centered Direction Loss (CPU). -> <out>/loss_checks.json"""
import argparse
import json
import os

import torch

from src.e6dir.loss import centered_dir, centered_dir_flatten_ref, dir_terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    R, N, D = 6, 576, 768
    T = torch.randn(R, N, D) * 1.5 + torch.randn(1, 1, D) * 3
    w = torch.rand(R, N)
    w[:, :100] = 0                                    # letterbox padding cells
    w[:, 100:150] = torch.rand(1, 50) * 0.5           # fractional coverage
    res = {}
    c, _, _ = centered_dir(T.clone(), T, w)
    res["identical_pattern_c"] = c.tolist()
    S = torch.randn(R, N, D)
    c0, _, _ = centered_dir(S, T, w)
    c1, _, _ = centered_dir(S + torch.randn(R, 1, D) * 10, T, w)
    res["common_offset_max_abs_change"] = float((c1 - c0).abs().max())
    c2, _, _ = centered_dir(S * 3.7, T, w)
    res["positive_scale_max_abs_change"] = float((c2 - c0).abs().max())
    c3, _, _ = centered_dir(-S, T, w)
    res["negative_scale_c_plus_c0_max_abs"] = float((c3 + c0).abs().max())
    ref = centered_dir_flatten_ref(S, T, w)
    res["vs_flatten_sqrtw_reference_max_abs"] = float((c0.double() - ref).abs().max())
    # squared-weight implementation would differ (sanity that fractional w matters)
    ref_sq = centered_dir_flatten_ref(S, T, w ** 2)
    res["if_weights_were_squared_diff"] = float((c0.double() - ref_sq).abs().max())
    # padding cells do not matter
    S_pad = S.clone(); S_pad[:, :100] += 100.0
    c4, _, _ = centered_dir(S_pad, T, w)
    res["padding_cells_change_max_abs"] = float((c4 - c0).abs().max())
    # constant teacher / zero coverage -> excluded, finite forward / backward
    T2 = T.clone(); T2[0] = T2[0, 120:121].expand(N, D)   # constant over positions
    w2 = w.clone(); w2[1] = 0                              # zero coverage ROI
    S3 = torch.randn(R, N, D, requires_grad=True)
    lsum, nval, c5, valid, eT = dir_terms(S3, T2, w2, thr=1e-6)
    lsum.backward()
    res["constant_teacher_excluded"] = bool(not valid[0])
    res["zero_coverage_excluded"] = bool(not valid[1])
    res["n_valid"] = int(nval)
    res["forward_finite"] = bool(torch.isfinite(c5).all()) and bool(torch.isfinite(lsum))
    res["grad_finite"] = bool(torch.isfinite(S3.grad).all())
    res["grad_zero_on_excluded"] = float(S3.grad[:2].abs().max())
    # student with zero centred pattern (constant student): finite
    S4 = T.mean(1, keepdim=True).expand(R, N, D).clone().requires_grad_(True)
    l4, _, c6, _, _ = dir_terms(S4, T, w, thr=1e-6)
    l4.backward()
    res["constant_student_c"] = c6.tolist()
    res["constant_student_grad_finite"] = bool(torch.isfinite(S4.grad).all())
    # teacher gradient blocked
    Tg = T.clone().requires_grad_(True)
    Sg = torch.randn(R, N, D, requires_grad=True)
    lg, _, _, _, _ = dir_terms(Sg, Tg, w, thr=1e-6)
    lg.backward()
    res["teacher_grad_is_none"] = Tg.grad is None
    ok = (min(res["identical_pattern_c"]) > 1 - 1e-5 and res["common_offset_max_abs_change"] < 1e-4 and res["positive_scale_max_abs_change"] < 1e-5
          and res["negative_scale_c_plus_c0_max_abs"] < 1e-5 and res["vs_flatten_sqrtw_reference_max_abs"] < 1e-5
          and res["padding_cells_change_max_abs"] < 1e-6 and res["constant_teacher_excluded"] and res["zero_coverage_excluded"]
          and res["forward_finite"] and res["grad_finite"] and res["grad_zero_on_excluded"] == 0 and res["constant_student_grad_finite"]
          and res["teacher_grad_is_none"])
    res["ALL_PASS"] = bool(ok)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "loss_checks.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
