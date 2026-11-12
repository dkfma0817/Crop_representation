"""lambda calibration for the ATAS-inspired auxiliary losses (once, before training; separate process; no optimizer step).
Model = E6 at its seed-0 canonical init (= what B / C / D start from; output S = Bicubic at init). Batches = the first 16
rows of the 12k COCO stream (= the first 16 training steps, 64 ROIs each, full row in one process). S is re-leafed in fp32:
  g_old = ||dL_old/dS||_F, g_rel = ||dL_rel/dS||_F, g_con = ||dL_con/dS||_F   (training reductions)
  lambda_rel = 0.5 * median_b g_old / g_rel ,  lambda_con = 0.5 * median_b g_old / g_con
(the same rule as the e6_direction pilot: the auxiliary gradient is half the reconstruction gradient at init).
Also recorded: loss values, pairwise gradient cosines (conflict = cos < 0), teacher-teacher region cosine statistics,
false-negative exclusions, positive-negative margin and in-batch retrieval top-1 at init.

  CUDA_VISIBLE_DEVICES=2 python -m src.e6atas.calibrate --out <run>/calib
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import src.msd.common as MS
from src.config import COCO_TRAIN, FEAT, STREAM
from src.e6atas.losses import contrastive_terms, negative_mask, region_vec, relation_terms, sample_idx
from src.losses.feature import dense_cos, dense_mse
from src.msd.pretrain import eval_exclusions
from src.xdec import geometry as XG
from src.xdec import models as XM
from src.xdec.pretrain import STATS, teacher12

SEED_REL = 12345


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_batches", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    XG.install()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(STATS)
    dec = XM.build("E6").to(dev)
    dec.set_stats(torch.as_tensor(z["mean_12"], device=dev), torch.as_tensor(z["std_12"], device=dev))
    dec.eval()
    sum0 = float(sum(p.double().sum() for p in dec.parameters()))
    student = MS.Student(adapters=False).to(dev).eval()
    rng0 = torch.random.get_rng_state().clone()
    ex = set(eval_exclusions()["raev2_stream"])
    ds = MS.CocoStreamDS(pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet")), COCO_TRAIN)
    rec, tt_offdiag = [], []
    for k, row in enumerate(np.load(STREAM)[: a.n_batches]):
        b = MS.to_dev(ds[[int(i) for i in row if int(i) not in ex]], dev)
        g = b["geom"]
        with torch.no_grad():
            feats, _ = student(b["full"])
            T = teacher12(student.vit, b["crops"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                S0 = dec(feats, g).float()
        w = g["valid_w"].float()
        S = S0.detach().clone().requires_grad_(True)
        R = S.shape[0]
        l_old = (dense_mse(S, T, w) + 1.0 * (1 - dense_cos(S, T, w))).sum() / R
        gen = torch.Generator(device=dev).manual_seed(SEED_REL + k)
        idx, ok = sample_idx(w > 0, gen)
        rs, rn, _ = relation_terms(S, T, w, idx, ok)
        l_rel = rs / rn.clamp_min(1)
        t = region_vec(T, w)
        s = region_vec(S, w)
        key, box, pos = g["img_index"], g["roi_frame"][:, :4].float(), torch.arange(R, device=dev)
        neg_ok, st = negative_mask(t, key, box, pos, t, key, box, torch.ones(R, dtype=torch.bool, device=dev))
        cs, cn, _ = contrastive_terms(s, t, pos, neg_ok)
        l_con = cs / cn.clamp_min(1)
        go, = torch.autograd.grad(l_old, S, retain_graph=True)
        gr, = torch.autograd.grad(l_rel, S, retain_graph=True)
        gc, = torch.autograd.grad(l_con, S)
        cosf = lambda x, y: float(torch.nn.functional.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)))  # noqa: E731
        with torch.no_grad():
            sim = s.detach() @ t.T
            tt = t @ t.T
            offd = ~torch.eye(R, dtype=torch.bool, device=dev)
            tt_offdiag.append(tt[offd].cpu())
            negm = torch.where(neg_ok, sim, torch.full_like(sim, float("nan")))
            margin = (sim.diagonal() - torch.nanmean(negm, 1))
            top1 = (torch.where(neg_ok | torch.eye(R, dtype=torch.bool, device=dev), sim, torch.full_like(sim, -9)).argmax(1) == pos).float().mean()
        rec.append(dict(batch=k, n_roi=R, L_old=float(l_old), L_rel=float(l_rel), L_con=float(l_con), n_rel=int(rn), n_con=int(cn),
                        g_old=float(go.norm()), g_rel=float(gr.norm()), g_con=float(gc.norm()),
                        cos_old_rel=cosf(go, gr), cos_old_con=cosf(go, gc), cos_rel_con=cosf(gr, gc),
                        margin_init=float(margin.mean()), top1_init=float(top1), **{f"neg_{k2}": v for k2, v in st.items()}))
        print(json.dumps(rec[-1]), flush=True)
    r_rel = np.array([r["g_old"] / r["g_rel"] for r in rec])
    r_con = np.array([r["g_old"] / r["g_con"] for r in rec])
    tt = torch.cat(tt_offdiag).numpy()
    out = dict(lambda_rel=float(0.5 * np.median(r_rel)), lambda_con=float(0.5 * np.median(r_con)),
               ratio_rel=dict(median=float(np.median(r_rel)), min=float(r_rel.min()), max=float(r_rel.max()), cv=float(r_rel.std() / r_rel.mean())),
               ratio_con=dict(median=float(np.median(r_con)), min=float(r_con.min()), max=float(r_con.max()), cv=float(r_con.std() / r_con.mean())),
               grad_cos_median={k: float(np.median([r[k] for r in rec])) for k in ("cos_old_rel", "cos_old_con", "cos_rel_con")},
               grad_conflict_batches={k: int(sum(r[k] < 0 for r in rec)) for k in ("cos_old_rel", "cos_old_con", "cos_rel_con")},
               teacher_teacher_cos=dict(mean=float(tt.mean()), p50=float(np.median(tt)), p95=float(np.percentile(tt, 95)), max=float(tt.max()),
                                        frac_ge_0p95=float((tt >= 0.95).mean())),
               neg_excluded_iou=float(sum(r["neg_n_excl_iou"] for r in rec)), neg_excluded_cos=float(sum(r["neg_n_excl_cos"] for r in rec)),
               neg_candidates=float(sum(r["neg_n_cand"] for r in rec)), rois_without_negative=int(sum(r["n_roi"] - r["n_con"] for r in rec)),
               margin_init=float(np.mean([r["margin_init"] for r in rec])), top1_init=float(np.mean([r["top1_init"] for r in rec])),
               tau=0.07, k_rel=128, seed_rel=SEED_REL, batches=rec, device=str(dev),
               model_unchanged=float(sum(p.double().sum() for p in dec.parameters())) == sum0,
               rng_unchanged=bool(torch.equal(rng0, torch.random.get_rng_state())))
    out["near_identical_gradient_warning"] = {k: v > 0.95 for k, v in out["grad_cos_median"].items() if k.startswith("cos_old")}
    out["sane"] = bool(all(1e-3 <= out[k] <= 1e3 and np.isfinite(out[k]) for k in ("lambda_rel", "lambda_con"))
                       and out["ratio_rel"]["cv"] < 0.5 and out["ratio_con"]["cv"] < 0.5 and out["model_unchanged"])
    json.dump(out, open(os.path.join(a.out, "calib.json"), "w"), indent=1)
    print("CALIB", json.dumps({k: v for k, v in out.items() if k != "batches"}), flush=True)
    if not out["sane"]:
        raise SystemExit("calibration not sane -> stop and inspect (no arbitrary lambda)")


if __name__ == "__main__":
    main()
