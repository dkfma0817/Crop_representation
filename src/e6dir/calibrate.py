"""lambda_dir calibration (once, before training; no optimizer step, no model update, separate process -> training's init,
RNG and sampler order are untouched). Model = E6 at its seed-0 canonical init (exactly what training starts from).
Batches = the first 16 rows of the 12k COCO stream (= the first 16 training steps, 64 ROIs each, PACO-overlap images removed
as in training), full row in one process. Output S = decoder output under the training autocast (bf16 decoder, fp32 B),
then S is re-leafed in fp32 and both losses are differentiated w.r.t. S in fp32:
  L_old = sum_r [MSE_w + 1.0 (1 - cos)_w]_r / R      (the training reduction)
  L_dir = sum_{valid r} (1 - c_dir,r) / N_valid
  lambda_dir = 0.5 * median_b ||dL_old/dS_b||_F / ||dL_dir/dS_b||_F
Teacher-energy threshold (L_dir exclusion) fixed here first: thr = 1e-3 * median over the 16 batches' ROIs of e_T.

  CUDA_VISIBLE_DEVICES=2 python -m src.e6dir.calibrate --out <run>/calib
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import src.msd.common as MS
from src.config import COCO_TRAIN, FEAT, STREAM
from src.e6dir.loss import dir_terms
from src.losses.feature import dense_cos, dense_mse
from src.msd.pretrain import eval_exclusions
from src.xdec import geometry as XG
from src.xdec import models as XM
from src.xdec.pretrain import STATS, teacher12


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
    rng0 = torch.random.get_rng_state().clone()        # from here on (batches, forward, gradients) no RNG draw
    vit = student.vit
    ex = set(eval_exclusions()["raev2_stream"])
    ds = MS.CocoStreamDS(pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet")), COCO_TRAIN)
    rows = np.load(STREAM)[: a.n_batches]
    cache = []
    for row in rows:
        b = MS.to_dev(ds[[int(i) for i in row if int(i) not in ex]], dev)
        with torch.no_grad():
            feats, _ = student(b["full"])
            T = teacher12(vit, b["crops"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                S = dec(feats, b["geom"]).float()
        cache.append((S, T, b["geom"]["valid_w"].float()))
    # threshold from teacher centred energies of all calibration ROIs
    from src.e6dir.loss import centered_dir
    eTs = torch.cat([centered_dir(S, T, w)[1] for S, T, w in cache])
    thr = float(1e-3 * eTs.median())
    rec = []
    for k, (S, T, w) in enumerate(cache):
        Sl = S.detach().clone().requires_grad_(True)
        R = Sl.shape[0]
        l_old = (dense_mse(Sl, T, w) + 1.0 * (1 - dense_cos(Sl, T, w))).sum() / R
        lsum, nval, c, valid, eT = dir_terms(Sl, T, w, thr)
        l_dir = lsum / nval.clamp_min(1)
        g_old, = torch.autograd.grad(l_old, Sl, retain_graph=True)
        g_dir, = torch.autograd.grad(l_dir, Sl)
        n_old, n_dir = float(g_old.norm()), float(g_dir.norm())
        cosg = float(torch.nn.functional.cosine_similarity(g_old.reshape(1, -1), g_dir.reshape(1, -1)))
        rec.append(dict(batch=k, n_roi=R, n_valid_dir=int(nval), L_old=float(l_old), L_dir=float(l_dir), c_dir_mean=float(c[valid].mean()),
                        g_old=n_old, g_dir=n_dir, ratio=n_old / n_dir, grad_cos=cosg))
        print(json.dumps(rec[-1]), flush=True)
    ratios = np.array([r["ratio"] for r in rec])
    lam = float(0.5 * np.median(ratios))
    sum1 = float(sum(p.double().sum() for p in dec.parameters()))
    out = dict(lambda_dir=lam, median_ratio=float(np.median(ratios)), ratio_min=float(ratios.min()), ratio_max=float(ratios.max()),
               ratio_cv=float(ratios.std() / ratios.mean()), thr_eT=thr, eT_median=float(eTs.median()), eT_min=float(eTs.min()),
               n_excluded=int((eTs <= thr).sum()), n_roi=int(len(eTs)), batches=rec, device=str(dev),
               model_unchanged=sum1 == sum0, rng_unchanged=bool(torch.equal(rng0, torch.random.get_rng_state())),
               g_old_median=float(np.median([r["g_old"] for r in rec])), g_dir_median=float(np.median([r["g_dir"] for r in rec])),
               grad_cos_median=float(np.median([r["grad_cos"] for r in rec])))
    out["sane"] = bool(1e-3 <= lam <= 1e3 and out["ratio_cv"] < 0.5 and out["model_unchanged"] and np.isfinite(lam))
    json.dump(out, open(os.path.join(a.out, "calib.json"), "w"), indent=1)
    print("LAMBDA", json.dumps({k: v for k, v in out.items() if k != "batches"}), flush=True)
    if not out["sane"]:
        raise SystemExit("calibration not sane -> stop and inspect (no arbitrary lambda)")


if __name__ == "__main__":
    main()
