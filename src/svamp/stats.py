"""Feature statistics + transform checks on PACO VAL (no training, no labels used). CPU or one GPU.

Per ROI (w = C0 valid coverage, M = w > 0, fp32), for V0..V3, O2 and C0 itself (x) against the C0 teacher T:
  tok_norm  = coverage-weighted mean token L2 norm on M          pool_norm = ||mu_x||
  tok_cos   = coverage-weighted mean cos(x_i, T_i)               pool_cos  = cos(mu_x, mu_T)
  sv        = sum_w mean_c (x_i - mu_x)^2 / sum_w                 sv_ratio  = sv / sv(T)   (= xdec recon definition)
  e_cent    = sum_w mean_c ((x_i - mu_x) - (T_i - mu_T))^2 / sum_w   (centred difference)
  e_mean    = mean_c (mu_x - mu_T)^2                                 (mean difference)
  cent_cos  = coverage-weighted cos(x_i - mu_x, T_i - mu_T)
Checks: |mu_V - mu_B| (mean preservation), max |V - B| on invalid cells, R0 pooled input (cov_c path) equality,
cov_c (R0/R2 coverage) == valid_w, measured sv ratio V_k / V0 == k^2.

  CUDA_VISIBLE_DEVICES= python -m src.svamp.stats --out <run> [--limit N]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import src.msd.common as MS
import src.readout.common as RC
from src.svamp.feature_transform import CONDS, teacher, transform, wmean
from src.xdec import geometry as XG

EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"


def roi_stats(x, T, w):
    ws = w.sum(1)
    px, pt = wmean(x, w), wmean(T, w)
    rx, rt = x - px[:, None], T - pt[:, None]
    wm = lambda v: (v * w).sum(1) / ws  # noqa: E731
    sv, svt = wm(rx.pow(2).mean(-1)), wm(rt.pow(2).mean(-1))
    return dict(tok_norm=wm(x.norm(dim=-1)), pool_norm=px.norm(dim=-1), tok_cos=wm(F.cosine_similarity(x, T, dim=-1, eps=1e-6)),
                pool_cos=F.cosine_similarity(px, pt, dim=-1), sv=sv, sv_ratio=sv / svt.clamp_min(1e-12),
                e_cent=wm((rx - rt).pow(2).mean(-1)), e_mean=(px - pt).pow(2).mean(-1),
                cent_cos=wm(F.cosine_similarity(rx, rt, dim=-1, eps=1e-6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(96)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = RC.load_data(EXCL, a.limit)
    dva, Yva = data["val"]
    ds = MS.MSDImageDS(dva, Yva, flip=False, want_crops=True)
    student = MS.Student(adapters=False).to(dev).eval()
    vit = student.vit
    rows, chk = [], dict(mean_pres_max_abs=0.0, mean_pres_max_rel=0.0, invalid_max_abs=0.0, r0_pooled_max_abs=0.0,
                         cov_c_vs_valid_w_max_abs=0.0, sv_scale_err_max={}, o2_mean_max_abs=0.0, o2_sv_ratio_dev_max=0.0,
                         n_invalid_cells=0, n_roi=0)
    t0 = time.time()
    for s in range(0, len(ds), 8):
        b = MS.to_dev(ds[list(range(s, min(len(ds), s + 8)))], dev)
        g = b["geom"]
        with torch.no_grad():
            feats, _ = student(b["full"])
            T = teacher(vit, b["crops"])
            B = XG.bicubic_roi(feats[12], g)
        w = g["valid_w"].float()
        inv = ~(w > 0)
        chk["cov_c_vs_valid_w_max_abs"] = max(chk["cov_c_vs_valid_w_max_abs"], float((b["cov_c"].float() - w).abs().max()))
        muB = wmean(B, w)
        out = {}
        for c, (kind, k) in CONDS.items():
            X = transform(B, w, kind, k, T)
            out[c] = X
            mu = wmean(X, w)
            d = (mu - muB).abs().max()
            key = "o2_mean_max_abs" if kind == "oracle" else "mean_pres_max_abs"
            chk[key] = max(chk[key], float(d))
            if kind == "amp":
                chk["mean_pres_max_rel"] = max(chk["mean_pres_max_rel"], float(((mu - muB).norm(dim=-1) / muB.norm(dim=-1)).max()))
            chk["invalid_max_abs"] = max(chk["invalid_max_abs"], float((X - B).abs()[inv].max()) if inv.any() else 0.0)
            r0 = (X * b["cov_c"].float()[..., None]).sum(1) / b["cov_c"].float().sum(1, keepdim=True)          # = R0.pooled
            r0b = (B * b["cov_c"].float()[..., None]).sum(1) / b["cov_c"].float().sum(1, keepdim=True)
            chk["r0_pooled_max_abs"] = max(chk["r0_pooled_max_abs"], float((r0 - r0b).abs().max()))
        out["C0"] = T
        st = {c: {k: v.double().cpu().numpy() for k, v in roi_stats(X, T, w).items()} for c, X in out.items()}
        for c, (kind, k) in CONDS.items():
            if kind == "amp":
                e = float(np.nanmax(np.abs(st[c]["sv"] / st["V0"]["sv"] - k * k)))
                chk["sv_scale_err_max"][c] = max(chk["sv_scale_err_max"].get(c, 0.0), e)
        chk["o2_sv_ratio_dev_max"] = max(chk["o2_sv_ratio_dev_max"], float(np.abs(st["O2"]["sv_ratio"] - 1).max()))
        chk["n_invalid_cells"] += int(inv.sum())
        chk["n_roi"] += len(w)
        for c, sd in st.items():
            df = pd.DataFrame(sd)
            df["cond"] = c
            df["row"] = b["rows"].cpu().numpy()
            df["n_in"] = g["n_in"].cpu().numpy()
            df["part"] = b["lv"].cpu().numpy()
            rows.append(df)
        if s % 400 == 0:
            print(json.dumps(dict(images=s, s=round(time.time() - t0))), flush=True)
    res = pd.concat(rows, ignore_index=True)
    res.to_parquet(os.path.join(a.out, f"roi_feature_stats{a.tag}.parquet"))
    chk["time_s"] = round(time.time() - t0)
    chk["device"] = str(dev)
    json.dump(chk, open(os.path.join(a.out, f"transform_checks{a.tag}.json"), "w"), indent=1)
    # aggregate CSV: overall + object / part x D-bin
    dbin = np.select([res.n_in <= 4, res.n_in <= 16], ["D1-4", "D5-16"], "D17+")
    res["group"] = np.where(res.part == 1, "part ", "object ") + dbin
    keys = ["tok_norm", "pool_norm", "tok_cos", "pool_cos", "sv_ratio", "e_cent", "e_mean", "cent_cos"]
    agg = [res.groupby("cond")[keys].mean().assign(group="all")]
    agg.append(res.assign(group=np.where(res.part == 1, "part", "object")).groupby(["cond", "group"])[keys].mean().reset_index("group"))
    agg.append(res.groupby(["cond", "group"])[keys].mean().reset_index("group"))
    tab = pd.concat(agg).reset_index()
    sv_of_means = res.groupby("cond").sv.mean() / res[res.cond == "C0"].sv.mean()
    tab["sv_ratio_of_means_all"] = tab.cond.map(sv_of_means).where(tab.group == "all")
    tab["n_roi"] = tab.apply(lambda r: int(((res.cond == r.cond) & ((r.group == "all") | (res.group == r.group) |
                                                                     ((r.group in ("object", "part")) & (res.part == int(r.group == "part"))))).sum()), axis=1)
    tab.to_csv(os.path.join(a.out, f"feature_statistics{a.tag}.csv"), index=False)
    print("DONE", json.dumps(chk), flush=True)


if __name__ == "__main__":
    main()
