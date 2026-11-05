"""D0: training-free ROI-binding diagnostic on PACO VAL predictions (same pairs for every representation).

Pairs (pre-registered): within-image (i, j, a) with i != j, same image, same level, Y[i,a] = 1 and Y[j,a] = 0 (explicit
negative, attribute type known). Every within pair gets up to 5 MATCHED cross-image pairs (i, j', a): j' explicit
negative for a, same level, same category, same size stratum as j, from another image (rng seed 0).
  PA_w  = P(s_i,a > s_j,a)          within-image pair accuracy (ties 0.5)
  PA_x  = P(s_i,a > s_j',a)         same i, matched negatives from other images  -> "score / classification" level
  BD    = PA_x - PA_w               binding deficit: how much harder a negative is because it is in the SAME image
  dP    = sigmoid(s_i,a) - sigmoid(s_j,a)   prediction change when the ROI is replaced by its same-image partner
Strata: all; obj_small (object, SigLIP2 n_in 1-16) / obj_large (object 17+) / part; same vs different category;
overlapping (box IoU > 0) vs disjoint partner. Image bootstrap (1,000, seed 0; pair weight = multiplicity of i's image).
Comparison per stratum: F336 baseline (ml_r2) vs C0 crop (c0_r2): score gap = PA_x diff, binding gap = BD diff.
Decision (pre-registered): binding problem CONFIRMED -> E05 iff on 'all' pairs
  BD(ml_r2) >= 1.0 pp with CI lower > 0   AND   BD(ml_r2) - BD(c0_r2) > 0 with CI lower > 0;  otherwise E08.

  python -m src.readout.d0_binding --run <queue run dir> --out <run>/D0
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import src.readout.common as RC
from pom_common import image_bootstrap_weights  # noqa: E402  (path set by src.readout.common -> eval_tables)

REPS = {"ml_r2": "ctrl_f336/ml_r2", "r2_final": "ctrl_f336/r2_final", "c0_r2": "ctrl_c0/c0_r2", "f672_ml_r2": "ctrl_f672/f672_ml_r2"}


def final_scores(run, rel):
    d = os.path.join(run, rel)
    cv = [json.loads(l) for l in open(os.path.join(d, "curves.jsonl"))]
    ep = cv[-1]["epoch"]
    return np.load(os.path.join(d, f"scores_val_ep{ep}.npy")).astype(np.float32), ep


def iou(b1, b2):
    l, t = np.maximum(b1[:, 0], b2[:, 0]), np.maximum(b1[:, 1], b2[:, 1])
    r, b = np.minimum(b1[:, 2], b2[:, 2]), np.minimum(b1[:, 3], b2[:, 3])
    inter = np.clip(r - l, 0, None) * np.clip(b - t, 0, None)
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    return inter / np.maximum(a1 + a2 - inter, 1e-9)


def build_pairs(df, Y, nin, k_cross=5, seed=0):
    lv = df.level.values
    cat = df.category_id.values
    img = df.image_id.values
    size = np.where(lv == "part", "part", np.where(nin <= 16, "obj_small", "obj_large"))
    box = df[["l", "t", "r", "b"]].values.astype(np.float64)
    W = []
    for _, g in df.groupby("image_id", sort=False):
        rows = g.index.to_numpy()
        for li in ("object", "part"):
            rr = rows[lv[rows] == li]
            if len(rr) < 2:
                continue
            I, J = np.meshgrid(rr, rr, indexing="ij")
            I, J = I[I != J], J[I != J]
            m = (Y[I] == 1) & (Y[J] == 0)                           # (pairs, 59)
            pi, aa = np.nonzero(m)
            W.append(np.stack([I[pi], J[pi], aa], 1))
    P = pd.DataFrame(np.concatenate(W), columns=["i", "j", "a"])
    P["level"] = lv[P.i]
    P["stratum"] = size[P.i]
    P["same_cat"] = cat[P.i] == cat[P.j]
    P["overlap"] = iou(box[P.i], box[P.j]) > 0
    P["img"] = img[P.i]
    # matched cross-image negatives
    rng = np.random.default_rng(seed)
    key = pd.Series(list(zip(lv, cat, size)))
    neg_by = {}
    for a in range(Y.shape[1]):
        neg = np.flatnonzero(Y[:, a] == 0)
        if len(neg):
            neg_by[a] = pd.Series(neg).groupby(key.values[neg]).apply(lambda s: s.to_numpy()).to_dict()
    X = np.full((len(P), k_cross), -1, np.int64)
    for n, (i, j, a) in enumerate(P[["i", "j", "a"]].to_numpy()):
        cand = neg_by.get(a, {}).get((lv[j], cat[j], size[j]))
        if cand is None:
            continue
        cand = cand[(img[cand] != img[i])]
        if len(cand):
            X[n, :min(k_cross, len(cand))] = rng.choice(cand, min(k_cross, len(cand)), replace=False)
    return P, X


def pair_stats(S, P, X):
    si, sj = S[P.i.values, P.a.values], S[P.j.values, P.a.values]
    win = (si > sj) + 0.5 * (si == sj)
    valid = X >= 0
    sx = np.where(valid, S[np.clip(X, 0, None), P.a.values[:, None]], np.nan)
    wx = np.where(valid, (si[:, None] > sx) + 0.5 * (si[:, None] == sx), np.nan)
    cross = np.nanmean(wx, 1)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    return dict(win=win.astype(np.float64), cross=cross, dP=sig(si) - sig(sj))


def boot_mean(v, Wp, m):
    """weighted mean over pairs in mask m for every bootstrap row -> (B+1,)."""
    ok = m & np.isfinite(v)
    w = Wp[:, ok]
    return (w * v[ok]).sum(1) / np.maximum(w.sum(1), 1e-12)


def ci(x):
    b = x[1:][np.isfinite(x[1:])]
    return dict(point=float(x[0]), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))


def analyse(stats, P, Wp, base="ml_r2", ref="c0_r2"):
    strata = {"all": np.ones(len(P), bool)}
    for s in ("obj_small", "obj_large", "part"):
        strata[s] = (P.stratum == s).values
    strata["same_cat"], strata["diff_cat"] = P.same_cat.values, ~P.same_cat.values
    strata["overlap"], strata["disjoint"] = P.overlap.values, ~P.overlap.values
    has_x = np.isfinite(stats[base]["cross"]) & np.isfinite(stats[ref]["cross"]) if ref in stats else np.isfinite(stats[base]["cross"])
    out = {}
    for sn, m in strata.items():
        m = m & has_x
        rec = dict(n_pairs=int(m.sum()), n_images=int(P.img[m].nunique()))
        B = {}
        for rep, st in stats.items():
            pw, px, dp = boot_mean(st["win"], Wp, m), boot_mean(st["cross"], Wp, m), boot_mean(st["dP"], Wp, m)
            B[rep] = dict(PA_w=100 * pw, PA_x=100 * px, BD=100 * (px - pw), dP=dp)
            rec[rep] = {k: ci(v) for k, v in B[rep].items()}
        for rep in stats:
            if rep == ref or ref not in B:
                continue
            rec[f"{rep}-{ref}"] = {f"{k}_diff": ci(B[rep][k] - B[ref][k]) for k in ("PA_w", "PA_x", "BD")}
        out[sn] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = RC.load_data(a.exclude or None, splits=("train", "val"))
    dva, Yva = data["val"]
    nin = np.load(RC.NIN_SG.format("val"))
    assert len(nin) == len(dva)
    S, eps = {}, {}
    for rep, rel in REPS.items():
        try:
            S[rep], eps[rep] = final_scores(a.run, rel)
        except FileNotFoundError:
            print("missing", rep, flush=True)
    assert "ml_r2" in S, "baseline predictions missing"
    P, X = build_pairs(dva, Yva, nin)
    P.assign(**{f"x{k}": X[:, k] for k in range(X.shape[1])}).to_parquet(os.path.join(a.out, "pairs.parquet"), index=False)
    Wimg = image_bootstrap_weights(dva.image_id.values, 1000, 0).astype(np.float32)
    Wimg = np.concatenate([np.ones((1, len(dva)), np.float32), Wimg])
    Wp = Wimg[:, P.i.values]
    stats = {rep: pair_stats(s, P, X) for rep, s in S.items()}
    res = analyse(stats, P, Wp)
    al = res["all"]
    bd = al["ml_r2"]["BD"]
    gap = al.get("ml_r2-c0_r2", {}).get("BD_diff")
    confirmed = bool(bd["point"] >= 1.0 and bd["lo"] > 0 and gap is not None and gap["lo"] > 0)
    decision = dict(binding_problem_confirmed=confirmed, slot1="E05" if confirmed else "E08",
                    rule="E05 iff all-pairs BD(ml_r2) >= 1.0pp, CI lo > 0, and BD(ml_r2)-BD(c0_r2) CI lo > 0",
                    BD_ml_r2=bd, BD_gap_vs_c0=gap, score_gap_vs_c0=al.get("ml_r2-c0_r2", {}).get("PA_x_diff"),
                    epochs_used=eps, reps=list(S))
    json.dump(dict(decision=decision, strata=res), open(os.path.join(a.out, "d0.json"), "w"), indent=1)
    open(os.path.join(a.out, "slot1.txt"), "w").write(decision["slot1"] + "\n")
    lines = ["# D0 ROI binding diagnostic (PACO val, final-epoch predictions)", "",
             f"Decision: **{decision['slot1']}** (binding problem confirmed = {confirmed}). Rule: {decision['rule']}", "",
             "| stratum | pairs | rep | PA_w | PA_x | BD = PA_x - PA_w | dP |", "|---|---|---|---|---|---|---|"]
    f = lambda d: f"{d['point']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"  # noqa: E731
    for sn, rec in res.items():
        for rep in S:
            r = rec[rep]
            lines.append(f"| {sn} | {rec['n_pairs']} | {rep} | {f(r['PA_w'])} | {f(r['PA_x'])} | {f(r['BD'])} | {f(r['dP'])} |")
    lines += ["", "Differences vs C0 (score gap = PA_x diff, binding gap = BD diff):", "",
              "| stratum | rep - c0 | PA_w diff | PA_x diff (score) | BD diff (binding) |", "|---|---|---|---|---|"]
    for sn, rec in res.items():
        for k, v in rec.items():
            if k.endswith("-c0_r2"):
                lines.append(f"| {sn} | {k} | {f(v['PA_w_diff'])} | {f(v['PA_x_diff'])} | {f(v['BD_diff'])} |")
    open(os.path.join(a.out, "d0.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=1), flush=True)


if __name__ == "__main__":
    main()
