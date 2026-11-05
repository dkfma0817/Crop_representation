"""IVC analysis (CPU): PACO VAL performance, feature differences, prediction-linked analysis, figures -> tables.md + analysis.json.

Rules (fixed in configs/ivc/protocol.md before results):
  * epoch per (condition, reader) = max PACO-dev M0 Overall over epochs 1..10 (MSD rule); val numbers at that epoch
  * M0 / M2 = src.readout.evaluate.Boot (eval_tables definitions), paired image bootstrap B = 1000, seed 0, same draws
  * D-bins = manifest n_in (cov > 0 cells of the 24 x 24 F336 grid): D1-4 / D5-16 / D17+
  * correctness of a ROI-attribute prediction: threshold per (condition, reader, level, attribute) = F1-maximising
    threshold on PACO dev (the 1,000 held-out train images, same epoch); unit must be an M0 unit and have >= 5 dev
    positives and >= 5 dev negatives, else the pair is excluded
  * score link: rank margin r (positive label: share of the same unit's val negatives scored below it, ties 1/2;
    negative label: share of the unit's val positives scored above it) -> scale-free, AP-linked;
    secondary = dev-calibrated logit margin (2y - 1)(s - tau). Per ROI: mean over its positive labels (primary)
  * correlations at ROI level (one value per ROI), Spearman + partial Spearman (rank n_in and level regressed out),
    image-cluster bootstrap 95% CI (1000, seed 0)

  python -m src.ivc.analyze --run output/2026-11-05/ivc
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import src.readout.common as RC
from src.ivc.common import BANDS, DBINS, EXCL
from src.readout.evaluate import Boot, summarize

CONDS = ("bil", "bic", "c0")
READERS = ("r0", "r2")
LAB = {"bil": "Bilinear", "bic": "Bicubic", "c0": "C0 Crop"}
COLS = [("Overall", "all", "overall"), ("Object", "all", "object"), ("Part", "all", "part"), ("Obj D5-16", "D5-16", "object"),
        ("Part D1-4", "D1-4", "part"), ("Obj D17+", "D17+", "object"), ("All D17+", "D17+", "overall"), ("Color", "all", "color"),
        ("Material", "all", "material"), ("Pattern", "all", "pattern_marking")]
CONTRASTS = [("bic", "bil"), ("c0", "bic"), ("c0", "bil")]
NB, SEED = 1000, 0


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def f2(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def fci(d, nd=2):
    if d is None or not np.isfinite(d["point"]):
        return "n/a"
    if not np.isfinite(d["lo"]):
        return f"{d['point']:+.{nd}f}"
    return f"{d['point']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def dev_split(data):
    """Exactly train.py: PACO dev = 1,000 train images, rng 0, from the sorted unique image ids of the filtered train set."""
    dtr, Ytr = data["train"]
    ids = np.array(sorted(dtr.image_id.unique()))
    dev_ids = set(np.random.default_rng(0).choice(ids, 1000, replace=False).tolist())
    m = dtr.image_id.isin(dev_ids).values
    return dtr[m].reset_index(drop=True), Ytr[m]


def load_probe(R, c, rd):
    d = os.path.join(R, f"probe_{c}", rd)
    dv = jl(os.path.join(d, "curves_dev.jsonl"))
    dv = [x for x in dv if x["epoch"] >= 1]
    if not dv or not os.path.exists(os.path.join(R, f"probe_{c}", "done.json")):
        return None
    best = max(dv, key=lambda x: x["dev"]["M0"]["overall"])
    ep = best["epoch"]
    return dict(epoch=ep, dev_M0=best["dev"]["M0"]["overall"], Sv=np.load(os.path.join(d, f"scores_val_ep{ep}.npy")).astype(np.float32),
                Sd=np.load(os.path.join(d, f"scores_dev_ep{ep}.npy")).astype(np.float32),
                curves_dev=[(x["epoch"], x["dev"]["M0"]["overall"]) for x in dv],
                curves_val=[(x["epoch"], x["val"]["M0"]["overall"]) for x in jl(os.path.join(d, "curves.jsonl")) if x["epoch"] >= 1])


# ------------------------------------------------------------------------------------------------ thresholds / groups
def f1_thresholds(Sd, ddev, Ydev, units):
    """-> tau (2, 59) logit thresholds (NaN = unavailable) for the M0 units."""
    tau = np.full((2, 59), np.nan)
    lvd = (ddev.level.values == "part").astype(int)
    for lv, a in units:
        k = int(lv == "part")
        m = (lvd == k) & (Ydev[:, a] >= 0)
        s, y = Sd[m, a], Ydev[m, a] == 1
        P, Nn = int(y.sum()), int((~y).sum())
        if P < 5 or Nn < 5:
            continue
        o = np.argsort(-s, kind="stable")
        tp = np.cumsum(y[o])
        kk = np.arange(1, len(s) + 1)
        f1 = 2 * tp / (kk + P)
        best = int(np.argmax(f1))
        tau[k, a] = s[o][best]
    return tau


def rank_margin(S, Y, lvi, units):
    """-> (N, 59) float, higher = better separated: positive label = share of the same unit's val negatives scored below
    it; negative label = share of the unit's positives scored above it (ties 1/2); NaN outside M0 units / unknown labels."""
    out = np.full(S.shape, np.nan, np.float32)
    for lv, a in units:
        k = int(lv == "part")
        m = np.flatnonzero((lvi == k) & (Y[:, a] >= 0))
        s, y = S[m, a].astype(np.float64), Y[m, a] == 1
        neg, pos = np.sort(s[~y]), np.sort(s[y])
        sp, sn = s[y], s[~y]
        rp = (np.searchsorted(neg, sp, "left") + np.searchsorted(neg, sp, "right")) / 2 / max(len(neg), 1)
        rn = (len(pos) - (np.searchsorted(pos, sn, "left") + np.searchsorted(pos, sn, "right")) / 2) / max(len(pos), 1)
        out[m[y], a] = rp
        out[m[~y], a] = rn          # negative: share of the unit's positives scored ABOVE it (higher = better separated)
    return out


def boot_weights(df, nb=NB, seed=SEED):
    from pom_common import image_bootstrap_weights
    return image_bootstrap_weights(df.image_id.values, nb, seed).astype(np.float32)        # (nb, N)


# ------------------------------------------------------------------------------------------------ correlation helpers
def _resid(r, Z):
    beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
    return r - Z @ beta


def spearman(x, y, ctrl=None, w=None):
    """(weighted by integer bootstrap counts w via repetition) Spearman / partial Spearman (controls = columns of ctrl)."""
    if w is not None:
        idx = np.repeat(np.arange(len(x)), w.astype(int))
        x, y = x[idx], y[idx]
        ctrl = None if ctrl is None else ctrl[idx]
    rx, ry = rankdata(x), rankdata(y)
    if ctrl is not None:
        Z = np.column_stack([np.ones(len(x))] + [rankdata(c) if np.unique(c).size > 2 else c for c in ctrl.T])
        rx, ry = _resid(rx, Z), _resid(ry, Z)
    return float(np.corrcoef(rx, ry)[0, 1])


def corr_ci(x, y, img, ctrl=None, nb=300, seed=SEED):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, img = x[ok], y[ok], img[ok]
    ctrl = None if ctrl is None else ctrl[ok]
    if len(x) < 20:
        return dict(rho=float("nan"), lo=float("nan"), hi=float("nan"), n=int(len(x)), n_img=int(np.unique(img).size))
    pt = spearman(x, y, ctrl)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(img, return_inverse=True)
    bs = []
    for _ in range(nb):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        bs.append(spearman(x, y, ctrl, cnt))
    return dict(rho=pt, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(x)), n_img=int(len(uniq)))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corr_boot", type=int, default=300)
    a = ap.parse_args()
    R = a.run
    FIG = os.path.join(R, "figures")
    os.makedirs(FIG, exist_ok=True)
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    Ytr_by = RC.ytr_by_level(data)
    ddev, Ydev = dev_split(data)
    N = len(dva)
    lvi = (dva.level.values == "part").astype(int)
    dbin = np.select([dva.n_in.values <= 4, dva.n_in.values <= 16], ["D1-4", "D5-16"], "D17+")
    L, OUT = [], {}

    # ============================================================ B. performance
    B = Boot(dva, Yva, Ytr_by)
    BB = {nm: Boot(dva, Yva, Ytr_by, mask=(dva.n_in.values >= lo) & (dva.n_in.values <= hi)) for nm, lo, hi in DBINS}
    units = list(B.keys_o)
    P, RES = {}, {}
    for c in CONDS:
        for rd in READERS:
            p = load_probe(R, c, rd)
            if p is None:
                continue
            assert p["Sv"].shape == (N, 59) and p["Sd"].shape == (len(ddev), 59), (c, rd, p["Sv"].shape, p["Sd"].shape)
            P[(c, rd)] = p
            RES[(c, rd)] = dict(m0={"all": B.m0(p["Sv"]), **{nm: bb.m0(p["Sv"]) for nm, bb in BB.items()}},
                                m2={"all": B.m2(p["Sv"]), **{nm: bb.m2(p["Sv"]) for nm, bb in BB.items()}})
    cfgs = {c: json.load(open(os.path.join(R, f"probe_{c}", "config.json"))) for c in CONDS if os.path.exists(os.path.join(R, f"probe_{c}", "config.json"))}
    same = {k: len({json.dumps(v[k]) for v in cfgs.values()}) == 1 for k in ("n_train_rows", "n_val_rows", "n_val_images", "n_train_images", "steps_per_epoch")}
    L += ["## B. PACO validation performance", "",
          f"Same ROI / label check: val rows {N}, every score file (N, 59) in manifest row order; probe configs agree on "
          + ", ".join(f"{k}={v}" for k, v in same.items()) + f"; label matrix sha = {hash(Yva.tobytes()) & 0xffffffff:08x} (one shared array).", ""]

    def val(c, rd, metric, bin_, g):
        r = RES.get((c, rd))
        if r is None or r[metric][bin_] is None:
            return None
        return r[metric][bin_][g]

    nunits = {nm: (len(bb.keys_o), sum(1 for lv, _ in bb.keys_o if lv == "object"), sum(1 for lv, _ in bb.keys_o if lv == "part")) for nm, bb in BB.items()}
    for metric in ("m0", "m2"):
        L += [f"### {metric.upper()} (val, dev-selected epoch)", "",
              "| Method | Reader | epoch (dev M0) | " + " | ".join(c[0] for c in COLS) + " |", "|---|---|---|" + "---|" * len(COLS)]
        for rd in READERS:
            for c in CONDS:
                if (c, rd) not in P:
                    continue
                cells = [f2(float(v[0])) if (v := val(c, rd, metric, b, g)) is not None else "n/a" for _, b, g in COLS]
                L.append(f"| {LAB[c]} | {rd.upper()} | {P[(c, rd)]['epoch']} ({P[(c, rd)]['dev_M0']:.2f}) | " + " | ".join(cells) + " |")
        L.append("")
        L += [f"**Paired image bootstrap ({metric.upper()}, B={NB}, seed {SEED}): difference [95% CI]**", "",
              "| Contrast | Reader | " + " | ".join(c[0] for c in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
        for rd in READERS:
            for x, y in CONTRASTS:
                if (x, rd) not in P or (y, rd) not in P:
                    continue
                cells = []
                for _, b, g in COLS:
                    vx, vy = val(x, rd, metric, b, g), val(y, rd, metric, b, g)
                    cells.append("n/a" if vx is None or vy is None else fci(summarize(vx, vy)))
                L.append(f"| {LAB[x]} - {LAB[y]} | {rd.upper()} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["M0 units per D-bin (all / object / part): " + "; ".join(f"{k}: {v}" for k, v in nunits.items()) +
          f"; all ROIs: {len(B.keys_o)}. M2 cells are NaN where a bin has no eligible category unit (never imputed).", ""]
    OUT["performance"] = {f"{c}/{rd}": dict(epoch=P[(c, rd)]["epoch"], dev_M0=P[(c, rd)]["dev_M0"],
                                             M0={cn: (None if (v := val(c, rd, "m0", b, g)) is None else float(v[0])) for cn, b, g in COLS},
                                             M2={cn: (None if (v := val(c, rd, "m2", b, g)) is None else float(v[0])) for cn, b, g in COLS})
                          for (c, rd) in P}
    OUT["contrasts"] = {f"{x}-{y}/{rd}/{m}": {cn: summarize(val(x, rd, m, b, g), val(y, rd, m, b, g)) for cn, b, g in COLS
                                               if val(x, rd, m, b, g) is not None and val(y, rd, m, b, g) is not None}
                        for rd in READERS for x, y in CONTRASTS for m in ("m0", "m2") if (x, rd) in P and (y, rd) in P}
    # reproduction check vs the 11-05 MSD probes (same code path for bilinear / C0 R2)
    msd = {}
    for c, k in (("bil", "bil"), ("c0", "c0")):
        p = load_probe("output/2026-11-05/msd", k, "r2")
        if p is not None and (c, "r2") in P:
            msd[c] = dict(msd_epoch=p["epoch"], msd_M0=float(B.m0(p["Sv"])["overall"][0]), ivc_epoch=P[(c, "r2")]["epoch"],
                          ivc_M0=float(RES[(c, "r2")]["m0"]["all"]["overall"][0]))
    if msd:
        L += ["Reproduction check (R2, val M0 Overall): " + "; ".join(f"{LAB[c]}: MSD probe {v['msd_M0']:.2f} (ep {v['msd_epoch']}) vs this run {v['ivc_M0']:.2f} (ep {v['ivc_epoch']})" for c, v in msd.items()) + ".", ""]
    OUT["msd_reproduction"] = msd
    L += ["Dev / val M0 Overall per epoch (R2 | R0):", "", "| Method | Reader | dev M0 ep1..10 | val M0 ep1..10 |", "|---|---|---|---|"]
    for (c, rd), p in P.items():
        L.append(f"| {LAB[c]} | {rd.upper()} | " + " ".join(f"{v:.1f}" for _, v in p["curves_dev"]) + " | " + " ".join(f"{v:.1f}" for _, v in p["curves_val"]) + " |")
    L.append("")

    # ============================================================ C. feature differences
    FD = os.path.join(R, "features")
    met = pd.read_parquet(os.path.join(FD, "roi_metrics.parquet"))
    assert (met.ann_id.values == dva.ann_id.values).all(), "feature rows != manifest rows"
    tm = np.load(os.path.join(FD, "token_maps.npz"))
    assert (tm["ann_id"] == dva.ann_id.values).all()
    cov = tm["cov"].astype(np.float32)
    for k in ("bil", "bic"):
        met[f"{k}_erank_ratio"] = met[f"{k}_erank_a"] / met[f"{k}_erank_c"]
        met[f"{k}_log_sv_ratio"] = np.log(met[f"{k}_sv_ratio"])
        tot_a = sum(met[f"{k}_dct_a_{b}"] for b, _, _ in BANDS)
        tot_c = sum(met[f"{k}_dct_c_{b}"] for b, _, _ in BANDS)
        tot_d = sum(met[f"{k}_dct_d_{b}"] for b, _, _ in BANDS)
        for b, _, _ in BANDS:
            met[f"{k}_fa_{b}"] = met[f"{k}_dct_a_{b}"] / tot_a
            met[f"{k}_fd_{b}"] = met[f"{k}_dct_d_{b}"] / tot_d
        met["c0_fc_" + "low"] = met[f"{k}_dct_c_low"] / tot_c
        met["c0_fc_mid"] = met[f"{k}_dct_c_mid"] / tot_c
        met["c0_fc_high"] = met[f"{k}_dct_c_high"] / tot_c
        met[f"{k}_high_ratio"] = met[f"{k}_dct_a_high"] / met[f"{k}_dct_c_high"]
    W = boot_weights(dva)                                                       # (NB, N) image counts per ROI
    img = dva.image_id.values

    def mean_ci(x, sel=None):
        sel = np.isfinite(x) if sel is None else (sel & np.isfinite(x))
        xs, Ws = x[sel], W[:, sel]
        bs = (Ws * xs).sum(1) / Ws.sum(1)
        return dict(point=float(xs.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(sel.sum()))

    ROWS_C = [("token cosine (w, M)", "tok_cos"), ("token cosine p10 within ROI", "tok_cos_p10"), ("token MSE (w, M)", "tok_mse"),
              ("pooled cosine", "pool_cos"), ("pooled L2", "pool_l2"), ("E_mean (pooled part of MSE)", "e_mean"),
              ("E_cent (centred spatial part)", "e_cent"), ("E_cent / MSE", "frac_cent"), ("centred token cosine", "cent_cos"),
              ("self-sim mean abs(S_a - S_c) (M1)", "ss_diff"), ("spatial var ratio a/c", "sv_ratio"), ("eff. rank ratio a/c (M1)", "erank_ratio"),
              ("all-576 token cosine", "tok_cos_all"), ("padding-cell token cosine", "tok_cos_pad")]
    L += ["## C. Feature difference (PACO val, all ROIs, raw final-layer features, fp32)", "",
          "Per-ROI metric, then mean over ROIs [image-bootstrap 95% CI]; median in parentheses. M = cells with C0 coverage > 0 "
          "(weights = coverage), M1 = fully covered rectangle. Bicubic - Bilinear = paired per-ROI difference.", "",
          "| metric | Bilinear vs C0 | Bicubic vs C0 | Bicubic - Bilinear |", "|---|---|---|---|"]
    OUT["features"] = {}
    for nm, k in ROWS_C:
        xa, xb = met[f"bil_{k}"].values.astype(np.float64), met[f"bic_{k}"].values.astype(np.float64)
        ca, cb, cd = mean_ci(xa), mean_ci(xb), mean_ci(xb - xa)
        OUT["features"][k] = dict(bil=ca, bic=cb, bic_minus_bil=cd, bil_median=float(np.nanmedian(xa)), bic_median=float(np.nanmedian(xb)))
        L.append(f"| {nm} | {ca['point']:.4f} [{ca['lo']:.4f}, {ca['hi']:.4f}] ({np.nanmedian(xa):.4f}) | {cb['point']:.4f} [{cb['lo']:.4f}, {cb['hi']:.4f}] ({np.nanmedian(xb):.4f}) | "
                 f"{cd['point']:+.4f} [{cd['lo']:+.4f}, {cd['hi']:+.4f}] |")
    L.append("")
    # structure: a vs c side by side
    L += ["**Spatial structure (M1; per-ROI means)**", "", "| quantity | Bilinear | Bicubic | C0 |", "|---|---|---|---|"]
    for nm, ka, kc in (("spatial variance (M, per channel)", "sv_a", "sv_c"), ("effective rank (M1)", "erank_a", "erank_c"),
                       ("adjacent-token cosine (M1)", "adj_a", "adj_c"), ("mean off-diag self-similarity (M1)", "ss_mean_a", "ss_mean_c"),
                       ("token L2 norm (M, w)", "norm_a", "norm_c")):
        L.append(f"| {nm} | {np.nanmean(met[f'bil_{ka}']):.4f} | {np.nanmean(met[f'bic_{ka}']):.4f} | {np.nanmean(met[f'bic_{kc}']):.4f} |")
    L += ["", "**2-D DCT on M1 (DC removed; rho = max(u/h, v/w); low < 1/3 <= mid < 2/3 <= high)**: share of non-DC energy per band "
          "(mean over ROIs with |M1| >= 4), and where the difference (a - C0) energy sits", "",
          "| band | Bilinear share | Bicubic share | C0 share | Bilinear/C0 abs. energy (median) | Bicubic/C0 abs. energy (median) | share of (Bil-C0) energy | share of (Bic-C0) energy |",
          "|---|---|---|---|---|---|---|---|"]
    for b, _, _ in BANDS:
        L.append(f"| {b} | {np.nanmean(met[f'bil_fa_{b}']):.3f} | {np.nanmean(met[f'bic_fa_{b}']):.3f} | {np.nanmean(met[f'c0_fc_{b}']):.3f} | "
                 f"{np.nanmedian(met[f'bil_dct_a_{b}'] / met[f'bil_dct_c_{b}']):.3f} | {np.nanmedian(met[f'bic_dct_a_{b}'] / met[f'bic_dct_c_{b}']):.3f} | "
                 f"{np.nanmean(met[f'bil_fd_{b}']):.3f} | {np.nanmean(met[f'bic_fd_{b}']):.3f} |")
    L.append("")
    # by size bin x level
    KEYS = [("tok_cos", "tok cos"), ("pool_cos", "pool cos"), ("tok_mse", "MSE"), ("e_mean", "E_mean"), ("e_cent", "E_cent"), ("frac_cent", "E_cent/MSE"),
            ("ss_diff", "abs dS"), ("sv_ratio", "var a/c"), ("erank_ratio", "erank a/c"), ("high_ratio", "high-band a/c")]
    L += ["**By size bin (manifest n_in) and level: per-ROI means, Bilinear / Bicubic**", "",
          "| level | bin | n ROI | " + " | ".join(n for _, n in KEYS) + " |", "|---|---|---|" + "---|" * len(KEYS)]
    for lv in ("object", "part", "all"):
        for nm, lo, hi in DBINS:
            sel = ((dva.level.values == lv) | (lv == "all")) & (dva.n_in.values >= lo) & (dva.n_in.values <= hi)
            cells = []
            for k, _ in KEYS:
                if k == "high_ratio":
                    cells.append(f"{np.nanmedian(met.loc[sel, f'bil_{k}']):.2f} / {np.nanmedian(met.loc[sel, f'bic_{k}']):.2f} (med)")
                else:
                    cells.append(f"{np.nanmean(met.loc[sel, f'bil_{k}']):.3f} / {np.nanmean(met.loc[sel, f'bic_{k}']):.3f}")
            L.append(f"| {lv} | {nm} | {int(sel.sum())} | " + " | ".join(cells) + " |")
    L.append("")
    # fine n_in relation
    from scipy.stats import spearmanr
    rel = {}
    for k in ("tok_cos", "e_cent", "e_mean", "ss_diff", "frac_cent"):
        rel[k] = {c: float(spearmanr(dva.n_in.values, met[f"{c}_{k}"].values, nan_policy="omit")[0]) for c in ("bil", "bic")}
    L += ["Spearman(n_in, metric) over all val ROIs: " + "; ".join(f"{k}: bil {v['bil']:+.3f} / bic {v['bic']:+.3f}" for k, v in rel.items()), ""]
    OUT["size_spearman"] = rel
    # token-level distribution (cov-weighted over all valid cells)
    hist = {}
    edges = np.linspace(-0.2, 1.0, 61)
    for c in ("bil", "bic"):
        cs = tm[f"cos_{c}"].astype(np.float32)
        for nm, lo, hi in (("all", 1, 10 ** 9),) + DBINS:
            sel = (dva.n_in.values >= lo) & (dva.n_in.values <= hi)
            v, w = cs[sel][cov[sel] > 0], cov[sel][cov[sel] > 0]
            h, _ = np.histogram(np.clip(v, -0.2, 1.0), edges, weights=w)
            q = np.interp([0.1, 0.25, 0.5, 0.75, 0.9], np.cumsum(w[np.argsort(v)]) / w.sum(), np.sort(v))
            hist[(c, nm)] = (h / h.sum(), q)
    L += ["**Token cosine distribution over all valid cells (coverage-weighted quantiles p10 / p25 / p50 / p75 / p90)**", "",
          "| bin | Bilinear | Bicubic |", "|---|---|---|"]
    for nm in ("all",) + tuple(d[0] for d in DBINS):
        L.append(f"| {nm} | " + " | ".join(" / ".join(f"{x:.3f}" for x in hist[(c, nm)][1]) for c in ("bil", "bic")) + " |")
    L.append("")
    # mask regions
    mf = np.load(os.path.join(FD, "mask_frac.npz"))
    assert (mf["ann_id"] == dva.ann_id.values).all()
    frac = mf["frac"]
    regions = {"interior": frac >= 0.9, "boundary": (frac > 0.1) & (frac < 0.9), "background": frac <= 0.1}
    L += ["**PACO mask regions on the C0 grid (cells with coverage > 0; interior frac >= 0.9, boundary 0.1-0.9, background <= 0.1)**",
          "Per ROI: coverage-weighted mean token cosine / MSE inside each region, then mean over ROIs that have >= 1 such cell.", "",
          "| level | bin | region | n ROI | mean share of M cells | Bilinear cos | Bicubic cos | Bilinear MSE | Bicubic MSE |", "|---|---|---|---|---|---|---|---|---|"]
    OUT["mask_regions"] = {}
    valid = cov > 0
    for lv in ("object", "part"):
        for nm, lo, hi in (("all", 1, 10 ** 9),) + DBINS:
            sel = (dva.level.values == lv) & (dva.n_in.values >= lo) & (dva.n_in.values <= hi)
            for rg, rm in regions.items():
                m = rm & valid & sel[:, None]
                w = np.where(m, cov, 0)
                has = w.sum(1) > 0
                share = (m.sum(1) / np.maximum(valid.sum(1), 1))[sel].mean()
                vals = {}
                for c in ("bil", "bic"):
                    for q, key in (("cos", f"cos_{c}"), ("mse", f"se_{c}")):
                        x = (w * tm[key].astype(np.float32)).sum(1) / np.maximum(w.sum(1), 1e-9)
                        vals[(c, q)] = float(x[has].mean()) if has.any() else float("nan")
                OUT["mask_regions"][f"{lv}/{nm}/{rg}"] = dict(n=int(has.sum()), **{f"{c}_{q}": v for (c, q), v in vals.items()})
                L.append(f"| {lv} | {nm} | {rg} | {int(has.sum())} | {share:.2f} | {vals[('bil', 'cos')]:.3f} | {vals[('bic', 'cos')]:.3f} | "
                         f"{vals[('bil', 'mse')]:.3f} | {vals[('bic', 'mse')]:.3f} |")
    L.append("")
    # paired within-ROI interior vs background
    w_i = np.where(regions["interior"] & valid, cov, 0)
    w_b = np.where(regions["background"] & valid, cov, 0)
    both = (w_i.sum(1) > 0) & (w_b.sum(1) > 0)
    for c in ("bil", "bic"):
        ci_ = (w_i * tm[f"cos_{c}"]).sum(1) / np.maximum(w_i.sum(1), 1e-9)
        cb_ = (w_b * tm[f"cos_{c}"]).sum(1) / np.maximum(w_b.sum(1), 1e-9)
        d = mean_ci((ci_ - cb_).astype(np.float64), both)
        OUT["mask_regions"][f"{c}_interior_minus_background_cos"] = d
        L.append(f"- {LAB[c]}: within-ROI token cosine interior - background = {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] over {d['n']} ROIs with both regions")
    L.append("")

    # ============================================================ D. prediction-linked
    FEATS = [("1 - tok_cos", lambda c: 1 - met[f"{c}_tok_cos"].values), ("tok_mse", lambda c: met[f"{c}_tok_mse"].values),
             ("1 - pool_cos", lambda c: 1 - met[f"{c}_pool_cos"].values), ("E_mean", lambda c: met[f"{c}_e_mean"].values),
             ("E_cent", lambda c: met[f"{c}_e_cent"].values), ("E_cent/MSE", lambda c: met[f"{c}_frac_cent"].values),
             ("abs dS self-sim", lambda c: met[f"{c}_ss_diff"].values), ("log var a/c", lambda c: met[f"{c}_log_sv_ratio"].values),
             ("erank a/c", lambda c: met[f"{c}_erank_ratio"].values), ("log high-band a/c", lambda c: np.log(met[f"{c}_high_ratio"].values)),
             ("Δhigh share (a-c)", lambda c: met[f"{c}_fa_high"].values - met["c0_fc_high"].values)]
    OUT["groups"], OUT["corr"] = {}, {}
    L += ["## D. Prediction-linked analysis (Bicubic vs C0 on the same ROI-attribute)", ""]
    for rd in READERS:
        if ("bic", rd) not in P or ("c0", rd) not in P:
            continue
        tau = {c: f1_thresholds(P[(c, rd)]["Sd"], ddev, Ydev, units) for c in ("bic", "c0", "bil")}
        ok_unit = np.zeros((2, 59), bool)
        for lv, at in units:
            ok_unit[int(lv == "part"), at] = True
        app = (Yva >= 0) & ok_unit[lvi] & np.isfinite(tau["bic"][lvi]) & np.isfinite(tau["c0"][lvi])
        cor = {c: (P[(c, rd)]["Sv"] >= tau[c][lvi]) == (Yva == 1) for c in ("bic", "c0")}
        grp = np.zeros(Yva.shape, np.int8)                              # 0 = not applicable
        grp[app & ~cor["bic"] & cor["c0"]] = 1
        grp[app & cor["bic"] & cor["c0"]] = 2
        grp[app & ~cor["bic"] & ~cor["c0"]] = 3
        grp[app & cor["bic"] & ~cor["c0"]] = 4
        np.savez_compressed(os.path.join(R, f"groups_{rd}.npz"), grp=grp, tau_bic=tau["bic"], tau_c0=tau["c0"], ann_id=dva.ann_id.values)
        n_excl_units = int(sum(1 for lv, at in units if not (np.isfinite(tau["bic"][int(lv == 'part'), at]) and np.isfinite(tau["c0"][int(lv == 'part'), at]))))
        L += [f"### Reader {rd.upper()}", "",
              f"Thresholds: F1-max on PACO dev per (level, attribute) for each condition; {len(units)} M0 units, {n_excl_units} excluded (< 5 dev positives or negatives). "
              f"Applicable ROI-attribute pairs: {int(app.sum()):,} ({int((app & (Yva == 1)).sum()):,} positive / {int((app & (Yva == 0)).sum()):,} negative).", "",
              "| polarity | G1 Bic wrong / C0 right | G2 both right | G3 both wrong | G4 Bic right / C0 wrong | net (G1 - G4) |", "|---|---|---|---|---|---|"]
        for pol, pm in (("positive", Yva == 1), ("negative", Yva == 0)):
            cnt = [int(((grp == g) & pm).sum()) for g in (1, 2, 3, 4)]
            tot = max(sum(cnt), 1)
            L.append(f"| {pol} | " + " | ".join(f"{c_:,} ({100 * c_ / tot:.1f}%)" for c_ in cnt) + f" | {cnt[0] - cnt[3]:+,} |")
        L.append("")
        # group means of ROI-level features (pair-weighted), G1 - G2 raw and size/level-stratified, image bootstrap CI
        strata = np.array([f"{l}|{b}" for l, b in zip(dva.level.values, dbin)])
        OUT["groups"][rd] = {}
        for pol, pm in (("positive", Yva == 1), ("negative", Yva == 0)):
            n_g = {g: ((grp == g) & pm).sum(1).astype(np.float64) for g in (1, 2, 3, 4)}        # pairs per ROI
            L += [f"**{rd.upper()} {pol} labels: pair-weighted mean of the ROI's Bicubic-vs-C0 feature metric per group; "
                  f"G1 - G2 raw and stratified by level x D-bin (weights = G1 pairs per stratum), standardized by the ROI-level SD; image bootstrap 95% CI**", "",
                  "| metric | G1 | G2 | G3 | G4 | G1 - G2 raw (SD units) | G1 - G2 stratified (SD units) |", "|---|---|---|---|---|---|---|"]
            rows_ = [("n_in (ROI size)", dva.n_in.values.astype(np.float64))] + [(nm, fn("bic").astype(np.float64)) for nm, fn in FEATS]
            for nm, x in rows_:
                fin = np.isfinite(x)
                sd = np.nanstd(x)
                gm = {g: float((n_g[g] * np.where(fin, x, 0)).sum() / max((n_g[g] * fin).sum(), 1)) for g in (1, 2, 3, 4)}

                def g12(Wt, strat):
                    num1, den1 = Wt @ (n_g[1] * np.where(fin, x, 0)), Wt @ (n_g[1] * fin)
                    num2, den2 = Wt @ (n_g[2] * np.where(fin, x, 0)), Wt @ (n_g[2] * fin)
                    if not strat:
                        return (num1 / np.maximum(den1, 1e-9) - num2 / np.maximum(den2, 1e-9)) / sd
                    out, wsum = 0, 0
                    for s in np.unique(strata):
                        m = (strata == s) & fin
                        a1, b1 = Wt @ (n_g[1] * np.where(m, x, 0)), Wt @ (n_g[1] * m)
                        a2, b2 = Wt @ (n_g[2] * np.where(m, x, 0)), Wt @ (n_g[2] * m)
                        okk = (b1 > 0) & (b2 > 0)
                        out = out + np.where(okk, (a1 / np.maximum(b1, 1e-9) - a2 / np.maximum(b2, 1e-9)) * b1, 0)
                        wsum = wsum + np.where(okk, b1, 0)
                    return out / np.maximum(wsum, 1e-9) / sd
                one = np.ones((1, N), np.float32)
                res = {}
                for strat in (False, True):
                    pt = float(g12(one, strat)[0])
                    bs = g12(W, strat)
                    res[strat] = dict(point=pt, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
                OUT["groups"][rd][f"{pol}/{nm}"] = dict(means=gm, g1_g2_raw=res[False], g1_g2_strat=res[True])
                L.append(f"| {nm} | " + " | ".join(f"{gm[g]:.4f}" for g in (1, 2, 3, 4)) + f" | {fci(res[False], 3)} | {fci(res[True], 3)} |")
            L.append("")
        # -------------------------------- correlations (ROI level)
        rm = {c: rank_margin(P[(c, rd)]["Sv"], Yva, lvi, units) for c in ("bil", "bic", "c0")}
        okp = (Yva == 1) & ok_unit[lvi]
        okn = (Yva == 0) & ok_unit[lvi]
        _t = __import__("aggregate").T_OF
        T_OF = np.array([_t[i] for i in range(59)])

        def roi_mean(D, msk):
            n = msk.sum(1)
            return np.where(n > 0, np.where(msk, D, 0).sum(1) / np.maximum(n, 1), np.nan)
        d_pos = roi_mean(rm["c0"] - rm["bic"], okp)
        d_neg = roi_mean(rm["c0"] - rm["bic"], okn)
        lm = {c: (2 * (Yva == 1) - 1) * (P[(c, rd)]["Sv"] - np.nan_to_num(tau[c][lvi], nan=0.0)) for c in ("bic", "c0")}
        d_lpos = roi_mean(lm["c0"] - lm["bic"], okp & app)
        # scale check
        sdr = []
        for lv, at in units:
            k = int(lv == "part")
            m = (lvi == k) & (Yva[:, at] >= 0)
            sdr.append(np.std(P[("c0", rd)]["Sv"][m, at]) / max(np.std(P[("bic", rd)]["Sv"][m, at]), 1e-9))
        sdr = np.array(sdr)
        ctrl = np.column_stack([np.log(dva.n_in.values.astype(np.float64)), lvi.astype(np.float64)])
        L += [f"**{rd.upper()} correlations (one value per ROI). Score change = C0 - Bicubic rank margin of the ROI's positive labels (mean over its M0-unit positives); "
              f"logit SD ratio C0/Bicubic over units: median {np.median(sdr):.2f} (IQR {np.percentile(sdr, 25):.2f}-{np.percentile(sdr, 75):.2f}) -> raw logits not used; "
              f"Spearman rho [image-bootstrap 95% CI, {a.corr_boot} reps]; partial = rank n_in and level regressed out**", "",
              f"Mean per-ROI rank-margin change C0 - Bic (positives): {np.nanmean(d_pos):+.4f} over {int(np.isfinite(d_pos).sum())} ROIs; negatives {np.nanmean(d_neg):+.4f}.", "",
              "| feature (Bic vs C0) | all ROIs rho | partial rho (size, level) | object | part | D1-4 | D5-16 | D17+ | negatives: partial rho | logit-margin: partial rho |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        OUT["corr"][rd] = {}
        sub = {"object": dva.level.values == "object", "part": dva.level.values == "part", **{nm: dbin == nm for nm, _, _ in DBINS}}
        for nm, fn in FEATS:
            x = fn("bic").astype(np.float64)
            r_all = corr_ci(x, d_pos, img, nb=a.corr_boot)
            r_par = corr_ci(x, d_pos, img, ctrl, nb=a.corr_boot)
            r_sub = {k: corr_ci(np.where(m, x, np.nan), d_pos, img, ctrl, nb=a.corr_boot) for k, m in sub.items()}
            r_neg = corr_ci(x, d_neg, img, ctrl, nb=a.corr_boot)
            r_lm = corr_ci(x, d_lpos, img, ctrl, nb=a.corr_boot)
            OUT["corr"][rd][nm] = dict(all=r_all, partial=r_par, **{f"partial_{k}": v for k, v in r_sub.items()}, neg_partial=r_neg, logit_partial=r_lm)
            cc = lambda d: "n/a" if not np.isfinite(d["rho"]) else f"{d['rho']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
            L.append(f"| {nm} | {cc(r_all)} | {cc(r_par)} | " + " | ".join(cc(r_sub[k]) for k in ("object", "part", "D1-4", "D5-16", "D17+")) + f" | {cc(r_neg)} | {cc(r_lm)} |")
        L += ["", f"n ROIs (positives): all {int(np.isfinite(d_pos).sum())}; " + ", ".join(f"{k} {int((np.isfinite(d_pos) & m).sum())}" for k, m in sub.items()), ""]
        # by attribute type
        L += [f"**{rd.upper()} by attribute type: partial rho (size, level) of the ROI's per-type positive rank-margin change**", "",
              "| feature | color | material | pattern |", "|---|---|---|---|"]
        for nm, fn in FEATS:
            x = fn("bic").astype(np.float64)
            cells = []
            for t in ("color", "material", "pattern_marking"):
                dt = roi_mean(rm["c0"] - rm["bic"], okp & (T_OF[None, :] == t))
                r_ = corr_ci(x, dt, img, ctrl, nb=a.corr_boot)
                OUT["corr"][rd][f"{nm}/{t}"] = r_
                cells.append("n/a" if not np.isfinite(r_["rho"]) else f"{r_['rho']:+.3f} [{r_['lo']:+.3f}, {r_['hi']:+.3f}] (n {r_['n']})")
            L.append(f"| {nm} | " + " | ".join(cells) + " |")
        L.append("")
        # Bicubic - Bilinear: does the per-ROI feature gain go with a per-ROI score gain?
        d_bb = roi_mean(rm["bic"] - rm["bil"], okp)
        fx = (met["bic_tok_cos"] - met["bil_tok_cos"]).values
        r_bb = corr_ci(fx, d_bb, img, ctrl, nb=a.corr_boot)
        OUT["corr"][rd]["bic_minus_bil"] = dict(mean_score_change=float(np.nanmean(d_bb)), partial=r_bb)
        L += [f"Bicubic - Bilinear ({rd.upper()}): mean per-ROI positive rank-margin change {np.nanmean(d_bb):+.4f}; partial rho(Δ token cosine, Δ rank margin) = "
              f"{r_bb['rho']:+.3f} [{r_bb['lo']:+.3f}, {r_bb['hi']:+.3f}] (n {r_bb['n']}).", ""]
        if rd == "r2":
            OUT["_fig"] = dict(d_pos=d_pos, grp=grp)
    open(os.path.join(R, "tables.md"), "w").write("\n".join(L) + "\n")
    figs(R, FIG, dva, met, hist, edges, OUT.pop("_fig", None), FEATS)
    json.dump(OUT, open(os.path.join(R, "analysis.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n".join(L))


def figs(R, FIG, dva, met, hist, edges, fg, FEATS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"bil": "#8c8c8c", "bic": "#e07b00", "object": "#1f77b4", "part": "#d62728"}
    n = dva.n_in.values
    # 10. ROI size vs feature difference
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    for k, (key, nm) in enumerate((("tok_cos", "token cosine to C0"), ("e_mean", "E_mean (pooled part)"), ("e_cent", "E_cent (centred spatial part)"), ("ss_diff", "|S_a - S_c| self-similarity"))):
        for lv in ("object", "part"):
            m = dva.level.values == lv
            ax[k].scatter(n[m], met.loc[m, f"bic_{key}"], s=2, alpha=0.25, c=col[lv], label=f"Bicubic, {lv}")
        ub = np.unique(np.round(np.geomspace(1, n.max(), 16)).astype(int))
        for c in ("bil", "bic"):
            med = [np.nanmedian(met.loc[(n >= lo) & (n < hi), f"{c}_{key}"]) if ((n >= lo) & (n < hi)).any() else np.nan for lo, hi in zip(ub[:-1], ub[1:])]
            ax[k].plot(np.sqrt(ub[:-1] * ub[1:]), med, c="k" if c == "bic" else "grey", lw=2, ls="-" if c == "bic" else "--", label=f"{'Bicubic' if c == 'bic' else 'Bilinear'} median")
        ax[k].set_xscale("log")
        ax[k].set_xlabel("ROI tokens n_in (F336 grid, log)")
        ax[k].set_title(nm)
        for x in (4.5, 16.5):
            ax[k].axvline(x, c="k", lw=0.5, ls=":")
    ax[0].legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_size_vs_featdiff.png"), dpi=110)
    plt.close(fig)
    # token cosine distributions
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    cen = (edges[1:] + edges[:-1]) / 2
    for k, nm in enumerate(("all", "D1-4", "D5-16", "D17+")):
        for c in ("bil", "bic"):
            ax[k].plot(cen, hist[(c, nm)][0], c=col[c], label="Bilinear" if c == "bil" else "Bicubic")
        ax[k].set_title(f"token cosine to C0, {nm} (coverage-weighted)")
        ax[k].set_xlabel("cosine")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_tokcos_dist.png"), dpi=110)
    plt.close(fig)
    # decomposition by bin
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    labs, em, ec = [], [], []
    for lv in ("object", "part"):
        for nm, lo, hi in DBINS:
            sel = (dva.level.values == lv) & (n >= lo) & (n <= hi)
            for c in ("bil", "bic"):
                labs.append(f"{lv[:3]} {nm}\n{'bil' if c == 'bil' else 'bic'}")
                em.append(np.nanmean(met.loc[sel, f"{c}_e_mean"]))
                ec.append(np.nanmean(met.loc[sel, f"{c}_e_cent"]))
    x = np.arange(len(labs))
    ax[0].bar(x, em, label="E_mean (pooled)")
    ax[0].bar(x, ec, bottom=em, label="E_cent (centred spatial)")
    ax[0].set_xticks(x, labs, fontsize=7)
    ax[0].set_ylabel("token MSE to C0 (per channel)")
    ax[0].legend()
    ax[0].set_title("MSE = E_mean + E_cent (exact), per-ROI means")
    for c in ("bil", "bic"):
        ax[1].hist(met[f"{c}_frac_cent"], bins=50, range=(0, 1), histtype="step", lw=2, color=col[c], label="Bilinear" if c == "bil" else "Bicubic")
    ax[1].set_title("share of MSE that is centred spatial difference (per ROI)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_decomposition.png"), dpi=110)
    plt.close(fig)
    # DCT bands
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    bands = [b for b, _, _ in BANDS]
    for k, (nm, lo, hi) in enumerate(DBINS):
        sel = (n >= lo) & (n <= hi)
        xx = np.arange(3)
        ax[k].bar(xx - 0.27, [np.nanmean(met.loc[sel, f"bil_dct_a_{b}"]) for b in bands], 0.27, color=col["bil"], label="Bilinear")
        ax[k].bar(xx, [np.nanmean(met.loc[sel, f"bic_dct_a_{b}"]) for b in bands], 0.27, color=col["bic"], label="Bicubic")
        ax[k].bar(xx + 0.27, [np.nanmean(met.loc[sel, f"bic_dct_c_{b}"]) for b in bands], 0.27, color="#1f77b4", label="C0")
        ax[k].set_xticks(xx, bands)
        ax[k].set_yscale("log")
        ax[k].set_title(f"DCT energy per band (M1, DC removed), {nm}")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dct_bands.png"), dpi=110)
    plt.close(fig)
    # 11. feature difference vs score difference (R2)
    if fg is not None:
        d = fg["d_pos"]
        fig, ax = plt.subplots(1, 4, figsize=(22, 5))
        for k, (nm, fn) in enumerate([FEATS[i] for i in (0, 3, 4, 6)]):
            xv = fn("bic")
            for lv in ("object", "part"):
                m = (dva.level.values == lv) & np.isfinite(d)
                ax[k].scatter(xv[m], d[m], s=3, alpha=0.3, c=col[lv], label=lv)
            ok = np.isfinite(d) & np.isfinite(xv)
            qs = np.nanpercentile(xv[ok], np.linspace(0, 100, 11))
            mids, means = [], []
            for lo, hi in zip(qs[:-1], qs[1:]):
                m = ok & (xv >= lo) & (xv <= hi)
                mids.append(np.median(xv[m]))
                means.append(np.mean(d[m]))
            ax[k].plot(mids, means, "k-o", lw=2, label="decile mean")
            ax[k].axhline(0, c="k", lw=0.5)
            ax[k].set_xlabel(f"Bicubic vs C0: {nm}")
            ax[k].set_ylabel("C0 - Bicubic rank margin, positives (R2)")
        ax[0].legend(markerscale=4)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig11_featdiff_vs_scorediff_r2.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
