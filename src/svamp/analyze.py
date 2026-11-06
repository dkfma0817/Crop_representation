"""Spatial-variance pilot analysis (CPU) -> <run>/paco_results.csv, bootstrap_results.csv, tables.md.

Probes: V0 = IVC probe_bic (same Bicubic tensor, same src.ivc.train protocol; R0 + R2), V1-V3 / O2 = this run (R2 only),
C0 Full Oracle = IVC probe_c0 (R0 + R2); references R00 / E3 / E6 = xdec probes (R2 + R0).
R0 for V1-V3 / O2: the R0 input (coverage-weighted ROI mean) is identical to V0's (checked to <= 4e-6), so V0's R0 applies.
Epoch = max PACO-dev M0 (1..10). Paired image bootstrap B = 1000, seed 0, same draws for all (src.readout.evaluate.Boot).

  python -m src.svamp.analyze --run output/2026-11-05/svamp
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import src.readout.common as RC
from src.ivc.analyze import dev_split, f2, fci, load_probe
from src.readout.evaluate import Boot, summarize

EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
COLS = [("Overall", "all", "all", "overall"), ("Obj D5-16", "D5-16", "all", "object"), ("Obj D1-4", "D1-4", "all", "object"),
        ("Obj D17+", "D17+", "all", "object"), ("Part D1-4", "D1-4", "all", "part"), ("Part D5-16", "D5-16", "all", "part"),
        ("Object", "all", "all", "object"), ("Part", "all", "all", "part"), ("Color", "all", "all", "color"),
        ("Material", "all", "all", "material"), ("Pattern", "all", "all", "pattern_marking")]
DB = (("D1-4", 1, 4), ("D5-16", 5, 16), ("D17+", 17, 10 ** 9))
CONTRASTS = [("V1", "V0"), ("V2", "V0"), ("V3", "V0"), ("O2", "V0"), ("O2", "V2"), ("V2", "V1"), ("V3", "V2"), ("C0 Full", "O2")]
LAB = {"V0": "V0 Bicubic", "V1": "V1 x1.25", "V2": "V2 x1.5", "V3": "V3 x2.0", "O2": "O2 C0 Spatial Oracle", "C0 Full": "C0 Full Oracle",
       "R00": "R00 (xdec)", "E3": "E3 (xdec)", "E6": "E6 (xdec)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = a.run
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    Ytr_by = RC.ytr_by_level(data)
    ddev, _ = dev_split(data)
    BB = {"all": Boot(dva, Yva, Ytr_by)}
    for nm, lo, hi in DB:
        BB[nm] = Boot(dva, Yva, Ytr_by, mask=(dva.n_in.values >= lo) & (dva.n_in.values <= hi))
    SRC = {"V0": ("output/2026-11-05/ivc", "bic", ("r2", "r0")), "V1": (os.path.join(R, "results"), "V1", ("r2",)),
           "V2": (os.path.join(R, "results"), "V2", ("r2",)), "V3": (os.path.join(R, "results"), "V3", ("r2",)),
           "O2": (os.path.join(R, "results"), "O2", ("r2",)), "C0 Full": ("output/2026-11-05/ivc", "c0", ("r2", "r0")),
           "R00": ("output/2026-11-05/xdec/results", "R00", ("r2", "r0")), "E3": ("output/2026-11-05/xdec/results", "E3", ("r2", "r0")),
           "E6": ("output/2026-11-05/xdec/results", "E6", ("r2", "r0"))}
    P, RES, rows = {}, {}, []
    for m, (root, name, rds) in SRC.items():
        for rd in rds:
            p = load_probe(root, name, rd)
            if p is None:
                continue
            assert p["Sv"].shape == (len(dva), 59)
            P[(m, rd)] = p
            RES[(m, rd)] = {met: {b: getattr(bb, met)(p["Sv"]) for b, bb in BB.items()} for met in ("m0", "m2")}
            for met in ("m0", "m2"):
                for cn, b, _, g in COLS:
                    v = RES[(m, rd)][met][b]
                    rows.append(dict(method=LAB[m], reader=rd.upper(), metric=met.upper(), column=cn, epoch=p["epoch"], dev_M0=p["dev_M0"],
                                     value=(float(v[g][0]) if v is not None and g in v else np.nan),
                                     source=f"{root}/probe_{name}", note=("R0 input identical to V0 (reused)" if False else "")))
    for m in ("V1", "V2", "V3", "O2"):           # R0 = V0's (identical input)
        if ("V0", "r0") in P:
            for r in [r for r in rows if r["method"] == LAB["V0"] and r["reader"] == "R0"]:
                rows.append({**r, "method": LAB[m], "note": "R0 input identical to V0 by construction (max diff <= 4e-6): V0 R0 reused"})
    pd.DataFrame(rows).to_csv(os.path.join(R, "paco_results.csv"), index=False)

    def val(m, rd, met, b, g):
        r = RES.get((m, rd))
        if r is None or r[met][b] is None or g not in r[met][b]:
            return None
        return r[met][b][g]

    brows = []
    for x, y in CONTRASTS:
        for rd in ("r2",):
            for met in ("m0", "m2"):
                for cn, b, _, g in COLS:
                    vx, vy = val(x, rd, met, b, g), val(y, rd, met, b, g)
                    if vx is None or vy is None:
                        continue
                    s = summarize(vx, vy)
                    brows.append(dict(contrast=f"{x} - {y}", reader="R2", metric=met.upper(), column=cn, **s))
    # trend V0 -> V3 at the draw level: share of bootstrap draws where V2 > V0 etc. are in the CIs above; add monotone check
    pd.DataFrame(brows).to_csv(os.path.join(R, "bootstrap_results.csv"), index=False)

    st = pd.read_csv(os.path.join(R, "feature_statistics.csv")) if os.path.exists(os.path.join(R, "feature_statistics.csv")) else None
    chk = json.load(open(os.path.join(R, "transform_checks.json"))) if os.path.exists(os.path.join(R, "transform_checks.json")) else {}
    xrec = None
    p = "output/2026-11-05/xdec/results/recon/paco_val_roi_metrics.parquet"
    if os.path.exists(p):
        xrec = pd.read_parquet(p).groupby("cond")[["sv_ratio", "tok_cos", "pool_cos", "e_cent"]].mean()

    def pv(m, rd="r2", met="m0", b="all", g="overall"):
        v = val(m, rd, met, b, g)
        return None if v is None else float(v[0])

    L = ["# Spatial Variance Amplification & C0 Spatial Oracle: tables (auto-generated)", "",
         "PACO **val** only (test never evaluated). Reader epoch = max PACO-dev M0. R2 primary. Paired image bootstrap B=1000, seed 0.", "",
         "## 1. Main table", "", "| Method | Spatial Var / C0 | Token Cos | Pooled Cos | R2 M0 | Obj D5–16 | R2 M2 | R2 epoch |", "|---|---|---|---|---|---|---|---|"]
    stc = {"V0": "V0", "V1": "V1", "V2": "V2", "V3": "V3", "O2": "O2", "C0 Full": "C0"}
    for m in ("V0", "V1", "V2", "V3", "O2", "C0 Full"):
        s = st[(st.cond == stc[m]) & (st.group == "all")].iloc[0] if st is not None else None
        L.append(f"| {LAB[m]} | {f2(s.sv_ratio, 3) if s is not None else 'n/a'} | {f2(s.tok_cos, 3) if s is not None else 'n/a'} | "
                 f"{f2(s.pool_cos, 3) if s is not None else 'n/a'} | {f2(pv(m)) if pv(m) is not None else 'n/a'} | "
                 f"{f2(pv(m, b='D5-16', g='object')) if pv(m) is not None else 'n/a'} | {f2(pv(m, met='m2')) if pv(m) is not None else 'n/a'} | "
                 f"{P[(m, 'r2')]['epoch'] if (m, 'r2') in P else 'n/a'} |")
    L += ["", "Reference (xdec decoders, same probe protocol; feature stats from xdec recon on PACO val):", "",
          "| Method | Spatial Var / C0 | Token Cos | Pooled Cos | R2 M0 | Obj D5–16 | R2 M2 | R0 M0 |", "|---|---|---|---|---|---|---|---|"]
    for m in ("R00", "E3", "E6"):
        x = xrec.loc[m] if xrec is not None else None
        L.append(f"| {LAB[m]} | {f2(x.sv_ratio, 3) if x is not None else 'n/a'} | {f2(x.tok_cos, 3) if x is not None else 'n/a'} | "
                 f"{f2(x.pool_cos, 3) if x is not None else 'n/a'} | {f2(pv(m))} | {f2(pv(m, b='D5-16', g='object'))} | {f2(pv(m, met='m2'))} | {f2(pv(m, rd='r0'))} |")
    L += ["", f"R0 (secondary): V0 = V1 = V2 = V3 = O2 input -> M0 {f2(pv('V0', rd='r0'))} (Obj D5–16 {f2(pv('V0', rd='r0', b='D5-16', g='object'))}); "
          f"C0 Full R0 {f2(pv('C0 Full', rd='r0'))}.", ""]
    for met in ("m0", "m2"):
        L += [f"## 2. {met.upper()} (R2), all columns", "", "| Method | " + " | ".join(c[0] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
        for m in SRC:
            if (m, "r2") in P:
                L.append(f"| {LAB[m]} | " + " | ".join(f2(v) if (v := pv(m, met=met, b=b, g=g)) is not None else "n/a" for _, b, _, g in COLS) + " |")
        L.append("")
    for met in ("m0", "m2"):
        L += [f"## 3. Paired bootstrap ({met.upper()}, R2): difference [95% CI]", "", "| Contrast | " + " | ".join(c[0] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
        for x, y in CONTRASTS:
            cells = []
            for _, b, _, g in COLS:
                vx, vy = val(x, "r2", met, b, g), val(y, "r2", met, b, g)
                cells.append("n/a" if vx is None or vy is None else fci(summarize(vx, vy)))
            L.append(f"| {x} − {y} | " + " | ".join(cells) + " |")
        L.append("")
    if st is not None:
        keys = ["tok_norm", "pool_norm", "tok_cos", "pool_cos", "sv_ratio", "e_cent", "e_mean", "cent_cos"]
        L += ["## 4. Feature statistics (PACO val, per-ROI coverage-weighted, mean over ROIs; vs C0 teacher)", "",
              "| Cond | group | " + " | ".join(keys) + " | sv ratio of means |", "|---|---|" + "---|" * (len(keys) + 1)]
        for _, r in st.iterrows():
            L.append(f"| {r.cond} | {r.group} | " + " | ".join(f"{r[k]:.4f}" for k in keys) + f" | {'' if pd.isna(r.sv_ratio_of_means_all) else f'{r.sv_ratio_of_means_all:.4f}'} |")
        L.append("")
    if chk:
        L += ["## 5. Transform checks (PACO val, all ROIs)", "", "```", json.dumps(chk, indent=1), "```", ""]
    open(os.path.join(R, "tables.md"), "w").write("\n".join(L) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
