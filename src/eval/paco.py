"""Evaluation (CPU) for the RAEv2 CoordMLP A/B/C pilot. Copy of 2026-11-03/visual_recon_loss/code/eval_vr.py (metric code =
paco_feat_upsample eval_fu -> paco_region_reader eval_rr: M2 official-style category-wise AP, test/val-primary, paired image
bootstrap 1,000 seed 0). Changed: condition list / labels / comparisons (all conditions are RAE-latent probes of this project),
no checkpoint selection (A/B/C fixed at 12k; 4k dev-only), plus auxiliary RAE-grid n_in bins (R1-4 / R5-16 / R17+).
Primary size bins = the existing F336 bins (D1-4 part, D5-16 object, D17+); SigLIP2 bins secondary.

  python -m src.eval.paco dev A_4000 B_4000 C_4000 A_12000 B_12000 C_12000   # val-primary / full-val point estimates
  python -m src.eval.paco final                                              # selection (best-val epoch per run) -> boot -> tables
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from src.config import CACHE, FEAT_UPSAMPLE_CODE, PROBE, RES  # noqa: E402
sys.path.insert(0, FEAT_UPSAMPLE_CODE)   # earlier project (not on this server)
import eval_fu as EF  # noqa: E402

ER, G, ET = EF.ER, EF.G, EF.ER.ET
ER.CACHE, ER.RES, ER.FIG = CACHE, RES, os.path.join(RES, "figures")
EF.CACHE, EF.RES = CACHE, RES
FINAL = {"decA": "A_12000", "decB": "B_12000", "decC": "C_12000"}
BASE = {c: os.path.join(PROBE, c) for c in ("full", "bil", "crop")}
LAB = {"full": "Full (RAE 16x16, GT-ROI cells, direct read-out)", "bil": "Bilinear (RAE full map -> crop grid)", "crop": "GT crop oracle (RAE)",
       "decA": "A Feature-only (12k)", "decB": "B Pixel-only (12k)", "decC": "C Feature+Pixel (12k)"}
PAT = {"r2": "r2_lr0.0003_s*", "r0": "r0_lr0.0001_s*"}
GROUPS = ("overall", "object", "part", "color", "material", "pattern_marking")
BINS_D = (("part", "D1-4"), ("object", "D5-16"), ("object", "D17+"))
BINS_S = (("part", "1-4"), ("object", "5-16"), ("object", "17+"))
BINS_R = (("part", "R1-4"), ("object", "R5-16"), ("object", "R17+"))
RBINS = (("R1-4", 1, 4), ("R5-16", 5, 16), ("R17+", 17, 10 ** 9))


def bin_mask(df, name, split):
    nd = df.n_in.values
    nin = np.load(EF.NIN_SG.format(split))
    for nm, lo, hi in EF.DBINS:
        if nm == name:
            return (nd >= lo) & (nd <= hi)
    for nm, lo, hi in EF.BINS:
        if nm == name:
            return (nin >= lo) & (nin <= hi)
    nr = rae_nin(split)
    for nm, lo, hi in RBINS:
        if nm == name:
            return (nr >= lo) & (nr <= hi)
    raise KeyError(name)


def rae_nin(split):
    """n_in of every manifest row on the RAE 16x16 grid (rae_common.roi_geometry on the unflipped box), cached."""
    f = os.path.join(CACHE, "eval", f"nin_rae_{split}.npy")
    if not os.path.exists(f):
        import src.rae.common as RC
        df = G["df"][split]
        v = np.array([RC.roi_geometry(int(W), int(H), (int(l), int(t), int(r), int(b)))["n_in"] for W, H, l, t, r, b in
                      zip(df.W, df.H, df.l, df.t, df.r, df.b)])
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.save(f, v)
    return np.load(f)


_prep0 = EF.prep_units


def prep_units(nb=1000):
    """eval_fu.prep_units + auxiliary RAE-grid bins."""
    _prep0(nb)
    for sub in ("testp", "valp"):
        split, primary = EF.SUBSETS[sub]
        df, Y = G["df"][split], G["Y"][split]
        mask = ~df.in_decoder_pool.values if primary else np.ones(len(df), bool)
        nr = rae_nin(split)
        for name, lo, hi in RBINS:
            G["units"][sub][name] = ER.build_units(df, Y, G["Ytr_by"], mask & (nr >= lo) & (nr <= hi))


ER.prep_units = prep_units


def point_metrics(score_list, split, primary=True):
    """M2 groups overall + per bin (val/test-primary), unit APs averaged over the given score arrays (= seeds)."""
    df, Y = G["df"][split], G["Y"][split]
    base = ~df.in_decoder_pool.values if primary else np.ones(len(df), bool)
    out = {}
    r = ET.evaluate(score_list, df, Y, base, G["Ytr_by"])
    for g in GROUPS:
        out[g] = r["M2"][g]
    out["M0_overall"] = r["M0"]["overall"]
    for g, b in BINS_D + BINS_S + BINS_R:
        try:
            rb = ET.evaluate(score_list, df, Y, base & bin_mask(df, b, split), G["Ytr_by"])
            out[f"{b} {g}"] = rb["M2"][g]
        except ValueError:                      # no evaluable unit in this bin / subset
            out[f"{b} {g}"] = float("nan")
    return out


def run_dirs(root, kind, seeds=None):
    ds = sorted(glob.glob(os.path.join(root, PAT[kind])))
    if seeds is not None:
        ds = [d for d in ds if int(d.rsplit("_s", 1)[1]) in seeds]
    return ds


def dev(tags):
    rows = []
    for kind in ("r2", "r0"):
        for nm, root in list(BASE.items()) + [(t, os.path.join(PROBE, t)) for t in tags]:
            for seeds, slab in (((0,), "s0"), ((0, 1, 2), "s012")):
                ds = run_dirs(root, kind, seeds)
                if len(ds) != len(seeds) or (nm in tags and slab == "s012"):
                    continue
                eps = [ER.best_of(d) for d in ds]
                S = [np.load(os.path.join(d, f"scores_val_ep{e[0]}.npy")).astype(np.float32) for d, e in zip(ds, eps)]
                m = point_metrics(S, "val")
                mf = point_metrics(S, "val", primary=False)
                m.update({f"fullval {k}": v for k, v in mf.items()})
                rows.append(dict(cond=nm, reader=kind, seeds=slab, best_epochs="/".join(str(e[0]) for e in eps),
                                 val_pooled=float(np.mean([e[1] for e in eps])), **m))
                print(nm, kind, slab, round(m["overall"], 2), round(m["D5-16 object"], 2), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "dev_valp_M2.csv"), index=False)
    cols = ["overall", "object", "part", "D5-16 object", "color", "material", "pattern_marking", "fullval overall", "fullval object", "fullval part",
            "fullval D5-16 object", "fullval D17+ object", "fullval D1-4 part", "fullval color", "fullval material", "fullval pattern_marking"]
    L = ["# Dev M2 (val-primary = primary rule; fullval = all 11,109 val ROIs, secondary, less noisy), point estimates (auto: code/eval_vr.py dev)\n", "| cond | reader | seeds | best ep | val pooled | " + " | ".join(cols) + " |", "|---" * (len(cols) + 5) + "|"]
    for _, r in df.iterrows():
        L.append(f"| {r.cond} | {r.reader} | {r.seeds} | {r.best_epochs} | {r.val_pooled:.2f} | " + " | ".join(f"{r[c]:.2f}" for c in cols) + " |")
    open(os.path.join(RES, "dev_valp_M2.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# --------------------------------------------------------------------------------------------- final (after selection)
def select():
    src = dict(BASE, **{k: os.path.join(PROBE, v) for k, v in FINAL.items()})
    sel = {"runs": {}, "conds": {}}
    for kind in ("r2", "r0"):
        for cond, root in src.items():
            for d in run_dirs(root, kind):
                if cond.startswith("dec") and not glob.glob(os.path.join(d, "scores_test_ep*.npy")):
                    print("SKIP (no test scores)", d)
                    continue
                nm = os.path.basename(d)
                ep, v, n = ER.best_of(d)
                key = f"{cond}/{nm}"
                sel["runs"][key] = dict(cond=cond, kind=kind, seed=int(nm.rsplit("_s", 1)[1]), best_epoch=ep, val=v, epochs_logged=n, dir=d)
                sel["conds"].setdefault(f"{cond}_{kind}", []).append(key)
    json.dump(sel, open(os.path.join(RES, "selection.json"), "w"), indent=1)
    print({k: [(r.split("/")[1], sel["runs"][r]["best_epoch"], round(sel["runs"][r]["val"], 2)) for r in v] for k, v in sel["conds"].items()})


def comparisons():
    c = []
    for k in ("r2", "r0"):
        c += [(f"C - A: pixel added to feature ({k})", f"decC_{k}", f"decA_{k}"), (f"B - A: pixel-only vs feature-only ({k})", f"decB_{k}", f"decA_{k}"),
              (f"C - B ({k})", f"decC_{k}", f"decB_{k}")]
        for d in ("decA", "decB", "decC"):
            c += [(f"{d[-1]} - Full ({k})", f"{d}_{k}", f"full_{k}"), (f"{d[-1]} - Bilinear ({k})", f"{d}_{k}", f"bil_{k}"), (f"{d[-1]} - Crop ({k})", f"{d}_{k}", f"crop_{k}")]
        c += [(f"Crop - Full ({k})", f"crop_{k}", f"full_{k}"), (f"Bilinear - Full ({k})", f"bil_{k}", f"full_{k}"), (f"Crop - Bilinear ({k})", f"crop_{k}", f"bil_{k}")]
    return [x for x in c if x[1] in G["sel"]["conds"] and x[2] in G["sel"]["conds"]]


ER.comparisons, ER.all_conds = comparisons, lambda: list(G["sel"]["conds"])


def tables():
    sel = G["sel"]
    Rb = pd.read_csv(os.path.join(RES, "ap_by_bin.csv"))
    Dd = pd.read_csv(os.path.join(RES, "paired_diffs.csv"))
    P = pd.read_csv(os.path.join(RES, "per_seed_testp_M2.csv"))

    def val(c, b, g, met="M2", sub="testp"):
        q = Rb[(Rb.subset == sub) & (Rb.bin == b) & (Rb.metric == met) & (Rb.group == g) & (Rb.cond == c)]
        return float(q.value.iloc[0]) if len(q) else np.nan

    def dif(name, b, g, met="M2", sub="testp"):
        q = Dd[(Dd.subset == sub) & (Dd.bin == b) & (Dd.metric == met) & (Dd.group == g) & (Dd.comparison == name)]
        return ER.fmt(q.value.iloc[0], q.ci_lo.iloc[0], q.ci_hi.iloc[0], q.sig.iloc[0]) if len(q) else "–"

    cols = (("overall", "all"), ("object", "all"), ("part", "all"), ("object", "D5-16"), ("object", "D17+"), ("part", "D1-4"),
            ("color", "all"), ("material", "all"), ("pattern_marking", "all"), ("object", "5-16"), ("object", "R5-16"), ("part", "R1-4"))
    hdr = "| overall | object | part | Obj D5-16 | Obj D17+ | Part D1-4 | color | material | pattern | Obj 5-16 (SigLIP2 bin) | Obj R5-16 (RAE grid) | Part R1-4 (RAE grid) |"
    order = ("full", "bil", "crop", "decA", "decB", "decC")
    rows = []
    L = ["# RAEv2 CoordMLP A/B/C pilot — final PACO tables (auto: src/eval/paco.py final)\n"]
    for sub, title in (("testp", "test-primary"), ("valp", "val-primary")):
        L += [f"\n## M2 {title}, reader seeds averaged at the unit-AP level (best-val epoch per run)\n", "| condition | reader | best ep " + hdr, "|---" * (len(cols) + 3) + "|"]
        for k in ("r2", "r0"):
            for c in order:
                ck = f"{c}_{k}"
                if ck in sel["conds"]:
                    eps = "/".join(str(sel["runs"][r]["best_epoch"]) for r in sel["conds"][ck])
                    vals = [val(ck, b, g, sub=sub) for g, b in cols]
                    L.append(f"| {LAB[c]} | {k.upper()} | {eps} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
                    rows.append(dict(subset=sub, cond=c, reader=k, n_seeds=len(sel["conds"][ck]), best_epochs=eps, **{f"{g}|{b}": v for (g, b), v in zip(cols, vals)}))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "main_table.csv"), index=False)
    L += ["\n## Paired differences, M2 test-primary (95% paired image-bootstrap CI, * = excludes 0)\n", "| comparison " + hdr, "|---" * (len(cols) + 1) + "|"]
    for name, _, _ in comparisons():
        L.append(f"| {name} | " + " | ".join(dif(name, b, g) for g, b in cols) + " |")
    L += ["\n## Paired differences, M2 val-primary\n", "| comparison | overall | Obj D5-16 |", "|---|---|---|"]
    for name, _, _ in comparisons():
        L.append(f"| {name} | {dif(name, 'all', 'overall', sub='valp')} | {dif(name, 'D5-16', 'object', sub='valp')} |")
    L += ["\n## M0 test-primary paired differences\n", "| comparison | overall | Obj D5-16 |", "|---|---|---|"]
    for name, _, _ in comparisons():
        L.append(f"| {name} | {dif(name, 'all', 'overall', 'M0')} | {dif(name, 'D5-16', 'object', 'M0')} |")
    L += ["\n## Per-seed test-primary M2 (overall / SigLIP2 5-16 object)\n", "| condition | reader | per seed | mean of seeds |", "|---|---|---|---|"]
    for k in ("r2", "r0"):
        for c in order:
            ck = f"{c}_{k}"
            if ck not in sel["conds"]:
                continue
            cells, a = [], []
            for r, ep in ER.cond_runs(ck):
                p = P[(P.run == r) & (P.bin == "all")].overall.iloc[0]
                p5 = P[(P.run == r) & (P.bin == "5-16")].object.iloc[0]   # per-seed table keeps eval_rr's SigLIP2 bins
                cells.append(f"{p:.2f}/{p5:.2f}")
                a.append(p)
            L.append(f"| {LAB[c]} | {k.upper()} | " + ", ".join(cells) + f" | {np.mean(a):.2f} |")
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    G["df"], G["Y"] = {}, {}
    for sp in ("train", "val", "test"):
        G["df"][sp], G["Y"][sp] = ER.load_split(sp)
    G["Ytr_by"] = {lv: G["Y"]["train"][(G["df"]["train"].level == lv).values] for lv in ER.LEVELS}
    os.makedirs(RES, exist_ok=True)
    if sys.argv[1] == "dev":
        dev(sys.argv[2:])
    elif sys.argv[1] == "final":
        select()
        G["sel"] = json.load(open(os.path.join(RES, "selection.json")))
        ER.boot()
        tables()
    elif sys.argv[1] == "tables":
        G["sel"] = json.load(open(os.path.join(RES, "selection.json")))
        tables()
