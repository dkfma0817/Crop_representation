"""Tables for the fixed-readout substitution study -> <root>/results.md body pieces (printed) and <root>/results.json.

  python -m src.fixedro.report --root output/2026-11-12/fixed_readout_all
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["F768", "F384NN", "F384BIL", "S1", "S2", "S3L2", "S3KL", "T768M", "FAA", "FAB", "P1", "P2"]
LABEL = dict(F768="F768 (reader's own input)", F384NN="F384 nearest ×2", F384BIL="F384 bilinear ×2 (= B0)", S1="S1", S2="S2", S3L2="S3-L2", S3KL="S3-Distill",
             T768M="T768-MeanOnly", FAA="FA-A", FAB="FA-B", P1="P1", P2="P2")
# independently retrained R2 (official; epoch on Val, Test once). None = not evaluated in that protocol.
R2 = dict(F768=(56.40, 43.66), F384NN=(None, None), F384BIL=(54.25, 38.14), S1=(54.80, None), S2=(54.09, None), S3L2=(55.19, 39.05), S3KL=(55.65, 38.70),
          T768M=(56.07, 41.28), FAA=(54.55, 39.54), FAB=(54.82, 39.72), P1=(50.39, 34.04), P2=(54.68, 39.69))
R2_F384_24 = (54.21, 39.28)


def row(path):
    h = json.loads(Path(path).read_text())["hierarchical"]
    o, p = h["obj-attrs"], h["obj-part-attrs"]
    return dict(overall=(h["obj-attr-AP"] + h["obj-attr-part-AP"]) / 2, object=h["obj-attr-AP"], part=h["obj-attr-part-AP"],
                pattern=(o["pattern_marking"] + p["pattern_marking"]) / 2, material=(o["material"] + p["material"]) / 2, color=(o["color"] + p["color"]) / 2)


def boot(path, a, b):
    d = json.loads(Path(path).read_text())
    bt, pt = d["boot"], d["points"]
    out = {}
    for k in ("overall", "obj", "part"):
        if k == "overall":
            x = (np.array(bt[f"{a}_r2"]["obj"]) + np.array(bt[f"{a}_r2"]["part"])) / 2 - (np.array(bt[f"{b}_r2"]["obj"]) + np.array(bt[f"{b}_r2"]["part"])) / 2
            p = (pt[f"{a}_r2"]["obj"] + pt[f"{a}_r2"]["part"]) / 2 - (pt[f"{b}_r2"]["obj"] + pt[f"{b}_r2"]["part"]) / 2
        else:
            x = np.array(bt[f"{a}_r2"][k]) - np.array(bt[f"{b}_r2"][k])
            p = pt[f"{a}_r2"][k] - pt[f"{b}_r2"][k]
        lo, hi = np.nanpercentile(x, [2.5, 97.5])
        out[k] = (p, float(lo), float(hi))
    return out


def fmt(t):
    p, lo, hi = t
    return f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("*" if lo > 0 or hi < 0 else "")


def ranks(vals):
    s = pd.Series(vals).dropna()
    return s.rank(ascending=False, method="min").astype(int).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    res = {}
    for m in METHODS:
        res[m] = {sp: row(root / f"eval_boot/{m}_r2/{sp}/official_results.json") for sp in ("val", "test") if (root / f"eval_boot/{m}_r2/{sp}/official_results.json").exists()}
    md = []
    md += ["## Fixed-readout official AP (frozen F768 R2 reader, epoch 6; only the input changes)", "",
           "| Method | Val Overall | Val Obj / Part | Test Overall | Test Obj / Part | Test Pattern / Material / Color |", "|---|---|---|---|---|---|"]
    for m in METHODS:
        v, t = res[m].get("val"), res[m].get("test")
        md.append(f"| {LABEL[m]} | {v['overall']:.2f} | {v['object']:.2f} / {v['part']:.2f} | {t['overall']:.2f} | {t['object']:.2f} / {t['part']:.2f} | {t['pattern']:.2f} / {t['material']:.2f} / {t['color']:.2f} |" if v and t else f"| {LABEL[m]} | pending | | | | |")
    bt = {}
    for sp in ("val", "test"):
        f = root / f"boot_{sp}.json"
        if f.exists():
            bt[sp] = {f"{m}-F384NN": boot(f, m, "F384NN") for m in METHODS if m != "F384NN"}
            bt[sp].update({f"F768-{m}": boot(f, "F768", m) for m in METHODS if m != "F768"})
    if bt:
        md += ["", "## Paired image bootstrap (B = 1,000, same resamples for every method, official AP recomputed; * = 95 % CI excludes 0)", "",
               "| Contrast | Val Overall | Test Overall | Test Object | Test Part |", "|---|---|---|---|---|"]
        for m in METHODS:
            if m == "F384NN":
                continue
            k = f"{m}-F384NN"
            md.append(f"| {LABEL[m]} − F384 nearest | {fmt(bt['val'][k]['overall']) if 'val' in bt else ''} | {fmt(bt['test'][k]['overall']) if 'test' in bt else ''} | {fmt(bt['test'][k]['obj']) if 'test' in bt else ''} | {fmt(bt['test'][k]['part']) if 'test' in bt else ''} |")
        md += [""]
        for m in METHODS:
            if m == "F768":
                continue
            k = f"F768-{m}"
            md.append(f"| F768 − {LABEL[m]} | {fmt(bt['val'][k]['overall']) if 'val' in bt else ''} | {fmt(bt['test'][k]['overall']) if 'test' in bt else ''} | {fmt(bt['test'][k]['obj']) if 'test' in bt else ''} | {fmt(bt['test'][k]['part']) if 'test' in bt else ''} |")
    # size bins
    sz = {}
    for m in METHODS:
        f = root / f"size/roi_size_test_{m}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["mean4"] = d[["Color", "Pattern", "Material", "Transparency"]].mean(1)
            sz[m] = d
    if sz and "F384NN" in sz:
        base = sz["F384NN"]
        md += ["", "## Test AP by ROI size (mean of Color / Pattern / Material / Transparency AP; box area in 16-px tokens at 384)", "",
               "| Level | Bin | #ROI | " + " | ".join(LABEL[m] for m in METHODS if m in sz) + " |", "|---|---|---|" + "---|" * len(sz)]
        for i in range(len(base)):
            md.append(f"| {base.level[i]} | {base.size_bin[i]} | {base.roi_count[i]:,} | " + " | ".join(f"{sz[m].mean4[i]:.2f}" for m in METHODS if m in sz) + " |")
    # side by side + ranks
    if all("test" in res[m] for m in METHODS):
        fv = {m: res[m]["val"]["overall"] for m in METHODS}
        ft = {m: res[m]["test"]["overall"] for m in METHODS}
        rv = {m: R2[m][0] for m in METHODS}
        rt = {m: R2[m][1] for m in METHODS}
        rk = dict(fv=ranks(fv), ft=ranks(ft), rv=ranks(rv), rt=ranks(rt))
        md += ["", "## Fixed readout vs independently retrained R2 (Overall AP; rank in brackets, 1 = best within the column)", "",
               "| Method | Fixed Val | Retrained Val | Fixed Test | Retrained Test | Fixed − Retrained (Test) |", "|---|---|---|---|---|---|"]
        for m in METHODS:
            def c(v, r):
                return "–" if v is None else f"{v:.2f} ({r})"
            md.append(f"| {LABEL[m]} | {c(fv[m], rk['fv'][m])} | {c(rv[m], rk['rv'].get(m))} | {c(ft[m], rk['ft'][m])} | {c(rt[m], rk['rt'].get(m))} | {'' if rt[m] is None else f'{ft[m] - rt[m]:+.2f}'} |")
        md.append(f"| (reference) F384 24 × 24 read, retrained R2 | | 54.21 | | 39.28 | |")
        both_t = [m for m in METHODS if rt[m] is not None]
        both_v = [m for m in METHODS if rv[m] is not None]
        sp_t = pd.Series({m: ft[m] for m in both_t}).corr(pd.Series({m: rt[m] for m in both_t}), method="spearman")
        sp_v = pd.Series({m: fv[m] for m in both_v}).corr(pd.Series({m: rv[m] for m in both_v}), method="spearman")
        md += ["", f"Spearman rank correlation between the two protocols: Test {sp_t:.2f} over {len(both_t)} methods, Val {sp_v:.2f} over {len(both_v)} methods."]
        res["_ranks"] = dict(spearman_test=sp_t, spearman_val=sp_v, fixed_test=ft, retrained_test=rt, fixed_val=fv, retrained_val=rv)
    res["_boot"] = {sp: {k: {kk: list(vv) for kk, vv in v.items()} for k, v in d.items()} for sp, d in bt.items()}
    (root / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
