"""Tables for the Coarse-Fine pilot from the finished evaluations. Official PACO GT-ROI attribute AP only (no M0 / M2). CI = paired image bootstrap (B=1000), 95%.

  python -m src.hcf.report --run <run>       -> <run>/results/*.csv, <run>/REPORT_tables.md, P/results, M/results
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

ROWS = [("Original Backbone (ROIAlign R2)", "orig_r2"), ("G1-pool (ROIAlign R2)", "g1pool_r2"), ("Full-ML (frozen Original, DINOv3, reader Fuse)", "orig_r2ml"),
        ("M — Retrained Full-ML (trained fuse Q)", "M_r2Q"), ("M — reader Fuse (fairness check)", "M_r2ml"), ("P — Coarse-only", "P_r2coarse"), ("P — Coarse+Fine", "P_r2fused"), ("GT Crop C0", "c0_r2")]
PAIRS = [("P_r2fused", "M_r2Q", "P-Fused − M"), ("P_r2fused", "M_r2ml", "P-Fused − M (reader Fuse)"), ("P_r2fused", "P_r2coarse", "P-Fused − P-Coarse"), ("P_r2fused", "P_r2fused_nofine", "P-Fused − same reader without Fine"),
         ("P_r2coarse", "M_r2Q", "P-Coarse − M"), ("M_r2Q", "g1pool_r2", "M − G1-pool"), ("P_r2fused", "g1pool_r2", "P-Fused − G1-pool"), ("P_r2fused", "c0_r2", "P-Fused − GT Crop"), ("g1pool_r2", "orig_r2", "G1-pool − Original")]


def to_md(df):
    """GitHub table (no tabulate dependency in this env)."""
    fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)  # noqa: E731
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    return "\n".join([head] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.values.tolist()])


def off(run, n, kind, ep=None):
    f = f"{run}/eval_test/{n}/test/official_results.json" if kind == "test" else f"{run}/eval_val/{n}/ep{ep}/official_results.json"
    h = json.load(open(f))["hierarchical"]
    return h["obj-attr-AP"], h["obj-attr-part-AP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = a.run
    os.makedirs(f"{run}/results", exist_ok=True)
    sel = json.load(open(f"{run}/selection.json"))
    B = json.load(open(f"{run}/boot_test.json"))
    rows = []
    for lab, n in ROWS + [("P-Fused reader, Fine removed", "P_r2fused_nofine")]:
        o, p = off(run, n, "test")
        ci = lambda k: tuple(np.percentile(B["boot"][n][k], [2.5, 97.5]))  # noqa: E731
        rows.append(dict(method=lab, key=n, epoch=sel[n]["epoch"], val_obj=sel[n].get("val_obj"), val_part=sel[n].get("val_part"), test_AP_obj=o, test_AP_opart=p,
                         test_obj_lo=ci("obj")[0], test_obj_hi=ci("obj")[1], test_opart_lo=ci("part")[0], test_opart_hi=ci("part")[1]))
    T1 = pd.DataFrame(rows)
    T1.to_csv(f"{run}/results/paco_attribute_ap.csv", index=False)
    d = []
    for x, y, lab in PAIRS:
        for k, name in (("obj", "AP_obj"), ("part", "AP_opart")):
            dv = np.array(B["boot"][x][k]) - np.array(B["boot"][y][k])
            lo, hi = np.percentile(dv, [2.5, 97.5])
            d.append(dict(comparison=lab, metric=name, diff=B["points"][x][k] - B["points"][y][k], lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0)))
    D = pd.DataFrame(d)
    D.to_csv(f"{run}/results/paco_differences.csv", index=False)
    D[D.comparison.str.contains("Fine|Coarse")].to_csv(f"{run}/results/ablation.csv", index=False)
    pres = []
    A0 = json.load(open("output/2026-11-07/g1_leakcheck_ade20k/ade/done.json"))["results"]
    I0 = json.load(open("output/2026-11-07/g1_imagenet_generalization/results.json"))["independent"]
    try:
        A1 = json.load(open(f"{run}/ade/done.json"))["results"]
    except FileNotFoundError:
        A1 = {}
    try:
        I1 = json.load(open(f"{run}/imagenet/linear/linear_results.json"))["independent"]
    except FileNotFoundError:
        I1 = {}
    top1 = lambda v: v["val"]["top1"] if isinstance(v["val"], dict) and "top1" in v["val"] else v["val"]  # noqa: E731
    pres.append(dict(method="Original (existing, same protocol)", ade_miou=A0["original"]["mIoU"], imagenet_top1=I0["original"]["val_top1"]))
    pres.append(dict(method="G1-pool (existing, same protocol)", ade_miou=A0["g1pool"]["mIoU"], imagenet_top1=I0["g1pool"]["val_top1"]))
    if "original" in I1:
        pres.append(dict(method="Original (re-extracted here)", ade_miou=None, imagenet_top1=top1(I1["original"])))
    for m, k in (("M", "m"), ("P", "p")):
        pres.append(dict(method=m, ade_miou=A1.get(m, {}).get("mIoU"), imagenet_top1=top1(I1[k]) if k in I1 else None))
    P = pd.DataFrame(pres)
    P.to_csv(f"{run}/results/preservation.csv", index=False)
    if os.path.exists(f"{run}/efficiency.csv"):
        shutil.copy(f"{run}/efficiency.csv", f"{run}/results/efficiency.csv")
    for c in ("P", "M"):
        os.makedirs(f"{run}/{c}/results", exist_ok=True)
        for f in ("paco_attribute_ap.csv", "preservation.csv", "efficiency.csv", "ablation.csv"):
            if os.path.exists(f"{run}/results/{f}"):
                shutil.copy(f"{run}/results/{f}", f"{run}/{c}/results/{f}")
    md = ["# Coarse-Fine pilot: automatic tables", "", "## Table 1. PACO GT-ROI attribute AP (official evaluator, GT box + category oracle; TEST)", "",
          to_md(T1[["method", "epoch", "val_obj", "val_part", "test_AP_obj", "test_AP_opart"]].round(2)), "", "## Paired differences (TEST, 95% CI)", "", to_md(D.round(2)), "",
          "## Table 2. Preservation", "", to_md(P.round(2)), ""]
    if os.path.exists(f"{run}/efficiency.csv"):
        md += ["## Table 3. Cost", "", to_md(pd.read_csv(f"{run}/efficiency.csv").round(2)), ""]
    open(f"{run}/REPORT_tables.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
