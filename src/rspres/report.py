"""Tables for the RSP-Residual comparison (reads only saved results; CPU). -> <comparison>/results/tables.md + summary.json

  python -m src.rspres.report --root output/2026-11-12/rsp_residual_comparison
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.rsp.report import f2, hier


def _v(d, key):
    if key == "overall":
        return (np.asarray(d["obj"]) + np.asarray(d["part"])) / 2
    if key in ("obj", "part"):
        return np.asarray(d[key])
    a = {"pattern": "pattern_marking"}.get(key, key)
    return (np.asarray(d[f"obj_{a}"]) + np.asarray(d[f"part_{a}"])) / 2


def contrast(B, a, b, key):
    if B is None or a not in B["points"] or b not in B["points"]:
        return None
    pt = float(_v(B["points"][a], key) - _v(B["points"][b], key))
    lo, hi = np.percentile(_v(B["boot"][a], key) - _v(B["boot"][b], key), [2.5, 97.5])
    return f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("*" if lo > 0 or hi < 0 else "")

NAMES = [("F384", "F384_r2"), ("F768 (T768)", "T768_r2"), ("RSP-Masked", "RSPPIX_r2"), ("Step-0 (Δ=0)", "RSPRES0_r2"), ("RSP-Residual", "RSPRES_r2")]
CONTRASTS = [("RSP-Residual − F384", "RSPRES_r2", "F384_r2"), ("RSP-Residual − RSP-Masked", "RSPRES_r2", "RSPPIX_r2"),
             ("RSP-Residual − Step-0", "RSPRES_r2", "RSPRES0_r2"), ("Step-0 − F384", "RSPRES0_r2", "F384_r2"), ("F768 − RSP-Residual", "T768_r2", "RSPRES_r2")]
TEST_SRC = [Path("output/2026-11-11/fine_aware_shared_s0/eval/eval_test"), Path("output/2026-11-12/rsp_masked_comparison/eval/eval_test"),
            Path("output/2026-11-11/token_split_x2_s0/eval/eval_test")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    C = Path(a.root)
    E = C / "eval"
    sel = json.loads((E / "selection_frozen.json").read_text())
    L, S = [], {}
    L.append("## Official PACO AP (GT box + GT category oracle attribute AP; R2 reader 10 epochs, seed 0; epoch = max official Val (Obj+Part)/2)\n")
    L.append("| Method | R2 ep | Val Overall | Val Obj / Part | Test Overall | Test Obj / Part | Test Pattern / Material / Color |")
    L.append("|---|---|---|---|---|---|---|")
    for disp, n in NAMES:
        ep = sel[n]["epoch"]
        v = hier(E / "eval_val" / n / f"ep{ep}" / "official_results.json")
        tp = next((p / n / "official_results.json" for p in [E / "eval_test"] + TEST_SRC if (p / n / "official_results.json").exists()), None)
        t = hier(tp) if tp else None
        S[disp] = dict(epoch=ep, val=v, test=t, test_source=str(tp) if t else None)
        L.append(f"| {disp} | {ep} | {f2(v['overall'])} | {f2(v['obj'])} / {f2(v['part'])} | {f2(t and t['overall'])} | {f2(t and t['obj'])} / {f2(t and t['part'])} | "
                 f"{f2(t and t['pattern'])} / {f2(t and t['material'])} / {f2(t and t['color'])} |")
    curves = json.loads((E / "val_curves.json").read_text()) if (E / "val_curves.json").exists() else {}
    L.append("\nOfficial Val (Obj+Part)/2 by R2 epoch 1–10:\n")
    for disp, n in NAMES:
        if n in curves:
            L.append(f"- {disp}: " + " ".join(f"{(c['obj'] + c['part']) / 2:.2f}" for e, c in sorted(curves[n].items(), key=lambda kv: int(kv[0]))))
    Bv = json.loads((E / "boot_val.json").read_text()) if (E / "boot_val.json").exists() else None
    Bt = json.loads((E / "boot_test.json").read_text()) if (E / "boot_test.json").exists() else None
    L.append("\n## Paired image bootstrap (B = 1,000, official AP recomputed per resample, same resamples for every method; * = 95% CI excludes 0)\n")
    L.append("| Contrast | Val Overall | Test Overall | Test Object | Test Part | Test Pattern | Test Material | Test Color |")
    L.append("|---|---|---|---|---|---|---|---|")
    S["bootstrap"] = {}
    for disp, x, y in CONTRASTS:
        row = [contrast(Bv, x, y, "overall"), contrast(Bt, x, y, "overall"), contrast(Bt, x, y, "obj"), contrast(Bt, x, y, "part"), contrast(Bt, x, y, "pattern"), contrast(Bt, x, y, "material"), contrast(Bt, x, y, "color")]
        S["bootstrap"][disp] = row
        L.append(f"| {disp} | " + " | ".join(r or "–" for r in row) + " |")
    sz = sorted((E / "size").glob("*_test.csv")) if (E / "size").exists() else []
    if sz:
        rows = []
        for p in sz:
            rows += list(csv.DictReader(open(p)))
        L.append("\n## Official Test AP by ROI size (box area at the 384 scale; mean of Color/Pattern/Material/Transparency AP per bin)\n")
        bins = sorted({r["size_bin"] for r in rows}, key=lambda b: ["tiny", "small", "medium", "large"].index(b.split("_")[0]))
        models = [(d, n) for d, n in NAMES if any(r["model"] == n for r in rows)]
        L.append("| Level | Size bin | #ROI | " + " | ".join(d for d, _ in models) + " |")
        L.append("|---|---|---|" + "---|" * len(models))
        S["size"] = {}
        for lv in ("Object", "Part"):
            for bn in bins:
                vals, cnt = [], ""
                for _, mname in models:
                    r = [r for r in rows if r["model"] == mname and r["level"] == lv and r["size_bin"] == bn]
                    if r:
                        cnt = r[0]["roi_count"]
                        v = np.mean([float(r[0][k]) for k in ("Color", "Pattern", "Material", "Transparency")])
                        vals.append(f"{v:.2f}")
                        S["size"].setdefault(mname, {})[f"{lv}/{bn}"] = v
                    else:
                        vals.append("–")
                L.append(f"| {lv} | {bn} | {cnt} | " + " | ".join(vals) + " |")
    fa = C / "feature_analysis.json"
    if fa.exists():
        d = json.loads(fa.read_text())
        L.append("\n## Held-out feature analysis vs the p_T = 4 dense teacher (300 COCO dev images, same ROIs as RSP analysis, all valid fine cells)\n")
        L.append("| Features | ROI bucket | cos | L1 (normalised) | MSE raw | norm ratio | within-parent residual R² | residual cos | parent-mean MSE |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for n in ("RSP-Residual", "RSP-Masked", "Step-0 (= F384-bil)"):
            for bk in ("all", "train_1to6", "large_7to12", "side_1to2", "side_3to4", "side_5to6"):
                r = d.get(n, {}).get(bk)
                if r:
                    L.append(f"| {n} | {bk} | {r['cos']:.4f} | {r['l1_norm']:.4f} | {r['mse_raw']:.4f} | {r['norm_ratio']:.3f} | {r['within_parent_residual_r2']:.3f} | {r['res_cos']:.3f} | {r['parent_mean_mse']:.4f} |")
        L.append("\n| Δ statistic | ROI bucket | |Δ| / |F384_bil| | cos(Δ, T − F384_bil) | #ROI |")
        L.append("|---|---|---|---|---|")
        for bk in ("all", "train_1to6", "large_7to12", "side_1to2", "side_3to4", "side_5to6"):
            r = d["_delta"].get(bk)
            if r:
                L.append(f"| RSP-Residual | {bk} | {r['delta_over_bil_norm']:.3f} | {r['delta_cos_to_teacher_minus_bil']:.3f} | {r['n_roi']} |")
        S["features"] = d
    p = C.parent / "rsp_residual_s0" / "train" / "train.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in open(p)]
        pick = [r for r in rows if r["step"] in (1, 500, 1000, 3000, 6000, 9000, 12000)]
        L.append("\n## Training log RSP-Residual (COCO train batches; ref = F384 bilinear read at the same queries = Step-0 prediction)\n")
        L.append("| step | loss | cos | L1 norm | norm ratio | |Δ|/|F384_bil| | ref cos | ref L1 norm | s/step |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in pick:
            L.append(f"| {r['step']} | {r['loss']:.4f} | {r['cos']:.4f} | {r['l1_norm']:.4f} | {r['norm_ratio']:.3f} | {r.get('delta_over_bil_norm', float('nan')):.3f} | "
                     f"{r['ref_f384bil_cos']:.4f} | {r['ref_f384bil_l1_norm']:.4f} | {r['sec_per_step']:.3f} |")
        S["train_curve"] = [{k: r.get(k) for k in ("step", "loss", "cos", "l1_norm", "norm_ratio", "delta_over_bil_norm", "ref_f384bil_cos", "ref_f384bil_l1_norm")} for r in rows]
    (C / "results").mkdir(exist_ok=True)
    (C / "results" / "tables.md").write_text("\n".join(L) + "\n")
    (C / "results" / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
