"""Tables for the RSP comparison (reads only saved results; CPU). -> <comparison>/results/tables.md + summary.json

  python -m src.rsp.report --root output/2026-11-12/rsp_masked_comparison
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

NAMES = [("F384", "F384_r2"), ("F768 (T768)", "T768_r2"), ("FA-A", "FAA_r2"), ("FA-B", "FAB_r2"), ("RSP-Masked", "RSPPIX_r2"), ("RSP-Masked-noPix", "RSPNOPIX_r2")]
CONTRASTS = [("RSP-Masked − F384", "RSPPIX_r2", "F384_r2"), ("RSP-Masked-noPix − F384", "RSPNOPIX_r2", "F384_r2"), ("RSP-Masked − RSP-Masked-noPix", "RSPPIX_r2", "RSPNOPIX_r2"),
             ("RSP-Masked − FA-B", "RSPPIX_r2", "FAB_r2"), ("RSP-Masked − FA-A", "RSPPIX_r2", "FAA_r2"), ("F768 − RSP-Masked", "T768_r2", "RSPPIX_r2"), ("F768 − RSP-Masked-noPix", "T768_r2", "RSPNOPIX_r2")]


def hier(p):
    h = json.loads(Path(p).read_text())["hierarchical"]
    o, pa = h["obj-attrs"], h["obj-part-attrs"]
    return dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"], overall=(h["obj-attr-AP"] + h["obj-attr-part-AP"]) / 2,
                **{k: (o[a] + pa[a]) / 2 for k, a in (("pattern", "pattern_marking"), ("material", "material"), ("color", "color"), ("transparency", "transparency"))})


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def ci(boot, a, b, key):
    if key == "overall":
        d = (np.array(boot[a]["obj"]) + np.array(boot[a]["part"])) / 2 - (np.array(boot[b]["obj"]) + np.array(boot[b]["part"])) / 2
    else:
        d = np.array(boot[a][key]) - np.array(boot[b][key])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return lo, hi


def contrast(B, a, b, key):
    if B is None or a not in B["points"] or b not in B["points"]:
        return None
    pa, pb = B["points"][a], B["points"][b]
    pt = ((pa["obj"] + pa["part"]) - (pb["obj"] + pb["part"])) / 2 if key == "overall" else pa[key] - pb[key]
    lo, hi = ci(B["boot"], a, b, key)
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    C = Path(a.root)
    E = C / "eval"
    sel = json.loads((E / "selection_frozen.json").read_text()) if (E / "selection_frozen.json").exists() else json.loads((E / "selection.json").read_text())
    fa_test = Path("output/2026-11-11/fine_aware_shared_s0/eval/eval_test")
    L, S = [], {}
    L.append("## Official PACO AP (GT box + GT category oracle attribute AP; R2 reader 10 epochs, seed 0; epoch = max official Val (Obj+Part)/2)\n")
    L.append("| Method | R2 ep | Val Overall | Val Obj / Part | Test Overall | Test Obj / Part | Test Pattern / Material / Color |")
    L.append("|---|---|---|---|---|---|---|")
    for disp, n in NAMES:
        if n not in sel:
            continue
        ep = sel[n]["epoch"]
        v = hier(E / "eval_val" / n / f"ep{ep}" / "official_results.json")
        tp = E / "eval_test" / n / "official_results.json"
        if not tp.exists():
            tp = fa_test / n / "official_results.json"
        if not tp.exists():
            tp = Path(f"output/2026-11-11/token_split_x2_s0/eval/eval_test/{n}/official_results.json")
        t = hier(tp) if tp.exists() else None
        S[disp] = dict(epoch=ep, val=v, test=t, test_source=str(tp) if t else None)
        L.append(f"| {disp} | {ep} | {f2(v['overall'])} | {f2(v['obj'])} / {f2(v['part'])} | {f2(t and t['overall'])} | {f2(t and t['obj'])} / {f2(t and t['part'])} | "
                 f"{f2(t and t['pattern'])} / {f2(t and t['material'])} / {f2(t and t['color'])} |")
    curves = json.loads((E / "val_curves.json").read_text()) if (E / "val_curves.json").exists() else {}
    L.append("\nOfficial Val (Obj+Part)/2 by R2 epoch 1–10:\n")
    for disp, n in NAMES[4:]:
        if n in curves:
            L.append(f"- {disp}: " + " ".join(f"{(c['obj'] + c['part']) / 2:.2f}" for e, c in sorted(curves[n].items(), key=lambda kv: int(kv[0]))))
    Bv = json.loads((E / "boot_val.json").read_text()) if (E / "boot_val.json").exists() else None
    Bt = json.loads((E / "boot_test.json").read_text()) if (E / "boot_test.json").exists() else None
    L.append("\n## Paired image bootstrap (B = 1,000, official AP recomputed per resample, same resamples for every method; * = 95% CI excludes 0)\n")
    L.append("| Contrast | Val Overall | Test Overall | Test Object | Test Part |")
    L.append("|---|---|---|---|---|")
    S["bootstrap"] = {}
    for disp, x, y in CONTRASTS:
        row = [contrast(Bv, x, y, "overall"), contrast(Bt, x, y, "overall"), contrast(Bt, x, y, "obj"), contrast(Bt, x, y, "part")]
        S["bootstrap"][disp] = row
        L.append(f"| {disp} | " + " | ".join(r or "–" for r in row) + " |")
    sz = sorted((E / "size").glob("*_test.csv")) if (E / "size").exists() else []
    if sz:
        rows = []
        for p in sz:
            rows += list(csv.DictReader(open(p)))
        L.append("\n## Official Test AP by ROI size (box area at the 384 scale; mean of Color/Pattern/Material/Transparency AP per bin)\n")
        bins = sorted({r["size_bin"] for r in rows}, key=lambda b: ["tiny", "small", "medium", "large"].index(b.split("_")[0]))
        models = [n for _, n in NAMES if any(r["model"] == n for r in rows)]
        L.append("| Level | Size bin | #ROI | " + " | ".join(models) + " |")
        L.append("|---|---|---|" + "---|" * len(models))
        S["size"] = {}
        for lv in ("Object", "Part"):
            for bn in bins:
                vals, cnt = [], ""
                for mname in models:
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
        L.append("\n## Held-out feature analysis vs the dense teacher (COCO dev images, all valid fine cells of each ROI)\n")
        L.append("| Features | ROI bucket | cos | L1 (normalised) | MSE raw | norm ratio | within-parent residual R² | residual cos | parent-mean MSE |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for n in ("RSP-Masked", "RSP-Masked-noPix", "F384-bil", "F384-NN", "F768-bil (vs teacher)"):
            for bk in ("train_1to6", "large_7to12", "side_1to2", "side_3to4", "side_5to6"):
                r = d.get(n, {}).get(bk)
                if r:
                    L.append(f"| {n} | {bk} | {r['cos']:.4f} | {r['l1_norm']:.4f} | {r['mse_raw']:.4f} | {r['norm_ratio']:.3f} | {r['within_parent_residual_r2']:.3f} | {r['res_cos']:.3f} | {r['parent_mean_mse']:.4f} |")
        r = d.get("F384-bil vs F768-bil", {}).get("all")
        if r:
            L.append(f"\nContext: F384-bil vs F768-bil at the same fine points: cos {r['cos']:.4f}, within-parent residual R² {r['within_parent_residual_r2']:.3f}.")
        S["features"] = d
    ef = C / "efficiency.json"
    if ef.exists():
        d = json.loads(ef.read_text())
        L.append(f"\n## Cost ({d['setup']})\n")
        L.append("| Component | ms | GFLOPs |")
        L.append("|---|---|---|")
        for k in ("DINOv3_384_per_image", "F768_reencode_per_image", "C0_crop_reencode_per_ROI"):
            L.append(f"| {k} | {d[k]['ms']:.2f} | {d[k]['gflops']:.1f} |")
        for m in ("RSP-Masked", "RSP-Masked-noPix"):
            r = d[m]
            L.append(f"| {m} context stream / image | {r['context_per_image']['ms']:.2f} | {r['context_per_image']['gflops']:.2f} |")
            if "evidence_conv_per_ROI" in r:
                L.append(f"| {m} evidence conv / ROI | {r['evidence_conv_per_ROI']['ms']:.3f} | {r['evidence_conv_per_ROI']['gflops']:.4f} |")
            for k, v in r.items():
                if "roi_" in k:
                    L.append(f"| {m} {k}: L1 / queries / total incl. DINOv3 384 | {v['L1_stream']['ms']:.2f} / {v['queries']['ms']:.2f} / {v['total_incl_dino384']['ms']:.2f} | "
                             f"{v['L1_stream']['gflops']:.2f} / {v['queries']['gflops']:.2f} / {v['total_incl_dino384']['gflops']:.1f} |")
        for k, v in d.items():
            if k.startswith("dense_teacher_direct"):
                L.append(f"| {k} | {v['ms']:.2f} | {v['gflops']:.1f} |")
        S["efficiency"] = d
    for tag, run in (("RSP-Masked", "rsp_masked_pix_s0"), ("RSP-Masked-noPix", "rsp_masked_nopix_s0")):
        p = C.parent / run / "train" / "train.jsonl"
        if p.exists():
            rows = [json.loads(l) for l in open(p)]
            pick = [r for r in rows if r["step"] in (1, 1000, 3000, 6000, 9000, 12000)]
            if pick:
                L.append(f"\n## Training log {tag} (COCO train batches; ref = F384 bilinear read at the same queries)\n")
                L.append("| step | loss | cos | L1 norm | norm ratio | ref cos | ref L1 norm | s/step |")
                L.append("|---|---|---|---|---|---|---|---|")
                for r in pick:
                    L.append(f"| {r['step']} | {r['loss']:.4f} | {r.get('cos', float('nan')):.4f} | {r.get('l1_norm', float('nan')):.4f} | {r.get('norm_ratio', float('nan')):.3f} | "
                             f"{r.get('ref_f384bil_cos', float('nan')):.4f} | {r.get('ref_f384bil_l1_norm', float('nan')):.4f} | {r['sec_per_step']:.3f} |")
    (C / "results").mkdir(exist_ok=True)
    (C / "results" / "tables.md").write_text("\n".join(L) + "\n")
    (C / "results" / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
