"""Tables for J1 / W1 / LowFlow on PACO official Val and Test (GT box + GT category oracle attribute AP; epoch fixed on official Val). Overall = (Object + Part) / 2.
Every model below goes through the same R2 trainer (src.g1up.train_bb), inference (src.overnight6.official_infer) and official evaluator. A0 = A0bb, PixelDiT = W1a000 (bit-identical Hybrid encoder), E2-L = the overnight_six run.

  --fast   point estimates so far -> eval/results/fast_tables.md
  (none)   + bootstrap, ROI size, gradient summary, baseline audit -> eval/results/tables.md, delta_ap.csv, provisional README.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.overnight6.report import COLS, from_results, md, value

LABEL = {"A0bb_r2": "A0 (Crop)", "W1a000_r2": "PixelDiT Hybrid (Crop + Flow)", "E2L_r2": "E2-L (Crop + Grid)", "J1_r2": "J1 (Crop + Flow + Grid)", "LowFlow_r2": "P-LowFlow (Crop + 0.25 Flow)",
         "W1a025_r2": "W1 alpha 0.25", "W1a050_r2": "W1 alpha 0.50", "W1a075_r2": "W1 alpha 0.75", "W1a100_r2": "W1 alpha 1.00 (E2-L encoder, R2 rerun)",
         "A0old_r2": "audit: A0 stored R2", "Hybridold_r2": "audit: Hybrid stored R2", "E2Lold_r2": "audit: E2-L stored R2"}
JOINT = ["A0bb_r2", "W1a000_r2", "E2L_r2", "J1_r2", "LowFlow_r2"]
W1 = ["W1a000_r2", "W1a025_r2", "W1a050_r2", "W1a075_r2", "W1a100_r2"]
PAIRS_JOINT = [("J1_r2", "W1a000_r2"), ("J1_r2", "E2L_r2"), ("J1_r2", "A0bb_r2"), ("W1a000_r2", "A0bb_r2"), ("E2L_r2", "A0bb_r2"), ("LowFlow_r2", "W1a000_r2")]
PRIMARY = ["Obj Color", "Obj Pattern", "Part Color", "Part Pattern"]
NOTE = "Official evaluator, GT box + GT category oracle attribute AP, R2 reader, encoder seed 0 (single run). Epoch = max (Object + Part) / 2 on official Val, ties earlier; Test once at that epoch. Reflectance = evaluator key `transparency`. Multiple comparisons are not corrected."


def load(root):
    sel, res = {}, dict(val={}, test={})
    for path in sorted((root / "selection").glob("*.json")):
        m = path.stem
        sel[m] = json.loads(path.read_text())
        val, test = root / f"eval_val/{m}/ep{sel[m]['epoch']}/official_results.json", root / f"fast_test/{m}/official_results.json"
        if val.exists():
            res["val"][m] = from_results(val)
        if test.exists():
            res["test"][m] = from_results(test)
    return sel, res


def table(points, sel, models):
    return pd.DataFrame([dict(Model=LABEL.get(m, m), epoch=sel[m]["epoch"], **{c: float(value(points[m], c)) for c in COLS}) for m in models if m in points])


def w1_pairs(run):
    path = run / "w1_alpha_choice.json"
    if not path.exists():
        return []
    c = json.loads(path.read_text())["chosen"] + "_r2"
    return [(c, "W1a000_r2"), (c, "E2L_r2"), ("W1a100_r2", "E2L_r2")]


def fast(run):
    root, out = run / "eval", run / "eval/results"
    out.mkdir(parents=True, exist_ok=True)
    sel, res = load(root)
    pairs = PAIRS_JOINT + w1_pairs(run)
    lines = ["# J1 / W1 / P-LowFlow: official AP (point estimates; bootstrap and ROI size follow)", "", NOTE, ""]
    audits = sorted(run.glob("verification/baseline_audit_*.json"))
    if audits:
        lines += ["## Baseline audit (stored R2 readers, re-inferred on the shared path)", "", "| Name | Val Obj new/old | Val Part new/old | Test Obj new/old | Test Part new/old | max abs diff Val / Test | Pearson vs probe scores |", "|---|---|---|---|---|---|---|"]
        for p in audits:
            a = json.loads(p.read_text())
            v, t = a["official"]["val"], a["official"]["test"]
            lines.append(f"| {a['name']} | {v['new_obj']:.2f} / {v['old_obj']:.2f} | {v['new_part']:.2f} / {v['old_part']:.2f} | {t['new_obj']:.2f} / {t['old_obj']:.2f} | {t['new_part']:.2f} / {t['old_part']:.2f} | {v['max_abs_diff']:.3f} / {t['max_abs_diff']:.3f} | {a['reproduction_pearson_vs_probe_scores']} |")
        lines.append("")
    for split in ("val", "test"):
        if res[split]:
            lines += [f"## Official {split.capitalize()}: Joint 2x2 + LowFlow", "", md(table(res[split], sel, JOINT)), "", f"## Official {split.capitalize()}: W1 weight mixing (separate from Joint)", "", md(table(res[split], sel, W1)), ""]
            delta = [dict(Contrast=f"{x} - {y}", **{c: f"{float(value(res[split][x], c) - value(res[split][y], c)):+.2f}" for c in COLS}) for x, y in pairs if x in res[split] and y in res[split]]
            if delta:
                lines += ["### Delta (point estimates)", "", md(pd.DataFrame(delta)), ""]
    choice = run / "w1_alpha_choice.json"
    if choice.exists():
        lines += ["## W1 alpha choice (official Val only)", "", "```json", choice.read_text(), "```", ""]
    (out / "fast_ap.json").write_text(json.dumps(dict(selection=sel, **res), indent=1))
    tmp = out / "fast_tables.md.tmp"
    tmp.write_text("\n".join(lines))
    tmp.replace(out / "fast_tables.md")
    print("\n".join(lines), flush=True)


def gradients(run):
    out = {}
    for n in ("J1", "LowFlow"):
        path = run / n / "logs/gradients.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        out[n] = {r["step"]: {k: v for k, v in r.items() if k.startswith(("coarse/", "fine/")) and ("weighted" in k or "/cos_" in k or "_raw" in k)} for r in rows if r["step"] in (1, 2, 3, 1000, 3000, 6000, 9000, 12000)}
    return out


def grad_md(g):
    lines = []
    for n, steps in g.items():
        keys = [k for k in next(iter(steps.values())) if k.split("/")[0] in ("coarse", "fine")]
        lines += [f"**{n}** (norms are with the loss coefficient; cosines are coefficient-free; student parameters only)", "", "| step | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
        lines += ["| " + str(s) + " | " + " | ".join(f"{v[k]:+.3f}" if "cos" in k else f"{v[k]:.3f}" for k in keys) + " |" for s, v in steps.items()]
        lines.append("")
    return lines


def full(run):
    root, out = run / "eval", run / "eval/results"
    sel, res = load(root)
    pairs = PAIRS_JOINT + w1_pairs(run)
    lines = ["# J1 / W1 / P-LowFlow: PACO official AP", "", NOTE + " Paired image bootstrap B = 1000, seed 0.", ""]
    deltas = []
    for split in ("val", "test"):
        boot = json.loads((root / f"boot_{split}.json").read_text())
        models = list(boot["points"])
        t = pd.DataFrame([dict(Model=LABEL.get(m, m), epoch=sel[m]["epoch"], **{c: float(value(boot["points"][m], c)) for c in COLS}) for m in models])
        t.to_csv(out / f"{split}_ap.csv", index=False)
        rows = []
        for x, y in pairs:
            if x in models and y in models:
                row = dict(Contrast=f"{x} - {y}")
                for c in COLS:
                    d = float(value(boot["points"][x], c) - value(boot["points"][y], c))
                    lo, hi = np.percentile(value(boot["boot"][x], c) - value(boot["boot"][y], c), [2.5, 97.5])
                    row[c] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                    deltas.append(dict(split=split, contrast=f"{x} - {y}", metric=c, delta=d, lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0), primary=c in PRIMARY))
                rows.append(row)
        lines += [f"## Official {split.capitalize()}", "", md(t), "", "### Delta (95% CI)", "", md(pd.DataFrame(rows)), ""]
        size = pd.concat([pd.read_csv(out / f"roi_size_{split}_{m}.csv") for m in models], ignore_index=True)
        size.to_csv(out / f"{split}_roi_size.csv", index=False)
        lines += ["### ROI size (GT-box area on the 384 canvas in 16 px tokens; point estimates)", ""]
        for level in ("Object", "Part"):
            for col in ("Color", "Pattern", "Material"):
                p = size[size.level == level].pivot(index="size_bin", columns="model", values=col).reindex(columns=models).reindex(["tiny_lt4_tokens", "small_4to16_tokens", "medium_16to64_tokens", "large_ge64_tokens"])
                cnt = size[(size.level == level) & (size.model == models[0])].set_index("size_bin").roi_count
                tt = p.reset_index()
                tt.insert(1, "ROIs", [int(cnt[s]) for s in p.index])
                tt.columns = ["Size", "ROIs"] + [m[:-3] for m in models]
                lines += [f"**{level} {col}**", "", md(tt), ""]
    pd.DataFrame(deltas).to_csv(out / "delta_ap.csv", index=False)
    (out / "official_ap.json").write_text(json.dumps(dict(selection=sel, **res), indent=1))
    g = gradients(run)
    (run / "verification/gradient_summary.json").write_text(json.dumps(g, indent=1))
    lines += ["## Gradient diagnostics", ""] + grad_md(g)
    (out / "tables.md").write_text("\n".join(lines))
    readme = run / "README.md"
    if not readme.exists() or "(해석 섹션은" in readme.read_text():
        readme.write_text("\n".join(["# J1 / W1 / P-LowFlow", "", NOTE, "", "Tables: [eval/results/tables.md](eval/results/tables.md), point estimates: [eval/results/fast_tables.md](eval/results/fast_tables.md)", "", "(해석 섹션은 결과 검토 후 추가됩니다.)", ""]))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    (fast if a.fast else full)(Path(a.run))


if __name__ == "__main__":
    main()
