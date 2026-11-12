"""Interaction ablation tables on PACO official Val and Test (GT box + GT category oracle attribute AP). Overall = (Object + Part) / 2. No M0 / M2.

  --fast   point estimates only, from official_results.json (no bootstrap), written to eval/results/fast_tables.md
  (none)   full tables with image-level paired bootstrap 95% CI, ROI size and training summary, written to eval/results/tables.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["A0_r2", "A1_r2", "A2_r2", "A0_coarse", "A1_coarse", "A2_coarse"]
LABEL = dict(A0_r2="A0 Bidirectional (Coarse+Fine)", A1_r2="A1 Coarse->Fine (Coarse+Fine)", A2_r2="A2 No Interaction (Coarse+Fine)", A0_coarse="A0 Coarse-only", A1_coarse="A1 Coarse-only", A2_coarse="A2 Coarse-only")
PAIRS = [("A1_r2", "A0_r2"), ("A2_r2", "A0_r2"), ("A1_r2", "A2_r2"), ("A0_r2", "A0_coarse"), ("A1_r2", "A1_coarse"), ("A2_r2", "A2_coarse"), ("A1_coarse", "A0_coarse"), ("A2_coarse", "A0_coarse")]
TYPES = [("color", "Color"), ("pattern_marking", "Pattern"), ("material", "Material"), ("transparency", "Transparency")]
COLS = ["Overall", "Object", "Part"] + [f"Obj {n}" for _, n in TYPES] + [f"Part {n}" for _, n in TYPES]
HEAD = "GT box + GT category oracle attribute AP, R2 reader. Epoch = max (Object + Part) / 2 on official Val, ties earlier; Test uses that epoch, once. Encoder seed 0."


def value(p, c):
    if c == "Overall":
        return (np.array(p["obj"]) + np.array(p["part"])) / 2
    if c == "Object":
        return np.array(p["obj"])
    if c == "Part":
        return np.array(p["part"])
    level, name = c.split(" ", 1)
    return np.array(p[("obj_" if level == "Obj" else "part_") + dict((b, a) for a, b in TYPES)[name]])


def md(df):
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |" for row in df.values])


def from_results(path):
    h = json.loads(Path(path).read_text())["hierarchical"]
    return dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"], **{f"obj_{k}": h["obj-attrs"][k] for k, _ in TYPES}, **{f"part_{k}": h["obj-part-attrs"][k] for k, _ in TYPES})


def fast(run):
    root, out = run / "eval", run / "eval/results"
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / "selection_frozen.json").read_text())
    lines = ["# Interaction ablation: official AP (point estimates; bootstrap and ROI size follow)", "", HEAD, ""]
    paths = dict(val={m: root / f"eval_val/{m}/ep{selection[m]['epoch']}/official_results.json" for m in MODELS}, test={m: root / "fast_test" / m / "official_results.json" for m in MODELS})
    result = {}
    for split in ("val", "test"):
        points = {m: from_results(paths[split][m]) for m in MODELS}
        result[split] = points
        table = pd.DataFrame([dict(Model=LABEL[m], epoch=selection[m]["epoch"], **{c: float(value(points[m], c)) for c in COLS}) for m in MODELS])
        delta = pd.DataFrame([dict(Contrast=f"{x} - {y}", **{c: f"{float(value(points[x], c) - value(points[y], c)):+.2f}" for c in COLS}) for x, y in PAIRS])
        lines += [f"## Official {split.capitalize()}", "", md(table), "", "### Contrasts (point estimates)", "", md(delta), ""]
    (out / "fast_ap.json").write_text(json.dumps(dict(selection=selection, **result), indent=1))
    (out / "fast_tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def summarise_training(run):
    rows = []
    logs = dict(A0="output/2026-11-08/hcf_material_causal/r384_cosine/P/logs/train.jsonl", A1=run / "A1/logs/train.jsonl", A2=run / "A2/logs/train.jsonl")
    for arch, path in logs.items():
        recs = [json.loads(x) for x in Path(path).read_text().splitlines()]
        first, last = recs[0], recs[-1]
        rows.append(dict(arch=arch, crop_cos_first=first["crop_cos_loss"], crop_cos_last=last["crop_cos_loss"], global_first=first["global_loss"], global_last=last["global_loss"], coarse_cos_first=first["coarse_cos_loss"], coarse_cos_last=last["coarse_cos_loss"], grad_coarse_last=last.get("grad_coarse"), grad_fine_last=last.get("grad_fine"), grad_interaction_last=last.get("grad_interaction"), grad_fusion_last=last.get("grad_fusion"), fine_contribution_last=last.get("fine_contribution"), peak_gib=max(r["mem_gb"] for r in recs), step_seconds=float(np.median([r["step_seconds"] for r in recs[1:]]))))
    return pd.DataFrame(rows)


def full(run):
    root, out = run / "eval", run / "eval/results"
    selection = json.loads((root / "selection_frozen.json").read_text())
    lines = ["# Coarse-Fine interaction ablation: PACO official AP", "", HEAD + " Paired image bootstrap B = 1000, seed 0. A0 fused is the reused r384_cosine result, same protocol.", ""]
    for split in ("val", "test"):
        boot = json.loads((root / f"boot_{split}.json").read_text())
        table = pd.DataFrame([dict(Model=LABEL[m], epoch=selection[m]["epoch"], **{c: float(value(boot["points"][m], c)) for c in COLS}) for m in MODELS])
        rows, csv = [], []
        for x, y in PAIRS:
            row = dict(Contrast=f"{x} - {y}")
            for c in COLS:
                d = float(value(boot["points"][x], c) - value(boot["points"][y], c))
                lo, hi = np.percentile(value(boot["boot"][x], c) - value(boot["boot"][y], c), [2.5, 97.5])
                row[c] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                csv.append(dict(split=split, contrast=f"{x} - {y}", metric=c, delta=d, lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0)))
            rows.append(row)
        table.to_csv(out / f"{split}_ap.csv", index=False)
        pd.DataFrame(csv).to_csv(out / f"{split}_contrasts.csv", index=False)
        size = pd.concat([pd.read_csv(out / f"roi_size_{split}_{m}.csv") for m in MODELS], ignore_index=True)
        size.to_csv(out / f"{split}_roi_size.csv", index=False)
        lines += [f"## Official {split.capitalize()}", "", md(table), "", "### Contrasts (95% CI)", "", md(pd.DataFrame(rows)), "", "### ROI size (GT-box area on the 384 canvas, point estimates)", ""]
        for level in ("Object", "Part"):
            for col in ("Pattern", "Material"):
                p = size[size.level == level].pivot(index="size_bin", columns="model", values=col).reindex(columns=MODELS).reindex(["tiny_lt4_tokens", "small_4to16_tokens", "medium_16to64_tokens", "large_ge64_tokens"])
                count = size[(size.level == level) & (size.model == MODELS[0])].set_index("size_bin").roi_count
                t = p.reset_index()
                t.insert(1, "ROIs", [int(count[s]) for s in p.index])
                t.columns = ["Size", "ROIs"] + MODELS
                lines += [f"**{level} {col}**", "", md(t), ""]
    train = summarise_training(run)
    train.to_csv(out / "training_summary.csv", index=False)
    lines += ["## Training", "", md(train), ""]
    eff = out / "efficiency.json"
    if eff.exists():
        lines += ["## Efficiency", "", "```json", eff.read_text(), "```", ""]
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    (fast if a.fast else full)(Path(a.run))


if __name__ == "__main__":
    main()
