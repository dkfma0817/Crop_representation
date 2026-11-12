"""Tables for G1-Uniform / G1-ROI on PACO official Val and Test (GT box + GT category oracle attribute AP; epoch fixed on official Val). Overall = (Object + Part) / 2. No M0/M2.

  python -m src.g1up.report --run output/2026-11-09/g1_upscaled_teacher_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["Cosine_r2", "Hybrid_r2", "G1U_r2", "G1R_r2", "C0_r2"]
LABEL = {"Cosine_r2": "P384 Crop Distillation", "Hybrid_r2": "PixelDiT + Crop Cosine", "G1U_r2": "G1-Uniform", "G1R_r2": "G1-ROI", "C0_r2": "C0 Crop Oracle"}
PAIRS = [("G1U", "Cosine"), ("G1R", "Cosine"), ("G1U", "Hybrid"), ("G1R", "Hybrid"), ("G1R", "G1U"), ("C0", "G1U"), ("C0", "G1R"), ("C0", "Cosine")]
TYPES = [("color", "Color"), ("pattern_marking", "Pattern"), ("material", "Material"), ("transparency", "Transparency")]
COLS = ["Overall", "Object", "Part"] + [f"Obj {n}" for _, n in TYPES] + [f"Part {n}" for _, n in TYPES]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    root, out = run / "eval", run / "eval/results"
    selection = json.loads((root / "selection.json").read_text())
    lines = ["# G1 Upscaled Teacher: PACO official AP", "", "GT box + GT category oracle attribute AP. Epoch = max (Object + Part) / 2 on official Val, ties earlier; Test uses that epoch, once. Paired image bootstrap B = 1000, seed 0. Encoder seed 0 only.", ""]
    for split in ("val", "test"):
        boot = json.loads((root / f"boot_{split}.json").read_text())
        table = pd.DataFrame([dict(Model=LABEL[m], epoch=selection[m]["epoch"], **{c: float(value(boot["points"][m], c)) for c in COLS}) for m in MODELS])
        rows, csv = [], []
        for x, y in PAIRS:
            X, Y = f"{x}_r2", f"{y}_r2"
            row = dict(Contrast=f"{x} - {y}")
            for c in COLS:
                d = float(value(boot["points"][X], c) - value(boot["points"][Y], c))
                lo, hi = np.percentile(value(boot["boot"][X], c) - value(boot["boot"][Y], c), [2.5, 97.5])
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
                count = size[(size.level == level) & (size.model == "G1U_r2")].set_index("size_bin").roi_count
                t = p.reset_index()
                t.insert(1, "ROIs", [int(count[s]) for s in p.index])
                t.columns = ["Size", "ROIs"] + MODELS
                lines += [f"**{level} {col}**", "", md(t), ""]
    decision = json.loads((run / "g0/decision.json").read_text())
    lam = json.loads((run / "lambda.json").read_text())
    lines += ["## G0 (official Val, frozen DINOv3, ROIAlign 24 x 24 + R2)", "", "```json", json.dumps(decision, indent=1), "```", "", f"## lambda_grid = {lam['lambda_grid']} (median ||g_crop||/||g_grid|| = {lam['median_ratio']:.3f})", "", "## Training", ""]
    for c in ("G1U", "G1R"):
        rec = [json.loads(x) for x in (run / c / "logs/train.jsonl").read_text().splitlines()]
        last = [r for r in rec if r["step"] > 11000]
        checks = [json.loads(x) for x in (run / c / "logs/diagnostics.jsonl").read_text().splitlines()]
        done = json.loads((run / c / "done.json").read_text())
        lines += [f"**{c}** crop {np.mean([r['crop_cos_loss'] for r in last]):.4f}, grid {np.mean([r['grid_loss'] for r in last]):.4f}, global {np.mean([r['global_loss'] for r in last]):.4f}, step {np.median([r['step_seconds'] for r in rec[1:]]):.3f} s, peak {done['peak_gib']:.1f} GiB, wall {done['time_s']} s", "", "| step | grid loss | crop loss | matched cos | shifted cos | fine ratio | s_d | grad grid coarse | grad grid fine tx | grad grid fine proj | cos(crop,grid) | grid-read vs inference |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in checks:
            if r["step"] in (1, 2, 3, 3000, 6000, 9000, 12000):
                lines.append(f"| {r['step']} | {r['grid']:.4f} | {r['crop']:.4f} | {r['align_cos_matched']:.3f} | {r['align_cos_shifted_1cell']:.3f} | {r['fine_contribution_ratio']:.3f} | {r['s_d']:.3f} | {r['grad_grid_coarse']:.3f} | {r['grad_grid_fine_transformer']:.3f} | {r['grad_grid_fine_projection']:.3f} | {r['cos_crop_grid_total']:+.3f} | {r['grid_vs_inference_read_cos']:.3f} |")
        lines.append("")
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
