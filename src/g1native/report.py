"""G1-Native tables on PACO official Val and Test (GT box + GT category oracle attribute AP; epoch fixed on official Val). No M0 / M2. CPU only.

  --g0     G0-Native768 vs the existing G0-Full384 / G0-Up768, official Val, written to g0/compare.md
  --fast   point estimates without bootstrap, written to eval/results/fast_tables.md
  (none)   full tables with image-level paired bootstrap 95% CI, ROI size, losses, alignment and efficiency, written to eval/results/tables.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

G1 = Path("/home/irteam/Crop_representation/output/2026-11-09/g1_upscaled_teacher_s0")
MODELS = ["Cosine_r2", "G1U_r2", "G1N_r2", "Hybrid_r2", "C0_r2"]
LABEL = {"Cosine_r2": "Crop Cosine P384", "G1U_r2": "G1U (Upscaled 768 Teacher)", "G1N_r2": "G1-Native (Native 768 Teacher)", "Hybrid_r2": "PixelDiT + Crop Cosine", "C0_r2": "C0 Crop Oracle"}
PAIRS = [("G1N", "G1U"), ("G1N", "Cosine"), ("G1U", "Cosine"), ("G1N", "Hybrid"), ("C0", "G1N"), ("C0", "G1U"), ("C0", "Cosine")]
TYPES = [("color", "Color"), ("pattern_marking", "Pattern"), ("material", "Material"), ("transparency", "Reflectance/Transparency")]
COLS = ["Overall", "Object", "Part"] + [f"Obj {n}" for _, n in TYPES] + [f"Part {n}" for _, n in TYPES]
PRIORITY = ["Overall", "Obj Pattern", "Obj Material", "Part Pattern", "Part Material"]
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


def g0(run):
    root = run / "g0"
    sel = json.loads((root / "selection.json").read_text())
    names = dict(Full384_r2="G0-Full384", Up768_r2="G0-Up768 (existing)", Native768_r2="G0-Native768")
    pts = {n: from_results(root / f"eval_val/{n}/ep{sel[n]['epoch']}/official_results.json") for n in names}
    table = pd.DataFrame([dict(Model=names[n], epoch=sel[n]["epoch"], **{c: float(value(pts[n], c)) for c in COLS}) for n in names])
    delta = pd.DataFrame([dict(Contrast=f"{names[x]} - {names[y]}", **{c: f"{float(value(pts[x], c) - value(pts[y], c)):+.2f}" for c in COLS}) for x, y in (("Native768_r2", "Up768_r2"), ("Native768_r2", "Full384_r2"), ("Up768_r2", "Full384_r2"))])
    text = "\n".join(["# G0: frozen DINOv3 teacher input (official Val, ROIAlign 24 x 24 + R2, same protocol)", "", "Verification only: this result does not change G1-Native.", "", md(table), "", md(delta), ""])
    (root / "compare.md").write_text(text)
    (root / "compare.json").write_text(json.dumps(dict(selection=sel, points=pts), indent=1))
    print(text, flush=True)


def fast(run):
    root, out = run / "eval", run / "eval/results"
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((root / "selection_frozen.json").read_text())
    paths = dict(val={m: root / f"eval_val/{m}/ep{sel[m]['epoch']}/official_results.json" for m in MODELS}, test={m: (root / "fast_test/G1N_r2/official_results.json" if m == "G1N_r2" else G1 / f"eval/eval_boot/{m}/test/official_results.json") for m in MODELS})
    lines, result = ["# G1-Native official AP (point estimates; bootstrap and ROI size follow)", "", HEAD, ""], {}
    for split in ("val", "test"):
        pts = {m: from_results(paths[split][m]) for m in MODELS}
        result[split] = pts
        table = pd.DataFrame([dict(Model=LABEL[m], epoch=sel[m]["epoch"], **{c: float(value(pts[m], c)) for c in COLS}) for m in MODELS])
        delta = pd.DataFrame([dict(Contrast=f"{x} - {y}", **{c: f"{float(value(pts[x + '_r2'], c) - value(pts[y + '_r2'], c)):+.2f}" for c in COLS}) for x, y in PAIRS])
        lines += [f"## Official {split.capitalize()}", "", md(table), "", "### Contrasts (point estimates)", "", md(delta), ""]
    (out / "fast_ap.json").write_text(json.dumps(dict(selection=sel, **result), indent=1))
    (out / "fast_tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def full(run):
    root, out = run / "eval", run / "eval/results"
    sel = json.loads((root / "selection_frozen.json").read_text())
    lines = ["# G1-Native: PACO official AP", "", HEAD + " Paired image bootstrap B = 1000, seed 0.", ""]
    for split in ("val", "test"):
        boot = json.loads((root / f"boot_{split}.json").read_text())
        table = pd.DataFrame([dict(Model=LABEL[m], epoch=sel[m]["epoch"], **{c: float(value(boot["points"][m], c)) for c in COLS}) for m in MODELS])
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
        size = pd.concat([pd.read_csv(out / f"roi_size_{split}_G1N_r2.csv")] + [pd.read_csv(G1 / f"eval/results/roi_size_{split}_{m}.csv") for m in MODELS if m != "G1N_r2"], ignore_index=True)
        size.to_csv(out / f"{split}_roi_size.csv", index=False)
        first = pd.DataFrame([r for r in rows if r["Contrast"] == "G1N - G1U"])[["Contrast"] + PRIORITY]
        lines += [f"## Official {split.capitalize()}", "", "**G1-Native - G1U (priority)**", "", md(first), "", md(table), "", "### Contrasts (95% CI)", "", md(pd.DataFrame(rows)), "", "### ROI size (GT-box area on the 384 canvas, point estimates)", ""]
        for level in ("Object", "Part"):
            for col in ("Pattern", "Material"):
                p = size[size.level == level].pivot(index="size_bin", columns="model", values=col).reindex(columns=MODELS).reindex(["tiny_lt4_tokens", "small_4to16_tokens", "medium_16to64_tokens", "large_ge64_tokens"])
                count = size[(size.level == level) & (size.model == "G1N_r2")].set_index("size_bin").roi_count
                t = p.reset_index()
                t.insert(1, "ROIs", [int(count[s]) for s in p.index])
                t.columns = ["Size", "ROIs"] + MODELS
                lines += [f"**{level} {col}**", "", md(t), ""]
    lines += ["## G0", "", (run / "g0/compare.md").read_text(), "", "## Training and alignment", ""]
    for name, path in (("G1N", run / "G1N"), ("G1U", G1 / "G1U")):
        rec = [json.loads(x) for x in (path / "logs/train.jsonl").read_text().splitlines()]
        last = [r for r in rec if r["step"] > 11000]
        done = json.loads((path / "done.json").read_text())
        lines += [f"**{name}** crop {np.mean([r['crop_cos_loss'] for r in last]):.4f}, grid {np.mean([r['grid_loss'] for r in last]):.4f}, global {np.mean([r['global_loss'] for r in last]):.4f}, step {np.median([r['step_seconds'] for r in rec[1:]]):.3f} s, peak {max(r['mem_gb'] for r in rec):.1f} GiB, wall {done['time_s']} s", "", "| step | grid | crop | matched cos | shifted cos | fine ratio | cos(crop,grid) coarse | grid-read vs inference |", "|---|---|---|---|---|---|---|---|"]
        for r in (json.loads(x) for x in (path / "logs/diagnostics.jsonl").read_text().splitlines()):
            if r["step"] in (1, 3000, 6000, 9000, 12000):
                lines.append(f"| {r['step']} | {r['grid']:.4f} | {r['crop']:.4f} | {r['align_cos_matched']:.3f} | {r['align_cos_shifted_1cell']:.3f} | {r['fine_contribution_ratio']:.3f} | {r['cos_crop_grid_coarse']:.3f} | {r['grid_vs_inference_read_cos']:.3f} |")
        lines.append("")
    for f in ("geometry/geometry_check.json", "ddp_decision.json", "eval/results/efficiency.json"):
        p = run / f
        if p.exists():
            obj = json.loads(p.read_text())
            obj = obj.get("summary", obj)
            lines += [f"## {f}", "", "```json", json.dumps(obj, indent=1), "```", ""]
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--g0", action="store_true")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    (g0 if a.g0 else fast if a.fast else full)(Path(a.run))


if __name__ == "__main__":
    main()
