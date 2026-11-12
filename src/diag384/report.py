"""D1 / D2 tables: PACO official GT box + GT category oracle attribute AP only. Reflectance = the evaluator's `transparency` family. Overall = (Object + Part) / 2.

  --part d1|d2 --fast   point estimates from official_results.json -> <part>/eval/results/fast_tables.md
  --part d1|d2          bootstrap CI, ROI size (and D1 RGB) -> <part>/eval/results/tables.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = [("color", "Color"), ("pattern_marking", "Pattern"), ("material", "Material"), ("transparency", "Reflectance")]
COLS = ["Overall", "Object", "Part"] + [f"Obj {n}" for _, n in TYPES] + [f"Part {n}" for _, n in TYPES]
SPEC = dict(
    d1=dict(models=["B0_r2", "C1_r2", "C0_r2"], label=dict(B0_r2="B0 Full384", C1_r2="C1 Crop-from-384", C0_r2="C0 Native Crop"), pairs=[("C1_r2", "B0_r2"), ("C0_r2", "C1_r2"), ("C0_r2", "B0_r2")], title="Table 1: Input Bottleneck"),
    d2=dict(models=["A1_coarse", "A1_fine", "A1_r2"], label=dict(A1_coarse="A1 Coarse-only", A1_fine="A1 Fine-only", A1_r2="A1 Coarse+Fine"), pairs=[("A1_r2", "A1_coarse"), ("A1_fine", "A1_coarse"), ("A1_fine", "A1_r2")], title="Table 3: Representation / Readout Bottleneck"),
)
BINS = ["tiny_lt4_tokens", "small_4to16_tokens", "medium_16to64_tokens", "large_ge64_tokens"]
G0 = Path("output/2026-11-09/g1_upscaled_teacher_s0/g0")
HEAD = "PACO official GT Box + GT Category Oracle Attribute AP, R2 reader, seed 0. Epoch = max (Object + Part) / 2 on official Val, ties earlier; Test at that epoch, once. Reflectance = evaluator family `transparency`."


def from_results(path):
    h = json.loads(Path(path).read_text())["hierarchical"]
    return dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"], **{f"obj_{k}": h["obj-attrs"][k] for k, _ in TYPES}, **{f"part_{k}": h["obj-part-attrs"][k] for k, _ in TYPES})


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


def ap_table(points, spec, selection):
    return pd.DataFrame([dict(Model=spec["label"][m], epoch=selection[m]["epoch"], **{c: float(value(points[m], c)) for c in COLS}) for m in spec["models"]])


def contrast_table(points, spec):
    return pd.DataFrame([dict(Contrast=f"{x} - {y}", **{c: f"{float(value(points[x], c) - value(points[y], c)):+.2f}" for c in COLS}) for x, y in spec["pairs"]])


def fast(root, spec):
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / "selection_frozen.json").read_text())
    paths = dict(val={m: root / f"eval_val/{m}/ep{selection[m]['epoch']}/official_results.json" for m in spec["models"]}, test={m: root / "fast_test" / m / "official_results.json" for m in spec["models"]})
    lines = [f"# {spec['title']} (point estimates; bootstrap and ROI size follow)", "", HEAD, ""]
    result = {}
    for split in ("val", "test"):
        points = {m: from_results(paths[split][m]) for m in spec["models"]}
        result[split] = points
        lines += [f"## Official {split.capitalize()}", "", md(ap_table(points, spec, selection)), "", "### Contrasts (point estimates)", "", md(contrast_table(points, spec)), ""]
    (out / "fast_ap.json").write_text(json.dumps(dict(selection=selection, **result), indent=1))
    (out / "fast_tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def size_lines(out, spec, split):
    size = pd.concat([pd.read_csv(out / f"roi_size_{split}_{m}.csv") for m in spec["models"]], ignore_index=True)
    size.to_csv(out / f"{split}_roi_size.csv", index=False)
    lines = []
    for level in ("Object", "Part"):
        for col in ("Pattern", "Material"):
            p = size[size.level == level].pivot(index="size_bin", columns="model", values=col).reindex(columns=spec["models"]).reindex(BINS)
            count = size[(size.level == level) & (size.model == spec["models"][0])].set_index("size_bin").roi_count
            t = p.reset_index()
            t.insert(1, "ROIs", [int(count[s]) for s in BINS])
            for x, y in spec["pairs"]:
                t[f"{x} - {y}"] = (p[x] - p[y]).values
            lines += [f"**{level} {col}** (Table 4, {split})", "", md(t), ""]
    return lines, size


def rgb_lines(run, sizes):
    t = pd.read_csv(run / "d1/checks/rgb_table.csv")
    lines = ["## Table 2: RGB difference, C0 vs C1", "", "Final crop RGB in [0, 1], same output rectangle, content only (padding excluded). PSNR uses peak 1. `lt1px` = ROIs under one Full384 pixel on a side; `round_collapse` = ROIs that integer rounding of the box would collapse (the C1 path does not round). MSE / PSNR do not measure how much discriminative information was lost.", ""]
    cols = ["split", "level", "size_bin", "rois", "mse_mean", "psnr_of_mean_mse", "psnr_median", "full384_w_mean", "full384_h_mean", "lt1px", "lt2px", "round_collapse"]
    show = t[cols].sort_values(["split", "level", "size_bin"]).copy()
    show["mse_mean"] = [f"{v:.5f}" for v in show.mse_mean]
    lines += [md(show), ""]
    lines += ["### RGB difference next to Pattern AP gap", ""]
    for split, size in sizes.items():
        rows = []
        for level, lv in (("object", "Object"), ("part", "Part")):
            s = size[size.level == lv]
            p = s.pivot(index="size_bin", columns="model", values="Pattern")
            for name in BINS:
                r = t[(t.split == split) & (t.level == level) & (t.size_bin == name)].iloc[0]
                rows.append(dict(split=split, level=lv, size_bin=name, rois=int(r.rois), rgb_mse=f"{r.mse_mean:.5f}", psnr=r.psnr_of_mean_mse, pattern_C1_minus_B0=p.C1_r2[name] - p.B0_r2[name], pattern_C0_minus_C1=p.C0_r2[name] - p.C1_r2[name]))
        lines += [md(pd.DataFrame(rows)), ""]
    return lines


def full(run, part, root, spec):
    out = root / "results"
    selection = json.loads((root / "selection_frozen.json").read_text())
    lines = [f"# {spec['title']}", "", HEAD + " Paired image bootstrap B = 1000, seed 0.", ""]
    sizes = {}
    for split in ("val", "test"):
        boot = json.loads((root / f"boot_{split}.json").read_text())
        table = ap_table(boot["points"], spec, selection)
        rows, csv = [], []
        for x, y in spec["pairs"]:
            row = dict(Contrast=f"{x} - {y}")
            for c in COLS:
                d = float(value(boot["points"][x], c) - value(boot["points"][y], c))
                lo, hi = np.percentile(value(boot["boot"][x], c) - value(boot["boot"][y], c), [2.5, 97.5])
                row[c] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                csv.append(dict(split=split, contrast=f"{x} - {y}", metric=c, delta=d, lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0)))
            rows.append(row)
        table.to_csv(out / f"{split}_ap.csv", index=False)
        pd.DataFrame(csv).to_csv(out / f"{split}_contrasts.csv", index=False)
        size, sizes[split] = size_lines(out, spec, split)
        lines += [f"## Official {split.capitalize()}", "", md(table), "", "### Contrasts (95% CI)", "", md(pd.DataFrame(rows)), "", "### ROI size (GT-box area on the 384 canvas, point estimates)", ""] + size
    if part == "d1":
        lines += rgb_lines(run, sizes)
        s = json.loads((G0 / "selection.json").read_text())
        p = {n: from_results(G0 / f"eval_val/{n}/ep{s[n]['epoch']}/official_results.json") for n in ("Full384_r2", "Up768_r2")}
        c1 = from_results(root / f"eval_val/C1_r2/ep{selection['C1_r2']['epoch']}/official_results.json")
        t = pd.DataFrame([dict(Model="G0 Full384 (existing, ep5)", **{c: float(value(p["Full384_r2"], c)) for c in COLS}), dict(Model="G0 Up768 (existing, ep7)", **{c: float(value(p["Up768_r2"], c)) for c in COLS}), dict(Model=f"C1 Crop-from-384 (ep{selection['C1_r2']['epoch']})", **{c: float(value(c1, c)) for c in COLS})])
        t.loc[len(t)] = dict(Model="C1 - Up768", **{c: float(value(c1, c) - value(p["Up768_r2"], c)) for c in COLS})
        lines += ["## Q5: C1 vs existing Up768 (official Val only; Up768 has no Test evaluation)", "", md(t), ""]
    else:
        f = pd.read_csv(root.parent / "checks/fine_size.csv")
        lines += ["## Fine contribution size (Val, frozen A1, no training)", "", md(f), "", "`fine_check.json` is in `d2/checks/`.", ""]
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--part", required=True, choices=tuple(SPEC))
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    run, root = Path(a.run), Path(a.run) / a.part / "eval"
    if a.fast:
        fast(root, SPEC[a.part])
    else:
        full(run, a.part, root, SPEC[a.part])


if __name__ == "__main__":
    main()
