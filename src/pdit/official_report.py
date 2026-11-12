"""Tables for P1 / P2 on PACO official Val (GT box + GT category oracle attribute AP). Overall = (Object AP + Part AP) / 2. Reads no Test file.

  python -m src.pdit.official_report --run output/2026-11-09/pixeldit_cond_p1p2_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["Initial", "Cosine", "Hybrid", "P1", "P2", "C0"]
LABEL = dict(Initial="Initial Coarse-Fine P384", Cosine="Crop Cosine P384", Hybrid="PixelDiT + Crop Cosine P384", P1="P1 Dense ROI Memory", P2="P2 Spatial Conditioning", C0="C0 Crop Oracle")
PAIRS = [("P1", "Hybrid"), ("P2", "P1"), ("P2", "Hybrid"), ("P1", "Initial"), ("P2", "Initial"), ("Hybrid", "Initial"), ("Cosine", "Initial"), ("C0", "P1"), ("C0", "P2"), ("C0", "Hybrid"), ("C0", "Initial")]
TYPES = [("color", "Color"), ("pattern_marking", "Pattern"), ("material", "Material"), ("transparency", "Transparency")]
COLUMNS = ["Overall", "Object", "Part"] + [f"Obj {n}" for _, n in TYPES] + [f"Part {n}" for _, n in TYPES]


def load(path):
    return json.loads(Path(path).read_text())


def boot_values(sample, col):
    if col == "Overall":
        return (np.array(sample["obj"]) + np.array(sample["part"])) / 2
    if col == "Object":
        return np.array(sample["obj"])
    if col == "Part":
        return np.array(sample["part"])
    level, name = col.split(" ", 1)
    return np.array(sample[("obj_" if level == "Obj" else "part_") + dict((b, a) for a, b in TYPES)[name]])


def point(p, col):
    if col == "Overall":
        return (p["obj"] + p["part"]) / 2
    if col == "Object":
        return p["obj"]
    if col == "Part":
        return p["part"]
    level, name = col.split(" ", 1)
    return p[("obj_" if level == "Obj" else "part_") + dict((b, a) for a, b in TYPES)[name]]


def md(df):
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |" for row in df.values])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    root = Path(a.run) / "official_val"
    out = root / "results"
    selection = load(root / "selection.json")
    lines = ["# PACO official Val (GT box + GT category oracle attribute AP)", "", "Epoch = max (Object + Part) / 2 on official Val, ties earlier. Paired image bootstrap B = 1000, seed 0. Overall = (Object + Part) / 2. Encoder seed 0 only. No Test.", ""]
    results = {}
    for key, name in (("r2", "R2 Cross-attention Reader"), ("linear", "Mean + Linear Probe")):
        boot = load(root / f"boot_val_{key}.json")
        table = pd.DataFrame([dict(Model=LABEL[m], epoch=selection[f"{m}_{key}"]["epoch"], **{c: float(point(boot["points"][f"{m}_{key}"], c)) for c in COLUMNS}) for m in MODELS])
        rows = []
        for x, y in PAIRS:
            X, Y = boot["boot"][f"{x}_{key}"], boot["boot"][f"{y}_{key}"]
            row = dict(Contrast=f"{x} - {y}")
            for c in COLUMNS:
                d = float(point(boot["points"][f"{x}_{key}"], c) - point(boot["points"][f"{y}_{key}"], c))
                lo, hi = np.percentile(boot_values(X, c) - boot_values(Y, c), [2.5, 97.5])
                row[c] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                rows.append(dict(reader=key, contrast=f"{x} - {y}", metric=c, delta=d, lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0)))
            results.setdefault(key, []).append(row)
        table.to_csv(out / f"val_ap_{key}.csv", index=False)
        pd.DataFrame([r for r in rows]).to_csv(out / f"val_contrasts_{key}.csv", index=False)
        contrast = pd.DataFrame(results[key])
        size = pd.concat([pd.read_csv(out / f"roi_size_{m}_{key}.csv") for m in MODELS], ignore_index=True)
        size.to_csv(out / f"val_roi_size_{key}.csv", index=False)
        lines += [f"## {name}", "", md(table), "", "### Contrasts (95% CI)", "", md(contrast), "", "### ROI size (GT-box area on the 384 canvas; existing bins, point estimates)", ""]
        for level in ("Object", "Part"):
            for col in ("Pattern", "Material"):
                p = size[size.level == level].pivot(index="size_bin", columns="model", values=col).reindex(columns=[f"{m}_{key}" for m in MODELS])
                count = size[(size.level == level) & (size.model == f"P1_{key}")].set_index("size_bin").roi_count
                p = p.reindex(["tiny_lt4_tokens", "small_4to16_tokens", "medium_16to64_tokens", "large_ge64_tokens"])
                t = p.reset_index()
                t.insert(1, "ROIs", [int(count[s]) for s in p.index])
                t.columns = ["Size", "ROIs"] + MODELS
                lines += [f"**{level} {col}**", "", md(t), ""]
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
