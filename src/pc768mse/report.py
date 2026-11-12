"""Tables for PC768 and M1 on PACO official Val and Test (GT box + GT category oracle attribute AP; epoch fixed on official Val). Overall = (Object + Part) / 2.

  --fast   point estimates of finished runs -> eval/results/fast_tables.md
  (none)   + paired bootstrap, ROI size, training / gradient / feature-norm summaries -> eval/results/tables.md, delta_ap.csv, verification/diagnostics_summary.json, provisional README.md
Pre-specified contrasts: PC768 - Hybrid, M1 - A0. PC768 - E1a (same Crop768 teacher without Pixel Flow) and Hybrid - A0 are exploratory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.overnight6.report import COLS, from_results, md, value

NEW = ["PC768", "M1"]
MODELS = ["PC768_r2", "M1_r2", "Hybrid_r2", "A0_r2", "E1a_r2"]
LABEL = {"PC768_r2": "P-C768 PixelDiT + Crop768", "M1_r2": "M1 Raw MSE Crop384", "Hybrid_r2": "PixelDiT Hybrid (baseline)", "A0_r2": "A0 Crop Cosine (baseline)", "E1a_r2": "E1a Crop768 Cosine (reference)"}
BASELINE = dict(PC768_r2="Hybrid_r2", M1_r2="A0_r2")
PAIRS = [("PC768_r2", "Hybrid_r2"), ("M1_r2", "A0_r2"), ("PC768_r2", "E1a_r2"), ("Hybrid_r2", "A0_r2")]
NOTE = "Official evaluator, GT box + GT category oracle attribute AP, R2 reader, encoder seed 0 (single run). Epoch = max (Object + Part) / 2 on official Val, ties earlier; Test once at that epoch. Transparency = evaluator key `transparency` (shown as Reflectance in the column names). Multiple comparisons are not corrected. PC768 - Hybrid and M1 - A0 are pre-specified; PC768 - E1a and Hybrid - A0 are exploratory."


def tables(root):
    sel, res = {}, dict(val={}, test={})
    for m in MODELS:
        path = root / "selection" / f"{m}.json"
        if not path.exists():
            continue
        sel[m] = json.loads(path.read_text())
        val, test = root / f"eval_val/{m}/ep{sel[m]['epoch']}/official_results.json", root / f"fast_test/{m}/official_results.json"
        if val.exists():
            res["val"][m] = from_results(val)
        if test.exists():
            res["test"][m] = from_results(test)
    return sel, res


def status(run, sel, res):
    s = run / "state"
    return pd.DataFrame([dict(Experiment=n, Encoder="done" if (s / f"train_{n}.done.json").exists() else "pending", R2="done" if (s / f"probe_{n}.done.json").exists() else "pending", Val_Best_Epoch=sel.get(f"{n}_r2", {}).get("epoch", "-"), Official_Test="done" if f"{n}_r2" in res["test"] else "pending") for n in NEW])


def ap_table(points, sel):
    return pd.DataFrame([dict(Model=LABEL[m], epoch=sel[m]["epoch"], **{c: float(value(points[m], c)) for c in COLS}) for m in MODELS if m in points])


def fast(run):
    root, out = run / "eval", run / "eval/results"
    out.mkdir(parents=True, exist_ok=True)
    sel, res = tables(root)
    lines = ["# P-C768 / M1: official AP (point estimates; bootstrap and ROI size follow)", "", NOTE, "", md(status(run, sel, res)), ""]
    for split in ("val", "test"):
        if res[split]:
            lines += [f"## Official {split.capitalize()}", "", md(ap_table(res[split], sel)), ""]
            delta = [dict(Contrast=f"{x} - {y}", **{c: f"{float(value(res[split][x], c) - value(res[split][y], c)):+.2f}" for c in COLS}) for x, y in PAIRS if x in res[split] and y in res[split]]
            if delta:
                lines += [md(pd.DataFrame(delta)), ""]
    (out / "fast_ap.json").write_text(json.dumps(dict(selection=sel, **res), indent=1))
    tmp = out / "fast_tables.md.tmp"
    tmp.write_text("\n".join(lines))
    tmp.replace(out / "fast_tables.md")
    print("\n".join(lines), flush=True)


def diagnostics(run):
    out = {}
    for n in NEW:
        path = run / n / "logs/diagnostics.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        keys = [k for k in rows[0] if k.startswith(("grad_", "cos_", "student_norm", "teacher", "norm_", "raw_mse", "mse", "weighted_mse", "crop", "flow", "glob", "conditioning"))]
        out[n] = {k: {str(r["step"]): r[k] for r in rows if r["step"] in (1, 3000, 6000, 9000, 12000) and k in r} for k in keys}
    return out


def training(run):
    rows = []
    for n in NEW:
        path = run / n / "logs/train.jsonl"
        if not path.exists():
            continue
        rec = [json.loads(x) for x in path.read_text().splitlines()]
        first, last = rec[0], [r for r in rec if r["step"] > 11000]
        avg = lambda k: float(np.mean([r[k] for r in last]))  # noqa: E731
        done = json.loads((run / n / "done.json").read_text())
        rows.append(dict(Experiment=n, crop_cos_first=first["crop_cos_loss"], crop_cos_last=avg("crop_cos_loss"), flow_last=avg("flow_loss"), raw_mse_first=first["raw_mse"], raw_mse_last=avg("raw_mse"), global_last=avg("global_loss"), student_norm_first=first["student_norm"], student_norm_last=avg("student_norm"), teacher_norm_last=avg("teacher_norm"), norm_absdiff_first=first["norm_absdiff"], norm_absdiff_last=avg("norm_absdiff"), cos_t768_t384=avg("cos_teacher768_teacher384") if n == "PC768" else None, step_seconds=float(np.median([r["step_seconds"] for r in rec[1:]])), time_s=done["time_s"]))
    return pd.DataFrame(rows)


def full(run):
    root, out = run / "eval", run / "eval/results"
    sel, res = tables(root)
    models = [m for m in MODELS if m in res["test"]]
    lines = ["# P-C768 / M1: PACO official AP", "", NOTE + " Paired image bootstrap B = 1000, seed 0.", ""]
    deltas = []
    for split in ("val", "test"):
        boot = json.loads((root / f"boot_{split}.json").read_text())
        table = pd.DataFrame([dict(Model=LABEL[m], epoch=sel[m]["epoch"], **{c: float(value(boot["points"][m], c)) for c in COLS}) for m in models])
        table.to_csv(out / f"{split}_ap.csv", index=False)
        rows = []
        for x, y in PAIRS:
            if x in models and y in models:
                row = dict(Contrast=f"{x} - {y}")
                for c in COLS:
                    d = float(value(boot["points"][x], c) - value(boot["points"][y], c))
                    lo, hi = np.percentile(value(boot["boot"][x], c) - value(boot["boot"][y], c), [2.5, 97.5])
                    row[c] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                    deltas.append(dict(split=split, contrast=f"{x} - {y}", metric=c, delta=d, lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0)))
                rows.append(row)
        lines += [f"## Official {split.capitalize()}", "", md(table), "", "### Delta (95% CI)", "", md(pd.DataFrame(rows)), ""]
        size = pd.concat([pd.read_csv(out / f"roi_size_{split}_{m}.csv") for m in models], ignore_index=True)
        size.to_csv(out / f"{split}_roi_size.csv", index=False)
        lines += ["### ROI size (GT-box area on the 384 canvas in 16 px tokens; point estimates)", ""]
        for level in ("Object", "Part"):
            for col in ("Pattern", "Material", "Color"):
                p = size[size.level == level].pivot(index="size_bin", columns="model", values=col).reindex(columns=models).reindex(["tiny_lt4_tokens", "small_4to16_tokens", "medium_16to64_tokens", "large_ge64_tokens"])
                count = size[(size.level == level) & (size.model == models[0])].set_index("size_bin").roi_count
                t = p.reset_index()
                t.insert(1, "ROIs", [int(count[s]) for s in p.index])
                t.columns = ["Size", "ROIs"] + [m[:-3] for m in models]
                lines += [f"**{level} {col}**", "", md(t), ""]
    pd.DataFrame(deltas).to_csv(out / "delta_ap.csv", index=False)
    (out / "official_ap.json").write_text(json.dumps(dict(selection=sel, **res), indent=1))
    train = training(run)
    train.to_csv(out / "training_summary.csv", index=False)
    (run / "verification/diagnostics_summary.json").write_text(json.dumps(diagnostics(run), indent=1))
    lines += ["## Training", "", md(train), "", md(status(run, sel, res)), ""]
    (out / "tables.md").write_text("\n".join(lines))
    readme = run / "README.md"
    if not readme.exists() or "(해석 섹션은" in readme.read_text():
        top = pd.DataFrame([dict(Method=n, Best_Epoch=sel[f"{n}_r2"]["epoch"], Val_Overall=float(value(res["val"][f"{n}_r2"], "Overall")), Test_Overall=float(value(res["test"][f"{n}_r2"], "Overall")), Test_Obj_Pattern=float(value(res["test"][f"{n}_r2"], "Obj Pattern")), Test_Part_Pattern=float(value(res["test"][f"{n}_r2"], "Part Pattern"))) for n in NEW if f"{n}_r2" in res["test"]])
        readme.write_text("\n".join(["# P-C768 / M1", "", md(top), "", NOTE, "", "Full tables: [eval/results/tables.md](eval/results/tables.md)", "", "(해석 섹션은 결과 검토 후 추가됩니다.)", ""]))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    (fast if a.fast else full)(Path(a.run))


if __name__ == "__main__":
    main()
