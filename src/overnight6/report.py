"""Tables for the six pilots on PACO official Val and Test (GT box + GT category oracle attribute AP; epoch fixed on official Val). Overall = (Object + Part) / 2. No M0 / M2.
"Reflectance" = the official evaluator's `transparency` attribute type.

  --fast   point estimates of whatever is finished so far -> eval/results/fast_tables.md
  (none)   + paired bootstrap, ROI size, training and gradient summaries -> eval/results/tables.md, delta_ap.csv, verification/gradient_summary.json, top of README.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NEW = ["E2", "E2L", "E1a", "E1b", "F1", "S1"]
MODELS = [f"{n}_r2" for n in NEW] + ["A0_r2", "G1U_r2"]
LABEL = {"E2_r2": "E2 G1 Gradient Routing", "E2L_r2": "E2-L Lower Grid Weight", "E1a_r2": "E1a Native Crop768 Pooled", "E1b_r2": "E1b Native Crop768 Spatial", "F1_r2": "F1 Fine-focused Crop", "S1_r2": "S1 Small-ROI Balanced", "A0_r2": "A0 Crop384 (baseline)", "G1U_r2": "G1U (baseline)"}
BASELINE = dict(E2_r2="G1U_r2", E2L_r2="G1U_r2", E1a_r2="A0_r2", E1b_r2="A0_r2", F1_r2="A0_r2", S1_r2="A0_r2")
PAIRS = [("E2_r2", "G1U_r2"), ("E2L_r2", "G1U_r2"), ("E2_r2", "E2L_r2"), ("E1a_r2", "A0_r2"), ("E1b_r2", "E1a_r2"), ("E1b_r2", "A0_r2"), ("F1_r2", "A0_r2"), ("S1_r2", "A0_r2")]
TYPES = [("color", "Color"), ("pattern_marking", "Pattern"), ("material", "Material"), ("transparency", "Reflectance")]
COLS = ["Overall", "Object", "Part"] + [f"Obj {n}" for _, n in TYPES] + [f"Part {n}" for _, n in TYPES]
NOTE = "Official evaluator, GT box + GT category oracle attribute AP, R2 reader, encoder seed 0 (single run). Epoch = max (Object + Part) / 2 on official Val, ties earlier; Test once at that epoch. Reflectance = evaluator key `transparency`. Multiple comparisons are not corrected."


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


def selection(root):
    out = {}
    for m in MODELS:
        path = root / "selection" / f"{m}.json"
        if path.exists():
            out[m] = json.loads(path.read_text())
    return out


def point_tables(root):
    sel, res = selection(root), dict(val={}, test={})
    for m, s in sel.items():
        val = root / f"eval_val/{m}/ep{s['epoch']}/official_results.json"
        test = root / f"fast_test/{m}/official_results.json"
        if val.exists():
            res["val"][m] = from_results(val)
        if test.exists():
            res["test"][m] = from_results(test)
    return sel, res


def status_table(run, sel, res):
    rows = []
    state = run / "state"
    for n in NEW:
        m = f"{n}_r2"
        probe_state = "done" if (state / f"probe_{n}.done.json").exists() else "pending"
        val_ok, test_ok = (state / f"official_val_{m}_10.done.json").exists(), m in res["test"]
        rows.append(dict(Experiment=n, Encoder="done" if (state / f"train_{n}.done.json").exists() else ("failed" if (state / f"train_{n}.failed.json").exists() else "pending"), R2=probe_state, Official_Val="done" if val_ok else "pending", Val_Best_Epoch=sel.get(m, {}).get("epoch", "-"), Official_Test="done" if test_ok else "pending"))
    return pd.DataFrame(rows)


def ap_table(points, sel):
    return pd.DataFrame([dict(Model=LABEL[m], epoch=sel[m]["epoch"], **{c: float(value(points[m], c)) for c in COLS}) for m in MODELS if m in points])


def fast(run):
    root, out = run / "eval", run / "eval/results"
    out.mkdir(parents=True, exist_ok=True)
    sel, res = point_tables(root)
    lines = ["# Six pilots: official AP (point estimates of finished runs; bootstrap and ROI size follow)", "", NOTE, "", "## Status", "", md(status_table(run, sel, res)), ""]
    for split in ("val", "test"):
        if not res[split]:
            continue
        lines += [f"## Official {split.capitalize()}", "", md(ap_table(res[split], sel)), "", "### Delta vs baseline (point estimates)", ""]
        delta = [dict(Contrast=f"{x} - {y}", **{c: f"{float(value(res[split][x], c) - value(res[split][y], c)):+.2f}" for c in COLS}) for x, y in PAIRS if x in res[split] and y in res[split]]
        if delta:
            lines += [md(pd.DataFrame(delta)), ""]
    (out / "fast_ap.json").write_text(json.dumps(dict(selection=sel, **res), indent=1))
    tmp = out / "fast_tables.md.tmp"
    tmp.write_text("\n".join(lines))
    tmp.replace(out / "fast_tables.md")
    print("\n".join(lines), flush=True)


def gradient_summary(run):
    out = {}
    for n in NEW:
        path = run / n / "logs/diagnostics.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        summary = {}
        routed = [r["routing"] for r in rows if "routing" in r]
        if routed:
            summary["routing_checks"] = len(routed)
            summary["routing_blocked_changed_norm_max"] = max(r["blocked_changed_by_extra_norm"] for r in routed)
            summary["routing_fine_extra_norm_mean"] = float(np.mean([r["fine_extra_norm"] for r in routed]))
        for key in ("coarse_grid_over_crop_grad", "cos_crop_grid_coarse", "cos_crop_grid_fine_transformer", "cos_crop_spatial_coarse", "cos_crop_spatial_fine_transformer", "spatial_roi_fraction", "spatial_cells_mean", "small_roi_fraction"):
            vals = [r[key] for r in rows if key in r]
            if vals:
                summary[key + "_by_step"] = {str(r["step"]): r[key] for r in rows if key in r and r["step"] in (1, 3000, 6000, 9000, 12000)}
                summary[key + "_mean"] = float(np.mean(vals))
        out[n] = summary
    return out


def training_summary(run):
    rows = []
    for n in NEW:
        path = run / n / "logs/train.jsonl"
        if not path.exists():
            continue
        rec = [json.loads(x) for x in path.read_text().splitlines()]
        last = [r for r in rec if r["step"] > 11000]
        done = json.loads((run / n / "done.json").read_text()) if (run / n / "done.json").exists() else {}
        rows.append(dict(Experiment=n, crop_last=float(np.mean([r["crop_cos_loss"] for r in last])), crop_plain_last=float(np.mean([r["crop_cos_plain"] for r in last])), global_last=float(np.mean([r["global_loss"] for r in last])), grid_last=float(np.mean([r["grid_loss"] for r in last])), spatial_last=float(np.mean([r["spatial_loss"] for r in last])), step_seconds=float(np.median([r["step_seconds"] for r in rec[1:]])), time_s=done.get("time_s"), peak_gib=done.get("peak_gib")))
    return pd.DataFrame(rows)


def full(run):
    root, out = run / "eval", run / "eval/results"
    sel, res = point_tables(root)
    models = [m for m in MODELS if m in res["test"]]
    lines = ["# Six pilots: PACO official AP", "", NOTE + " Paired image bootstrap B = 1000, seed 0.", ""]
    deltas = []
    tables = {}
    for split in ("val", "test"):
        boot = json.loads((root / f"boot_{split}.json").read_text())
        table = pd.DataFrame([dict(Model=LABEL[m], epoch=sel[m]["epoch"], **{c: float(value(boot["points"][m], c)) for c in COLS}) for m in models])
        tables[split] = table
        rows = []
        for x, y in PAIRS:
            if x not in models or y not in models:
                continue
            row = dict(Contrast=f"{x} - {y}")
            for c in COLS:
                d = float(value(boot["points"][x], c) - value(boot["points"][y], c))
                lo, hi = np.percentile(value(boot["boot"][x], c) - value(boot["boot"][y], c), [2.5, 97.5])
                row[c] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                deltas.append(dict(split=split, contrast=f"{x} - {y}", metric=c, delta=d, lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0)))
            rows.append(row)
        table.to_csv(out / f"{split}_ap.csv", index=False)
        lines += [f"## Official {split.capitalize()}", "", md(table), "", "### Delta vs baseline (95% CI)", "", md(pd.DataFrame(rows)), ""]
        size = pd.concat([pd.read_csv(out / f"roi_size_{split}_{m}.csv") for m in models], ignore_index=True)
        size.to_csv(out / f"{split}_roi_size.csv", index=False)
        lines += ["### ROI size (GT-box area on the 384 canvas, tokens = 16 px cells; point estimates)", ""]
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
    train = training_summary(run)
    train.to_csv(out / "training_summary.csv", index=False)
    grad = gradient_summary(run)
    (run / "verification/gradient_summary.json").write_text(json.dumps(grad, indent=1))
    lines += ["## Training", "", md(train), "", "## Status", "", md(status_table(run, sel, res)), ""]
    (out / "tables.md").write_text("\n".join(lines))
    top = pd.DataFrame([dict(Experiment=n, Best_Epoch=sel[f"{n}_r2"]["epoch"], Val_Overall=float(value(res["val"][f"{n}_r2"], "Overall")), Test_Overall=float(value(res["test"][f"{n}_r2"], "Overall")), Test_Obj_Pattern=float(value(res["test"][f"{n}_r2"], "Obj Pattern")), Test_Part_Pattern=float(value(res["test"][f"{n}_r2"], "Part Pattern")), Baseline=BASELINE[f"{n}_r2"][:-3], Delta_Test_Overall=f"{float(value(res['test'][f'{n}_r2'], 'Overall') - value(res['test'][BASELINE[f'{n}_r2']], 'Overall')):+.2f}") for n in NEW if f"{n}_r2" in res["test"]])
    readme = run / "README.md"
    head = ["# Overnight six pilots", "", md(top), "", NOTE, "", "Full tables: [eval/results/tables.md](eval/results/tables.md). Bootstrap CIs and deltas: [eval/results/delta_ap.csv](eval/results/delta_ap.csv).", "", "(해석 섹션은 결과 검토 후 추가됩니다.)", ""]
    if not readme.exists() or "(해석 섹션은" in readme.read_text():
        readme.write_text("\n".join(head))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    (fast if a.fast else full)(Path(a.run))


if __name__ == "__main__":
    main()
