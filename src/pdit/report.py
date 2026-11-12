"""CPU tables for P1 / P2: PACO Val project M2 (R2 and Mean + Linear), epoch = max dev M0 overall (ties earlier), paired image bootstrap B = 1000 seed 0.

The size bins are the existing M2 `size` column (GT-box tokens on the DINOv2 336/14 grid: 1-4, 5-16, 17+). A size / level group without enough eligible units stays NaN and is not imputed;
for those cells the table also gives the pooled per-attribute AP (M0 style), labelled as such.

  OMP_NUM_THREADS=1 python -m src.pdit.report --run output/2026-11-09/pixeldit_cond_p1p2_s0
"""
import argparse
import csv
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import src.readout.common as RC
from src.ivc.common import EXCL
from src.readout.evaluate import Boot, summarize

R = Path("output/2026-11-08/hcf_material_causal")
ORDER = ["Initial", "Cosine", "Hybrid", "P1", "P2", "C0"]
LABEL = dict(Initial="Initial Coarse-Fine P384", Cosine="Crop Cosine P384", Hybrid="PixelDiT + Crop Cosine P384", P1="P1 Dense ROI Memory", P2="P2 Spatial Conditioning", C0="C0 Crop Oracle")
METRICS = ["overall", "object", "part", "object/color", "object/pattern_marking", "object/material", "object/transparency", "part/color", "part/pattern_marking", "part/material", "part/transparency"]
CONTRASTS = [("P1", "Hybrid"), ("P2", "P1"), ("P2", "Hybrid"), ("P1", "Initial"), ("P2", "Initial"), ("Hybrid", "Initial"), ("Cosine", "Initial"), ("C0", "P1"), ("C0", "P2"), ("C0", "Hybrid"), ("C0", "Initial")]
SIZES = ["1-4", "5-16", "17+"]
SIZE_KEYS = ["object/pattern_marking", "object/material", "part/pattern_marking", "part/material"]
G = {}


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def dirs(run):
    return {"Initial": R / "r384_initial/probe", "Cosine": R / "r384_cosine/probe", "Hybrid": R / "r384_pixeldit_hybrid/probe", "P1": run / "P1/probe", "P2": run / "P2/probe", "C0": Path("output/2026-11-06/ivc_v3/probe_c0")}


def reader_dir(root, model, kind):
    return root / ("r0" if kind == "mean" else ("r2" if model == "C0" else "r2fused"))


def task(args):
    boot, model, kind, path = args
    scores = np.load(path).astype(np.float32)
    assert scores.shape == G["Y"].shape, (path, scores.shape)
    b = G["boots"][boot]
    m2 = b.m2(scores)
    m0 = b.m0(scores) if boot != "all" else None
    return boot, model, kind, m2, m0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "results"
    out.mkdir(exist_ok=True)
    data = RC.load_data(EXCL)
    df, Y = data["val"]
    ytr = RC.ytr_by_level(data)
    G["Y"] = Y
    G["boots"] = {"all": Boot(df, Y, ytr, nb=1000, seed=0)}
    for s in SIZES:
        G["boots"][s] = Boot(df, Y, ytr, nb=1000, seed=0, mask=df["size"].values == s)
    root = dirs(run)
    selected, tasks = {}, []
    for model in ORDER:
        cfg = json.loads((root[model] / "config.json").read_text())
        assert cfg["world"] == 1 and cfg["n_train_images"] == 37472 and cfg["n_train_rows"] == 200088 and cfg["n_val_rows"] == 11109 and cfg["total_steps"] == 15594, model
        frozen = json.loads((root[model] / "frozen_check.json").read_text())
        assert frozen["unchanged"], model
        for kind in ("r2", "mean"):
            d = reader_dir(root[model], model, kind)
            dev = [r for r in lines(d / "curves_dev.jsonl") if r["epoch"] >= 1]
            assert len(dev) == 10, (model, kind, len(dev))
            epoch = max(dev, key=lambda r: (r["dev"]["M0"]["overall"], -r["epoch"]))["epoch"]
            val = next(r for r in lines(d / "curves.jsonl") if r["epoch"] == epoch)
            selected[(model, kind)] = dict(epoch=epoch, dev_M0=next(r for r in dev if r["epoch"] == epoch)["dev"]["M0"]["overall"], logged=val["val"]["M2"])
            for boot in ["all"] + SIZES:
                tasks.append((boot, model, kind, str(d / f"scores_val_ep{epoch}.npy")))
    with get_context("fork").Pool(a.procs) as pool:
        result = pool.map(task, tasks, chunksize=1)
    S = {(b, m, k): (m2, m0) for b, m, k, m2, m0 in result}
    report = dict(protocol="PACO Val project M2 (11,109 rows, 2,051 images); epoch = max dev M0 overall, ties earlier; paired image bootstrap B=1000 seed 0; encoder seed 0 only. No Test.", selected={f"{m}/{k}": v for (m, k), v in selected.items()}, n_size_rows={s: {lv: int(((df["size"] == s) & (df.level == lv)).sum()) for lv in ("object", "part")} for s in SIZES})
    tables = {}
    rows = []
    for kind in ("r2", "mean"):
        table = {}
        for model in ORDER:
            m2 = S[("all", model, kind)][0]
            for k in METRICS:
                assert abs(m2[k][0] - selected[(model, kind)]["logged"][k]) < 1e-5, (model, kind, k, m2[k][0], selected[(model, kind)]["logged"][k])
            table[model] = {k: summarize(m2[k]) for k in METRICS}
            rows += [dict(reader=kind, model=model, metric=k, value=table[model][k]["point"], lo=table[model][k]["lo"], hi=table[model][k]["hi"], epoch=selected[(model, kind)]["epoch"]) for k in METRICS]
        cmp_ = {f"{x}-{y}": {k: summarize(S[("all", x, kind)][0][k], S[("all", y, kind)][0][k]) for k in METRICS} for x, y in CONTRASTS}
        closure = {f"{x}_closure_over_Initial": {k: float((S[("all", x, kind)][0][k][0] - S[("all", "Initial", kind)][0][k][0]) / (S[("all", "C0", kind)][0][k][0] - S[("all", "Initial", kind)][0][k][0])) for k in METRICS} for x in ("Hybrid", "P1", "P2")}
        tables[kind] = dict(models=table, contrasts=cmp_, closure=closure)
    report["tables"] = tables
    size = {}
    for kind in ("r2", "mean"):
        size[kind] = {}
        for s in SIZES:
            size[kind][s] = {}
            for model in ORDER:
                m2, m0 = S[(s, model, kind)]
                size[kind][s][model] = {k: float(m2[k][0]) if k in m2 else float("nan") for k in SIZE_KEYS} | {"M0/" + k: float(m0[k.split("/")[-1]][0]) if m0 and k.split("/")[-1] in m0 else float("nan") for k in SIZE_KEYS}
            size[kind][s]["contrasts"] = {f"{x}-{y}": {k: summarize(S[(s, x, kind)][0][k], S[(s, y, kind)][0][k]) if k in S[(s, x, kind)][0] else dict(point=float("nan"), lo=float("nan"), hi=float("nan")) for k in SIZE_KEYS} for x, y in [("P1", "Hybrid"), ("P2", "P1"), ("P2", "Initial"), ("P1", "Initial")]}
    report["size"] = size
    train = {}
    for name, path, done in [("Hybrid", R / "r384_pixeldit_hybrid/P/logs/train.jsonl", R / "r384_pixeldit_hybrid/P/done.json"), ("P1", run / "P1/logs/train.jsonl", run / "P1/done.json"), ("P2", run / "P2/logs/train.jsonl", run / "P2/done.json")]:
        rec = [r for r in lines(path) if r["step"] > 1]
        last = [r for r in rec if r["step"] > 11000]
        row = dict(flow_last1k=float(np.mean([r["method_loss"] for r in last])), crop_last1k=float(np.mean([r["crop_cos_loss"] for r in last])), global_last1k=float(np.mean([r["global_loss"] for r in last])), step_seconds_median=float(np.median([r["step_seconds"] for r in rec])), peak_gib=max(r["mem_gb"] for r in rec), wall_s=json.loads(done.read_text())["time_s"])
        for g in ("grad_encoder", "grad_fine", "grad_head"):
            row[g + "_mean"] = float(np.mean([r[g] for r in rec])) if g in rec[0] else None
        train[name] = row
    report["training"] = train
    report["objective_checks"] = {name: lines(p) for name, p in [("Hybrid", R / "r384_pixeldit_hybrid/P/logs/objective_checks.jsonl"), ("P1", run / "P1/logs/objective_checks.jsonl"), ("P2", run / "P2/logs/objective_checks.jsonl")]}
    report["diag"] = {name: json.loads((run / f"diag/{f}.json").read_text()) for name, f in [("Hybrid", "baseline"), ("P1", "P1"), ("P2", "P2")]}
    report["efficiency"] = json.loads((run / "efficiency.json").read_text())
    (out / "report.json").write_text(json.dumps(report, indent=1))
    with (out / "paco_val_m2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = [f"# Tables (auto)\n\n{report['protocol']}\n"]
    for kind, name in (("r2", "R2 Cross-attention Reader"), ("mean", "Mean + Linear Probe")):
        t = tables[kind]
        md.append(f"\n## {name}\n\n| Model | epoch | " + " | ".join(METRICS) + " |\n|---|---|" + "---|" * len(METRICS))
        for m in ORDER:
            md.append(f"| {LABEL[m]} | {selected[(m, kind)]['epoch']} | " + " | ".join(f"{t['models'][m][k]['point']:.2f}" for k in METRICS) + " |")
        md.append("\n| Contrast | " + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS))
        for c, row in t["contrasts"].items():
            md.append(f"| {c} | " + " | ".join(f"{row[k]['point']:+.2f} [{row[k]['lo']:+.2f}, {row[k]['hi']:+.2f}]" for k in METRICS) + " |")
        md.append("\n| Share of C0 - Initial recovered | " + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS))
        for c, row in t["closure"].items():
            md.append(f"| {c} | " + " | ".join(f"{row[k]:.2f}" for k in METRICS) + " |")
        md.append(f"\n### {name}: ROI size (existing M2 `size` bins, tokens on 336/14)\n")
        for s in SIZES:
            md.append(f"\n**{s}** rows {report['n_size_rows'][s]}\n\n| Model | " + " | ".join(SIZE_KEYS) + " | M0: pattern / material |\n|---|" + "---|" * (len(SIZE_KEYS) + 1))
            for m in ORDER:
                v = size[kind][s][m]
                md.append(f"| {LABEL[m]} | " + " | ".join(f"{v[k]:.2f}" for k in SIZE_KEYS) + f" | {v['M0/object/pattern_marking']:.2f} / {v['M0/object/material']:.2f} |")
            md.append("\n| Contrast | " + " | ".join(SIZE_KEYS) + " |\n|---|" + "---|" * len(SIZE_KEYS))
            for c, row in size[kind][s]["contrasts"].items():
                md.append(f"| {c} | " + " | ".join(f"{row[k]['point']:+.2f} [{row[k]['lo']:+.2f}, {row[k]['hi']:+.2f}]" for k in SIZE_KEYS) + " |")
    md.append("\n## Training\n\n| Model | flow (last 1k) | crop | global | grad encoder | grad fine | grad head | step s | peak GiB | wall s |\n|---|---|---|---|---|---|---|---|---|---|")
    for n, r in train.items():
        md.append(f"| {n} | {r['flow_last1k']:.4f} | {r['crop_last1k']:.4f} | {r['global_last1k']:.4f} | {r['grad_encoder_mean']} | {r['grad_fine_mean']} | {r['grad_head_mean']} | {r['step_seconds_median']:.3f} | {r['peak_gib']:.1f} | {r['wall_s']} |")
    md.append("\n## Conditioning dependency (dev GT ROIs, frozen student + head)\n")
    for n, d in report["diag"].items():
        o = d["overall"]
        md.append(f"\n**{n}** overall A {o['A_correct']:.4f}, B zero {o['B_zero']:.4f}, C other {o['C_other_image']:.4f}, B-A {o['B_minus_A']:+.4f} {o['B_minus_A_ci']}, C-A {o['C_minus_A']:+.4f} {o['C_minus_A_ci']}\n\n| t | A | B | C | B-A | C-A |\n|---|---|---|---|---|---|")
        for b, v in d["per_bin"].items():
            md.append(f"| {b} | {v['A_correct']:.4f} | {v['B_zero']:.4f} | {v['C_other_image']:.4f} | {v['B_minus_A']:+.4f} | {v['C_minus_A']:+.4f} |")
    md.append("\n## Efficiency\n\n```json\n" + json.dumps(report["efficiency"], indent=1) + "\n```\n")
    (out / "tables.md").write_text("\n".join(md))
    print("\n".join(md[:60]), flush=True)


if __name__ == "__main__":
    main()
