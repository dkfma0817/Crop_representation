"""Tables for P1/P2: PACO Val project M2, dev-M0-selected epochs, paired image bootstrap B=1000 seed 0, ROI-size bins, conditioning diagnosis and training logs. Never reads Test."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.readout.common as RC
from src.ivc.common import DBINS, EXCL
from src.readout.evaluate import Boot, summarize

METRICS = ["overall", "object", "part"] + [f"{lv}/{t}" for lv in ("object", "part") for t in ("color", "pattern_marking", "material", "transparency")]
MAIN = ["object/pattern_marking", "object/material", "part/pattern_marking", "part/material"]
CAUSAL = Path("output/2026-11-08/hcf_material_causal")
DIRS = {"Initial": CAUSAL / "r384_initial/probe", "Cosine": CAUSAL / "r384_cosine/probe", "PixelDiT": CAUSAL / "r384_pixeldit_hybrid/probe", "C0": Path("output/2026-11-06/ivc_v3/probe_c0")}
MODELS = ["Initial", "Cosine", "PixelDiT", "P1", "P2", "C0"]
PAIRS = [("P1", "PixelDiT"), ("P2", "PixelDiT"), ("P2", "P1"), ("P1", "Initial"), ("P2", "Initial"), ("PixelDiT", "Initial"), ("P1", "Cosine"), ("P2", "Cosine"), ("C0", "PixelDiT"), ("C0", "P1"), ("C0", "P2"), ("C0", "Initial")]
SIZE_PAIRS = [("P1", "PixelDiT"), ("P2", "P1"), ("P2", "Initial"), ("C0", "P1"), ("C0", "P2")]


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def dev_epoch(directory):
    curve = [r for r in lines(directory / "curves_dev.jsonl") if r["epoch"] >= 1]
    return max(curve, key=lambda r: (r["dev"]["M0"]["overall"], -r["epoch"]))["epoch"]


def reader_dir(model, run, reader):
    if model == "C0":
        return DIRS["C0"] / ("r2" if reader == "r2fused" else reader)
    if model in DIRS:
        return DIRS[model] / reader
    return run / f"probe_{model}" / reader


def md(df):
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join("" if pd.isna(v) else f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |" for row in df.values])


def contrasts(samples, pairs, metrics):
    rows = []
    for x, y in pairs:
        for k in metrics:
            if k in samples[x] and k in samples[y]:
                s = summarize(samples[x][k], samples[y][k])
                rows.append(dict(comparison=f"{x} − {y}", metric=k, delta=s["point"], ci95_low=s["lo"], ci95_high=s["hi"], excludes_zero=bool(s["lo"] > 0 or s["hi"] < 0) if np.isfinite(s["lo"]) else False))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "results"
    out.mkdir(exist_ok=True)
    data = RC.load_data(EXCL)
    df, Y = data["val"]
    ytr = RC.ytr_by_level(data)
    boot = Boot(df, Y, ytr, nb=1000, seed=0)
    bins = {name: Boot(df, Y, ytr, nb=1000, seed=0, mask=(df.n_in.values >= lo) & (df.n_in.values <= hi)) for name, lo, hi in DBINS}
    scores, epochs = {}, {}
    for reader in ("r0", "r2fused"):
        for m in MODELS:
            directory = reader_dir(m, run, reader)
            ep = dev_epoch(directory)
            epochs[f"{m}/{reader}"] = ep
            s = np.load(directory / f"scores_val_ep{ep}.npy").astype(np.float32)
            assert s.shape == Y.shape, (m, reader, s.shape)
            scores[(m, reader)] = s
    results = dict(protocol="PACO Val project M2; dev M0 epoch; encoder seed 0; paired image bootstrap B=1000 seed 0; no Test", epochs=epochs)
    text = []
    for reader, title in (("r2fused", "R2 Cross-attention Reader"), ("r0", "Mean + Linear Probe")):
        samples = {m: boot.m2(scores[(m, reader)]) for m in MODELS}
        t = pd.DataFrame([dict(model=m, epoch=epochs[f"{m}/{reader}"], **{k: float(samples[m][k][0]) if k in samples[m] else float("nan") for k in METRICS}) for m in MODELS])
        c = contrasts(samples, PAIRS, METRICS)
        t.to_csv(out / f"m2_{reader}.csv", index=False)
        c.to_csv(out / f"m2_contrasts_{reader}.csv", index=False)
        results[f"m2/{reader}"] = dict(table=t.to_dict("records"), contrasts=c.to_dict("records"))
        text += [f"## PACO Val M2 — {title}", md(t), "", md(c[c.metric.isin(["overall", "object", "part"] + MAIN)]), ""]
        rows = []
        for name, bb in bins.items():
            bb_samples = {m: bb.m2(scores[(m, reader)]) for m in MODELS}
            for k in ["object", "part"] + MAIN:
                row = dict(reader=reader, bin=name, metric=k)
                for m in MODELS:
                    row[m] = float(bb_samples[m][k][0]) if k in bb_samples[m] else float("nan")
                for x, y in SIZE_PAIRS:
                    if k in bb_samples[x] and k in bb_samples[y]:
                        s = summarize(bb_samples[x][k], bb_samples[y][k])
                        row[f"{x}-{y}"] = s["point"]
                        row[f"{x}-{y} CI"] = f"[{s['lo']:+.2f}, {s['hi']:+.2f}]"
                rows.append(row)
        sz = pd.DataFrame(rows)
        sz.to_csv(out / f"size_bins_{reader}.csv", index=False)
        results[f"size/{reader}"] = sz.to_dict("records")
        text += [f"## ROI size bins — {title}", md(sz), ""]
    results["bootstrap"] = dict(B=1000, seed=0, n_images=int(df.image_id.nunique()), n_rows=len(df))
    diag = {c: json.loads((run / "diag" / f"{c}.json").read_text()) for c in ("B", "P1", "P2")}
    results["diag"] = diag
    rows = [dict(model=c, block=k, correct=v["correct"], zero=v["zero"], different=v["different"], zero_minus_correct=v["zero_minus_correct"]["point"], zero_ci=f"[{v['zero_minus_correct']['lo']:.4f}, {v['zero_minus_correct']['hi']:.4f}]", different_minus_correct=v["different_minus_correct"]["point"], different_ci=f"[{v['different_minus_correct']['lo']:.4f}, {v['different_minus_correct']['hi']:.4f}]") for c, d in diag.items() for k, v in d.items() if isinstance(v, dict) and "correct" in v]
    pd.DataFrame(rows).to_csv(out / "conditioning_dependency.csv", index=False)
    text += ["## Conditioning dependency (flow MSE, PACO Val, frozen head and encoder)", md(pd.DataFrame(rows)), ""]
    training = {}
    for c in ("P1", "P2"):
        log = pd.read_json(run / c / "logs/train.jsonl", lines=True)
        training[c] = dict(log=log.to_dict("records"), checks=lines(run / c / "logs/objective_checks.jsonl"), done=json.loads((run / c / "done.json").read_text()), config=json.loads((run / c / "config.json").read_text()))
    baseline = pd.read_json(CAUSAL / "r384_pixeldit_hybrid/P/logs/train.jsonl", lines=True)
    training["B"] = dict(log=baseline.to_dict("records"), checks=lines(CAUSAL / "r384_pixeldit_hybrid/P/logs/objective_checks.jsonl"), done=json.loads((CAUSAL / "r384_pixeldit_hybrid/P/done.json").read_text()), config=json.loads((CAUSAL / "r384_pixeldit_hybrid/P/config.json").read_text()))
    (out / "training_summary.json").write_text(json.dumps(training, default=float))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (out / "tables.md").write_text("\n".join(text))
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
