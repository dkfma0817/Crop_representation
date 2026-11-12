"""Tables for H0/H1: PACO Val project M2, dev-M0-selected epochs, paired image bootstrap B=1000, ROI-size bins, ablations and training logs. Never reads Test."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.readout.common as RC
from src.hier.ablate import MODES, dev_epoch
from src.ivc.common import DBINS, EXCL
from src.readout.evaluate import Boot, summarize

METRICS = ["overall", "object", "part"] + [f"{lv}/{t}" for lv in ("object", "part") for t in ("color", "pattern_marking", "material", "transparency")]
MAIN = ["object/pattern_marking", "object/material", "part/pattern_marking", "part/material"]
P384 = Path("output/2026-11-08/hcf_material_causal/r384_cosine/probe")
C0 = Path("output/2026-11-06/ivc_v3/probe_c0")
PAIRS = [("H0", "P384"), ("H1", "H0"), ("H1", "P384"), ("C0", "H1"), ("C0", "H0"), ("C0", "P384")]


def md(df):
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join("" if pd.isna(v) else f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |" for row in df.values])


def reader_dir(model, run, reader):
    if model == "P384":
        return P384 / reader
    if model == "C0":
        return C0 / ("r2" if reader == "r2fused" else reader)
    return run / f"probe_{model}" / reader


def table(samples, metrics):
    return pd.DataFrame([dict(model=m, **{k: float(v[k][0]) if k in v else float("nan") for k in metrics}) for m, v in samples.items()])


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
    bins = {nm: Boot(df, Y, ytr, nb=1000, seed=0, mask=(df.n_in.values >= lo) & (df.n_in.values <= hi)) for nm, lo, hi in DBINS}
    models = ["P384", "H0", "H1", "C0"]
    scores, epochs = {}, {}
    for reader in ("r0", "r2fused"):
        for m in models:
            directory = reader_dir(m, run, reader)
            ep = dev_epoch(directory)
            epochs[f"{m}/{reader}"] = ep
            s = np.load(directory / f"scores_val_ep{ep}.npy").astype(np.float32)
            assert s.shape == Y.shape, (m, reader, s.shape)
            scores[(m, reader)] = s
    results = {}
    lines = []
    for reader in ("r2fused", "r0"):
        samples = {m: boot.m2(scores[(m, reader)]) for m in models}
        t = table(samples, METRICS)
        c = contrasts(samples, PAIRS, METRICS)
        t.to_csv(out / f"m2_{reader}.csv", index=False)
        c.to_csv(out / f"m2_contrasts_{reader}.csv", index=False)
        results[f"m2/{reader}"] = dict(table=t.to_dict("records"), contrasts=c.to_dict("records"))
        lines += [f"## PACO Val M2 ({'R2' if reader == 'r2fused' else 'Mean + Linear'})", md(t), "", md(c[c.metric.isin(["overall", "object", "part"] + MAIN)]), ""]
        size_rows = []
        for name, bb in bins.items():
            bb_samples = {m: bb.m2(scores[(m, reader)]) for m in models}
            for k in ["object", "part"] + MAIN:
                row = dict(reader=reader, bin=name, metric=k)
                for m in models:
                    row[m] = float(bb_samples[m][k][0]) if k in bb_samples[m] else float("nan")
                for x, y in (("H0", "P384"), ("H1", "H0"), ("H1", "P384"), ("C0", "H1")):
                    if k in bb_samples[x] and k in bb_samples[y]:
                        s = summarize(bb_samples[x][k], bb_samples[y][k])
                        row[f"{x}-{y}"] = s["point"]
                        row[f"{x}-{y} CI"] = f"[{s['lo']:+.2f}, {s['hi']:+.2f}]"
                size_rows.append(row)
        sz = pd.DataFrame(size_rows)
        sz.to_csv(out / f"size_bins_{reader}.csv", index=False)
        results[f"size/{reader}"] = sz.to_dict("records")
        lines += [f"## ROI size bins (n_in, {reader})", md(sz), ""]
    ablation = {}
    for cond in ("H0", "H1"):
        rep = json.loads((run / "ablation" / cond / "ablation.json").read_text())
        ablation[cond] = rep
        for reader in ("r2fused", "r0"):
            samples = {mode: boot.m2(np.load(run / "ablation" / cond / f"{mode}_{reader}.npy").astype(np.float32)) for mode in MODES}
            c = contrasts(samples, [("normal", "zero"), ("normal", "shuffle")], METRICS)
            t = table(samples, METRICS)
            t.to_csv(out / f"ablation_{cond}_{reader}.csv", index=False)
            c.to_csv(out / f"ablation_contrasts_{cond}_{reader}.csv", index=False)
            ablation[f"{cond}/{reader}/table"] = t.to_dict("records")
            ablation[f"{cond}/{reader}/contrasts"] = c.to_dict("records")
            lines += [f"## Ablation {cond} ({reader})", md(t), "", md(c[c.metric.isin(["overall"] + MAIN)]), ""]
    results["ablation"] = ablation
    train = {}
    for c in ("H0", "H1"):
        log = pd.read_json(run / c / "logs/train.jsonl", lines=True)
        train[c] = dict(log=log.to_dict("records"), checks=[json.loads(x) for x in (run / c / "logs/grad_checks.jsonl").read_text().splitlines()], done=json.loads((run / c / "done.json").read_text()), config=json.loads((run / c / "config.json").read_text()))
    p384 = pd.read_json("output/2026-11-08/hcf_material_causal/r384_cosine/P/logs/train.jsonl", lines=True)
    train["P384"] = dict(log=p384.to_dict("records"), done=json.loads(Path("output/2026-11-08/hcf_material_causal/r384_cosine/P/done.json").read_text()))
    (out / "training_summary.json").write_text(json.dumps(train, default=float))
    results["epochs"] = epochs
    results["bootstrap"] = dict(B=1000, seed=0, n_images=int(df.image_id.nunique()), n_rows=len(df))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
