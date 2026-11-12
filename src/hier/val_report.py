"""Tables for H0/H1 on PACO official Val (GT-ROI / GT-category oracle attribute AP), paired image bootstrap B=1000. Reads no Test file and no project M2."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.siglip_extra.report import ATTRS, TYPES, flat
from src.siglip_pm.report import md

MODELS = ["P384", "H0", "H1", "C0"]
PAIRS = [("H0", "P384"), ("H1", "H0"), ("H1", "P384"), ("C0", "H1"), ("C0", "H0"), ("C0", "P384")]
GT = Path("/home/irteam/datasets/paco/paco_lvis_v1_val.json")


def load(path):
    return json.loads(Path(path).read_text())


def row(name, h):
    r = flat(name, h)
    r["Overall AP"] = (r["Object AP"] + r["Part AP"]) / 2
    return r


COLUMNS = ["Model", "Overall AP", "Object AP", "Part AP"] + [f"{lv} {n}" for lv in ("Obj", "Part") for _, n in TYPES]


def boot_key(col):
    if col == "Object AP":
        return "obj"
    if col == "Part AP":
        return "part"
    lv, n = col.split(" ", 1)
    return ("obj_" if lv == "Obj" else "part_") + dict((b, a) for a, b in TYPES)[n]


def contrast(x, y, X, Y, Bx, By, B):
    rows = []
    for col in COLUMNS[1:]:
        z = (np.array(Bx["obj"]) + np.array(Bx["part"]) - np.array(By["obj"]) - np.array(By["part"])) / 2 if col == "Overall AP" else np.array(Bx[boot_key(col)]) - np.array(By[boot_key(col)])
        lo, hi = np.percentile(z, [2.5, 97.5])
        rows.append(dict(comparison=f"{x} − {y}", metric=col, delta=float(X[col] - Y[col]), ci95_low=float(lo), ci95_high=float(hi), excludes_zero=bool(lo > 0 or hi < 0), B=B))
    return rows


def attributes(root, key):
    gt = load(GT)
    by_id = {v["id"]: v["name"] for v in gt["attributes"]}
    categories = {v["id"]: v["name"] for v in gt["categories"]}
    per = {m: load(root / f"eval_boot/{m}_{key}/val/official_results.json")["hierarchical"] for m in MODELS}
    rows = []
    for t, label in TYPES:
        for i in gt["attr_type_to_attr_idxs"][t]:
            for level, part, field in [("Object", False, "obj-attr-AP-per-class"), ("Part", True, "obj-attr-part-AP-per-class")]:
                pos = [a for a in gt["annotations"] if i in a.get("attribute_ids", []) and (":" in categories[a["category_id"]]) == part]
                r = dict(level=level, type=label, attribute=by_id[i], positive_ROIs=len(pos))
                for m in MODELS:
                    r[m] = per[m][field].get(by_id[i], float("nan"))
                rows.append(r)
    df = pd.DataFrame(rows)
    for x, y in (("H0", "P384"), ("H1", "H0"), ("H1", "P384"), ("C0", "H1")):
        df[f"{x}−{y}"] = df[x] - df[y]
    return df


def sizes(root, key):
    data = pd.concat([pd.read_csv(root / f"results/roi_size_{key}_{m}.csv") for m in MODELS], ignore_index=True)
    rows = []
    for x, y in (("H0", "P384"), ("H1", "H0"), ("H1", "P384"), ("C0", "H1")):
        for lv in ("Object", "Part"):
            for name in data.size_bin.unique():
                a = data[(data.model == x) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                b = data[(data.model == y) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                rows.append(dict(comparison=f"{x} − {y}", level=lv, size_bin=name, roi_count=int(a.roi_count), **{m: float(a[m] - b[m]) for m in ("Color", "Pattern", "Material", "Transparency")}))
    return data, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    root = run / "val_eval"
    out = root / "results"
    results, lines = {}, []
    for key, name in (("r2", "R2"), ("linear", "Mean + Linear")):
        boot = load(root / f"boot_val_{key}.json")
        T = {m: row(m, load(root / f"eval_boot/{m}_{key}/val/official_results.json")["hierarchical"]) for m in MODELS}
        table = pd.DataFrame(list(T.values()))[COLUMNS]
        c = pd.DataFrame([r for x, y in PAIRS for r in contrast(x, y, T[x], T[y], boot["boot"][f"{x}_{key}"], boot["boot"][f"{y}_{key}"], boot["B"])])
        attr = attributes(root, key)
        size_data, size_delta = sizes(root, key)
        ab_rows, ab_contrast = [], []
        for cond in ("H0", "H1"):
            normal = T[cond]
            ab_rows.append(dict(condition=f"{cond} normal", **{k: normal[k] for k in COLUMNS[1:]}))
            for mode in ("zero", "shuffle"):
                h = load(root / f"eval_boot/{cond}_{key}_{mode}/val/official_results.json")["hierarchical"]
                r = row(f"{cond} {mode}", h)
                ab_rows.append(dict(condition=f"{cond} {mode}", **{k: r[k] for k in COLUMNS[1:]}))
                ab_contrast += contrast(f"{cond} normal", f"{cond} {mode}", normal, r, boot["boot"][f"{cond}_{key}"], boot["boot"][f"{cond}_{key}_{mode}"], boot["B"])
        ablation, ablation_c = pd.DataFrame(ab_rows), pd.DataFrame(ab_contrast)
        for label, frame in (("ap", table), ("contrasts", c), ("attribute_ap", attr), ("roi_size_ap", size_data), ("roi_size_contrasts", size_delta), ("ablation_ap", ablation), ("ablation_contrasts", ablation_c)):
            frame.to_csv(out / f"val_{label}_{key}.csv", index=False)
        focus = attr[(attr.level == "Object") & attr.attribute.isin(ATTRS)][["attribute", "type", "positive_ROIs", "P384", "H0", "H1", "C0", "H0−P384", "H1−H0", "H1−P384", "C0−H1"]]
        results[key] = dict(ap=table.to_dict("records"), contrasts=c.to_dict("records"), attributes=attr.to_dict("records"), roi_size=size_data.to_dict("records"), roi_size_contrasts=size_delta.to_dict("records"), ablation=ablation.to_dict("records"), ablation_contrasts=ablation_c.to_dict("records"))
        lines += [f"## {name}: official Val AP", md(table), "", "## Contrasts (paired image bootstrap B=1000, 95% CI)", md(c.drop(columns=["B"])), "", "## Six Object attributes (point estimates)", md(focus), "",
                  "## ROI size", md(size_data), "", md(size_delta), "", "## Ablation", md(ablation), "", md(ablation_c.drop(columns=["B"])), ""]
    train = {}
    for c in ("H0", "H1"):
        log = pd.read_json(run / c / "logs/train.jsonl", lines=True)
        train[c] = dict(log=log.to_dict("records"), checks=[json.loads(x) for x in (run / c / "logs/grad_checks.jsonl").read_text().splitlines()], done=load(run / c / "done.json"), config=load(run / c / "config.json"), ablation=load(root / f"ablation/{c}/ablation_meta.json"))
    p384 = pd.read_json("output/2026-11-08/hcf_material_causal/r384_cosine/P/logs/train.jsonl", lines=True)
    train["P384"] = dict(log=p384.to_dict("records"), done=load("output/2026-11-08/hcf_material_causal/r384_cosine/P/done.json"))
    (out / "training_summary.json").write_text(json.dumps(train, default=float))
    results.update(selection=load(root / "selection.json"), efficiency=load(run / "results/efficiency.json"), input_audit=load(run / "input_audit.json"), design=load(run / "design.json"))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    lines += ["## Epoch selection", json.dumps(results["selection"], indent=1), "", "## Efficiency", json.dumps(results["efficiency"], indent=1)]
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
