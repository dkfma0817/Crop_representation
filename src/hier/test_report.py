"""Official Test tables at the epochs fixed on official Val (R2 and Mean probe), paired image bootstrap B=1000. Used by the H0/H1 and S1/S2 follow-ups."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.siglip_extra.report import ATTRS, TYPES, flat
from src.siglip_pm.report import md

GT = Path("/home/irteam/datasets/paco/paco_lvis_v1_test.json")
COLUMNS = ["Model", "Overall AP", "Object AP", "Part AP"] + [f"{lv} {n}" for lv in ("Obj", "Part") for _, n in TYPES]


def load(path):
    return json.loads(Path(path).read_text())


def boot_key(col):
    if col == "Object AP":
        return "obj"
    if col == "Part AP":
        return "part"
    lv, n = col.split(" ", 1)
    return ("obj_" if lv == "Obj" else "part_") + dict((b, a) for a, b in TYPES)[n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="holds eval_test/<model>_r2 and boot_test.json")
    ap.add_argument("--mean-root", required=True, help="holds eval_test/<model>_linear and boot_test.json")
    ap.add_argument("--models", required=True)
    ap.add_argument("--pairs", required=True, help="x-y,x-y")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = a.models.split(",")
    pairs = [tuple(p.split("-")) for p in a.pairs.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = load(GT)
    by_id = {v["id"]: v["name"] for v in gt["attributes"]}
    categories = {v["id"]: v["name"] for v in gt["categories"]}
    results, lines = {}, []
    for key, root, name in (("r2", Path(a.root), "R2"), ("linear", Path(a.mean_root), "Mean + Linear")):
        boot = load(root / "boot_test.json")
        hier = {m: load(root / f"eval_test/{m}_{key}/test/official_results.json")["hierarchical"] for m in models}
        T = {}
        for m, h in hier.items():
            r = flat(m, h)
            r["Overall AP"] = (r["Object AP"] + r["Part AP"]) / 2
            T[m] = r
        table = pd.DataFrame(list(T.values()))[COLUMNS]
        rows = []
        for x, y in pairs:
            X, Y = boot["boot"][f"{x}_{key}"], boot["boot"][f"{y}_{key}"]
            for col in COLUMNS[1:]:
                z = (np.array(X["obj"]) + np.array(X["part"]) - np.array(Y["obj"]) - np.array(Y["part"])) / 2 if col == "Overall AP" else np.array(X[boot_key(col)]) - np.array(Y[boot_key(col)])
                lo, hi = np.percentile(z, [2.5, 97.5])
                rows.append(dict(comparison=f"{x} − {y}", metric=col, delta=float(T[x][col] - T[y][col]), ci95_low=float(lo), ci95_high=float(hi), excludes_zero=bool(lo > 0 or hi < 0), B=boot["B"]))
        contrasts = pd.DataFrame(rows)
        attr_rows = []
        for t, label in TYPES:
            for i in gt["attr_type_to_attr_idxs"][t]:
                for level, part, field in [("Object", False, "obj-attr-AP-per-class"), ("Part", True, "obj-attr-part-AP-per-class")]:
                    pos = [v for v in gt["annotations"] if i in v.get("attribute_ids", []) and (":" in categories[v["category_id"]]) == part]
                    attr_rows.append(dict(level=level, type=label, attribute=by_id[i], positive_ROIs=len(pos), **{m: hier[m][field].get(by_id[i], float("nan")) for m in models}))
        attr = pd.DataFrame(attr_rows)
        table.to_csv(out / f"test_ap_{key}.csv", index=False)
        contrasts.to_csv(out / f"test_contrasts_{key}.csv", index=False)
        attr.to_csv(out / f"test_attribute_ap_{key}.csv", index=False)
        focus = attr[(attr.level == "Object") & attr.attribute.isin(ATTRS)]
        results[key] = dict(ap=table.to_dict("records"), contrasts=contrasts.to_dict("records"), attributes=attr.to_dict("records"))
        lines += [f"## {name}: official Test AP", md(table), "", "## Contrasts (paired image bootstrap B=1000, 95% CI)", md(contrasts.drop(columns=["B"])), "", "## Six Object attributes (point estimates)", md(focus), ""]
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
