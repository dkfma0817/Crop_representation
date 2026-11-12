"""Small official-AP tools (run in the paco_eval env): G0 decision, paired bootstrap, ROI-size AP.

  python -m src.g1up.support decide --root <run>/g0
  python -m src.g1up.support boot --root <root> --names a,b --split val --out <json>
  python -m src.g1up.support size --split val --pred <npz> --model <name> --out <csv>
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


def decide(a):
    """G0 gate, fixed before any G0 result exists. Pattern = (Object + Part Pattern) / 2, Material likewise, official Val, epoch by max (Object + Part) / 2. PASS only if Up768 beats Full384 on both."""
    root = Path(a.root)
    selection = json.loads((root / "selection.json").read_text())
    rows = {}
    for name in ("Full384_r2", "Up768_r2"):
        e = selection[name]["epoch"]
        h = json.loads((root / f"eval_val/{name}/ep{e}/official_results.json").read_text())["hierarchical"]
        rows[name] = dict(epoch=e, overall=(h["obj-attr-AP"] + h["obj-attr-part-AP"]) / 2, object=h["obj-attr-AP"], part=h["obj-attr-part-AP"], pattern=(h["obj-attrs"]["pattern_marking"] + h["obj-part-attrs"]["pattern_marking"]) / 2, material=(h["obj-attrs"]["material"] + h["obj-part-attrs"]["material"]) / 2)
    gain = {k: rows["Up768_r2"][k] - rows["Full384_r2"][k] for k in ("overall", "object", "part", "pattern", "material")}
    result = dict(status="PASS" if gain["pattern"] > 0 and gain["material"] > 0 else "FAIL", rule="official Val, Up768 - Full384: mean(Object, Part) Pattern gain > 0 AND Material gain > 0", gain=gain, models=rows, note="Point estimates, seed 0 readers. A FAIL holds G1 until the user decides.")
    (root / "decision.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


def boot(a):
    import src.pacoattr.boot as B
    B.RUN = str(a.root)
    sys.argv = ["boot", "--split", a.split, "--conds", a.names, "--B", "1000", "--evaldir", "eval_boot", "--procs", "24", "--out", str(a.out)]
    B.main()


def size(a):
    from paco.evaluation.paco_eval_api import PACO, PACOEval, PACOResults
    from paco.evaluation.utils.paco_utils import heirachrichal_APs

    from src.pacoattr.official_eval import ANN, make_dets
    from src.siglip_pm.size_diagnostic import BINS
    M = pd.read_parquet(f"output/2026-11-07/paco_oracle_attr/data/manifest_{a.split}.parquet")
    z = np.load(a.pred)
    assert np.array_equal(z["ann_id"], M.ann_id.values)
    gt = PACO(ANN.format(a.split))
    dt = PACOResults(gt, make_dets(M, z["probs"].astype(np.float64)), max_dets=300)
    wh = {im["id"]: max(im["width"], im["height"]) for im in gt.dataset["images"]}
    for dataset in [gt, dt]:
        for ann in dataset.dataset["annotations"]:
            box = ann["bbox"]
            ann["area"] = box[2] * box[3] * (384 / wh[ann["image_id"]]) ** 2
    E = PACOEval(gt, dt, "bbox", attr_ap_type="usual")
    E.params.area_rng = [[lo, hi] for _, lo, hi in BINS]
    E.params.area_rng_lbl = [n for n, _, _ in BINS]
    E.evaluate()
    E.evaluate_joint_attr()
    E.accumulate_joint_attr()
    rows, categories = [], {v["id"]: v["name"] for v in gt.dataset["categories"]}
    for i, (name, lo, hi) in enumerate(BINS):
        h = heirachrichal_APs(SimpleNamespace(lvis_gt=gt, lvis_dt=dt, eval_joint_attr={"precision": E.eval_joint_attr["precision"][:, :, :, i:i + 1]}))
        for lv, key in [("Object", "obj-attrs"), ("Part", "obj-part-attrs")]:
            count = sum(lo <= ann["area"] <= hi and (":" in categories[ann["category_id"]]) == (lv == "Part") for ann in gt.dataset["annotations"])
            rows.append(dict(model=a.model, level=lv, size_bin=name, roi_count=count, Color=h[key]["color"], Pattern=h[key]["pattern_marking"], Material=h[key]["material"], Transparency=h[key]["transparency"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("decide", "boot", "size"))
    ap.add_argument("--root")
    ap.add_argument("--names")
    ap.add_argument("--split")
    ap.add_argument("--out")
    ap.add_argument("--pred")
    ap.add_argument("--model")
    a = ap.parse_args()
    dict(decide=decide, boot=boot, size=size)[a.cmd](a)


if __name__ == "__main__":
    main()
