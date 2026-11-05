"""G1-G4 example pairs by the pre-registered rule (configs/ivc/sample_rois.json: group_example_rule): reader R2 (and R0),
POSITIVE-label ROI-attribute pairs, per group 3 pairs drawn uniformly with numpy default_rng(0) (one generator, groups in
order G1..G4) from the group's pairs in (row, attr) order. Writes <run>/group_examples_<reader>.json.

  python -m src.ivc.group_examples --run output/2026-11-05/ivc --reader r2
"""
import argparse
import json
import os

import numpy as np

import src.readout.common as RC
from src.config import PACO_DIR
from src.ivc.common import EXCL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reader", default="r2")
    a = ap.parse_args()
    dva, Yva = RC.load_data(EXCL, splits=("val",))["val"]
    z = np.load(os.path.join(a.run, f"groups_{a.reader}.npz"))
    assert (z["ann_id"] == dva.ann_id.values).all()
    names = {x["id"]: x["name"] for x in json.load(open(os.path.join(PACO_DIR, "paco_lvis_v1_val.json")))["attributes"]}
    rng = np.random.default_rng(0)
    out = []
    for g in (1, 2, 3, 4):
        pairs = np.argwhere((z["grp"] == g) & (Yva == 1))            # row-major = (row, attr) order
        for k in rng.choice(len(pairs), min(3, len(pairs)), replace=False):
            r, at = (int(v) for v in pairs[k])
            out.append(dict(group=f"G{g}", row=r, ann_id=int(dva.ann_id.values[r]), attr=at, attr_name=names[at], level=dva.level.values[r],
                            category=dva.category.values[r], n_in=int(dva.n_in.values[r])))
    json.dump(out, open(os.path.join(a.run, f"group_examples_{a.reader}.json"), "w"), indent=1)
    print(",".join(str(o["ann_id"]) for o in out))
    print(",".join(f"{o['group']}_{a.reader}_{o['attr_name']}" for o in out))


if __name__ == "__main__":
    main()
