"""SELF analysis (not the PACO main metric): attribute AP by ORIGINAL BOX PIXEL AREA, COCO bins small < 32^2 <= medium
< 96^2 <= large. Runs the UNMODIFIED official evaluator (env /home/irteam/envs/paco_eval) on the same oracle detections
(src.pacoattr.official_eval.make_dets) with a COPY of the official val GT whose `area` field is replaced by the GT box
area w * h (the official file uses the segmentation area; detections of iou_type bbox already get area = w * h in
PACOResults). The evaluator's own area ranges (all / small / medium / large) then give per-bin precision; each bin is
aggregated with the official heirachrichal_APs (pair -> attr -> type -> mean of types), exactly as the 'all' slice.
Sanity: the 'all' slice must equal the main official result (area plays no role for [0, 1e10]).

  PYTHONPATH=<paco>:<lvis-api>:. python -m src.maskemb.area_eval --conds clip_full,clip_c0,maskembed_full
"""
import argparse
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from paco.evaluation.paco_eval_api import PACO, PACOEval, PACOResults
from paco.evaluation.utils.paco_utils import heirachrichal_APs
from src.pacoattr.official_eval import ANN, MAXDET, make_dets

RUN = "output/2026-11-07/maskembed_paco"
DATA = "output/2026-11-07/paco_oracle_attr/data"


def box_area_gt(out, split):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(out):
        J = json.load(open(ANN.format(split)))
        for a in J["annotations"]:
            a["area_segm_official"] = a["area"]
            a["area"] = float(a["bbox"][2] * a["bbox"][3])
        json.dump(J, open(out, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    a = ap.parse_args()
    gt_path = box_area_gt(os.path.join(RUN, "data", f"paco_lvis_v1_{a.split}_boxarea.json"), a.split)
    gt = PACO(gt_path)
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    res = {}
    for c in a.conds.split(","):
        z = np.load(os.path.join(RUN, "preds", c, f"probs_{a.split}.npz"))
        assert np.array_equal(z["ann_id"], M.ann_id.values)
        dt = PACOResults(gt, copy.deepcopy(make_dets(M, z["probs"].astype(np.float64))), max_dets=MAXDET)
        E = PACOEval(gt, dt, "bbox", attr_ap_type="usual")
        E.run()
        P = E.eval_joint_attr["precision"]
        lbl = list(E.params.area_rng_lbl)
        out = {}
        for i, nm in enumerate(lbl):
            ev = SimpleNamespace(lvis_gt=gt, lvis_dt=SimpleNamespace(dataset={"attr_type_to_attr_idxs": gt.dataset["attr_type_to_attr_idxs"]}),
                                 eval_joint_attr={"precision": P[..., i][..., None]})
            h = heirachrichal_APs(ev)
            valid = (P[..., i] > -1).any(axis=(0, 1))
            out[nm] = dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"], obj_types=h["obj-attrs"], part_types=h["obj-part-attrs"],
                           n_valid_pairs=int(valid.sum()), area_rng=list(E.params.area_rng[i]))
        res[c] = out
        print(c, {k: (round(v["obj"], 2), round(v["part"], 2), v["n_valid_pairs"]) for k, v in out.items()}, flush=True)
    # GT counts per bin (attribute-annotated objects / parts)
    anns = gt.dataset["annotations"]
    cnt = {}
    for x in anns:
        ar = x["area"]
        b = "small" if ar < 32 ** 2 else "medium" if ar < 96 ** 2 else "large"
        lv = "object" if x.get("obj_ann_id", x["id"]) == x["id"] else "part"
        k = f"{lv}/{b}"
        cnt.setdefault(k, [0, 0])
        cnt[k][0] += 1
        cnt[k][1] += int(len(x.get("attribute_ids", [])) > 0)
    res["_gt_counts_box_area"] = {k: dict(anns=v[0], with_attributes=v[1]) for k, v in sorted(cnt.items())}
    os.makedirs(os.path.join(RUN, "eval_area"), exist_ok=True)
    json.dump(res, open(os.path.join(RUN, "eval_area", f"{a.split}_box_area_bins.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
