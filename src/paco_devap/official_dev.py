"""Official-protocol Dev AP, step 3 (CPU, env /home/irteam/envs/paco_eval): the UNMODIFIED official PACO attribute
evaluator (src.pacoattr.official_eval: make_dets adapter, PACOResults max_dets 300, PACOEval attr_ap_type 'usual',
heirachrichal_APs) on the dev GT built by src.paco_devap.build_dev. GT box + category oracle, score 1.0, attribute_probs
= the model's sigmoid outputs. Submission order (fixed rule): within each image the annotations carrying any known
attribute label first, then the rest (official order otherwise) -> the max_dets cut (1 dev image > 300) never removes a
labelled GT. Reported as "Official-protocol Dev AP" (dev = PACO train-internal, NOT an official split).

Per prediction file -> <dir>/official_dev.json: Object / Object-Part attribute AP, the four official type APs per level,
evaluable pairs per level (pairs with >= 1 positive dev GT; the evaluator drops the others), evaluable attributes per
type, dropped detections; and boot_pack.pkl (per-image official matching, for paired bootstrap).

  PYTHONPATH=<paco>:<lvis-api>:. python -m src.paco_devap.official_dev --pred <.../ep6/pred_dev.npz> [--pred ...]
"""
import argparse
import copy
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from paco.evaluation.paco_eval_api import PACO, PACOEval, PACOResults
from paco.evaluation.utils.paco_utils import heirachrichal_APs
from src.pacoattr.official_eval import MAXDET, make_dets, pack_boot

RUN = "output/2026-11-07/paco_devap"
GT = os.path.join(RUN, "data", "paco_dev_gt.json")
G = {}


def labeled_first(M, Y):
    lab = (Y >= 0).any(1)
    first = pd.Series(np.arange(len(M))).groupby(M.image_id.values).transform("min").values
    return np.lexsort((np.arange(len(M)), ~lab, first))


def evaluate(pred):
    gt, M, Y = G["gt"], G["M"], G["Y"]
    z = np.load(pred)
    assert np.array_equal(z["ann_id"], M.ann_id.values), "prediction rows != dev manifest"
    o = labeled_first(M, Y)
    dets = make_dets(M.iloc[o].reset_index(drop=True), z["probs"][o].astype(np.float64))
    t0 = time.time()
    dt = PACOResults(gt, copy.deepcopy(dets), max_dets=MAXDET)
    E = PACOEval(gt, dt, "bbox", attr_ap_type="usual")
    E.run()
    res = heirachrichal_APs(E)
    kept = {d["gt_ann_id"] for d in dt.dataset["annotations"]}
    drop = M.ann_id.isin(set(M.ann_id) - kept).values
    P = E.eval_joint_attr["precision"][:, :, :, 0]
    valid = (P > -1).any(axis=(0, 1))
    jids = list(E.params.joint_attr_ids)
    part_cat = {c["id"]: (":" in c["name"]) for c in gt.dataset["categories"]}
    j2 = {p["obj-attr"]: (p["obj"], p["attr"]) for p in gt.dataset["joint_obj_attribute_categories"]}
    t_of = {a: t for t, ix in gt.dataset["attr_type_to_attr_idxs"].items() for a in ix}
    names = {x["id"]: x["name"] for x in gt.dataset["attributes"]}
    cnt = {"obj": {}, "part": {}}
    for k, v in zip(jids, valid):
        if not v:
            continue
        o_, at = j2[k]
        lv = "part" if part_cat[o_] else "obj"
        cnt[lv].setdefault(t_of[at], set()).add(names[at])
    out = dict(pred=pred, protocol="Official-protocol Dev AP (PACO train-internal dev, 1,000 images; NOT an official split)",
               obj=res["obj-attr-AP"], part=res["obj-attr-part-AP"], obj_types=res["obj-attrs"], part_types=res["obj-part-attrs"],
               n_images=len(E.params.img_ids), n_gt_anns=len(gt.anns), n_dets_given=len(dets), n_dets_kept=len(dt.dataset["annotations"]),
               dropped=int(drop.sum()), dropped_with_known_label=int(((Y >= 0).any(1) & drop).sum()),
               n_pairs=len(jids), evaluable_pairs=dict(obj=int(sum(v and not part_cat[j2[k][0]] for k, v in zip(jids, valid))),
                                                      part=int(sum(v and part_cat[j2[k][0]] for k, v in zip(jids, valid)))),
               evaluable_attrs_per_type={lv: {t: len(s) for t, s in d.items()} for lv, d in cnt.items()},
               seconds=round(time.time() - t0, 1), hierarchical=res)
    d = os.path.dirname(pred)
    json.dump(out, open(os.path.join(d, "official_dev.json"), "w"), indent=1, default=float)
    pickle.dump(pack_boot(E), open(os.path.join(d, "boot_pack.pkl"), "wb"), protocol=4)
    return {k: out[k] for k in ("pred", "obj", "part", "evaluable_pairs", "dropped_with_known_label", "seconds")}


def init():
    G["gt"] = PACO(GT)
    G["M"] = pd.read_parquet(os.path.join(RUN, "data", "manifest_dev.parquet"))
    G["Y"] = np.load(os.path.join(RUN, "data", "labels_dev.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    if len(a.pred) == 1:
        init()
        print(json.dumps(evaluate(a.pred[0]), default=float))
        return
    with Pool(min(a.procs, len(a.pred)), initializer=init) as p:
        for r in p.imap(evaluate, a.pred):
            print(json.dumps(r, default=float), flush=True)


if __name__ == "__main__":
    main()
