"""PACO-LVIS GT-Box + Category Oracle Attribute AP (Official Evaluator). Runs in the dedicated env /home/irteam/envs/paco_eval
(numpy 1.23.5: the official accumulate_joint_attr uses np.float), with the UNMODIFIED official code on sys.path:
  PACO   github.com/facebookresearch/paco      @ cdefd31 (paco/evaluation/paco_eval_api, paco/evaluation/utils/paco_utils.py)
  lvis   github.com/lvis-dataset/lvis-api      @ 7d7f07d
Official path reproduced from paco/evaluation/paco_evaluation._evaluate_predictions_on_lvis (eval_attributes=True,
attr_ap_type='usual', iou_type='bbox', max_dets_per_image=None -> 300), minus its detectron2 logging wrapper:
  PACOResults(gt, dets, max_dets=300) -> PACOEval(gt, dt, 'bbox', attr_ap_type='usual').run() -> heirachrichal_APs(eval)

Oracle Prediction Adapter (outside the evaluator): ONE detection per official GT annotation (objects AND object-parts, all
of them, in official JSON annotation order): bbox = the original GT bbox, category_id = the GT category, score = 1.0,
attribute_probs = the model's 59 sigmoid outputs for that ROI (official attribute index order; verified equal to the
model head order). The GT ann id is carried as 'gt_ann_id' only to verify alignment; the evaluator ignores it and matches
by IoU as usual. Nothing is filtered or re-scored with GT attributes. The evaluator itself multiplies score x attr prob
(eval.py evaluate_img_attr, attr_ap_type 'usual'); with score = 1.0 that product is the attr prob (no double product).

  PYTHONPATH=<paco>:<lvis-api>:. python -m src.pacoattr.official_eval --split val --pred <probs.npy> --out <dir>
  ... --sanity perfect|constant   (sanity predictions from the official GT fields; no model)
"""
import argparse
import copy
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from paco.evaluation.paco_eval_api import PACO, PACOEval, PACOResults
from paco.evaluation.utils.paco_utils import heirachrichal_APs

ANN = "/home/irteam/datasets/paco/paco_lvis_v1_{}.json"
MAXDET = 300


def load_gt(split):
    return PACO(ANN.format(split))


def make_dets(M, probs):
    """M: data/manifest_<split>.parquet (official annotation order); probs (N, 59) aligned to M rows."""
    assert len(M) == len(probs) and probs.shape[1] == 59 and np.isfinite(probs).all()
    dets = []
    for r, p in zip(M.itertuples(index=False), probs):
        dets.append(dict(image_id=int(r.image_id), category_id=int(r.category_id), bbox=[float(r.bx), float(r.by), float(r.bw), float(r.bh)],
                         score=1.0, attribute_probs=[float(v) for v in p], gt_ann_id=int(r.ann_id)))
    return dets


def sanity_probs(Y, kind):
    if kind == "perfect":                     # explicit positive 1, explicit negative 0, unknown type 0.5
        return np.where(Y == 1, 1.0, np.where(Y == 0, 0.0, 0.5))
    if kind == "constant":
        return np.full(Y.shape, 0.5)
    raise ValueError(kind)


def run(split, dets, img_ids=None):
    gt = load_gt(split)
    t0 = time.time()
    dt = PACOResults(gt, copy.deepcopy(dets), max_dets=MAXDET)
    E = PACOEval(gt, dt, "bbox", attr_ap_type="usual")
    if img_ids is not None:                   # subset analysis only (params.img_ids, as the lvis/paco API allows)
        E.params.img_ids = sorted(int(i) for i in img_ids)
    E.run()
    res = heirachrichal_APs(E)
    return gt, dt, E, res, time.time() - t0


def pack_boot(E):
    """Per (joint pair, image) official evaluate_img_attr output for area 'all' (index 0): what accumulate_joint_attr
    consumes. Stored compactly for the weighted re-accumulation (src.pacoattr.boot)."""
    K, A, I = len(E.params.joint_attr_ids), len(E.params.area_rng), len(E.params.img_ids)
    out = []
    for k in range(K):
        for i in range(I):
            e = E.eval_imgs_joint_attr[k * A * I + 0 * I + i]
            if e is None:
                continue
            out.append((k, i, np.asarray(e["dt_scores"], np.float64), np.asarray(e["dt_matches"]) > 0,
                        np.asarray(e["dt_ignore"]).astype(bool), np.asarray(e["gt_ignore"]).astype(bool)))
    return dict(K=K, I=I, img_ids=list(E.params.img_ids), joint_attr_ids=list(E.params.joint_attr_ids),
                rec_thrs=np.asarray(E.params.rec_thrs), iou_thrs=np.asarray(E.params.iou_thrs), entries=out,
                precision=E.eval_joint_attr["precision"][:, :, :, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--data", default="output/2026-11-07/paco_oracle_attr/data")
    ap.add_argument("--pred", default="")
    ap.add_argument("--sanity", default="")
    ap.add_argument("--subset", default="", help="json list of image ids (subset analysis)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", action="store_true", help="save per-image matching for the bootstrap")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M = pd.read_parquet(os.path.join(a.data, f"manifest_{a.split}.parquet"))
    if a.sanity:
        probs = sanity_probs(np.load(os.path.join(a.data, f"labels_{a.split}.npy")), a.sanity)
    else:
        z = np.load(a.pred)
        probs = z["probs"] if isinstance(z, np.lib.npyio.NpzFile) else z
        if isinstance(z, np.lib.npyio.NpzFile):
            assert np.array_equal(z["ann_id"], M.ann_id.values), "prediction rows not aligned with the official manifest"
    dets = make_dets(M, probs.astype(np.float64))
    img_ids = json.load(open(a.subset)) if a.subset else None
    gt, dt, E, res, secs = run(a.split, dets, img_ids)
    # detection bookkeeping: input vs kept after the official max_dets cut
    kept = Counter(d["image_id"] for d in dt.dataset["annotations"])
    given = Counter(d["image_id"] for d in dets)
    dropped = {int(i): given[i] - kept.get(i, 0) for i in given if given[i] > kept.get(i, 0)}
    kept_ids = {d["gt_ann_id"] for d in dt.dataset["annotations"]}
    # alignment: every kept det box/category equals its GT annotation
    gtA = gt.anns
    mis = sum(1 for d in dt.dataset["annotations"] if gtA[d["gt_ann_id"]]["category_id"] != d["category_id"]
              or not np.allclose(gtA[d["gt_ann_id"]]["bbox"], d["bbox"]) or gtA[d["gt_ann_id"]]["image_id"] != d["image_id"])
    valid = E.eval_joint_attr["precision"][:, :, :, 0]
    n_valid_pairs = int(((valid > -1).any(axis=(0, 1))).sum())
    out = dict(split=a.split, pred=a.pred or f"sanity:{a.sanity}", subset=a.subset or None, n_eval_images=len(E.params.img_ids),
               n_gt_anns=len(gt.anns), n_dets_given=len(dets), n_dets_kept=len(dt.dataset["annotations"]),
               dets_dropped_by_max_dets=sum(dropped.values()), images_with_dropped=len(dropped), dropped_per_image=dropped,
               dropped_gt_ann_ids=sorted(set(d["gt_ann_id"] for d in dets) - kept_ids)[:2000],
               alignment_mismatch=mis, n_joint_pairs=len(E.params.joint_attr_ids), n_valid_pairs_area_all=n_valid_pairs,
               seconds=round(secs, 1), lvis_bbox_results={k: float(v) for k, v in E.results.items()},
               joint_attr_summary={k: float(v) for k, v in E.results_joint_attr.items()}, hierarchical=res)
    json.dump(out, open(os.path.join(a.out, "official_results.json"), "w"), indent=1, default=float)
    if a.boot:
        pickle.dump(pack_boot(E), open(os.path.join(a.out, "boot_pack.pkl"), "wb"), protocol=4)
    h = res
    print(json.dumps(dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"], obj_types=h["obj-attrs"], part_types=h["obj-part-attrs"],
                          valid_pairs=n_valid_pairs, dropped=out["dets_dropped_by_max_dets"], mis=mis, secs=out["seconds"]), default=float))


if __name__ == "__main__":
    main()
