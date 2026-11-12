"""Official evaluation with the OTHER server's submission order (for table compatibility): within every image, the oracle
detections of GT annotations that carry ANY known attribute label (labels_<split>.npy >= 0 in some column) are given first,
then the rest; otherwise the official JSON order. All scores are 1.0, and the official PACOResults keeps the first
max_dets = 300 per image after a stable sort by score, so this order decides which detections the cut removes (11 test
images). Predictions, adapter fields and the evaluator are unchanged (src.pacoattr.official_eval.make_dets / run).
Output: <run>/eval_labeled_first/<cond>/<split>/official_results.json (the original eval/ is not touched).

  PYTHONPATH=<paco>:<lvis-api>:. python -m src.maskemb.eval_labeled_first --run <run dir> --cond clip_full --split test
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.pacoattr.official_eval import make_dets, run

DATA = "output/2026-11-07/paco_oracle_attr/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    Y = np.load(os.path.join(DATA, f"labels_{a.split}.npy"))
    z = np.load(os.path.join(a.run, "preds", a.cond, f"probs_{a.split}.npz"))
    assert np.array_equal(z["ann_id"], M.ann_id.values)
    labeled = (Y >= 0).any(1)
    img_first = pd.Series(np.arange(len(M))).groupby(M.image_id.values).transform("min").values   # keep image blocks in official order
    order = np.lexsort((np.arange(len(M)), ~labeled, img_first))
    dets = make_dets(M.iloc[order].reset_index(drop=True), z["probs"][order].astype(np.float64))
    gt, dt, E, res, secs = run(a.split, dets)
    kept = {d["gt_ann_id"] for d in dt.dataset["annotations"]}
    dropped = sorted(set(M.ann_id.values) - kept)
    lab_drop = int(labeled[M.ann_id.isin(dropped).values].sum())
    out = dict(cond=a.cond, split=a.split, order="labeled-first within image (other server's adapter)", n_dets_given=len(dets),
               n_dets_kept=len(dt.dataset["annotations"]), dropped=len(dropped), dropped_with_known_label=lab_drop,
               obj=res["obj-attr-AP"], part=res["obj-attr-part-AP"], obj_types=res["obj-attrs"], part_types=res["obj-part-attrs"], seconds=round(secs, 1))
    od = os.path.join(a.run, "eval_labeled_first", a.cond, a.split)
    os.makedirs(od, exist_ok=True)
    json.dump(dict(**out, hierarchical=res), open(os.path.join(od, "official_results.json"), "w"), indent=1, default=float)
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
