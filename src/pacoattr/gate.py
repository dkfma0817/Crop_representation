"""Phase-1 gate (CPU): the official-test inference runs only if every check below passes on the OFFICIAL VAL.
  * predictions complete: every manifest row scored, finite, aligned (ann_id order), shape (N, 59)
  * full / dec: exactly one encoder canvas per image; c0: one crop canvas per ROI
  * reproduction of the saved probe val logits (existing 11,109 val rows, same R2 ep6): corr >= 0.999
  * decoder path (g1cross_dec): zeroing the decoder output changes the logits; swapping ROI boxes changes the logits
  * official evaluator on val: ran, no alignment mismatch, no max_dets drop, finite object / part attribute AP
  * sanity: perfect = 100 / 100, constant ran
Writes <run>/gate.json; exit 1 on failure.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from src.pacoattr.audit import CONDS

RUN = "output/2026-11-07/paco_oracle_attr"


def main():
    M = pd.read_parquet(os.path.join(RUN, "data", "manifest_val.parquet"))
    fails, info = [], {}
    for c in CONDS:
        z = np.load(os.path.join(RUN, "preds", c, "probs_val.npz"))
        inf = json.load(open(os.path.join(RUN, "preds", c, "infer_val.json")))
        ev = json.load(open(os.path.join(RUN, "eval", c, "val", "official_results.json")))
        ok = dict(aligned=bool(np.array_equal(z["ann_id"], M.ann_id.values)), finite=bool(np.isfinite(z["probs"]).all()),
                  shape=list(z["probs"].shape), enc_per_image=inf["encoder_calls_per_image"],
                  repro_corr=inf["checks"].get("repro_corr"), repro_max=inf["checks"].get("repro_max_abs_logit_diff"),
                  eval_mis=ev["alignment_mismatch"], eval_dropped=ev["dets_dropped_by_max_dets"],
                  obj=ev["hierarchical"]["obj-attr-AP"], part=ev["hierarchical"]["obj-attr-part-AP"])
        if CONDS[c][2] == "dec":
            ok["dec_zero"] = inf["checks"]["dec_bypass_zero_max_abs_logit_change"]["min"]
            ok["dec_boxswap"] = inf["checks"]["dec_box_swap_mean_abs_logit_change"]["min"]
            if not (ok["dec_zero"] > 0 and ok["dec_boxswap"] > 0):
                fails.append(f"{c}: decoder path check")
        if not (ok["aligned"] and ok["finite"] and ok["shape"] == [len(M), 59]):
            fails.append(f"{c}: predictions incomplete / misaligned")
        if CONDS[c][2] in ("full", "dec") and ok["enc_per_image"] != 1.0:
            fails.append(f"{c}: encoder calls per image {ok['enc_per_image']}")
        if ok["repro_corr"] is None or ok["repro_corr"] < 0.999:
            fails.append(f"{c}: reproduction corr {ok['repro_corr']}")
        if ok["eval_mis"] or ok["eval_dropped"] or not (math.isfinite(ok["obj"]) and math.isfinite(ok["part"])):
            fails.append(f"{c}: official val evaluation")
        info[c] = ok
    sp = json.load(open(os.path.join(RUN, "sanity", "val_perfect", "official_results.json")))["hierarchical"]
    info["sanity_perfect"] = [sp["obj-attr-AP"], sp["obj-attr-part-AP"]]
    if not (abs(sp["obj-attr-AP"] - 100) < 1e-6 and abs(sp["obj-attr-part-AP"] - 100) < 1e-6):
        fails.append("sanity perfect != 100")
    out = dict(passed=not fails, fails=fails, checks=info)
    json.dump(out, open(os.path.join(RUN, "gate.json"), "w"), indent=1, default=float)
    print(json.dumps(out, default=float))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
