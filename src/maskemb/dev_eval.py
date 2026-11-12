"""Dev metrics of ONE epoch's saved dev scores (CPU, separate process so the GPUs keep training): the same
RC.m2_eval(S, ddev, Ydev, Ytr_by) call the probe made in-process before, on the same deterministic dev split.
  python -m src.maskemb.dev_eval --out <probe dir> --ep <k> [--limit N]   -> <out>/r2/dev_ep<k>.json
"""
import argparse
import json
import os

import numpy as np

import src.readout.common as RC
from src.maskemb.probe import EXCLUDE, split_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_dev", type=int, default=1000)
    a = ap.parse_args()
    data = RC.load_data(EXCLUDE, a.limit, splits=("train",))
    Ytr_by = RC.ytr_by_level(data)
    _, _, ddev, Ydev = split_dev(*data["train"], a.n_dev)
    S = np.load(os.path.join(a.out, "r2", f"scores_dev_ep{a.ep}.npy"))
    assert S.shape == (len(ddev), 59)
    r = RC.m2_eval(S, ddev, Ydev, Ytr_by)
    if not np.isfinite(r["loss"]):
        raise FloatingPointError(f"non-finite dev loss at epoch {a.ep}")
    json.dump(dict(epoch=a.ep, dev=dict(loss=r["loss"], object_loss=r["object_loss"], part_loss=r["part_loss"], M0=r["M0"], M2=r["M2"])),
              open(os.path.join(a.out, "r2", f"dev_ep{a.ep}.json"), "w"), default=float)


if __name__ == "__main__":
    main()
