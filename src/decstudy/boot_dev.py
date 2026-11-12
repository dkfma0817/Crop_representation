"""Paired image bootstrap of the Official-protocol Dev AP (1,000 PACO-train dev images) for the decoder study = the
src.pacoattr.boot / src.paco_devap.compare procedure (official per-image matching re-accumulated, B = 1000, seed 0) on the
fixed epoch-6 boot packs <run>/preds_dev/<name>/ep6/boot_pack.pkl. Runs in /home/irteam/envs/paco_eval.

  python -m src.decstudy.boot_dev --names a,b,... --out <json>
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

import src.pacoattr.boot as BT
from paco.evaluation.paco_eval_api import PACO

RR = "output/2026-11-07/decoder_study"
GT = "output/2026-11-07/paco_devap/data/paco_dev_gt.json"


def _one(b):
    w = BT.G["W"][b]
    return {c: BT.hier(pk.precision(w)) for c, pk in BT.G["packs"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--epoch", type=int, default=6)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = a.names.split(",")
    BT.G["gt"] = PACO(GT)
    BT.G["packs"] = {n: BT.Pack(os.path.join(RR, "preds_dev", n, f"ep{a.epoch}", "boot_pack.pkl")) for n in names}
    pk0 = next(iter(BT.G["packs"].values()))
    for n, pk in BT.G["packs"].items():
        assert pk.img_ids == pk0.img_ids
        assert np.array_equal(pk.precision(np.ones(pk.I)), pk.official), n
    pts = {n: BT.hier(pk.precision(np.ones(pk.I))) for n, pk in BT.G["packs"].items()}
    rng = np.random.default_rng(0)
    I = pk0.I
    BT.G["W"] = rng.multinomial(I, np.full(I, 1.0 / I), size=a.B).astype(float)
    with Pool(a.procs) as p:
        R = p.map(_one, range(a.B), chunksize=4)
    res = dict(split="dev", epoch=a.epoch, B=a.B, n_images=I, points=pts, verified_w1_equals_official=True,
               boot={n: {k: [r[n][k] for r in R] for k in pts[n]} for n in names})
    json.dump(res, open(a.out, "w"), default=float)
    print(json.dumps(pts, default=float))


if __name__ == "__main__":
    main()
