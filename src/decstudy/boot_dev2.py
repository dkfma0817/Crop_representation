"""Paired image bootstrap of the Official-protocol Dev AP over arbitrary (name = boot_pack.pkl) entries (same procedure as
src.decstudy.boot_dev: official per-image matching re-accumulated, B = 1000, seed 0, w = 1 reproduces the official
precision). Env /home/irteam/envs/paco_eval.
  python -m src.decstudy.boot_dev2 --packs name=path ... --out <json>
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np

import src.pacoattr.boot as BT
from paco.evaluation.paco_eval_api import PACO

GT = "output/2026-11-07/paco_devap/data/paco_dev_gt.json"


def _one(b):
    w = BT.G["W"][b]
    return {c: BT.hier(pk.precision(w)) for c, pk in BT.G["packs"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    BT.G["gt"] = PACO(GT)
    BT.G["packs"] = {s.split("=", 1)[0]: BT.Pack(s.split("=", 1)[1]) for s in a.packs}
    pk0 = next(iter(BT.G["packs"].values()))
    for n, pk in BT.G["packs"].items():
        assert pk.img_ids == pk0.img_ids and np.array_equal(pk.precision(np.ones(pk.I)), pk.official), n
    pts = {n: BT.hier(pk.precision(np.ones(pk.I))) for n, pk in BT.G["packs"].items()}
    I = pk0.I
    BT.G["W"] = np.random.default_rng(0).multinomial(I, np.full(I, 1.0 / I), size=a.B).astype(float)
    with Pool(a.procs) as p:
        R = p.map(_one, range(a.B), chunksize=4)
    json.dump(dict(split="dev", B=a.B, n_images=I, points=pts, boot={n: {k: [r[n][k] for r in R] for k in pts[n]} for n in pts}), open(a.out, "w"), default=float)
    print(json.dumps(pts, default=float)[:2000])


if __name__ == "__main__":
    main()
