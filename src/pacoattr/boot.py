"""Image-level paired bootstrap of the official PACO attribute AP (runs in /home/irteam/envs/paco_eval).

No re-implementation of matching / ignore rules: the OFFICIAL per-(pair, image) results of evaluate_img_attr (area 'all',
saved by official_eval --boot) are re-accumulated with integer image multiplicities w (a resample draws images with
replacement; an image drawn m times counts m times = what the official evaluator gives on a dataset holding m copies of
that image with fresh image / annotation ids). The accumulation replicates accumulate_joint_attr line by line (concatenate
in image order, mergesort by -score, cumsum of tp / fp weighted by w, num_gt = sum w * #non-ignored gt, precision
envelope, searchsorted at the 101 recall thresholds, out-of-range -> 0, pairs with num_gt == 0 -> -1). The hierarchical
aggregation is the OFFICIAL heirachrichal_APs called on that precision array. Checks: (1) w = 1 reproduces the official
precision array exactly; (2) src.pacoattr.boot_verify runs the official evaluator on materialised resampled copies.
Valid pairs (num_gt > 0) depend only on GT and w, so they are the same for every model within a resample (paired).

  python -m src.pacoattr.boot --split test --conds frozen_full,... --B 1000 --subset <images json> --out <json>
"""
import argparse
import json
import os
import pickle
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np

from paco.evaluation.paco_eval_api import PACO
from paco.evaluation.utils.paco_utils import heirachrichal_APs

RUN = "output/2026-11-07/paco_oracle_attr"
ANN = "/home/irteam/datasets/paco/paco_lvis_v1_{}.json"
G = {}


class Pack:
    def __init__(self, fn):
        p = pickle.load(open(fn, "rb"))
        self.K, self.I, self.img_ids, self.rec = p["K"], p["I"], p["img_ids"], p["rec_thrs"]
        self.T = len(p["iou_thrs"])
        self.official = p["precision"]
        by = {}
        for k, i, s, m, ig, gi in p["entries"]:
            by.setdefault(k, []).append((i, s, m, ig, gi))
        self.pairs = {}
        for k, E in by.items():
            s = np.concatenate([e[1] for e in E])
            im = np.concatenate([np.full(len(e[1]), e[0]) for e in E]).astype(np.int64)
            m = np.concatenate([e[2].reshape(self.T, -1) for e in E], axis=1)
            ig = np.concatenate([e[3].reshape(self.T, -1) for e in E], axis=1)
            gi = np.concatenate([e[4].reshape(-1) for e in E])
            gim = np.concatenate([np.full(len(e[4].reshape(-1)), e[0]) for e in E]).astype(np.int64)
            o = np.argsort(-s, kind="mergesort")
            self.pairs[k] = (im[o], np.logical_and(m[:, o], ~ig[:, o]), np.logical_and(~m[:, o], ~ig[:, o]), gim[gi == 0])

    def precision(self, w):
        P = -np.ones((self.T, len(self.rec), self.K))
        for k, (im, tps, fps, gim) in self.pairs.items():
            num_gt = w[gim].sum()
            if num_gt == 0:
                continue
            wd = w[im]
            tp = np.cumsum(tps * wd, axis=1)
            fp = np.cumsum(fps * wd, axis=1)
            n = tp.shape[1]
            for t in range(self.T):
                rc = tp[t] / num_gt
                pr = tp[t] / (fp[t] + tp[t] + np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1] if n else pr
                idx = np.searchsorted(rc, self.rec, side="left")
                q = np.zeros(len(self.rec))
                ok = idx < n
                q[ok] = pr[idx[ok]]
                P[t, :, k] = q
        return P


def hier(P):
    ev = SimpleNamespace(lvis_gt=G["gt"], lvis_dt=SimpleNamespace(dataset={"attr_type_to_attr_idxs": G["gt"].dataset["attr_type_to_attr_idxs"]}),
                         eval_joint_attr={"precision": P[..., None]})
    r = heirachrichal_APs(ev)
    return dict(obj=r["obj-attr-AP"], part=r["obj-attr-part-AP"], **{f"obj_{k}": v for k, v in r["obj-attrs"].items()},
                **{f"part_{k}": v for k, v in r["obj-part-attrs"].items()})


def _one(b):
    w = G["W"][b]
    out = {}
    nvalid = None
    for c, pk in G["packs"].items():
        P = pk.precision(w)
        if nvalid is None:
            nvalid = int(((P > -1).any(axis=(0, 1))).sum())
        out[c] = hier(P)
    return out, nvalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--subset", default="")
    ap.add_argument("--evaldir", default="eval")
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conds = a.conds.split(",")
    G["gt"] = PACO(ANN.format(a.split))
    G["packs"] = {c: Pack(os.path.join(RUN, a.evaldir, c, a.split, "boot_pack.pkl")) for c in conds}
    pk0 = next(iter(G["packs"].values()))
    for c, pk in G["packs"].items():
        assert pk.img_ids == pk0.img_ids
        P1 = pk.precision(np.ones(pk.I))
        assert np.array_equal(P1, pk.official), f"{c}: weighted re-accumulation != official precision"
    img = np.array(pk0.img_ids)
    base = np.ones(len(img))
    if a.subset:
        keep = set(json.load(open(a.subset)))
        base = np.isin(img, list(keep)).astype(float)
    pts = {c: hier(G["packs"][c].precision(base)) for c in conds}
    rng = np.random.default_rng(0)
    idx = np.nonzero(base)[0]
    W = np.zeros((a.B, len(img)))
    W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=a.B)
    G["W"] = W
    with Pool(a.procs) as p:
        R = p.map(_one, range(a.B), chunksize=2)
    res = dict(split=a.split, subset=a.subset or None, B=a.B, n_images=int(base.sum()), points=pts,
               verified_w1_equals_official=True, valid_pairs=dict(point=int(((G["packs"][conds[0]].precision(base) > -1).any(axis=(0, 1))).sum()),
                                                                 boot=[r[1] for r in R]),
               boot={c: {k: [r[0][c][k] for r in R] for k in pts[c]} for c in conds})
    json.dump(res, open(a.out, "w"), default=float)
    print(json.dumps(dict(points=pts, valid_pairs_point=res["valid_pairs"]["point"],
                          valid_pairs_boot=[min(res["valid_pairs"]["boot"]), max(res["valid_pairs"]["boot"])]), default=float))


if __name__ == "__main__":
    main()
