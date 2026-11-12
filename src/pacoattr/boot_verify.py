"""Check of the bootstrap shortcut against the OFFICIAL evaluator on materialised resamples (val; paco_eval env).

For a resample (multinomial image multiplicities, rng seed s), build a new GT json holding m copies of each drawn image:
copy c of image i gets image id i + c * 10^7, every annotation id a -> a + c * 10^8 and obj_ann_id remapped the same way
(object-part -> parent links stay inside the copy); image-level lists (neg / not-exhaustive category ids) are copied; the
Oracle detections are copied with the new image ids. Run the official PACOResults / PACOEval / heirachrichal_APs and compare
with src.pacoattr.boot (weighted re-accumulation of the original per-image results with the same multiplicities).

  python -m src.pacoattr.boot_verify --cond frozen_full --seeds 1,2 --out <json>
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd

from paco.evaluation.paco_eval_api import PACO, PACOEval, PACOResults
from paco.evaluation.utils.paco_utils import heirachrichal_APs
from src.pacoattr import boot as B
from src.pacoattr.official_eval import make_dets

RUN = "output/2026-11-07/paco_oracle_attr"
SCR = "/tmp/claude-500/-home-irteam-Crop-representation/7786d5be-b6d3-441c-859f-03eba8485c73/scratchpad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = json.load(open(B.ANN.format("val")))
    B.G["gt"] = PACO(B.ANN.format("val"))
    pk = B.Pack(os.path.join(RUN, "eval", a.cond, "val", "boot_pack.pkl"))
    M = pd.read_parquet(os.path.join(RUN, "data", "manifest_val.parquet"))
    z = np.load(os.path.join(RUN, "preds", a.cond, "probs_val.npz"))
    dets = make_dets(M, z["probs"].astype(np.float64))
    by_img_ann, by_img_det = {}, {}
    for x in J["annotations"]:
        by_img_ann.setdefault(x["image_id"], []).append(x)
    for d in dets:
        by_img_det.setdefault(d["image_id"], []).append(d)
    imgs = {im["id"]: im for im in J["images"]}
    out = []
    for s in (int(v) for v in a.seeds.split(",")):
        rng = np.random.default_rng(1000 + s)
        w = rng.multinomial(pk.I, np.full(pk.I, 1.0 / pk.I)).astype(float)
        Jn = {k: v for k, v in J.items() if k not in ("images", "annotations")}
        Jn["images"], Jn["annotations"], Dn = [], [], []
        for i_idx, iid in enumerate(int(v) for v in pk.img_ids):
            for c in range(int(w[i_idx])):
                im = copy.deepcopy(imgs[iid])
                im["id"] = iid + c * 10 ** 7
                Jn["images"].append(im)
                for x in by_img_ann.get(iid, []):
                    y = copy.deepcopy(x)
                    y["id"], y["obj_ann_id"], y["image_id"] = x["id"] + c * 10 ** 8, x["obj_ann_id"] + c * 10 ** 8, im["id"]
                    Jn["annotations"].append(y)
                for d in by_img_det.get(iid, []):
                    e = copy.deepcopy(d)
                    e["image_id"] = im["id"]
                    Dn.append(e)
        fn = os.path.join(SCR, f"val_boot_s{s}.json")
        json.dump(Jn, open(fn, "w"))
        gt = PACO(fn)
        dt = PACOResults(gt, Dn, max_dets=300)
        E = PACOEval(gt, dt, "bbox", attr_ap_type="usual")
        E.run()
        r = heirachrichal_APs(E)
        off = dict(obj=r["obj-attr-AP"], part=r["obj-attr-part-AP"])
        wt = B.hier(pk.precision(w))
        out.append(dict(seed=s, images_drawn=int((w > 0).sum()), official=off, weighted=dict(obj=wt["obj"], part=wt["part"]),
                        abs_diff_obj=abs(off["obj"] - wt["obj"]), abs_diff_part=abs(off["part"] - wt["part"])))
        print(json.dumps(out[-1]), flush=True)
        os.remove(fn)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
