"""Rebuild the ROI lists on a new server (the originals came from other projects and are not available here).

  python -m src.data.make_rois        -> cache/data/{coco_rois.parquet, rae_rois.parquet, paco_valtest_ids.json, make_rois.json}

NOT the original ROI sets (the selection code of dinov2_oracle_region_decoder / rae_crop_pilot is not available), only the same
format and the same rules as documented in configs/protocol.md:
  coco_rois.parquet  split 'train': 60,000 COCO train2017 images x 4 ROIs (3 GT + 1 random)  -> decoder pool (make_stream.py
                                    later removes the PACO val/test images, as in the original)
                     split 'dev'  :    500 images x 4 ROIs = 2,000 ROIs, outside 'train' and PACO val/test
  rae_rois.parquet   split 'head_val': 1,000 images x up to 4 GT ROIs, outside 'train', 'dev' and PACO val/test
Rules chosen here (all seeded, deterministic):
  GT ROI   = COCO instance, iscrowd 0, integer box (floor l/t, ceil r/b, clipped), both sides >= MIN_SIDE px.
             Images need >= 3 such boxes. GT ROIs per image drawn uniformly without replacement.
  random   = area fraction log-uniform [0.005, 0.25], aspect log-uniform [1/3, 3], uniform position; cat_id -1.
  n_in     = F336 token count (same roi_geometry box rule at S=336, PATCH=14, G=24) -> size bins D1-4 / D5-16 / D17+.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import src.rae.common as RC  # noqa: E402

from src.config import COCO_ANN, PACO_DIR  # noqa: E402

OUT = os.path.join(RC.CACHE, "data")
SEED = 0
MIN_SIDE = 8
N_TRAIN, N_DEV, N_HVAL = 60000, 500, 1000


def n_in_f336(W, H, l, t, r, b, S=336, P=14):
    """Token count of the ROI box on the 24x24 grid of a 336 letterbox (roi_geometry's box rule)."""
    G = S // P
    s = S / max(W, H)
    nw, nh = min(S, max(1, int(round(W * s)))), min(S, max(1, int(round(H * s))))
    ox, oy, sx, sy = (S - nw) // 2, (S - nh) // 2, nw / W, nh / H
    j0, j1 = int(math.floor((ox + l * sx) / P)), int(math.ceil((ox + r * sx) / P)) - 1
    i0, i1 = int(math.floor((oy + t * sy) / P)), int(math.ceil((oy + b * sy) / P)) - 1
    j0, i0, j1, i1 = max(j0, 0), max(i0, 0), min(j1, G - 1), min(i1, G - 1)
    return (i1 - i0 + 1) * (j1 - j0 + 1)


def random_box(rng, W, H):
    for _ in range(100):
        a = math.exp(rng.uniform(math.log(0.005), math.log(0.25))) * W * H
        ar = math.exp(rng.uniform(math.log(1 / 3), math.log(3)))
        w, h = int(round(math.sqrt(a * ar))), int(round(math.sqrt(a / ar)))
        if MIN_SIDE <= w <= W and MIN_SIDE <= h <= H:
            l, t = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
            return l, t, l + w, t + h
    raise RuntimeError("no random box")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = json.load(open(COCO_ANN))
    imgs = {im["id"]: (im["width"], im["height"]) for im in d["images"]}
    names = {c["id"]: c["name"] for c in d["categories"]}
    gt = {}
    for a in d["annotations"]:
        if a["iscrowd"]:
            continue
        W, H = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        l, t = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
        r, b = min(W, int(math.ceil(x + w))), min(H, int(math.ceil(y + h)))
        if r - l >= MIN_SIDE and b - t >= MIN_SIDE:
            gt.setdefault(a["image_id"], []).append((a["id"], a["category_id"], l, t, r, b))
    for v in gt.values():
        v.sort()
    eligible = np.array(sorted(i for i, v in gt.items() if len(v) >= 3))

    paco = set()
    for sp in ("val", "test"):
        paco |= {im["id"] for im in json.load(open(os.path.join(PACO_DIR, f"paco_lvis_v1_{sp}.json")))["images"]}
    json.dump(sorted(paco), open(os.path.join(OUT, "paco_valtest_ids.json"), "w"))

    perm = rng.permutation(eligible)
    train = perm[:N_TRAIN]
    rest = [i for i in perm[N_TRAIN:] if i not in paco]
    dev, hval = rest[:N_DEV], rest[N_DEV:N_DEV + N_HVAL]
    assert len(hval) == N_HVAL

    def rows(ids, split, n_gt, with_random):
        out = []
        for im in ids:
            W, H = imgs[im]
            boxes = gt[im]
            pick = sorted(rng.choice(len(boxes), size=min(n_gt, len(boxes)), replace=False).tolist())
            cand = [("gt",) + boxes[k] for k in pick]
            if with_random:
                cand.append(("random", -1, -1) + random_box(rng, W, H))
            for k, (kind, ann_id, cat, l, t, r, b) in enumerate(cand):
                out.append(dict(image_id=int(im), roi_id=int(im) * 10 + k, split=split, kind=kind, ann_id=int(ann_id), cat_id=int(cat),
                                category=names.get(cat, ""), W=W, H=H, l=l, t=t, r=r, b=b,
                                n_in=n_in_f336(W, H, l, t, r, b), n_in_rae=RC.roi_geometry(W, H, (l, t, r, b))["n_in"]))
        return out

    coco = pd.DataFrame(rows(train, "train", 3, True) + rows(dev, "dev", 3, True))
    rae = pd.DataFrame(rows(hval, "head_val", 4, False))
    coco.to_parquet(os.path.join(OUT, "coco_rois.parquet"))
    rae.to_parquet(os.path.join(OUT, "rae_rois.parquet"))

    def bins(df):
        g = df[df.kind == "gt"].n_in
        return dict(d1_4=int((g <= 4).sum()), d5_16=int(((g >= 5) & (g <= 16)).sum()), d17p=int((g >= 17).sum()))

    info = dict(seed=SEED, min_side=MIN_SIDE, coco_images=len(imgs), eligible_images=len(eligible), paco_valtest_images=len(paco),
                train_images=int(len(train)), train_rois=int((coco.split == "train").sum()),
                train_paco_valtest_images=int(np.isin(train, list(paco)).sum()),
                dev_images=len(dev), dev_rois=int((coco.split == "dev").sum()), hval_images=len(hval), hval_rois=len(rae),
                overlap_train_dev=len(set(train) & set(dev)), overlap_train_hval=len(set(train) & set(hval)),
                overlap_dev_hval=len(set(dev) & set(hval)), overlap_devhval_paco=len((set(dev) | set(hval)) & paco),
                f336_bins_train_gt=bins(coco[coco.split == "train"]), f336_bins_dev_gt=bins(coco[coco.split == "dev"]), f336_bins_hval=bins(rae))
    json.dump(info, open(os.path.join(OUT, "make_rois.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
