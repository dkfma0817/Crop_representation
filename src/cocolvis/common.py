"""COCO 2017 / LVIS v1 GT-box region-classification headroom pilot (2026-11-07): paths and annotation loaders.

Instances (both datasets): one row per GT box usable as a region, in ORIGINAL image pixels.
  COCO : instances_{train,val}2017.json, iscrowd == 0 only (crowd regions are not single objects; cocoapi ignores them),
         80 classes, index = rank of the COCO category id.
  LVIS : lvis_v1_{train,val}.json (no iscrowd field), 1,203 classes, index = rank of the LVIS category id (1..1203);
         images are COCO 2017 files, path taken from the image's coco_url (train2017/ or val2017/).
Size bins (COCO definition, on the ORIGINAL GT box area w*h, NOT the segmentation 'area' field): S < 32^2 <= M < 96^2 <= L.
"""
import json
import os

import numpy as np
import pandas as pd

COCO_DIR = "/home/irteam/datasets/coco"
LVIS_DIR = "/home/irteam/datasets/lvis"
LVIS_API = "/home/irteam/datasets/lvis_src/lvis-api"
RUN = "output/2026-11-07/cocolvis_headroom"
ANN = {("coco", "train"): f"{COCO_DIR}/annotations/instances_train2017.json",
       ("coco", "val"): f"{COCO_DIR}/annotations/instances_val2017.json",
       ("lvis", "train"): f"{LVIS_DIR}/lvis_v1_train.json",
       ("lvis", "val"): f"{LVIS_DIR}/lvis_v1_val.json"}
NCLS = {"coco": 80, "lvis": 1203}
MAX_GROUP = 32          # images with more ROIs are split into same-image chunks of <= 32 ROIs (see data.groups)


def size_bin(area):
    return np.where(area < 32 ** 2, 0, np.where(area < 96 ** 2, 1, 2))           # 0 S, 1 M, 2 L


def image_path(ds, img):
    if ds == "coco":
        return None
    u = img["coco_url"].split("/")
    return os.path.join(COCO_DIR, u[-2], u[-1])


def load(ds, split, raw=None):
    """-> (inst DataFrame, images dict id -> record, categories list sorted by id). inst columns:
    image_id, ann_id, x, y, w, h (original px, as annotated), label (0..K-1), cat_id, box_area, size (0/1/2), path, W, H"""
    J = raw if raw is not None else json.load(open(ANN[(ds, split)]))
    cats = sorted(J["categories"], key=lambda c: c["id"])
    assert len(cats) == NCLS[ds]
    c2i = {c["id"]: i for i, c in enumerate(cats)}
    imgs = {im["id"]: im for im in J["images"]}
    rows = []
    for a in J["annotations"]:
        if ds == "coco" and a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        rows.append((a["image_id"], a["id"], x, y, w, h, c2i[a["category_id"]], a["category_id"]))
    D = pd.DataFrame(rows, columns=["image_id", "ann_id", "x", "y", "w", "h", "label", "cat_id"])
    D["box_area"] = D.w * D.h
    D["size"] = size_bin(D.box_area.values)
    folder = {"train": "train2017", "val": "val2017"}[split]
    if ds == "coco":
        D["path"] = [os.path.join(COCO_DIR, folder, imgs[i]["file_name"]) for i in D.image_id]
    else:
        D["path"] = [image_path(ds, imgs[i]) for i in D.image_id]
    D["W"] = [imgs[i]["width"] for i in D.image_id]
    D["H"] = [imgs[i]["height"] for i in D.image_id]
    D = D.sort_values(["image_id", "ann_id"], kind="mergesort").reset_index(drop=True)
    return D, imgs, cats
