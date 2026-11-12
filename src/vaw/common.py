"""VAW (CVPR'21) evaluation of the G1 encoders: shared paths / loaders (2026-11-07)."""
import json
import os

import numpy as np

VAW = "/home/irteam/datasets/vaw_src/vaw_dataset"
VAW_DATA = os.path.join(VAW, "data")
VG_DIR = "/home/irteam/datasets/visual_genome"
VG_META = os.path.join(VG_DIR, "image_data.json")
PACO_TRAIN = "/home/irteam/datasets/paco/paco_lvis_v1_train.json"
RUN = "output/2026-11-07/vaw_g1"
G1 = "output/2026-11-06/g1_cross_distill"
SPLITS = ("train", "val", "test")


def load_split(split):
    if split == "train":
        return (json.load(open(os.path.join(VAW_DATA, "train_part1.json")))
                + json.load(open(os.path.join(VAW_DATA, "train_part2.json"))))
    return json.load(open(os.path.join(VAW_DATA, f"{split}.json")))


def attr_index():
    """official attribute -> column index (attribute_index.json), and the inverse list."""
    a2i = json.load(open(os.path.join(VAW_DATA, "attribute_index.json")))
    i2a = [None] * len(a2i)
    for a, i in a2i.items():
        i2a[i] = a
    assert None not in i2a and len(i2a) == 620
    return a2i, i2a


def label_matrix(insts, a2i):
    """official encoding (README): 1 positive, 0 negative, 2 unlabeled -> (N, 620) int8"""
    Y = np.full((len(insts), len(a2i)), 2, dtype=np.int8)
    for r, x in enumerate(insts):
        for a in x["negative_attributes"]:
            Y[r, a2i[a]] = 0
        for a in x["positive_attributes"]:
            Y[r, a2i[a]] = 1
    return Y


SIZES = os.path.join(RUN, "leak", "image_sizes.json")


def image_sizes():
    """ACTUAL (W, H) of every VAW image read from the files (VG image_data.json is wrong for 17 VAW images)."""
    return {int(k): tuple(v) for k, v in json.load(open(SIZES)).items()}


def vg_meta():
    return {int(m["image_id"]): m for m in json.load(open(VG_META))}


def vg_path(m):
    """VG image file from the image_data url (VG_100K or VG_100K_2)."""
    u = m["url"].split("/")
    return os.path.join(VG_DIR, u[-2], u[-1])
