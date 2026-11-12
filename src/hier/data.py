"""Detail inputs of H0 / H1. Same canvas geometry, kernel (PIL bicubic = the P384 canvas kernel), normalization and padding.

  H0: original -> the P384 384 content -> content rect of the 640 canvas     (no pixel that the 384 input did not already have)
  H1: original -> content rect of the 640 canvas                             (identity when the original already has that size)
The 640 content rect is the 384 content rect x 5/3 (rounded), so the 640 canvas maps onto the P384 grids by a fixed factor.
"""
import math
import os

import numpy as np
import torch
from PIL import Image

from src.config import COCO_TRAIN
from src.hcf.distill import TrainDS
from src.viz.c0_pca import LB, MEAN_T, STD_T

CANVAS = 640
SCALE = CANVAS / 384


def rect(full):
    """384 letterbox -> (ox, oy, nw, nh) on the 640 canvas."""
    r = [math.floor(v * SCALE + .5) for v in (full.ox, full.oy, full.nw, full.nh)]
    assert r[0] + r[2] <= CANVAS and r[1] + r[3] <= CANVAS, r
    return tuple(r)


def detail_uint8(img, full, cond):
    _, _, nw, nh = rect(full)
    img = img.convert("RGB")
    if cond == "H0":
        return img.resize((full.nw, full.nh), resample=Image.BICUBIC).resize((nw, nh), resample=Image.BICUBIC)
    if cond == "H1":
        return img.resize((nw, nh), resample=Image.BICUBIC)
    raise ValueError(cond)


def detail_canvas(img, full, cond):
    ox, oy, nw, nh = rect(full)
    r = np.asarray(detail_uint8(img, full, cond), dtype=np.float32)
    x = torch.zeros(3, CANVAS, CANVAS)
    x[:, oy:oy + nh, ox:ox + nw] = (torch.from_numpy(r.copy()).permute(2, 0, 1) / 255.0 - MEAN_T) / STD_T
    return x


def shuffle_tiles(d, generator):
    """(B, 3, 640, 640) -> content 8 x 8 tiles of each image permuted (padding tiles stay); the ablation input."""
    B = len(d)
    t = d.unfold(2, 8, 8).unfold(3, 8, 8).permute(0, 2, 3, 1, 4, 5).reshape(B, 6400, 3, 8, 8)
    for i in range(B):
        keep = torch.nonzero(t[i].abs().flatten(1).sum(1) > 0).squeeze(1)
        t[i, keep] = t[i, keep[torch.randperm(len(keep), generator=generator, device=keep.device)]]
    return t.reshape(B, 80, 80, 3, 8, 8).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 80, 8, 80, 8).permute(0, 1, 2, 4, 3, 5).reshape(B, 3, 640, 640)


class HierDS(TrainDS):
    """P384 batch (same stream, boxes, teacher cache) + one Detail canvas per image."""

    def __init__(self, shared, cond):
        super().__init__(shared)
        self.cond = cond

    def __getitem__(self, item):
        b = super().__getitem__(item)
        _, slots, _ = item
        details = []
        for _, iid in slots:
            img = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
            details.append(detail_canvas(img, LB(*img.size, 384, 16), self.cond))
        b["detail"] = torch.stack(details)
        return b
