"""Native 768 teacher input: the ORIGINAL image resized directly into the layout of the student's 384 canvas at scale 2.

The content rectangle is exactly 2 x the 384 letterbox (nw, nh, ox, oy), so every student canvas coordinate maps to 2 x that coordinate here and the 48 x 48 teacher cells (16 px at 768)
cover the same image regions as the 48 x 48 student cells (8 px at 384). Same resampling filter and normalization as LB.canvas (PIL bicubic, ImageNet mean/std); no pixel of the 384 tensor is used.
"""
import numpy as np
from PIL import Image
import torch

from src.viz.c0_pca import MEAN_T, STD_T


def native_canvas(img, lb):
    assert lb.S == 384
    r = np.asarray(img.convert("RGB").resize((2 * lb.nw, 2 * lb.nh), resample=Image.BICUBIC), dtype=np.float32)
    x = torch.zeros(3, 768, 768)
    x[:, 2 * lb.oy:2 * lb.oy + 2 * lb.nh, 2 * lb.ox:2 * lb.ox + 2 * lb.nw] = (torch.from_numpy(r).permute(2, 0, 1) / 255.0 - MEAN_T) / STD_T
    return x
