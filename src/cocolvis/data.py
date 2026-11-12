"""COCO / LVIS inputs = the VAW R2 input pipeline (src.vaw.data.VawDS: identical letterbox, box mapping, R2 geometry, hflip),
with one adaptation: an image with more than MAX_GROUP (32) GT boxes is split into consecutive same-image chunks of
<= 32 ROIs (COCO/LVIS images can carry hundreds of boxes; the ~128-ROI step packer needs >= 4 groups per step).
Full mode re-encodes the same full image once per chunk (identical tokens, so the reader input is unchanged).
"""
import numpy as np

from src.cocolvis.common import MAX_GROUP
from src.vaw.data import VawDS, check  # noqa: F401  (geometry check re-used)


def chunk_groups(groups, cap=MAX_GROUP):
    out = []
    for g in groups:
        k = int(np.ceil(len(g) / cap))
        out += [c for c in np.array_split(g, k)] if k > 1 else [g]
    return out


class RegionDS(VawDS):
    def __init__(self, D, labels, flip, mode, S=384, P=16):
        inst = [dict(image_id=int(i), instance_bbox=[x, y, w, h]) for i, x, y, w, h in zip(D.image_id, D.x, D.y, D.w, D.h)]
        first = D.drop_duplicates("image_id")
        paths = dict(zip(first.image_id.astype(int), first.path))
        WH = {int(i): (int(w), int(h)) for i, w, h in zip(first.image_id, first.W, first.H)}
        super().__init__(inst, labels, paths, WH, flip, mode, S, P)
        self.groups = chunk_groups(self.groups)
        self.n_roi = np.array([len(g) for g in self.groups])
        assert self.n_roi.max() <= MAX_GROUP and self.n_roi.sum() == len(D)
