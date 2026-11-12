"""M1/M2 COCO batch: the P8 TrainDS batch plus a locally edited view of every image.

The P8 batch (full canvases, boxes, cached ROI teacher means, global targets) is produced by TrainDS unchanged. Added here:
  full_p        the same image after the local edit, resized with the P8 rule
  crops0/1      the original and edited crop of the edited ROI, with the P8 crop rule (for the frozen teacher)
  cue_index     position of the edited ROI in the batch's ROI list
  crop1_mask    ROIs whose teacher target is still known on the edited view: the edited ROI (online target) and ROIs that do not overlap it
  native0/1     (M2 only) the undistorted image in its own resolution, padded to NATIVE x NATIVE
"""
import os

import numpy as np
from PIL import Image
import torch

from src.config import COCO_TRAIN
from src.siglip_cue import transforms as T
from src.siglip_pm.data import LB, geometry
from src.siglip_pm.distill import TrainDS

NATIVE = 640
S, P = 384, 16


def to_tensor(arr):
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 127.5 - 1


def native_canvas(arr):
    """(H, W, 3) uint8 at native size -> (3, NATIVE, NATIVE) in the model range, padded bottom/right with 0 (mid grey, same value as canvas padding)."""
    h, w = arr.shape[:2]
    assert max(w, h) <= NATIVE, (w, h)
    x = torch.zeros(3, NATIVE, NATIVE)
    x[:, :h, :w] = to_tensor(arr)
    return x


def native_geo(w, h):
    fl = LB(w, h, S, P)
    return torch.tensor([w, h, fl.nw, fl.nh, fl.ox, fl.oy], dtype=torch.float32)


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def load_rgb(iid):
    return np.asarray(Image.open(os.path.join(COCO_TRAIN, f'{iid:012d}.jpg')).convert('RGB'))


def choose_box(boxes, rng):
    ok = [j for j, b in enumerate(boxes) if min(b[2] - b[0], b[3] - b[1]) >= T.MIN_SIDE]
    return int(rng.choice(ok)) if ok else -1


class CueDS(TrainDS):
    def __init__(self, shared, native=False):
        super().__init__(shared)
        self.native = native

    def edit_slot(self, step, column, iid, arr, donor, region='full'):
        """The one deterministic edit of one stream slot. Used by the batch and, with region != 'full', by the sanity controls."""
        i = self.pos[iid]
        nested = [tuple(int(x) for x in box) for box in self.NB[step, column] if box[2] > box[0]]
        boxes = [tuple(int(x) for x in box) for box in self.obj[i]] + nested
        rng = T.item_rng(step, column)
        j = choose_box(boxes, rng)
        if j < 0:                                                      # no box is large enough: this image has no edit (identical views)
            j = int(np.argmax([min(b[2] - b[0], b[3] - b[1]) for b in boxes]))
            return boxes, j, arr, dict(op=-1, sigma=0., strength=0., feather=0., luma_abs=0., chroma_abs=0.)
        edited, m = T.apply(arr, boxes[j], donor, rng, region)
        return boxes, j, edited, m

    def __getitem__(self, item):
        base = super().__getitem__(item)
        step, slots, _ = item
        arrays = [load_rgb(iid) for _, iid in slots]
        canvases, crops0, crops1, cov, grid_list, idx, img_idx, masks, meta, natives0, natives1, geos, cue_boxes = [], [], [], [], [], [], [], [], [], [], [], [], []
        offset = 0
        for k, (column, iid) in enumerate(slots):
            arr = arrays[k]
            H, W = arr.shape[:2]
            boxes, j, edited, m = self.edit_slot(step, column, iid, arr, arrays[(k + 1) % len(arrays)])
            bx = boxes[j]
            fl = LB(W, H, S, P)
            canvases.append(fl.canvas(Image.fromarray(edited)))
            crop, _, grid = geometry(W, H, bx, S, P)
            crops0.append(crop.canvas(Image.fromarray(arr).crop(bx)))
            crops1.append(crop.canvas(Image.fromarray(edited).crop(bx)))
            cov.append(crop.coverage())
            grid_list.append(grid)
            idx.append(offset + j)
            img_idx.append(k)
            masks.append([q == j or not overlaps(boxes[q], bx) for q in range(len(boxes))])
            meta.append([m['op'], m['sigma'], m['strength'], m['luma_abs'], m['chroma_abs'], bx[2] - bx[0], bx[3] - bx[1], max(W, H)])
            cue_boxes.append(bx)
            if self.native:
                natives0.append(native_canvas(arr))
                natives1.append(native_canvas(edited))
                geos.append(native_geo(W, H))
            offset += len(boxes)
        assert offset == len(base['tpool'])
        base.update(full_p=torch.stack(canvases), crops0=torch.stack(crops0), crops1=torch.stack(crops1), cue_cov=torch.from_numpy(np.stack(cov)),
                    cue_grid=torch.from_numpy(np.stack(grid_list)), cue_index=torch.tensor(idx), cue_img=torch.tensor(img_idx),
                    crop1_mask=torch.tensor(np.concatenate(masks), dtype=torch.bool), cue_meta=torch.tensor(np.array(meta), dtype=torch.float32),
                    cue_box=torch.tensor(cue_boxes, dtype=torch.float32))
        if self.native:
            base.update(native0=torch.stack(natives0), native1=torch.stack(natives1), geo=torch.stack(geos))
        return base
