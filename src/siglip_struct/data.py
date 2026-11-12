import numpy as np
import pandas as pd
import torch

from src.config import FEAT
from src.siglip_pm.data import LB
from src.siglip_pm.distill import TrainDS
from src.siglip_pm.bbdata import BBImageDS as OriginalBB
from src.siglip_struct.geometry import box_from_bilinear


def rectangle(w, h):
    lb = LB(int(w), int(h), 384, 16)
    return [lb.ox, lb.oy, lb.ox+lb.nw, lb.oy+lb.nh]


class TrainingDS(TrainDS):
    def __init__(self, shared):
        super().__init__(shared)
        df = pd.read_parquet(FEAT+'/stream_rois.parquet', columns=['image_id', 'W', 'H']).drop_duplicates('image_id')
        self.rects = {int(r.image_id): rectangle(r.W, r.H) for r in df.itertuples()}

    def __getitem__(self, item):
        b = super().__getitem__(item)
        b['valid_rect'] = torch.tensor([self.rects[i] for _, i in item[1]], dtype=torch.float32)
        return b


class BBImageDS(OriginalBB):
    def __getitem__(self, idx):
        b = super().__getitem__(idx)
        assert self.mode=='bil'
        b['box_c'] = box_from_bilinear(b)
        b['valid_rect'] = torch.tensor([rectangle(*self.WH[self.groups[i][0]]) for i in idx], dtype=torch.float32)
        return b


def micro_batches(b, size):
    """Split image slots and remap ROIs, preserving exact full-batch loss weights."""
    for lo in range(0, len(b['full']), size):
        hi = min(lo+size, len(b['full']))
        keep = (b['roi_img']>=lo) & (b['roi_img']<hi)
        out = {k: b[k][lo:hi] for k in ['full', 'fvalid', 'gembed', 'gmean', 'valid_rect']}
        out.update({k: b[k][keep] for k in ['tpool', 'box_c', 'nested'] if k in b})
        out['roi_img'] = b['roi_img'][keep]-lo
        yield out
