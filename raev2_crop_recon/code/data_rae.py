"""Cached COCO decoder data for the RAEv2 CoordMLP (training stream / dev / hval).

Item = one list of images (one stream row = 16 images, or a dev chunk). Per image: cached full raw latent (256,768) fp16;
per ROI (4 per image): cached teacher content cells scattered back to a (256,768) grid (padding cells = 0, weight 0),
roi_geometry (S=256), crop content rect, and optionally the GT crop RGB canvas (uint8 256 letterbox of the ORIGINAL crop,
the same u8_canvas used for the teacher encoder input).
Stream rows hold COCO image ids (make_stream.py).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import rae_common as RC

FEAT = os.path.join(RC.CACHE, "feat")


class CachedSet(Dataset):
    def __init__(self, name, rgb):
        self.name, self.rgb = name, rgb
        self.rois = pd.read_parquet(os.path.join(FEAT, f"{name}_rois.parquet"))
        self.by_slot = {s: g for s, g in self.rois.groupby("img_slot")}
        self.full = None
        self.teacher = None

    def _open(self):
        if self.full is None:
            self.full = np.load(os.path.join(FEAT, f"{self.name}_full.npy"), mmap_mode="r")
            self.teacher = np.load(os.path.join(FEAT, f"{self.name}_teacher.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.by_slot)

    def __getitem__(self, slots):
        self._open()
        full, T, geoms, idx, rects, rgb, roi_ids, nin = [], [], [], [], [], [], [], []
        for k, s in enumerate(slots):
            g = self.by_slot[s]
            full.append(torch.from_numpy(np.array(self.full[s])))
            img = None
            if self.rgb:
                r0 = g.iloc[0]
                img = Image.open(os.path.join(RC.COCO_TRAIN, f"{int(r0.image_id):012d}.jpg")).convert("RGB")
                assert img.size == (int(r0.W), int(r0.H))
            for r in g.itertuples():
                ltrb = (int(r.l), int(r.t), int(r.r), int(r.b))
                geo = RC.roi_geometry(int(r.W), int(r.H), ltrb)
                cmask = geo["valid_w"] > 0
                assert cmask.sum() == r.t_n
                t = np.zeros((RC.NCELL, RC.D), np.float16)
                t[cmask] = self.teacher[r.t_off:r.t_off + r.t_n]
                T.append(torch.from_numpy(t))
                geoms.append({kk: geo[kk] for kk in RC.GEOM_KEYS})
                idx.append(k)
                rects.append(geo["crop"].rect())
                roi_ids.append(r.roi_id)
                nin.append(geo["n_in"])
                if img is not None:
                    rgb.append(RC.u8_canvas(img.crop(ltrb), geo["crop"]))
        out = dict(full=torch.stack(full), teacher=torch.stack(T), geom=RC.collate_geoms(geoms, idx),
                   rects=torch.as_tensor(np.array(rects)), roi_ids=roi_ids, n_in_rae=torch.as_tensor(nin))
        if rgb:
            out["rgb"] = torch.stack(rgb)
        return out


def stream_slots(steps):
    """Stream rows (12k x 16 COCO image ids, make_stream.py) -> stream-cache image slots."""
    sr = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    slot = dict(zip(sr.image_id, sr.img_slot))
    st = np.load(RC.STREAM)[:steps]
    return [[int(slot[i]) for i in row] for row in st]


def loader(ds, batches, workers=10, prefetch=2):
    return DataLoader(ds, batch_size=None, sampler=batches, num_workers=workers, pin_memory=True,
                      prefetch_factor=prefetch if workers else None, collate_fn=None, persistent_workers=False)


def seq_slots(ds, per=16):
    n = len(ds)
    return [list(range(k, min(n, k + per))) for k in range(0, n, per)]


def to_dev(b, dev):
    o = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items() if k != "geom"}
    o["geom"] = RC.to_device(b["geom"], dev)
    o["full"], o["teacher"] = o["full"].float(), o["teacher"].float()
    if "rgb" in o:
        o["rgb"] = o["rgb"].float() / 255
    o["mask"] = RC.content_mask(b["rects"], dev)
    return o


def teacher_stats(n=3000):
    """CoordMLP output de-normalisation buffers mean_T / std_T (the existing CoordMLP convention: per-channel stats of the
    teacher over valid cells), from the first n ROIs of the stream cache (deterministic, no PACO data)."""
    sr = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    T = np.load(os.path.join(FEAT, "stream_teacher.npy"), mmap_mode="r")
    end = int(sr.t_off.iloc[n - 1] + sr.t_n.iloc[n - 1])
    x = torch.from_numpy(np.asarray(T[:end]).astype(np.float32))
    return x.mean(0), x.std(0) + 1e-6
