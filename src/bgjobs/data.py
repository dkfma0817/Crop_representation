"""PACO train ROI stream for the attribute-aware background jobs (A, A control, B share it exactly: same rows, order, flips and labels, so the label budget is identical).

Rows: PACO train minus the data_check exclusions (= RC.load_data(EXCL), the probes' train) minus the 1,000-image PACO dev (rng 0, = the probes' dev), so dev stays held out from the encoder.
Order: C.full_epoch_batches (the probes' image packs, ~128 ROIs/step, 4 shards), epochs 1, 2, ...; a step is a pure function of (seed, step); flips are a pure function of (seed, epoch, image).
Labels Y: 1 positive, 0 negative, -1 unknown. Unknown is never a negative (BCE mask, ranking pairs); the teacher logit KD covers all attributes, including unknown ones (no label is read there).
Geometry: R2 full-canvas (g1x.data.full_r2_geo), identical to the eval probe mode 'full'.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

import src.readout.common as RC
from src.g1x.data import full_r2_geo, roi_geom
from src.ivc.bbdata import crop_geo
from src.ivc.common import EXCL
from src.readout.common import C
from src.viz.c0_pca import LB


def load_rows():
    df, Y = RC.load_data(EXCL, splits=("train",))["train"]
    ids = np.array(sorted(df.image_id.unique()))
    dev = set(np.random.default_rng(0).choice(ids, 1000, replace=False).tolist())
    keep = ~df.image_id.isin(dev).values
    return df[keep].reset_index(drop=True), Y[keep]


class Plan:
    def __init__(self, df, seed, start_epoch=1):
        self.groups = C.image_groups(df)
        self.n_roi = np.array([len(g) for g in self.groups])
        self.seed = seed
        self.cum, self.cache = [0], {}                 # cum[e] = steps before epoch e + 1 ... (cum[0] = 0)
        self.cache_e = None
        self.cached = None

    def _epoch(self, e):
        if self.cache_e != e:
            self.cached = C.full_epoch_batches(self.n_roi, e, self.seed)
            self.cache_e = e
        return self.cached

    def locate(self, step):
        while self.cum[-1] <= step:
            e = len(self.cum)
            self.cum.append(self.cum[-1] + len(self._epoch(e)))
        e = next(i for i in range(1, len(self.cum)) if self.cum[i] > step)
        return e, step - self.cum[e - 1]

    def step_items(self, step):
        e, pos = self.locate(step)
        return e, pos, self._epoch(e)[pos]              # 4 shards of image indices

    def flip(self, e, gi):
        return bool(np.random.default_rng([self.seed, e, int(gi), 13]).random() < 0.5)


class StepSampler(Sampler):
    def __init__(self, plan, Y, start, stop, accum):
        self.plan, self.Y, self.start, self.stop, self.accum = plan, Y, start, stop, accum
        assert accum in (1, 2, 4)

    def __iter__(self):
        for g in range(self.start, self.stop):
            e, pos, shards = self.plan.step_items(g)
            k = 4 // self.accum
            rows = [r for sh in shards for gi in sh for r in self.plan.groups[gi]]
            nt = int((self.Y[rows] >= 0).sum())
            for m in range(self.accum):
                items = [(gi, self.plan.flip(e, gi)) for sh in shards[m * k:(m + 1) * k] for gi in sh]
                yield dict(step=g, m=m, epoch=e, items=items, nt=nt, n_rows=len(rows))

    def __len__(self):
        return (self.stop - self.start) * self.accum


class AttrDS(Dataset):
    """Item = sampler payload -> one micro step (one full canvas per image, R2 full-canvas geometry per ROI)."""

    def __init__(self, df, Y, S, P):
        self.groups = C.image_groups(df)
        self.p, self.box = df.path.values, df[["l", "t", "r", "b"]].values.astype(int)
        self.WH = df[["W", "H"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.cat = df.category_id.values.astype(np.int64)
        self.Y, self.S, self.P = torch.as_tensor(Y), S, P

    def __len__(self):
        return 1 << 40

    def __getitem__(self, pl):
        xs, fvalid, img_index, rows_all, flips = [], [], [], [], []
        per = {k: [] for k in ("geo", "box4", "cov")}
        for k, (gi, flip) in enumerate(pl["items"]):
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[rows[0]]), self.p[rows[0]]
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flb = LB(W, H, self.S, self.P)
            xs.append(flb.canvas(img))
            fvalid.append(flb.coverage() > 0)
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if flip else tuple(self.box[r])
                geo, box4, cov = full_r2_geo(roi_geom(W, H, bx, self.S, self.P), self.S, self.P)
                for kk, v in (("geo", geo), ("box4", box4), ("cov", cov)):
                    per[kk].append(v)
                img_index.append(k)
            rows_all.append(rows)
            flips += [int(flip)] * len(rows)
        rows_all = np.concatenate(rows_all)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items()}
        out.update(x=torch.stack(xs), fvalid=torch.from_numpy(np.stack(fvalid)), img_index=torch.as_tensor(img_index, dtype=torch.long), rows=torch.from_numpy(rows_all),
                   flips=torch.as_tensor(flips, dtype=torch.bool), lv=torch.from_numpy(self.lv[rows_all]), cat=torch.from_numpy(self.cat[rows_all]), Y=self.Y[rows_all],
                   step=pl["step"], m=pl["m"], epoch=pl["epoch"], nt=pl["nt"], n_rows=pl["n_rows"])
        return out


# ------------------------------------------------------------------------------------------------ teacher logits (crop encoder + C0 R2 reader; both flips)
class TeacherDS(Dataset):
    def __init__(self, df, flip, S, P):
        self.groups = C.image_groups(df)
        self.p, self.box = df.path.values, df[["l", "t", "r", "b"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.flip, self.S, self.P = flip, S, P

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        xs, rows_all, per = [], [], {k: [] for k in ("geo", "box4", "cov")}
        for gi in idx:
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert("RGB")
            W, H = img.size
            if self.flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if self.flip else tuple(self.box[r])
                crop = LB(bx[2] - bx[0], bx[3] - bx[1], self.S, self.P)
                xs.append(crop.canvas(img.crop(bx)))
                geo, box4, cov = crop_geo(crop, self.S, self.P)
                for kk, v in (("geo", geo), ("box4", box4), ("cov", cov)):
                    per[kk].append(v)
            rows_all.append(rows)
        rows_all = np.concatenate(rows_all)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items()}
        out.update(x=torch.stack(xs), rows=torch.from_numpy(rows_all), lv=torch.from_numpy(self.lv[rows_all]), img_index=torch.arange(len(xs)))
        return out


def cache_paths(cfg):
    d = cfg["data"]["teacher_cache"]
    return d, [os.path.join(d, f"flip{f}.npy") for f in (0, 1)], os.path.join(d, "done.json")
