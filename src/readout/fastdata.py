"""Fast input pipeline: pre-built uint8 letterbox canvases (same PIL code as the on-the-fly pipeline) + GPU geometry.

Cache (built once, shared by every job; default /dev/shm/readout_cache):
  full_{train,val}_{noflip,flip}.u8 : (n_img, 336, 336, 3) uint8 = np.asarray(img[.transpose(FLIP)].resize((nw, nh), BICUBIC))
                                      pasted at (ox, oy); padding pixels are never read (zeroed after normalisation)
  c0_{train,val}_{noflip,flip}.u8   : (n_roi, 336, 336, 3) uint8 GT crops (roi_geometry crop of the [flipped] image,
                                      flipped box = C.flip_box), same letterbox rule
  meta.npz                           : letterbox rects (ox, oy, nw, nh) per image / per crop, row order = manifest
On the GPU: x = (u8 / 255 - MEAN) / STD inside the content rect, 0 outside (= common.to_tensor_canvas, same fp32 ops);
geometry (geo, box4, cov) from the box in canvas pixels, computed in float64 then cast to float32 (= common_rr.roi_geo /
common_tu.canvas_geo / crop letterbox coverage). Flip decisions stay random per image per epoch (torch.rand in the worker,
p = 0.5) exactly as before; only the decode/resize work moved to the one-off cache.

  python -m src.readout.fastdata build --out /dev/shm/readout_cache [--c0]
  python -m src.readout.fastdata verify --cache /dev/shm/readout_cache
"""
import argparse
import json
import math
import os
import time
from multiprocessing import get_context

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import src.readout.common as RC
from src.readout.common import C, G, PATCH, S, Letterbox, roi_geometry

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"


# ------------------------------------------------------------------------------------------------ build
def _canvas(img, lb):
    out = np.zeros((S, S, 3), np.uint8)
    out[lb.oy:lb.oy + lb.nh, lb.ox:lb.ox + lb.nw] = np.asarray(img.resize((lb.nw, lb.nh), resample=Image.BICUBIC))
    return out


G_ = {}


def _job_full(i):
    p, W, H = G_["p"][i], G_["W"][i], G_["H"][i]
    img = Image.open(p).convert("RGB")
    assert img.size == (W, H)
    lb = Letterbox(W, H)
    G_["mm"]["noflip"][i] = _canvas(img, lb)
    if "flip" in G_["mm"]:
        G_["mm"]["flip"][i] = _canvas(img.transpose(Image.FLIP_LEFT_RIGHT), lb)
    return i


def _job_c0(gi):
    rows = G_["groups"][gi]
    p, W, H = G_["p"][rows[0]], G_["W"][rows[0]], G_["H"][rows[0]]
    img = Image.open(p).convert("RGB")
    imf = img.transpose(Image.FLIP_LEFT_RIGHT) if "flip" in G_["mm"] else None
    rect = []
    for r in rows:
        bx = tuple(G_["box"][r])
        g = roi_geometry(W, H, bx)
        G_["mm"]["noflip"][r] = _canvas(img.crop(g["ltrb"]), g["crop"])
        if imf is not None:
            gf = roi_geometry(W, H, C.flip_box(bx, W))
            G_["mm"]["flip"][r] = _canvas(imf.crop(gf["ltrb"]), gf["crop"])
        cl = g["crop"]
        rect.append((r, cl.ox, cl.oy, cl.nw, cl.nh))
    return rect


def build(out, with_c0, workers):
    os.makedirs(out, exist_ok=True)
    data = RC.load_data(EXCL)
    meta = {}
    for sp, (df, Y) in data.items():
        groups = C.image_groups(df)
        first = np.array([g[0] for g in groups])
        p, W, H = df.path.values[first], df.W.values[first].astype(int), df.H.values[first].astype(int)
        n = len(groups)
        mm = {k: np.lib.format.open_memmap(os.path.join(out, f"full_{sp}_{k}.npy"), "w+", np.uint8, (n, S, S, 3))
              for k in (("noflip", "flip") if sp == "train" else ("noflip",))}
        G_.update(p=p, W=W, H=H, mm=mm)
        t = time.time()
        with get_context("fork").Pool(workers) as pool:
            for _ in pool.imap_unordered(_job_full, range(n), chunksize=16):
                pass
        for m in mm.values():
            m.flush()
        lbs = [Letterbox(w, h) for w, h in zip(W, H)]
        meta[f"full_{sp}_rect"] = np.array([(l.ox, l.oy, l.nw, l.nh) for l in lbs], np.int32)
        meta[f"full_{sp}_image_id"] = df.image_id.values[first]
        print(sp, "full", n, round(time.time() - t, 1), "s", flush=True)
        if with_c0:
            n_r = len(df)
            mm = {k: np.lib.format.open_memmap(os.path.join(out, f"c0_{sp}_{k}.npy"), "w+", np.uint8, (n_r, S, S, 3))
                  for k in (("noflip", "flip") if sp == "train" else ("noflip",))}
            G_.update(p=df.path.values, W=df.W.values.astype(int), H=df.H.values.astype(int), mm=mm, groups=groups,
                      box=df[["l", "t", "r", "b"]].values.astype(int))
            rect = np.zeros((n_r, 4), np.int32)
            t = time.time()
            with get_context("fork").Pool(workers) as pool:
                for rr in pool.imap_unordered(_job_c0, range(n), chunksize=8):
                    for r, *v in rr:
                        rect[r] = v
            for m in mm.values():
                m.flush()
            meta[f"c0_{sp}_rect"] = rect
            print(sp, "c0", n_r, round(time.time() - t, 1), "s", flush=True)
    np.savez(os.path.join(out, "meta.npz"), **meta)
    json.dump(dict(done=True, c0=with_c0, time=time.strftime("%F %T"), exclude=EXCL), open(os.path.join(out, "done.json"), "w"))


# ------------------------------------------------------------------------------------------------ dataset
class GlobalStepSampler(RC.RankStepSampler):
    """RankStepSampler + the step's GLOBAL label / ROI counts (so no all-reduce is needed for the loss normaliser)."""

    def __init__(self, n_roi, rank, world, Ytr, groups, seed=0):
        super().__init__(n_roi, rank, world, seed)
        self.known = np.array([int((Ytr[g] >= 0).sum()) for g in groups], np.int64)

    def __iter__(self):
        k = 4 // self.world
        for st in C.full_epoch_batches(self.n_roi, self.epoch, self.seed):
            allimg = [i for sh in st for i in sh]
            mine = [i for sh in st[self.rank * k:(self.rank + 1) * k] for i in sh]
            yield (mine, float(self.known[allimg].sum()), float(self.n_roi[allimg].sum()))


class CachedDS(Dataset):
    def __init__(self, df, Y, cache, split, flip, mode):
        self.groups = C.image_groups(df)
        self.n_roi = np.array([len(g) for g in self.groups])
        self.box = df[["l", "t", "r", "b"]].values.astype(int)
        self.WH = df[["W", "H"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.Y, self.flip, self.mode, self.split = torch.as_tensor(Y), flip, mode, split
        meta = np.load(os.path.join(cache, "meta.npz"))
        kind = "c0" if mode == "c0" else "full"
        self.rect = meta[f"{kind}_{split}_rect"]
        if kind == "full":
            ids = meta[f"full_{split}_image_id"]
            assert (ids == df.image_id.values[[g[0] for g in self.groups]]).all(), "cache / manifest image order mismatch"
        else:
            assert len(self.rect) == len(df)
        self.files = {k: os.path.join(cache, f"{kind}_{split}_{k}.npy") for k in ("noflip", "flip")}
        self.mm = None

    def __len__(self):
        return len(self.groups)

    def _open(self):
        if self.mm is None:
            self.mm = {k: np.load(f, mmap_mode="r") for k, f in self.files.items() if os.path.exists(f)}

    def __getitem__(self, item):
        self._open()
        if isinstance(item, tuple):
            idx, ntg, nrg = item
        else:
            idx, ntg, nrg = item, None, None
        xs, rects, boxes, img_index, rows_all, grp = [], [], [], [], [], []
        for k, gi in enumerate(idx):
            rows = self.groups[gi]
            W = int(self.WH[rows[0], 0])
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            key = "flip" if f else "noflip"
            if self.mode == "c0":
                xs.append(self.mm[key][rows])
                rects.append(self.rect[rows])
                img_index += list(range(len(img_index), len(img_index) + len(rows)))
            else:
                xs.append(self.mm[key][gi][None])
                rects.append(self.rect[gi][None])
                lb = Letterbox(W, int(self.WH[rows[0], 1]))
                for r in rows:
                    bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                    X0, Y0 = lb.to_canvas(bx[0], bx[1])
                    X1, Y1 = lb.to_canvas(bx[2], bx[3])
                    boxes.append((float(X0), float(Y0), float(X1), float(Y1)))
                img_index += [k] * len(rows)
            rows_all.append(rows)
            grp += [k] * len(rows)
        rows_all = np.concatenate(rows_all)
        out = dict(u8=torch.from_numpy(np.concatenate(xs)), rect=torch.from_numpy(np.concatenate(rects)),
                   img_index=torch.as_tensor(img_index, dtype=torch.long), rows=torch.from_numpy(rows_all),
                   lv=torch.from_numpy(self.lv[rows_all]), Y=self.Y[rows_all], group=torch.as_tensor(grp, dtype=torch.long))
        if boxes:
            out["boxes"] = torch.tensor(boxes, dtype=torch.float64)
        if ntg is not None:
            out["nt_global"], out["nroi_global"] = torch.tensor(ntg), torch.tensor(nrg)
        return out


# ------------------------------------------------------------------------------------------------ GPU side
def gpu_geo(boxes, side):
    """boxes (n,4) float64 canvas px [X0,Y0,X1,Y1] -> geo (n,T,5), box4 (n,4), cov (n,T) float32 (roi_geo formula)."""
    Gc = side // PATCH
    dev = boxes.device
    e = torch.arange(Gc + 1, device=dev, dtype=torch.float64) * PATCH
    X0, Y0, X1, Y1 = (boxes[:, i:i + 1] for i in range(4))
    cx = torch.clamp(torch.minimum(e[1:], X1) - torch.maximum(e[:-1], X0), min=0) / PATCH
    cy = torch.clamp(torch.minimum(e[1:], Y1) - torch.maximum(e[:-1], Y0), min=0) / PATCH
    cov = (cy[:, :, None] * cx[:, None, :]).reshape(len(boxes), -1)
    bw, bh = X1 - X0, Y1 - Y0
    ci, cj = torch.div(torch.arange(Gc * Gc, device=dev), Gc, rounding_mode="floor"), torch.arange(Gc * Gc, device=dev) % Gc
    px, py = (cj + 0.5) * PATCH, (ci + 0.5) * PATCH
    n, T = len(boxes), Gc * Gc
    geo = torch.stack([(px[None] - X0) / bw, (py[None] - Y0) / bh, cov, torch.log(bw / PATCH).expand(n, T), torch.log(bh / PATCH).expand(n, T)], -1)
    box4 = torch.cat([X0 / side, Y0 / side, bw / side, bh / side], 1)
    return geo.float(), box4.float(), cov.float()


def gpu_prepare(b, dev, mode):
    """CPU batch from CachedDS -> the dict the readers / training loop expect (same keys as PacoImageDS)."""
    out = {k: v.to(dev, non_blocking=True) for k, v in b.items() if torch.is_tensor(v)}
    if "nt_global" in b:
        out["nt_cpu"] = float(b["nt_global"])                      # CPU value for logging (no device sync)
    u8 = out.pop("u8")
    x = u8.permute(0, 3, 1, 2).float()
    x = (x / 255.0 - MEAN.to(dev)) / STD.to(dev)
    rect = out.pop("rect").long()
    ar = torch.arange(S, device=dev)
    mx = (ar[None] >= rect[:, 0:1]) & (ar[None] < (rect[:, 0] + rect[:, 2])[:, None])
    my = (ar[None] >= rect[:, 1:2]) & (ar[None] < (rect[:, 1] + rect[:, 3])[:, None])
    x = x * (my[:, :, None] & mx[:, None, :])[:, None].float()
    out["x"] = x
    if mode == "c0":
        bx = torch.stack([rect[:, 0], rect[:, 1], rect[:, 0] + rect[:, 2], rect[:, 1] + rect[:, 3]], 1).double()
        side = S
    else:
        bx = out.pop("boxes")
        side = 672 if mode == "f672" else S
        if mode == "f672":
            bx = bx * 2.0                                       # common_tu.full_box_canvas: float(X0) * (672 / 336)
    out["geo"], out["box4"], out["cov"] = gpu_geo(bx, side)
    out["side"] = side
    return out


# ------------------------------------------------------------------------------------------------ verify
def verify(cache, n=64, modes=("full", "f672", "c0")):
    """Same images through the old (PacoImageDS) and new (cache + GPU) path: inputs and geometry must agree."""
    data = RC.load_data(EXCL)
    rep = {}
    dev = torch.device("cuda")
    for sp in ("train", "val"):
        df, Y = data[sp]
        for mode in modes:
            if mode == "c0" and not os.path.exists(os.path.join(cache, f"c0_{sp}_noflip.npy")):
                continue
            for flip in ((False, True) if sp == "train" else (False,)):
                old = RC.PacoImageDS(df, Y, flip=False, mode=mode)
                new = CachedDS(df, Y, cache, sp, flip=False, mode=mode)
                if flip:   # force flip on both: monkeypatch the coin
                    old.flip = new.flip = True
                idx = list(np.random.default_rng(1).choice(len(old), n, replace=False))
                real_rand = torch.rand
                if flip:
                    torch.rand = lambda *x, **k: torch.zeros(*x)          # coin < 0.5 -> every image flipped in BOTH paths
                try:
                    bo = old[idx]
                    bn = gpu_prepare(new[idx], dev, mode)
                finally:
                    torch.rand = real_rand
                d = {k: float((bo[k].to(dev).float() - bn[k].float()).abs().max()) for k in ("x", "geo", "box4", "cov")}
                d.update({k: bool(torch.equal(bo[k].to(dev), bn[k])) for k in ("rows", "lv", "Y", "img_index")})
                rep[f"{sp}/{mode}/{'flip' if flip else 'noflip'}"] = d
                print(sp, mode, flip, d, flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--out", default="/dev/shm/readout_cache")
    ap.add_argument("--cache", default="/dev/shm/readout_cache")
    ap.add_argument("--c0", action="store_true")
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.c0, a.workers)
    else:
        rep = verify(a.cache)
        json.dump(rep, open(os.path.join(a.cache, "verify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
