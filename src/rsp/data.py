"""RSP data.

COCO training stream (RSPStreamDS): identical image / flip / ROI draws to S3 and FA (src.tsplit.train.StreamDS): rows 0..11,999 of
src.config.STREAM, 16 images / step minus the PACO val/test duplicates, hflip with rng (step, slot), 8 parent rectangles per image,
H, W ~ U{1..6} parents with rng (step, slot, 1) (src.tsplit.train.sample_rects). No 768 canvas is built (not needed).
Per ROI:
  box   = rect ∩ letterbox content (canvas px, ltrb)              -> z and evidence geometry
  ev    = the box mapped to ORIGINAL image px, cropped from the original (flipped) RGB and resized to 64 x 64 (PIL bicubic with
          antialias), ImageNet-normalised. Only built when want_ev (RSP-Masked); noPix gets no RGB input at all.
  query = K = min(256, #valid) fine cells (4 px) of the rect drawn without replacement with rng (step, slot, 2, roi); valid = the fine
          cell overlaps the letterbox content. Same draws for both runs (independent of want_ev).
PACO (RSPContentDS): src.tsplit.features.ContentDS + per ROI 'ev' (GT box from the original image, 64 x 64) and 'boxc' (GT box on the
384 canvas, continuous). Flips are re-derived by replaying the dataset's own torch RNG draws; the recomputed bil_grid must equal the
parent's (asserted), so the evidence always comes from the same (flipped) image and box as the R2 read.
"""
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import COCO_TRAIN
from src.readout.common import C
from src.rsp.common import EV_SIDE, GF, K_TRAIN, PT, R_FP
from src.tsplit.features import ContentDS
from src.tsplit.train import ROIS_PER_IMG, sample_rects
from src.viz.c0_pca import LB, MEAN_T, STD_T, geometry


def evidence(img, lb, box):
    """img: PIL (already flipped), lb: its 384 LB, box: ltrb on the 384 canvas -> (3, 64, 64) normalised crop of ORIGINAL pixels."""
    W, H = img.size
    l, t, r, b = box
    lo = min(max((l - lb.ox) * W / lb.nw, 0.0), W)
    ro = min(max((r - lb.ox) * W / lb.nw, 0.0), W)
    to = min(max((t - lb.oy) * H / lb.nh, 0.0), H)
    bo = min(max((b - lb.oy) * H / lb.nh, 0.0), H)
    if ro - lo < 1e-3:
        ro = min(lo + 1e-3, W) if lo < W else lo
        lo = ro - 1e-3
    if bo - to < 1e-3:
        bo = min(to + 1e-3, H) if to < H else to
        to = bo - 1e-3
    a = np.asarray(img.resize((EV_SIDE, EV_SIDE), resample=Image.BICUBIC, box=(lo, to, ro, bo)), dtype=np.float32)
    return (torch.from_numpy(a).permute(2, 0, 1) / 255.0 - MEAN_T) / STD_T


def rect_box(rect, lb):
    py, px, H, W = rect
    return [max(px * 16, lb.ox), max(py * 16, lb.oy), min((px + W) * 16, lb.ox + lb.nw), min((py + H) * 16, lb.oy + lb.nh)]


def fine_valid_np(rect, lb):
    """parent rect -> (4H * 4W,) bool, row-major over the fine rectangle: fine cell overlaps the content."""
    py, px, H, W = rect
    uy = np.arange(R_FP * py, R_FP * (py + H))
    ux = np.arange(R_FP * px, R_FP * (px + W))
    oy = np.clip(np.minimum(uy * PT + PT, lb.oy + lb.nh) - np.maximum(uy * PT, lb.oy), 0, None) > 0
    ox = np.clip(np.minimum(ux * PT + PT, lb.ox + lb.nw) - np.maximum(ux * PT, lb.ox), 0, None) > 0
    return (oy[:, None] & ox[None, :]).reshape(-1)


def sample_queries(rng, valid, K=K_TRAIN):
    cand = np.flatnonzero(valid)
    k = min(K, len(cand))
    sel = np.sort(rng.choice(cand, k, replace=False))
    q = np.zeros(K, np.int64)
    ok = np.zeros(K, bool)
    q[:k], ok[:k] = sel, True
    return q, ok


class RSPStreamDS(Dataset):
    def __init__(self, stream, exclude, want_ev=True, K=K_TRAIN, flip=True, rois=ROIS_PER_IMG):
        self.stream, self.ex, self.want_ev, self.K, self.flip, self.rois = stream, set(int(i) for i in exclude), want_ev, K, flip, rois

    def __len__(self):
        return len(self.stream)

    def __getitem__(self, s):
        xs, content, rects, rimg, boxes, evs, qs, qoks, ids = [], [], [], [], [], [], [], [], []
        for c, iid in enumerate(self.stream[s]):
            iid = int(iid)
            if iid in self.ex:
                continue
            rng = np.random.default_rng([s, c])
            with Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")) as im:
                img = im.convert("RGB")
            if bool(rng.random() < 0.5) and self.flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lb = LB(img.size[0], img.size[1], 384, 16)
            k = len(xs)
            xs.append(lb.canvas(img))
            content.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
            for ri, r in enumerate(sample_rects(np.random.default_rng([s, c, 1]), lb, self.rois)):
                bx = rect_box(r, lb)
                rects.append(r)
                rimg.append(k)
                boxes.append(bx)
                if self.want_ev:
                    evs.append(evidence(img, lb, bx))
                q, ok = sample_queries(np.random.default_rng([s, c, 2, ri]), fine_valid_np(r, lb), self.K)
                qs.append(q)
                qoks.append(ok)
            ids.append(iid)
        out = dict(step=s, x=torch.stack(xs), content=torch.tensor(content, dtype=torch.float32), rects=torch.tensor(rects, dtype=torch.long),
                   rimg=torch.tensor(rimg, dtype=torch.long), box=torch.tensor(boxes, dtype=torch.float32), q=torch.from_numpy(np.stack(qs)),
                   q_ok=torch.from_numpy(np.stack(qoks)), ids=torch.tensor(ids))
        if self.want_ev:
            out["ev"] = torch.stack(evs)
        return out


def frects_of(rects):
    """parent rects (R, 4) [py, px, H, W] -> fine rects (R, 4) [fy, fx, h, w]."""
    return rects * R_FP


class RSPContentDS(ContentDS):
    """ContentDS ('bil') + ev (R, 3, 64, 64) and boxc (R, 4) ltrb on the 384 canvas, for the RSP R2 read."""
    WANT_EV = True

    def __getitem__(self, idx):
        assert self.mode == "bil" and self.full_size == self.S == 384
        state0 = torch.get_rng_state()
        out = super().__getitem__(idx)
        state1 = torch.get_rng_state()
        torch.set_rng_state(state0)
        flips = [bool(self.flip and torch.rand(1).item() < 0.5) for _ in idx]
        assert torch.equal(torch.get_rng_state(), state1), "flip replay consumed a different number of draws"
        evs, boxc, grids = [], [], []
        for gi, f in zip(idx, flips):
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert("RGB")
            W, H = img.size
            if f:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            full = LB(W, H, 384, 16)
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                _, _, grid = geometry(W, H, bx, 384, 16)
                grids.append(grid)
                l, t, rr, bb = bx
                cb = [full.ox + l * full.nw / W, full.oy + t * full.nh / H, full.ox + rr * full.nw / W, full.oy + bb * full.nh / H]
                boxc.append(cb)
                if self.WANT_EV:
                    evs.append(evidence(img, full, cb))
        assert np.array_equal(np.stack(grids), out["bil_grid"].numpy()), "evidence geometry differs from the R2 read geometry"
        out["boxc"] = torch.tensor(boxc, dtype=torch.float32)
        if self.WANT_EV:
            out["ev"] = torch.stack(evs)
        return out
