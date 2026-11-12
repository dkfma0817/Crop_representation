"""COCO train2017 OBJECT-ROI stream for the crop-aware pretraining.

Scope: COCO instance boxes only. PACO parts are NOT trained on; a future PACO-train part source only has to add rows
(image_id, W, H, l, t, r, b, area, source) to the ROI table built by build_roi_table -> the plan / dataset are source-agnostic.

Determinism / DDP: the data order is a PURE FUNCTION of (seed, global_step), independent of the world size:
  step g -> (epoch, pos); epoch permutation of images; images perm[pos*B : (pos+1)*B]; ROIs of an image = f(seed, epoch, image).
The B images of a step are cut into world*accum shards; rank r takes shard m*world + r at micro step m. Global normalisers
(n_roi, n_spatial) come from the plan, so the loss of a step does not depend on how it is split over GPUs.
Geometry = g1x.data.roi_geom (= the R2 / C0 geometry). Teacher crops and full canvases come from the original pixels only.
"""
import json
import math
import os
import signal
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.config import COCO_ANN, COCO_TRAIN, FEAT, PACO_DIR
from src.g1x.data import G, roi_geom
from src.pacoattr.leak import ham, thumb, thumb_flip
from src.readout.data_check import _job as key_job, coco_path_by_id
from src.viz.c0_pca import LB

THR_CORR, THR_HAM = 0.85, 4                   # = src.pacoattr.leak (near-duplicate candidates; here excluded without inspection)


# ------------------------------------------------------------------------------------------------ tables + exclusion
def build_roi_table(cfg, out, workers=24):
    """-> <out>/roi_table.parquet, exclude.json, data_report.json. Exclusion = exact (id / byte md5 / pixel md5 / dHash) and
    near-duplicate candidates of any official PACO val or test image, + the held-out COCO dev images of the feature diagnostics."""
    os.makedirs(out, exist_ok=True)
    d = json.load(open(COCO_ANN))
    imgs = {im["id"]: (im["width"], im["height"]) for im in d["images"]}
    dc = cfg["data"]
    bins = dc["area_bins_px2"]
    rows = []
    for a in d["annotations"]:
        if a["iscrowd"]:
            continue
        W, H = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        l, t = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
        r, b = min(W, int(math.ceil(x + w))), min(H, int(math.ceil(y + h)))
        if r - l >= dc["min_side"] and b - t >= dc["min_side"]:
            rows.append((a["image_id"], W, H, a["id"], a["category_id"], l, t, r, b, float(a["area"]), "coco_object"))
    t = pd.DataFrame(rows, columns=["image_id", "W", "H", "ann_id", "cat_id", "l", "t", "r", "b", "area", "source"])
    t["bin"] = np.digitize(t.area.values, bins, right=False)

    paco = {}
    for sp in ("val", "test"):
        paco[sp] = sorted(im["id"] for im in json.load(open(os.path.join(PACO_DIR, f"paco_lvis_v1_{sp}.json")))["images"])
    all_coco = sorted(imgs)
    print(f"computing image keys: {len(all_coco)} COCO train, {sum(map(len, paco.values()))} PACO val+test", flush=True)
    with Pool(workers) as pool:
        ck = pool.map(key_job, [(i, os.path.join(COCO_TRAIN, f"{i:012d}.jpg")) for i in all_coco], chunksize=64)
        pk = {sp: pool.map(key_job, [(i, coco_path_by_id(i)) for i in paco[sp]], chunksize=32) for sp in paco}
    assert all(e is None for *_, e in ck), [e for *_, e in ck if e][:3]
    for sp in paco:
        assert all(e is None for *_, e in pk[sp]), [e for *_, e in pk[sp] if e][:3]
    C = pd.DataFrame([dict(image_id=i, path=p, **k) for i, p, k, _ in ck]).set_index("image_id")
    P = pd.concat([pd.DataFrame([dict(image_id=i, path=p, split=sp, **k) for i, p, k, _ in pk[sp]]) for sp in paco]).set_index("image_id")

    exact = {k: set(C.index[C[k].isin(set(P[k]))]) for k in ("md5", "pix_md5", "dhash")}
    exact["image_id"] = set(C.index) & set(P.index)
    print({k: len(v) for k, v in exact.items()}, flush=True)

    print("near-duplicate screen (dHash Hamming, thumbnail correlation)", flush=True)
    paths = [C.path.loc[i] for i in all_coco]
    ppaths = list(P.path)
    with Pool(workers) as pool:
        Tc = np.stack(pool.map(thumb, paths, chunksize=64))
        Tf = np.stack(pool.map(thumb_flip, paths, chunksize=64))
        Tp = np.stack(pool.map(thumb, ppaths, chunksize=32))
    hc = np.array([int(h, 16) for h in C.dhash.loc[all_coco]], dtype=np.uint64)
    hp = np.array([int(h, 16) for h in P.dhash], dtype=np.uint64)
    near = set()
    pairs = []
    pid = list(P.index)
    for s in range(0, len(ppaths), 1000):
        Cm = np.maximum(Tp[s:s + 1000] @ Tc.T, Tp[s:s + 1000] @ Tf.T)
        Hm = ham(hp[s:s + 1000], hc)
        r, c = np.nonzero((Cm >= THR_CORR) | (Hm <= THR_HAM))
        for i, j in zip(r, c):
            near.add(int(all_coco[j]))
            pairs.append((int(pid[s + i]), int(all_coco[j]), float(Cm[i, j]), int(Hm[i, j])))
    dev = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"), columns=["image_id"]).image_id.unique().tolist()
    excl = sorted({int(i) for v in exact.values() for i in v} | near | {int(i) for i in dev})
    pd.DataFrame(pairs, columns=["paco_id", "coco_id", "corr", "hamming"]).to_csv(os.path.join(out, "near_candidates.csv"), index=False)
    json.dump(excl, open(os.path.join(out, "exclude.json"), "w"))
    t = t[~t.image_id.isin(set(excl))].reset_index(drop=True)
    t.to_parquet(os.path.join(out, "roi_table.parquet"))
    rep = dict(coco_train_images=len(all_coco), paco_val_images=len(paco["val"]), paco_test_images=len(paco["test"]),
               exact_excluded={k: len(v) for k, v in exact.items()}, near_candidate_images=len(near),
               near_pairs=len(pairs), dev_images_excluded=len(dev), total_excluded=len(excl), train_images=int(t.image_id.nunique()),
               train_rois=len(t), roi_bin_counts={str(k): int(v) for k, v in t.bin.value_counts().sort_index().items()},
               thresholds=dict(corr=THR_CORR, hamming=THR_HAM), scope="COCO train2017 object ROIs only (no PACO parts)")
    json.dump(rep, open(os.path.join(out, "data_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    return rep


def verify_exclusion(table_dir):
    """Hard assertion used at training start: no training image is an official PACO val/test image, nor a screened near-duplicate."""
    excl = set(json.load(open(os.path.join(table_dir, "exclude.json"))))
    t = pd.read_parquet(os.path.join(table_dir, "roi_table.parquet"), columns=["image_id"])
    ids = set(int(i) for i in t.image_id.unique())
    paco = set()
    for sp in ("val", "test"):
        paco |= {im["id"] for im in json.load(open(os.path.join(PACO_DIR, f"paco_lvis_v1_{sp}.json")))["images"]}
    assert not (ids & paco), f"{len(ids & paco)} PACO val/test images in the training table"
    assert not (ids & excl), "excluded images in the training table"
    return dict(train_images=len(ids), paco_valtest_overlap=0, excluded_overlap=0)


# ------------------------------------------------------------------------------------------------ step plan
class StepPlan:
    def __init__(self, table_path, cfg):
        t = pd.read_parquet(table_path)
        t = t.assign(bin=np.digitize((t.r - t.l).values * (t.b - t.t).values, cfg["data"]["area_bins_px2"], right=False))   # box area (what the readout sees), not the segment area
        order = np.argsort(t.image_id.values, kind="stable")     # numpy split instead of groupby: same rows and order, about 30 s faster at every (re)start
        uid, start = np.unique(t.image_id.values[order], return_index=True)
        self.by = dict(zip(map(int, uid), np.split(t[["l", "t", "r", "b", "bin"]].values.astype(np.int64)[order], start[1:])))
        self.wh = dict(zip(map(int, uid), zip(map(int, t.W.values[order][start]), map(int, t.H.values[order][start]))))
        self.ids = np.array(sorted(self.by))
        self.n = len(self.ids)
        self.B = int(cfg["train"]["global_batch_images"])
        self.R = int(cfg["data"]["rois_per_image"])
        self.probs = np.array(cfg["data"]["bin_probs"], float)
        self.seed = int(cfg["seed"])
        self.K = int(cfg["loss"]["crop"]["spatial_k"])
        self.min_cells = float(cfg["loss"]["crop"]["spatial_min_cells"])
        self.spe = self.n // self.B
        self._ep, self._perm = None, None

    def epoch_pos(self, step):
        return divmod(int(step), self.spe)

    def _order(self, ep):
        if self._ep != ep:
            self._perm = np.random.default_rng([self.seed, ep, 7]).permutation(self.n)
            self._ep = ep
        return self._perm

    def _pick(self, ep, iid):
        boxes = self.by[iid]
        rng = np.random.default_rng([self.seed, ep, iid, 11])
        left = list(range(len(boxes)))
        out = []
        for _ in range(min(self.R, len(boxes))):
            present = np.array([any(boxes[i, 4] == b for i in left) for b in range(len(self.probs))])
            p = self.probs * present
            b = rng.choice(len(self.probs), p=p / p.sum())
            cand = [i for i in left if boxes[i, 4] == b]
            k = cand[int(rng.integers(len(cand)))]
            out.append(boxes[k, :4])
            left.remove(k)
        return np.stack(out)

    def spatial_valid(self, W, H, rois, S=384, P=16):
        v = []
        for l, t, r, b in rois:
            g = roi_geom(W, H, (int(l), int(t), int(r), int(b)), S, P)
            X0, Y0, X1, Y1 = (float(x) for x in g["box_c"])
            c = g["crop_lb"]
            v.append(min(X1 - X0, Y1 - Y0) / self.K >= self.min_cells * P and min(c.nw, c.nh) / self.K >= self.min_cells * P)
        return np.array(v, bool)

    def step_items(self, step):
        ep, pos = self.epoch_pos(step)
        order = self._order(ep)
        ids = self.ids[order[pos * self.B:(pos + 1) * self.B]]
        items, n_roi, n_sp = [], 0, 0
        for iid in map(int, ids):
            W, H = self.wh[iid]
            rois = self._pick(ep, iid)
            sv = self.spatial_valid(W, H, rois)
            items.append(dict(iid=iid, W=W, H=H, rois=rois, sv=sv))
            n_roi += len(rois)
            n_sp += int(sv.sum())
        return items, dict(step=int(step), epoch=ep, pos=pos, n_roi=n_roi, n_sp=n_sp, n_img=len(items), order_head=int(ids[0]))

    def shard(self, items, world, accum, rank, m):
        n = world * accum
        idx = m * world + rank
        k = len(items) // n
        return items[idx * k:(idx + 1) * k]


class ShardSampler(Sampler):
    """Yields one micro-step payload per iteration, from (start_step, micro 0) to (total_steps) -> (step, m, rank's images, global counts)."""

    def __init__(self, plan, start, total, world, accum, rank):
        self.plan, self.start, self.total, self.world, self.accum, self.rank = plan, start, total, world, accum, rank

    def __iter__(self):
        for g in range(self.start, self.total):
            items, info = self.plan.step_items(g)
            for m in range(self.accum):
                yield dict(items=self.plan.shard(items, self.world, self.accum, self.rank, m), m=m, **info)

    def __len__(self):
        return (self.total - self.start) * self.accum


# ------------------------------------------------------------------------------------------------ dataset
def rect_cell_weights(x0, y0, x1, y1, P, Gr=G):
    """Rects (K2,) on a canvas -> (K2, G*G) overlap area with each token cell, in cell units (0..1 per cell)."""
    e = np.arange(Gr + 1) * P
    ox = np.clip(np.minimum(e[1:], x1[:, None]) - np.maximum(e[:-1], x0[:, None]), 0, None) / P
    oy = np.clip(np.minimum(e[1:], y1[:, None]) - np.maximum(e[:-1], y0[:, None]), 0, None) / P
    return (oy[:, :, None] * ox[:, None, :]).reshape(len(x0), -1).astype(np.float32)


def coarse_rects(X0, Y0, X1, Y1, K):
    """K x K rects of a box (row-major)."""
    fx = np.linspace(0, 1, K + 1)
    xa, xb = X0 + (X1 - X0) * fx[:-1], X0 + (X1 - X0) * fx[1:]
    ya, yb = Y0 + (Y1 - Y0) * fx[:-1], Y0 + (Y1 - Y0) * fx[1:]
    ix, iy = np.meshgrid(np.arange(K), np.arange(K))
    ix, iy = ix.reshape(-1), iy.reshape(-1)
    return xa[ix], ya[iy], xb[ix], yb[iy]


def coarse_weights(g, K, S, P):
    """-> student weights (K2, 576) on the full canvas, teacher weights (K2, 576) on the crop canvas. Cell (a, b) of both = the same
    rectangle of the original image, because the full canvas maps the box linearly and the crop canvas maps the content rect linearly."""
    X0, Y0, X1, Y1 = (float(v) for v in g["box_c"])
    c = g["crop_lb"]
    ws = rect_cell_weights(*coarse_rects(X0, Y0, X1, Y1, K), P)
    wt = rect_cell_weights(*coarse_rects(c.ox, c.oy, c.ox + c.nw, c.oy + c.nh, K), P)
    return ws, wt


class CocoRoiDS(Dataset):
    """item = payload from ShardSampler -> tensors for one rank's micro step."""

    def __init__(self, S=384, P=16, K=3, root=COCO_TRAIN):
        self.S, self.P, self.K, self.root = S, P, K, root

    def __len__(self):
        return 1 << 40

    def __getitem__(self, pl):
        full, fvalid, crops, roi_img = [], [], [], []
        per = {k: [] for k in ("ccov", "box_c", "wfull", "ws", "wt", "sv")}
        for k, it in enumerate(pl["items"]):
            img = Image.open(os.path.join(self.root, f"{it['iid']:012d}.jpg")).convert("RGB")
            W, H = img.size
            assert (W, H) == (it["W"], it["H"]), it["iid"]
            flb = LB(W, H, self.S, self.P)
            full.append(flb.canvas(img))
            fvalid.append(flb.coverage() > 0)
            for q, box in enumerate(it["rois"]):
                bx = tuple(int(v) for v in box)
                g = roi_geom(W, H, bx, self.S, self.P)
                crops.append(g["crop_lb"].canvas(img.crop(bx)))
                ws, wt = coarse_weights(g, self.K, self.S, self.P)
                per["ccov"].append(g["ccov"]); per["box_c"].append(g["box_c"].astype(np.float32))
                per["wfull"].append(g["wfull"]); per["ws"].append(ws); per["wt"].append(wt); per["sv"].append(bool(it["sv"][q]))
                roi_img.append(k)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items() if k != "sv"}
        out["sv"] = torch.as_tensor(per["sv"], dtype=torch.bool)
        out.update(full=torch.stack(full), fvalid=torch.from_numpy(np.stack(fvalid)), crops=torch.stack(crops),
                   roi_img=torch.as_tensor(roi_img, dtype=torch.long), step=pl["step"], m=pl["m"], n_roi=pl["n_roi"], n_sp=pl["n_sp"],
                   n_img=pl["n_img"], epoch=pl["epoch"])
        return out


def worker_init(_):
    """DataLoader workers ignore SIGINT/SIGTERM: stopping is decided by the main process at a step boundary."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
