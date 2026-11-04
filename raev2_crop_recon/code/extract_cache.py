"""Pre-cache frozen RAEv2 encoder outputs (official preprocess, fp32 encoder with TF32 off, stored fp16):
  full   : (n_img, 256, 768) all 16x16 cells of the full-image 256 letterbox canvas
  teacher: GT-crop latents, CONTENT cells only (crop letterbox coverage > 0; padding cells have loss weight 0 and are never
           used as targets), flat (n_cells, 768) + per-ROI offsets. Disk: ~21 GB + ~57 GB for the 12k stream.
Sets: 'stream' = the unique COCO decoder-pool images of the 12k stream (make_stream.py: seed 2027, PACO val/test images excluded), each with its
4 ROIs (3 GT + 1 random, coco_rois.parquet); 'dev' = coco_rois split 'dev' (2,000 ROIs); 'hval' = rae_crop_pilot head_val
(3,194 GT ROIs, 1,000 COCO images outside both decoder pools).

  python code/extract_cache.py plan                      # index files (deterministic)
  CUDA_VISIBLE_DEVICES=k python code/extract_cache.py run SET SHARD NSHARD
  python code/extract_cache.py check SET
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rae_common as RC  # noqa: E402

OUT = os.path.join(RC.CACHE, "feat")


def set_rois(name):
    if name in ("stream", "dev"):
        r = pd.read_parquet(RC.COCO_ROIS)
        if name == "dev":
            return r[r.split == "dev"].copy(), RC.COCO_TRAIN
        tr = r[r.split == "train"]
        used = np.unique(np.load(RC.STREAM))                    # image ids
        return tr[tr.image_id.isin(set(used.tolist()))].copy(), RC.COCO_TRAIN
    r = pd.read_parquet(RC.RAE_ROIS)
    return r[r.split == "head_val"].copy(), RC.COCO_TRAIN


def plan():
    os.makedirs(OUT, exist_ok=True)
    for name in ("dev", "hval", "stream"):
        rois, _ = set_rois(name)
        rois = rois.sort_values(["image_id", "roi_id"]).reset_index(drop=True)
        img_ids = np.array(sorted(rois.image_id.unique()))
        slot = {im: k for k, im in enumerate(img_ids)}
        rois["img_slot"] = rois.image_id.map(slot).astype(np.int64)
        ncell = np.array([int((RC.Letterbox(r - l, b - t).coverage() > 0).sum()) for l, t, r, b in zip(rois.l, rois.t, rois.r, rois.b)])
        off = np.concatenate([[0], np.cumsum(ncell)]).astype(np.int64)
        rois["t_off"], rois["t_n"] = off[:-1], ncell
        cr = np.array([RC.Letterbox(r - l, b - t).rect() for l, t, r, b in zip(rois.l, rois.t, rois.r, rois.b)])
        fr = np.array([RC.Letterbox(w, h).rect() for w, h in zip(rois.W, rois.H)])
        for k, c in enumerate(("ox", "oy", "nw", "nh")):          # 256-canvas content rects (pixel-loss mask / padding)
            rois[f"crop_{c}"], rois[f"full_{c}"] = cr[:, k], fr[:, k]
        rois["n_in_rae"] = [RC.roi_geometry(int(w), int(h), (int(l), int(t), int(r), int(b)))["n_in"]
                            for w, h, l, t, r, b in zip(rois.W, rois.H, rois.l, rois.t, rois.r, rois.b)]
        rois.to_parquet(os.path.join(OUT, f"{name}_rois.parquet"))
        info = dict(n_img=len(img_ids), n_roi=len(rois), n_cells=int(off[-1]), full_gb=len(img_ids) * 256 * 768 * 2 / 1e9,
                    teacher_gb=int(off[-1]) * 768 * 2 / 1e9)
        json.dump(info, open(os.path.join(OUT, f"{name}_info.json"), "w"), indent=1)
        print(name, info, flush=True)


def npy(name, what):
    """Cache arrays live in OUT; FEAT_WORK (e.g. /dev/shm/...) redirects them while extracting: scattered memmap writes on
    Lustre stall on synchronous page reads, so write in RAM and copy to OUT sequentially afterwards (run_extract_local.sh)."""
    return os.path.join(os.environ.get("FEAT_WORK", OUT), f"{name}_{what}.npy")


def mm(name, what, mode):
    info = json.load(open(os.path.join(OUT, f"{name}_info.json")))
    shape = (info["n_img"], 256, 768) if what == "full" else (info["n_cells"], 768)
    return np.lib.format.open_memmap(npy(name, what), mode=mode, dtype=np.float16, shape=shape if mode == "w+" else None)


class ImgDS(Dataset):
    def __init__(self, rois, root, slots):
        self.g = {s: d for s, d in rois.groupby("img_slot")}
        self.slots, self.root = slots, root

    def __len__(self):
        return len(self.slots)

    def __getitem__(self, i):
        s = self.slots[i]
        g = self.g[s]
        r0 = g.iloc[0]
        img = Image.open(os.path.join(self.root, f"{int(r0.image_id):012d}.jpg")).convert("RGB")
        assert img.size == (int(r0.W), int(r0.H))
        full = RC.u8_canvas(img, RC.Letterbox(*img.size))
        crops = [RC.u8_canvas(img.crop((int(r.l), int(r.t), int(r.r), int(r.b))), RC.Letterbox(int(r.r - r.l), int(r.b - r.t)))
                 for r in g.itertuples()]
        masks = [torch.from_numpy(RC.Letterbox(int(r.r - r.l), int(r.b - r.t)).coverage() > 0) for r in g.itertuples()]
        return dict(slot=s, full=full, crops=torch.stack(crops), roi_idx=torch.as_tensor(g.index.values), masks=torch.stack(masks))


def coll(items):
    return dict(slot=torch.as_tensor([it["slot"] for it in items]), full=torch.stack([it["full"] for it in items]),
                crops=torch.cat([it["crops"] for it in items]), roi_idx=torch.cat([it["roi_idx"] for it in items]),
                masks=torch.cat([it["masks"] for it in items]))


def run(name, shard, nshard):
    RC.set_fp32()
    rois = pd.read_parquet(os.path.join(OUT, f"{name}_rois.parquet"))
    info = json.load(open(os.path.join(OUT, f"{name}_info.json")))
    for what in ("full", "teacher"):
        if not os.path.exists(npy(name, what)):
            assert shard == 0, "shard 0 creates the files; start it first"
            os.makedirs(os.path.dirname(npy(name, what)), exist_ok=True)
            mm(name, what, "w+").flush()
    while not all(os.path.exists(npy(name, w)) for w in ("full", "teacher")):
        time.sleep(5)
    time.sleep(3)
    full_mm, t_mm = mm(name, "full", "r+"), mm(name, "teacher", "r+")
    slots = [s for s in range(info["n_img"]) if s % nshard == shard]
    done_f = os.path.join(OUT, f"{name}_done_{shard}of{nshard}.json")
    rae = RC.FrozenRAE("cuda", decoder=False)
    bs, nw = int(os.environ.get("EXTRACT_BS", 32)), int(os.environ.get("EXTRACT_WORKERS", 18))
    dl = DataLoader(ImgDS(rois, RC.COCO_TRAIN, slots), batch_size=bs, num_workers=nw, collate_fn=coll, pin_memory=True, prefetch_factor=4)
    t0, n = time.time(), 0
    off, tn = rois.t_off.values, rois.t_n.values
    for b in dl:
        zf = rae.encode(b["full"]).half().cpu().numpy()
        zc = rae.encode(b["crops"])                                    # (R,256,768) fp32 on GPU
        m = b["masks"]
        sel = zc[m.to(zc.device)].half().cpu().numpy()                 # content cells, ROI by ROI, ascending cell order
        full_mm[b["slot"].numpy()] = zf
        ri = b["roi_idx"].numpy()
        cnt = m.sum(1).numpy()
        assert (cnt == tn[ri]).all()
        pos = np.concatenate([[0], np.cumsum(cnt)])
        for k, r in enumerate(ri):
            t_mm[off[r]:off[r] + tn[r]] = sel[pos[k]:pos[k + 1]]
        n += len(b["slot"])
        if n % 3200 < bs:
            print(json.dumps(dict(set=name, shard=shard, n=n, of=len(slots), s=round(time.time() - t0))), flush=True)
    full_mm.flush()
    t_mm.flush()
    json.dump(dict(n=n, s=time.time() - t0), open(done_f, "w"))
    print("DONE", name, shard, n, round(time.time() - t0), flush=True)


def check(name):
    """Recompute 8 random images from scratch and compare with the cache (fp16 rounding only)."""
    RC.set_fp32()
    rois = pd.read_parquet(os.path.join(OUT, f"{name}_rois.parquet"))
    full_mm, t_mm = mm(name, "full", "r"), mm(name, "teacher", "r")
    rae = RC.FrozenRAE("cuda", decoder=False)
    ds = ImgDS(rois, RC.COCO_TRAIN, list(np.random.default_rng(0).choice(rois.img_slot.max() + 1, 8, replace=False)))
    err_f, err_t, zero_rows = [], [], 0
    for i in range(len(ds)):
        it = ds[i]
        zf = rae.encode(it["full"][None])[0].cpu()
        err_f.append(float((zf - torch.from_numpy(full_mm[it["slot"]].astype(np.float32))).abs().max()))
        zc = rae.encode(it["crops"]).cpu()
        for k, ri in enumerate(it["roi_idx"].tolist()):
            r = rois.iloc[ri]
            m = torch.from_numpy(RC.Letterbox(int(r.r - r.l), int(r.b - r.t)).coverage() > 0)
            c = torch.from_numpy(t_mm[r.t_off:r.t_off + r.t_n].astype(np.float32))
            err_t.append(float((zc[k][m] - c).abs().max()))
    sample = np.random.default_rng(1).choice(len(full_mm), min(2000, len(full_mm)), replace=False)
    zero_rows = int(sum(np.abs(full_mm[i]).max() == 0 for i in sample))
    out = dict(set=name, full_maxabs_err=max(err_f), teacher_maxabs_err=max(err_t), zero_full_rows_in_2000=zero_rows)
    json.dump(out, open(os.path.join(OUT, f"{name}_check.json"), "w"), indent=1)
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        plan()
    elif sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        check(sys.argv[2])
