"""Fresh frozen SigLIP2 crop teacher cache. Identical original object/nested ROI manifest. g_embed is native pretrained attention pooling; no CLS token or DINO features."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT, STREAM
from src.msd.pretrain import eval_exclusions
from src.siglip_pm.model import load_vit, MODEL_ID, REVISION
from src.siglip_pm.data import LB

S, P, NIMG_SLOT, NEST = 384, 16, 16, 4
MINP, MINS, FLO, FHI = 64, 24, 0.25, 0.6


def boxes_for(parents, rng):
    """parents: (M, 4) int ltrb -> (<=4, 4) boxes, n_adjacent"""
    out, nadj = [], 0
    ok = parents[(parents[:, 2] - parents[:, 0] >= MINP) & (parents[:, 3] - parents[:, 1] >= MINP)]
    if not len(ok):
        return out, 0
    idx = rng.permutation(len(ok))[:2]
    if len(idx) == 1:
        idx = np.array([idx[0], idx[0]])                          # one eligible parent -> both pairs from it
    for pi in idx:
        pl, pt, pr, pb = (int(v) for v in ok[pi])
        pw, ph = pr - pl, pb - pt
        nw = int(np.clip(round(rng.uniform(FLO, FHI) * pw), MINS, pw))
        nh = int(np.clip(round(rng.uniform(FLO, FHI) * ph), MINS, ph))
        ax0 = int(rng.integers(pl, pr - nw + 1))
        ay0 = int(rng.integers(pt, pb - nh + 1))
        A = (ax0, ay0, ax0 + nw, ay0 + nh)
        placed = None
        for axis in rng.permutation(2):
            cand = [(A[2], ax0, 1), (ax0 - nw, ax0, -1)] if axis == 0 else [(A[3], ay0, 1), (ay0 - nh, ay0, -1)]
            for v, _, _d in cand:
                if axis == 0 and pl <= v and v + nw <= pr:
                    placed = (v, ay0, v + nw, ay0 + nh)
                elif axis == 1 and pt <= v and v + nh <= pb:
                    placed = (ax0, v, ax0 + nw, v + nh)
                if placed:
                    break
            if placed:
                break
        if placed is None:                                        # no adjacent room: fresh random box
            bx0, by0 = int(rng.integers(pl, pr - nw + 1)), int(rng.integers(pt, pb - nh + 1))
            B = (bx0, by0, bx0 + nw, by0 + nh)
        else:
            B = placed
            nadj += 1
        out += [A, B]
    return out, nadj


def build(a):
    os.makedirs(a.out, exist_ok=True)
    rois = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    by = {int(i): g.sort_values("k") for i, g in rois.groupby("image_id")}
    stream = np.load(STREAM)
    n, m = stream.shape
    assert (n, m) == (12000, NIMG_SLOT)
    ids = sorted(by)
    NB = np.full((n, m, NEST, 4), -1, np.int32)
    nadj = np.zeros((n, m), np.int8)
    for s in range(n):
        for c in range(m):
            iid = int(stream[s, c])
            g = by[iid]
            par = g[["l", "t", "r", "b"]].values.astype(int)
            rng = np.random.Generator(np.random.PCG64(1000003 * (s * m + c)))
            bx, ad = boxes_for(par, rng)
            for j, b in enumerate(bx):
                NB[s, c, j] = b
            nadj[s, c] = ad
    np.save(os.path.join(a.out, "nested_boxes.npy"), NB)
    np.save(os.path.join(a.out, "nested_nadj.npy"), nadj)
    np.save(os.path.join(a.out, "image_ids.npy"), np.array(ids, np.int64))
    # object-ROI order on disk: per image, k = 0..3
    obj = np.stack([by[i][["l", "t", "r", "b"]].values.astype(np.int32) for i in ids])
    np.save(os.path.join(a.out, "object_boxes.npy"), obj)
    w = NB[..., 2] - NB[..., 0]
    present = w > 0
    ex = eval_exclusions()
    info = dict(seed=0, n_steps=n, n_slots_per_step=m, images=len(ids), object_rois=int(rois.shape[0]), object_kinds=rois.kind.value_counts().to_dict(),
                nested_rois=int(present.sum()), nested_per_slot_mean=float(present.sum(-1).mean()), slots_without_nested=int((present.sum(-1) == 0).sum()),
                adjacent_fraction_of_pairs=float(nadj.sum() / max(1, present.sum() / 2)),
                nested_w_mean=float(w[present].mean()), nested_h_mean=float((NB[..., 3] - NB[..., 1])[present].mean()),
                excluded_stream_images=ex["raev2_stream"], rule=dict(minP=MINP, minS=MINS, frac=[FLO, FHI]), time=time.strftime("%F %T"))
    json.dump(info, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


class CropDS(Dataset):
    """item = image index -> full canvas + every crop canvas of that image (object ROIs + its nested ROIs of every visit)."""

    def __init__(self, ids, obj, NB, stream):
        self.ids, self.obj, self.NB = ids, obj, NB
        v = {}
        s, c = np.nonzero(np.isin(stream, list(ids)))
        pos = pd.Series(np.arange(len(ids)), index=ids)
        for ss, cc in zip(s, c):
            v.setdefault(int(pos[int(stream[ss, cc])]), []).append((int(ss), int(cc)))
        self.visits = v

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = int(self.ids[i])
        img = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
        W, H = img.size
        fl = LB(W, H, S, P)
        crops, tag = [], []
        for k, b in enumerate(self.obj[i]):
            crops.append(LB(b[2] - b[0], b[3] - b[1], S, P))
            tag.append((0, k, -1, -1, *b.tolist()))
        for s, c in self.visits.get(i, []):
            for j in range(NEST):
                b = self.NB[s, c, j]
                if b[2] <= b[0]:
                    continue
                crops.append(LB(b[2] - b[0], b[3] - b[1], S, P))
                tag.append((1, j, s, c, *b.tolist()))
        X = torch.stack([lb.canvas(img.crop(tuple(t[4:]))) for lb, t in zip(crops, tag)])
        cov = np.stack([lb.coverage() for lb in crops]).astype(np.float32)
        return dict(i=i, full=fl.canvas(img), fv=torch.from_numpy(fl.coverage() > 0), X=X, cov=torch.from_numpy(cov), tag=np.array(tag, np.int32))


def pool(x, w):
    return (x * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)


SHAPE = lambda N: {"t_object": (N, 4, 768), "t_nested": (12000, 16, NEST, 768), "g_embed": (N, 768), "g_mean": (N, 768)}


def alloc(a):
    """Create the NaN-filled memmaps once, before the independent GPU shards start."""
    for k, sh in SHAPE(len(np.load(f"{a.out}/image_ids.npy"))).items():
        m = np.lib.format.open_memmap(f"{a.out}/{k}.npy", "w+", np.float32, sh)
        m[...] = np.nan
        m.flush()
    print("alloc done", flush=True)


def cache(a):
    """One independent process per GPU (no process group): images [shard::nshards]; each process writes only its own memmap rows."""
    dev = torch.device("cuda")
    ids, obj, NB = np.load(f"{a.out}/image_ids.npy"), np.load(f"{a.out}/object_boxes.npy"), np.load(f"{a.out}/nested_boxes.npy")
    stream = np.load(STREAM)
    N = len(ids)
    maps = {k: np.load(f"{a.out}/{k}.npy", mmap_mode="r+") for k in SHAPE(N)}
    meta = json.load(open(f"{a.out}/backbone.json"))
    assert meta['model_id'] == MODEL_ID and meta['revision'] == REVISION
    enc0 = load_vit(dev).eval()
    for p in enc0.parameters():
        p.requires_grad_(False)
    all_mine = list(range(N))[a.shard::a.nshards]
    # g_mean is written last and serves as an image completion marker.
    mine = [i for i in all_mine if not np.isfinite(maps['g_mean'][i]).all()]
    print(json.dumps(dict(shard=a.shard,mode='batched-resume',already_complete=len(all_mine)-len(mine),remaining=len(mine),crop_batch=a.batch,image_batch=8)),flush=True)
    ds = CropDS(ids, obj, NB, stream)
    dl = DataLoader(ds, sampler=mine, batch_size=8, collate_fn=list, num_workers=a.workers, prefetch_factor=2, persistent_workers=False, pin_memory=True)
    t0, nc, ni = time.time(), 0, 0
    with torch.no_grad():
        for it, group in enumerate(dl):
            X=torch.cat([b['X'] for b in group])
            cov=torch.cat([b['cov'] for b in group])
            values=[]
            for lo in range(0,len(X),a.batch):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    t=enc0.forward_features(X[lo:lo+a.batch].to(dev,non_blocking=True)).float()
                values.append(pool(t,cov[lo:lo+a.batch].to(dev,non_blocking=True)).cpu().numpy())
            z=np.concatenate(values);offset=0
            for b in group:
                for r,v in zip(b['tag'],z[offset:offset+len(b['X'])]):
                    if r[0]==0:maps['t_object'][b['i'],r[1]]=v
                    else:maps['t_nested'][r[2],r[3],r[1]]=v
                offset+=len(b['X'])
            full=torch.stack([b['full'] for b in group]).to(dev,non_blocking=True)
            fv=torch.stack([b['fv'] for b in group]).to(dev,non_blocking=True).float()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                t=enc0.forward_features(full).float();gp=enc0.forward_head(t).float()
            gm=pool(t,fv).cpu().numpy();gp=gp.cpu().numpy()
            for k,b in enumerate(group):
                maps['g_embed'][b['i']]=gp[k]
                maps['g_mean'][b['i']]=gm[k]
            if it==0:
                # A different batch size may change bf16 rounding, but must preserve target direction.
                b=group[0]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    tt=enc0.forward_features(b['X'].to(dev)).float()
                    tf=enc0.forward_features(b['full'][None].to(dev)).float();gf=enc0.forward_head(tf).float()
                zp=pool(tt,b['cov'].to(dev)).cpu()
                crop_cos=float(torch.nn.functional.cosine_similarity(zp,torch.from_numpy(z[:len(zp)]),dim=-1).min())
                embed_cos=float(torch.nn.functional.cosine_similarity(gf.cpu(),torch.from_numpy(gp[:1]),dim=-1).min())
                assert min(crop_cos,embed_cos)>.9995,(crop_cos,embed_cos)
                json.dump(dict(crop_cosine_min=crop_cos,pooled_image_cosine_min=embed_cos,passed=True),open(f'{a.out}/batch_parity_shard{a.shard}.json','w'),indent=1)
            nc+=len(X);ni+=len(group)
            if it%25==0:
                print(json.dumps(dict(shard=a.shard,it=ni,n=len(mine),crops=nc,s=round(time.time()-t0),cps=round(nc/(time.time()-t0)))),flush=True)
    for m in maps.values():
        m.flush()
    print(f"shard {a.shard} DONE {round(time.time() - t0)}s", flush=True)


def verify(a):
    NB = np.load(f"{a.out}/nested_boxes.npy")
    N = len(np.load(f"{a.out}/image_ids.npy"))
    pres = NB[..., 2] > NB[..., 0]
    tn = np.load(f"{a.out}/t_nested.npy", mmap_mode="r")
    nan_present = int((~np.isfinite(tn[pres])).any(-1).sum())
    nan_absent_ok = bool(np.isnan(tn[~pres]).all())
    nan = {k: int((~np.isfinite(np.load(f"{a.out}/{k}.npy", mmap_mode="r"))).any(-1).sum()) for k in ("t_object", "g_embed", "g_mean")}
    res = dict(images=N, n_nested_present=int(pres.sum()), nested_present_with_nan=nan_present, absent_slots_all_nan=nan_absent_ok, nan_rows=nan, time=time.strftime("%F %T"))
    json.dump(res, open(f"{a.out}/teacher_cache_check.json", "w"), indent=1)
    print(json.dumps(res), flush=True)
    assert nan_present == 0 and not any(nan.values()), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "alloc", "cache", "verify"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    {"build": build, "alloc": alloc, "cache": cache, "verify": verify}[a.cmd](a)


if __name__ == "__main__":
    main()
