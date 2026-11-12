"""Build/resume exact 48x48 native teacher targets; each shard owns disjoint image rows."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.ivc.ft import run_enc
from src.g1x.distill import pool, cosd
from src.fgtrain.common import encoder, write_json
from src.fgtrain.data import Data


@torch.no_grad()
def build(root, initial, split='train', limit=0, batch=64, workers=8, shard=0, shards=1, ids_file=""):
    root = Path(root)
    ds = Data(root, split, crops=True)
    complete = np.load(root / 'completed.npy', mmap_mode='r+')
    sp = np.load(root / 'spatial.npy', mmap_mode='r+')
    pooled = np.load(root / 'pooled.npy', mmap_mode='r+')
    gc, gm = [np.load(root / f'{k}.npy', mmap_mode='r+') for k in ('g_cls', 'g_mean')]
    wanted = set(json.loads(Path(ids_file).read_text())) if ids_file else None
    ids = [i for i in sorted(ds.by) if not complete[ds.image_pos[i]] and (wanted is None or i in wanted)][shard::shards]
    if limit:
        ids = ids[:limit]
    model = encoder(initial)
    checks, global_checks = [], []
    t0, rois = time.time(), 0
    # One image per item limits host-side prefetch of 768x768 tensors.
    chunks = [ids[i:i+8] for i in range(0,len(ids),8)]
    dl = DataLoader(ds, sampler=chunks, batch_size=None, num_workers=workers, pin_memory=True,
                    **(dict(prefetch_factor=2) if workers else {}))
    for it, b in enumerate(dl):
        x = b['teacher_crops'].cuda(non_blocking=True)
        tok, _ = run_enc(model, x, chunk=batch)
        cov = b['ccov'].cuda()
        tp = pool(tok, cov)
        old = b['tpool'].cuda()
        good = torch.isfinite(old).all(-1)
        if good.any():
            c = 1 - cosd(tp[good], old[good]); checks.extend(c.cpu().tolist())
            assert c.min() > .999, ('native pooled cache mismatch', float(c.min()))
        for j, (ii,k) in enumerate(zip(b['cache_i'].tolist(),b['roi_k'].tolist())):
            off, n = ds.offsets[ii, k]
            valid = b['ccov'][j].numpy() > 0
            sp[off:off+n] = tok[j, valid].cpu().numpy().astype(np.float16)
            if not np.isfinite(pooled[ii,k]).all():
                pooled[ii, k] = tp[j].cpu().numpy()
        # Existing full-image teacher globals must also reproduce at the pinned initialization.
        ft, cls = run_enc(model, b['full'].cuda())
        mean = pool(ft, b['fvalid'].cuda())
        oldcls, oldmean = b['g_cls'].cuda(), b['g_mean'].cuda()
        if torch.isfinite(oldcls).all() and torch.isfinite(oldmean).all():
            c = torch.cat([1-cosd(cls, oldcls), 1-cosd(mean, oldmean)])
            global_checks.extend(c.cpu().tolist())
            assert c.min() > .999, ('global teacher cache mismatch', c.tolist())
        image_indices = [ds.image_pos[int(i)] for i in b['image_ids']]
        for j,ii in enumerate(image_indices):
            if not np.isfinite(gc[ii]).all():
                gc[ii], gm[ii] = cls[j].cpu().numpy(), mean[j].cpu().numpy()
        complete[image_indices] = True
        rois += len(x)
        if (it+1) % 25 == 0 or it+1 == len(chunks):
            sp.flush(); pooled.flush(); gc.flush(); gm.flush(); complete.flush()
            rec = dict(split=split, shard=shard, images=min((it+1)*8,len(ids)), total=len(ids), rois=rois, seconds=round(time.time()-t0),
                       pooled_min_cos=min(checks) if checks else None, global_min_cos=min(global_checks) if global_checks else None)
            write_json(root / f'progress_{split}_{shard}.json', rec)
            print(json.dumps(rec), flush=True)
    # Flush targets BEFORE the final readiness signal.
    sp.flush(); pooled.flush(); gc.flush(); gm.flush(); complete.flush()
    report = dict(split=split, images=len(ids), seconds=round(time.time()-t0), teacher_frozen=not any(p.requires_grad for p in model.parameters()),
                  pooled_min_cos=min(checks) if checks else None, global_min_cos=min(global_checks) if global_checks else None,
                  teacher_grid=[48,48,768], roi_native_rgb=True, padding_excluded=True)
    write_json(root / f'verified_{split}_{shard}.json', report)
    return report


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True);p.add_argument('--initial',required=True)
    p.add_argument('--split',default='train');p.add_argument('--limit',type=int,default=0);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--ids-file',default='');p.add_argument('--workers',type=int,default=8);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();build(**vars(a))

if __name__=='__main__': main()
