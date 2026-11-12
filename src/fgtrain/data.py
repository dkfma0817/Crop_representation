"""Existing stream/letterbox geometry, original GT + random ROIs, packed native Crop768 targets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.config import COCO_TRAIN, FEAT, STREAM
from src.g1x.data import CocoDS
from src.msd.pretrain import eval_exclusions
from src.viz.c0_pca import LB
from src.overnight6.teacher768 import cov48
from src.fgtrain.common import OLD768, OLDGLOBAL, write_json


def teacher_geometry(box):
    l, t, r, b = map(int, box)
    lb = LB(r - l, b - t, 768, 16)
    yy, xx = np.divmod(np.arange(2304), 48)
    qrel = np.stack([((xx + .5) * 16 - lb.ox) / lb.nw, ((yy + .5) * 16 - lb.oy) / lb.nh], -1).astype(np.float32)
    return lb, cov48(lb), qrel


def official_exclusions():
    """Full official image scope, never attribute/test scores, plus existing fingerprint and near-photo audit."""
    from src.pacoattr.subsets import NEAR
    legacy=eval_exclusions()
    from src.msd.pretrain import KEYS
    keys=pd.read_parquet(KEYS)
    official_ids={split:{int(i['id']) for i in json.load(open(f'/home/irteam/datasets/paco/paco_lvis_v1_{split}.json'))['images']} for split in ('val','test')}
    all_ids=set.union(*official_ids.values())
    near_train={int(v[0]) for split in NEAR.values() for v in split.values()}
    old=json.load(open('output/2026-11-07/paco_oracle_attr/leak/leak_auto.json'))
    # The official-scope fingerprint audit found exact matches with the same COCO id;
    # assert this before translating them to training exclusions.
    for split in ('val','test'):
        exact=old['exact'][split]
        assert set(exact['image_id'])==set(exact['md5'])==set(exact['pix_md5'])==set(exact['dhash'])
    result={}
    for source in ('raev2_stream','raev2_dev'):
        source_ids=set(map(int,keys[keys.source==source].image_id))
        result[source]=sorted(set(legacy[source]) | (source_ids & (all_ids | near_train)))
    return result


def prepare(root, dev_images=128):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ex = official_exclusions()
    excluded = ex['raev2_stream']
    train_all = pd.read_parquet(Path(FEAT) / 'stream_rois.parquet')
    dev = pd.read_parquet(Path(FEAT) / 'dev_rois.parquet')
    dev = dev[~dev.image_id.isin(ex['raev2_dev'])]
    dev_ids = sorted(map(int, dev.image_id.unique()))[:dev_images]
    train = train_all[~train_all.image_id.isin(excluded)].copy()
    dev = dev[dev.image_id.isin(dev_ids)].copy()
    train['target_split'], dev['target_split'] = 'train', 'dev'
    rois = pd.concat([train, dev], ignore_index=True).sort_values(['target_split', 'image_id', 'k']).reset_index(drop=True)
    ids = sorted(map(int, rois.image_id.unique()))
    positions = {iid: i for i, iid in enumerate(ids)}
    rois['cache_image'] = [positions[int(i)] for i in rois.image_id]
    oldids = np.load(OLD768 / 'image_ids.npy')
    oldbox = np.load(OLD768 / 'object_boxes.npy')
    oldpos = {int(i): k for k, i in enumerate(oldids)}
    # Verify ALL source boxes before considering pooled-target reuse.
    assert all(np.array_equal(g.sort_values('k')[['l','t','r','b']].values, oldbox[oldpos[int(i)]]) for i, g in train_all.groupby('image_id'))
    rois.to_parquet(root / 'rois.parquet', index=False)
    np.save(root / 'image_ids.npy', np.array(ids, np.int64))
    offsets = np.full((len(ids), 4, 2), -1, np.int64)
    total = 0
    for row in rois.itertuples():
        _, cov, _ = teacher_geometry((row.l, row.t, row.r, row.b))
        n = int((cov > 0).sum())
        offsets[row.cache_image, row.k] = [total, n]
        total += n
    np.save(root / 'offsets.npy', offsets)
    if not (root / 'pooled.npy').exists():
        pooled = np.lib.format.open_memmap(root / 'pooled.npy', 'w+', np.float32, (len(ids), 4, 768))
        gc = np.lib.format.open_memmap(root / 'g_cls.npy', 'w+', np.float32, (len(ids), 768))
        gm = np.lib.format.open_memmap(root / 'g_mean.npy', 'w+', np.float32, (len(ids), 768))
        pooled[:] = np.nan; gc[:] = np.nan; gm[:] = np.nan
        op = np.load(OLD768 / 't_object.npy', mmap_mode='r')
        oc = np.load(OLDGLOBAL / 'g_cls.npy', mmap_mode='r')
        om = np.load(OLDGLOBAL / 'g_mean.npy', mmap_mode='r')
        for row in rois.itertuples():
            if row.target_split == 'train':
                i = oldpos[int(row.image_id)]
                pooled[row.cache_image, row.k] = op[i, row.k]
                gc[row.cache_image], gm[row.cache_image] = oc[i], om[i]
        pooled.flush(); gc.flush(); gm.flush()
    if not (root / 'spatial.npy').exists():
        np.lib.format.open_memmap(root / 'spatial.npy', 'w+', np.float16, (total, 768)).flush()
        np.save(root / 'completed.npy', np.zeros(len(ids), bool))
    report = dict(images=len(ids), train_images=int(train.image_id.nunique()), train_rois=len(train), dev_images=int(dev.image_id.nunique()),
                  dev_rois=len(dev), source_rois=len(train_all), source_gt_rois=int((train_all.kind == 'gt').sum()),
                  roi_rule='Unchanged existing image stream, GT + random + fallback boxes; same B0/B1 batches',
                  excluded_train_images=excluded, excluded_dev_images=ex['raev2_dev'], spatial_cells=total,
                  spatial_bytes=total * 768 * 2, spatial_storage='float16 packed cells with coverage>0; full 48x48 coordinates retained',
                  pooled_reuse='All 50491 source image boxes verified equal to existing native Crop768 object_boxes',
                  original_teacher=str(OLD768), global_teacher=str(OLDGLOBAL))
    write_json(root / 'manifest.json', report)
    return report


class Data(CocoDS):
    def __init__(self, root, split='train', spatial=False, crops=False):
        self.root = Path(root)
        rois = pd.read_parquet(self.root / 'rois.parquet')
        rois = rois[rois.target_split == split]
        super().__init__(rois, COCO_TRAIN, 384, 16, want_crops=False)
        self.spatial, self.want_teacher_crops = spatial, crops
        self.image_pos = {int(i): j for j, i in enumerate(np.load(self.root / 'image_ids.npy'))}
        self.offsets = np.load(self.root / 'offsets.npy')
        self._maps = None

    def maps(self):
        if self._maps is None:
            self._maps = {k: np.load(self.root / f'{k}.npy', mmap_mode='r') for k in ['pooled', 'g_cls', 'g_mean'] + (['spatial', 'completed'] if self.spatial else [])}
        return self._maps

    def __getitem__(self, item):
        ids, ng = item if isinstance(item, tuple) else (item, None)
        ids = [int(i) for i in ids]
        base = super().__getitem__((ids, ng))
        maps = self.maps()
        pooled, covs, qrels, sp, tcrops, cache_i, ks = [], [], [], [], [], [], []
        for iid in ids:
            i = self.image_pos[iid]
            img = Image.open(Path(COCO_TRAIN) / f'{iid:012d}.jpg').convert('RGB') if self.want_teacher_crops else None
            for r in self.by[iid].itertuples():
                lb, cov, qr = teacher_geometry((r.l, r.t, r.r, r.b))
                pooled.append(np.array(maps['pooled'][i, r.k]))
                covs.append(cov); qrels.append(qr); cache_i.append(i); ks.append(r.k)
                if self.want_teacher_crops:
                    tcrops.append(lb.canvas(img.crop((r.l, r.t, r.r, r.b))))
                if self.spatial:
                    assert bool(maps['completed'][i]), f'incomplete teacher cache image {iid}'
                    off, n = self.offsets[i, r.k]
                    target = np.zeros((2304, 768), np.float16)
                    target[cov > 0] = maps['spatial'][off:off+n]
                    sp.append(target)
        base.update(tpool=torch.tensor(np.stack(pooled)), ccov=torch.tensor(np.stack(covs)), qrel=torch.tensor(np.stack(qrels)),
                    g_cls=torch.tensor(np.stack([maps['g_cls'][self.image_pos[i]] for i in ids])),
                    g_mean=torch.tensor(np.stack([maps['g_mean'][self.image_pos[i]] for i in ids])),
                    cache_i=torch.tensor(cache_i), roi_k=torch.tensor(ks))
        if sp:
            base['target'] = torch.tensor(np.stack(sp))
        if tcrops:
            base['teacher_crops'] = torch.stack(tcrops)
        # Full384/Patch16 area-equivalent tokens, and intersected grid cells, both recorded.
        base['roi_token_area'] = base['wfull'].sum(1)
        base['roi_token_cells'] = (base['wfull'] > 0).sum(1)
        return base
