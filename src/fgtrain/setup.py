"""Reproducible one-time setup. Existing allocations are never silently reindexed."""
import argparse
from pathlib import Path
import numpy as np
from src.config import STREAM
from src.fgtrain.common import RUN,initialise,write_json
from src.fgtrain.data import prepare


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN));p.add_argument('--dev-images',type=int,default=126);a=p.parse_args()
    run=Path(a.run);run.mkdir(parents=True,exist_ok=True)
    initialise(run)
    if not (run/'teacher/manifest.json').exists():prepare(run/'teacher',a.dev_images)
    import json,pandas as pd
    rois=pd.read_parquet(run/'teacher/rois.parquet')
    splits={sp:{int(i['id']) for i in json.load(open(f'/home/irteam/datasets/paco/paco_lvis_v1_{sp}.json'))['images']} for sp in ['val','test']}
    overlaps={sp:{ev:len(set(map(int,rois[rois.target_split==sp].image_id))&ids) for ev,ids in splits.items()} for sp in ['train','dev']}
    assert all(v==0 for group in overlaps.values() for v in group.values()),overlaps
    write_json(run/'official_split_exclusion.json',dict(official_images={k:len(v) for k,v in splits.items()},
              train_images=int(rois[rois.target_split=='train'].image_id.nunique()),train_rois=int((rois.target_split=='train').sum()),
              dev_images=int(rois[rois.target_split=='dev'].image_id.nunique()),train_official_overlap=overlaps['train'],dev_official_overlap=overlaps['dev'],score_or_attributes_read=False))
    write_json(run/'first_stream_ids.json',list(map(int,np.load(STREAM)[0])))

if __name__=='__main__':main()
