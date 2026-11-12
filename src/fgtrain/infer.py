"""Existing official-manifest adapter + unchanged R2/ROIAlign24; pure exported encoders."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import src.readout.common as RC
from src.fgtrain.common import encoder, write_json
from src.g1up.official_infer import DATA, load_reader
from src.pdit.bbdata import BBImageDS, bil_sample


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--encoder',required=True);p.add_argument('--probe',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['bil','c0'],default='bil');p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--epochs',type=int,nargs='+',choices=range(1,11),default=list(range(1,11)))
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert len(set(a.epochs))==len(a.epochs)
    if a.split=='test':assert len(a.epochs)==1, 'Test uses exactly one Val-selected Reader epoch'
    vit=encoder(a.encoder)
    readers={e:load_reader(Path(a.probe)/'r2'/f'ep{e}.pt','cuda') for e in a.epochs}
    manifest=pd.read_parquet(Path(DATA)/f'manifest_{a.split}.parquet')
    if a.limit:manifest=manifest[manifest.image_id.isin(manifest.image_id.unique()[:a.limit])].reset_index(drop=True)
    ds=BBImageDS(manifest,np.zeros((len(manifest),59),np.int8),False,a.mode,384,16)
    logits={e:torch.full((len(manifest),59),float('nan')) for e in readers}
    chunks=[list(range(i,min(len(ds),i+(4 if a.mode=='c0' else 12)))) for i in range(0,len(ds),4 if a.mode=='c0' else 12)]
    dl=DataLoader(ds,sampler=chunks,batch_size=1,num_workers=a.workers,collate_fn=RC.collate_one,pin_memory=True)
    counts=dict(images=0,encoder_canvases=0,rois=0)
    start=time.time()
    for b in dl:
        counts['images']+=len(set(b['group'].tolist()));counts['encoder_canvases']+=len(b['x']);counts['rois']+=len(b['rows'])
        if a.mode=='bil':assert len(b['x'])==len(set(b['group'].tolist()))
        b=RC.to_dev(b,torch.device('cuda'));rows=b['rows'].cpu()
        tok=[]
        for i in range(0,len(b['x']),64):
            with torch.autocast('cuda',dtype=torch.bfloat16):t=vit.forward_features(b['x'][i:i+64])
            tok.append(t[:,vit.num_prefix_tokens:].float())
        f=torch.cat(tok)
        if a.mode=='bil':
            f=bil_sample(f,b)
            b['img_index']=torch.arange(len(f),device='cuda')
        for e,r in readers.items():
            with torch.autocast('cuda',dtype=torch.bfloat16):v=r({12:f},b)['logits'].float().cpu()
            logits[e][rows]=v
    for e,v in logits.items():
        assert torch.isfinite(v).all()
        folder=out/f'ep{e}';folder.mkdir(exist_ok=True)
        np.savez(folder/f'probs_{a.split}.npz',ann_id=manifest.ann_id.values,image_id=manifest.image_id.values,category_id=manifest.category_id.values,
                 logits=v.numpy(),probs=v.sigmoid().numpy())
    correlations={}
    if not a.limit and a.split=='val':
        original=RC.load_data('output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json',splits=('val',))['val'][0]
        rows=pd.Series(np.arange(len(manifest)),index=manifest.ann_id).loc[original.ann_id].values
        for e in (1,10):
            if e not in readers:continue
            saved=np.load(Path(a.probe)/'r2'/f'scores_val_ep{e}.npy').astype(np.float64).ravel()
            fresh=logits[e][rows].numpy().astype(np.float64)
            corr=max(float(np.corrcoef(saved,fresh.ravel())[0,1]),float(np.corrcoef(saved,torch.from_numpy(fresh).sigmoid().numpy().ravel())[0,1]))
            assert corr>.999,(e,corr)
            correlations[str(e)]=corr
    if a.mode=='bil':assert counts['images']==counts['encoder_canvases']
    write_json(out/'inference_verification.json',dict(status='passed',mode=a.mode,split=a.split,reader_epochs=a.epochs,encoder=a.encoder,teacher_instantiated=False,
               fine_decoder_instantiated=False,pixel_decoder_instantiated=False,standard_encoder=True,counts=counts,
               roi_rgb_reencoding=a.mode=='c0',oracle_reference=a.mode=='c0',reproduction_pearson=correlations,seconds=round(time.time()-start)))

if __name__=='__main__':main()
