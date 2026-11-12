"""Frozen feature-only inference. Replay existing Mean Probe preprocessing and head."""
import os,sys,json,time,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT=Path('/home/irteam/Crop_representation');LEGACY=Path('/home/irteam/sg/crop-rep')
OUT=ROOT/'results/roi_pair_feature_difference_2026-11-09'
sys.path.insert(0,str(LEGACY));os.chdir(LEGACY)
import src.readout.common as RC
from src.ivc.bbdata import BBImageDS
from src.ivc.common import R0
from src.readout.models import Reader
from src.hcf.model import load_hcf
from src.hcf.probe import encode as hcf_encode
from src.viz.c0_pca import load_encoder

def fingerprint(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--condition',required=True);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    c=json.load(open(OUT/'provenance.json'))['conditions'][a.condition]
    dev=torch.device('cuda');df=pd.read_parquet(OUT/'roi_manifest.parquet');Y=np.load(OUT/'labels_val.npy')
    reader=R0().to(dev);ck=torch.load(c['mean']['checkpoint'],map_location=dev,weights_only=False);reader.load_state_dict(ck['reader']);reader.eval()
    r2=Reader('r2',layers=(12,)).to(dev);r2ck=torch.load(c['r2']['checkpoint'],map_location=dev,weights_only=False);r2.load_state_dict(r2ck['reader']);r2.eval()
    for p in list(reader.parameters())+list(r2.parameters()):p.requires_grad_(False)
    student=a.condition!='C0'
    if student:model,ec=load_hcf(c['encoder_checkpoint'],dev);assert ec['cond']=='P' and ec['kind']=='ema' and ec['step']==12000
    else:model=load_encoder('dinov3',dev);ec={}
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    before=fingerprint(model)
    ds=BBImageDS(df,Y,False,'bil' if student else 'c0',384,16,full_size=c['full_size'])
    batch=12 if student else 6
    chunks=[list(range(i,min(i+batch,len(ds)))) for i in range(0,len(ds),batch)]
    if a.limit:chunks=chunks[:a.limit]
    dl=DataLoader(ds,sampler=chunks,batch_size=1,num_workers=4,pin_memory=True,collate_fn=RC.collate_one)
    features=np.full((len(df),768),np.nan,np.float32);scores=np.full(Y.shape,np.nan,np.float32)
    r2scores=np.full(Y.shape,np.nan,np.float32)
    saved=np.load(c['mean']['scores']);start=time.monotonic();max_pool_error=0.
    for it,b in enumerate(dl):
        b=RC.to_dev(b,dev)
        if student:tok=hcf_encode(model,b,['Q'])['Q']
        else:
            parts=[]
            for i in range(0,len(b['x']),64):
                with torch.autocast('cuda',dtype=torch.bfloat16):z=model.forward_features(b['x'][i:i+64])
                parts.append(z[:,model.num_prefix_tokens:].float())
            tok=torch.cat(parts)[b['img_index']]
            b['img_index']=torch.arange(len(tok),device=dev)
        mean=R0.pooled(tok,b['cov']);rows=b['rows'].cpu().numpy()
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=reader({12:tok},b)['logits'].float()
        with torch.autocast('cuda',dtype=torch.bfloat16):r2prediction=r2({12:tok},b)['logits'].float()
        features[rows]=mean.cpu().numpy();scores[rows]=prediction.cpu().numpy()
        r2scores[rows]=r2prediction.cpu().numpy()
        # Independent explicit weighted reduction, identical pooling semantics.
        check=torch.einsum('rnd,rn->rd',tok.float(),b['cov'].float())/b['cov'].float().sum(1,keepdim=True)
        max_pool_error=max(max_pool_error,float((mean-check).abs().max()))
        if it%20==0 or it+1==len(dl):print(json.dumps(dict(condition=a.condition,batch=it+1,total=len(dl),seconds=round(time.monotonic()-start,1))),flush=True)
    complete=np.isfinite(features).all(1);err=np.abs(scores[complete]-saved[complete])
    after=fingerprint(model)
    audit=dict(condition=a.condition,n_rows=int(complete.sum()),complete=bool(complete.all()),seconds=round(time.monotonic()-start,2),feature_shape=list(features.shape),encoder_state_sha256_before=before,encoder_state_sha256_after=after,encoder_unchanged=before==after,requires_grad_any=any(p.requires_grad for p in model.parameters()),grads_present=any(p.grad is not None for p in model.parameters()),mean_probe_replay_max_abs_logit_error=float(err.max()),mean_probe_replay_mean_abs_logit_error=float(err.mean()),mean_probe_replay_exact=bool((err==0).all()),pooling_independent_max_abs_error=max_pool_error,source_encoder_step=ec.get('step'),source_encoder_kind=ec.get('kind'),head_epoch=c['mean']['epoch'],batch_images=batch,torch=str(torch.__version__),normalization='RGB ImageNet normalization in BBImageDS; final-LN tokens; existing fuse for students; fp32 coverage weighted spatial mean. Save raw means; no additional feature normalization here.')
    r2err=np.abs(r2scores[complete]-np.load(c['r2']['scores'])[complete])
    audit.update(r2_replay_max_abs_logit_error=float(r2err.max()),r2_replay_mean_abs_logit_error=float(r2err.mean()),r2_replay_exact=bool((r2err==0).all()),r2_epoch=c['r2']['epoch'])
    assert audit['encoder_unchanged'] and not audit['requires_grad_any'] and not audit['grads_present']
    assert audit['mean_probe_replay_max_abs_logit_error']<.1,audit
    assert audit['r2_replay_max_abs_logit_error']<.1,audit
    suffix='_smoke' if a.limit else ''
    first_pass=OUT/'features'/f'{a.condition}_first_pass.npy'
    if not a.limit and not first_pass.exists():np.save(first_pass,features)
    np.savez(OUT/'features'/f'{a.condition}{suffix}.npz',features=features,ann_id=df.ann_id.values,image_id=df.image_id.values,category_id=df.category_id.values,level=df.level.values,labels=Y,replayed_scores=scores,replayed_r2_scores=r2scores)
    (OUT/'features'/f'{a.condition}{suffix}_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
