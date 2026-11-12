"""Matched B0 training with P2 uniform / P3 region-selective patch anchoring."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.config import STREAM
from src.msd.common import StreamSampler
from src.ivc.ft import enc_param_groups, ema_update, run_enc
from src.fgtrain.common import encoder, state_sha, freeze_export, write_json
from src.fgtrain.audit import target_audit
from src.fgtrain.train import modules, terms, diagnose, grouped_norms
from src.fgtrain.patch_preserve import PatchData, patch_loss, drift_stats, gradient_trace, weight_stats
from src.fgtrain.patch_verify import batch0
from src.fgtrain.p1 import read, sha


def device_check(root, condition):
    physical = {'P2':0,'P3':1}[condition]
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(physical)
    assert torch.cuda.device_count() == 1
    props=torch.cuda.get_device_properties(0)
    write_json(Path(root)/'device_mapping.json',dict(physical_gpu=physical,CUDA_VISIBLE_DEVICES=str(physical),
               internal_device='cuda:0',name=props.name,uuid=str(getattr(props,'uuid','unavailable'))))


def train(root, base, shared, condition):
    root,base,shared=map(Path,[root,base,shared])
    device_check(root,condition)
    out=root/condition;out.mkdir(exist_ok=True)
    if (out/'done.json').exists():
        assert read(out/'done.json')['steps']==12000
        return
    selective=condition=='P3';total=12000
    torch.set_num_threads(2);random.seed(0);np.random.seed(0);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    scope=read(base/'official_split_exclusion.json')
    assert all(x==0 for x in scope['train_official_overlap'].values())
    assert all(x==0 for x in scope['dev_official_overlap'].values())
    write_json(out/'target_verification.json',target_audit(base))
    enc,_,_=modules(base/'initial_encoder.pt','B0')
    initial_sha=state_sha(enc.state_dict())
    assert initial_sha==read(base/'initialisation.json')['state_sha256']
    ema=copy.deepcopy(enc).eval().requires_grad_(False)
    # Constructing the frozen reference must not perturb B0's training RNG.
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    teacher=encoder(base/'initial_encoder.pt')
    torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng)
    teacher_sha=state_sha(teacher.state_dict());assert teacher_sha==initial_sha
    versions=[p._version for p in teacher.parameters()]
    calibration=read(shared/'lambda_preserve.json');lam=calibration['lambda_preserve']
    assert calibration['status']=='passed' and calibration['P2_P3_same_lambda']
    opt=torch.optim.AdamW(enc_param_groups(enc,5e-5,.75,.05),betas=(.9,.999))
    recovery=out/'recovery.pt';start=0
    if recovery.exists():
        ck=torch.load(recovery,map_location='cuda',weights_only=False)
        assert ck['condition']==condition and ck['total_steps']==total and ck['lambda_preserve']==lam
        enc.load_state_dict(ck['enc'],strict=True);ema.load_state_dict(ck['ema'],strict=True);opt.load_state_dict(ck['opt'])
        start=ck['step'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
    ds=PatchData(base/'teacher');dev=PatchData(base/'teacher','dev')
    excluded=read(base/'teacher/manifest.json')['excluded_train_images']
    sampler=StreamSampler(np.load(STREAM)[:total],0,1,exclude=excluded,start=start)
    dl=DataLoader(ds,sampler=sampler,batch_size=None,num_workers=4,pin_memory=False,prefetch_factor=2,persistent_workers=True)
    cfg=read(base/'B0/config.json')
    cfg.update(condition=condition,lambda_preserve=lam,physical_gpu=0 if condition=='P2' else 1,
               preservation='FP32 final-LN patch cosine, original frozen same Full384 reference',
               weighting='normalized 1-0.75*exact_ROI_union_coverage' if selective else 'uniform1',
               calibration=str(shared/'lambda_preserve.json'),calibration_sha256=sha(shared/'lambda_preserve.json'),
               denominator='sum fractional content coverage, then average nonzero-valid images',
               source_hash=read(shared/'source_manifest.json')['aggregate_sha256'])
    write_json(out/'config.json',cfg)
    write_json(out/'initialization_verification.json',dict(state_sha256=initial_sha,teacher_state_sha256=teacher_sha,
               Original=True,independent_start=True,calibration_checkpoint_used_for_initialization=False,
               trainable_encoder_parameters=sum(p.numel() for p in enc.parameters() if p.requires_grad),
               training_only_modules=[],source_hash=cfg['source_hash']))
    if start==0:diagnose(ema,None,None,dev,'cuda',0,out,'ema')
    def save(step):
        state=dict(enc=enc.state_dict(),ema=ema.state_dict(),opt=opt.state_dict(),step=step,condition=condition,
                   total_steps=total,lambda_preserve=lam,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        tmp=out/'recovery.tmp';torch.save(state,tmp);tmp.replace(recovery)
    t0=time.time();running=[]
    for step,b in enumerate(dl,start=start):
        b=RC.to_dev(b,'cuda')
        for g in opt.param_groups:g['lr']=RC.C.lr_at(step,g['base'],600,total)
        opt.zero_grad(set_to_none=True)
        o=terms(b,enc,None)
        if step==0:
            original=json.loads((base/'B0/per_loss_gradients.jsonl').read_text().splitlines()[0])
            check={k:math.isclose(float(o[k].mean().detach()),original[k]['loss'],rel_tol=1e-5,abs_tol=1e-7) for k in ['crop','glob']}
            write_json(out/'initial_B0_loss_match.json',check);assert all(check.values()),check
        with torch.no_grad():target,_=run_enc(teacher,b['full'])
        preserve,_,distance=patch_loss(o['tok'],target,b['patch_area'],b['roi_union'],selective)
        loss=o['crop'].mean()+.5*o['glob'].mean()+lam*preserve
        assert bool(torch.isfinite(loss)),(step,float(loss.detach()))
        diagnostic=step+1 in [1,6000,12000]
        if diagnostic:
            trace=gradient_trace(o['crop'].mean(),preserve,enc,lam)
            assert [p._version for p in teacher.parameters()]==versions
            assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            trace.update(step=step+1,**drift_stats(distance,b['patch_area'],b['roi_union']),
                         preserve_loss=float(preserve.detach()),image_ids=b['image_ids'].tolist(),
                         teacher_frozen=True,weight_stats=weight_stats(b['patch_area'],b['roi_union'],selective))
        loss.backward()
        if diagnostic:
            named=list(enc.named_parameters());trace['total_gradient']=grouped_norms(named,[p.grad for _,p in named])
            assert trace['total_gradient']['patch_embed']>0
            assert all(trace['total_gradient'][f'block_{i:02d}']>0 for i in range(12))
            with (out/'gradient_diagnostics.jsonl').open('a') as f:f.write(json.dumps(trace)+'\n')
        norm=float(torch.nn.utils.clip_grad_norm_(enc.parameters(),1.,error_if_nonfinite=True))
        opt.step();mt=1-(1-.999)*(math.cos(math.pi*step/total)+1)/2;ema_update(ema,enc,mt)
        rec=dict(loss=float(loss.detach()),crop=float(o['crop'].mean().detach()),glob=float(o['glob'].mean().detach()),
                 preserve=float(preserve.detach()),weighted_preserve=lam*float(preserve.detach()),encoder_grad=norm,
                 crop_cos=float((1-o['crop']).mean().detach()),cls_cos=float(o['cls_cos'].mean().detach()),
                 mean_cos=float(o['mean_cos'].mean().detach()),**drift_stats(distance,b['patch_area'],b['roi_union']))
        running.append(rec)
        if (step+1)%50==0:
            row=dict(step=step+1,seconds=round(time.time()-t0),**{k:float(np.mean([r[k] for r in running if r[k] is not None])) if any(r[k] is not None for r in running) else None for k in rec})
            with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write_json(out/'progress.json',row);print(json.dumps(row),flush=True);running=[]
        if (step+1)%500==0:save(step+1)
        if (step+1)%1000==0:diagnose(ema,None,None,dev,'cuda',step+1,out,'ema')
    save(total)
    assert state_sha(teacher.state_dict())==teacher_sha
    sample=b['full'][:2] if 'b' in locals() else batch0(base,'cuda')['full'][:2]
    exported=freeze_export(ema,out/'export',sample,base/'initial_encoder.pt',step=total)
    write_json(out/'done.json',dict(status='done',steps=total,export=exported,lambda_preserve=lam,
                                  teacher_frozen=True,initial_sha=initial_sha,final_ema_sha=exported['state_sha256']))


@torch.no_grad()
def drift(shared,base,p1):
    """Same excluded COCO dev126, same tensor, five frozen final encoders."""
    from src.fgtrain.gram import gram_loss
    shared,base,p1=map(Path,[shared,base,p1])
    torch.set_num_threads(2)
    teacher=encoder(base/'initial_encoder.pt')
    encs=dict(Original=teacher,B0=encoder(base/'B0/export/encoder.pt'),P1=encoder(p1/'P1/export/encoder.pt'),
              P2=encoder(shared/'P2/export/encoder.pt'),P3=encoder(shared/'P3/export/encoder.pt'))
    ds=PatchData(base/'teacher','dev');ids=list(ds.by);rows={k:[] for k in encs}
    patch_rows={k:[] for k in encs};areas=[];unions=[]
    for i in range(0,len(ids),8):
        b=RC.to_dev(ds[ids[i:i+8]],'cuda');target,_=run_enc(teacher,b['full'])
        areas.append(b['patch_area'].cpu().numpy());unions.append(b['roi_union'].cpu().numpy())
        for name,enc in encs.items():
            tok=target if name=='Original' else run_enc(enc,b['full'])[0]
            _,_,d=patch_loss(tok,target,b['patch_area'],b['roi_union'],False)
            patch_rows[name].append(torch.where(b['patch_area']>0,d,torch.nan).cpu().numpy())
            _,grams,_=gram_loss(tok,target,b['patch_area'])
            for j,iid in enumerate(b['image_ids'].tolist()):
                stats=drift_stats(d[j:j+1],b['patch_area'][j:j+1],b['roi_union'][j:j+1])
                rows[name].append(dict(image_id=iid,patch_cosine=1-stats['patch_distance'],gram_drift=float(grams[j]),**stats))
    means={name:{key:float(np.mean([r[key] for r in values if r[key] is not None])) for key in values[0] if key!='image_id'} for name,values in rows.items()}
    write_json(shared/'patch_drift.json',dict(split='Same excluded COCO dev126; no PACO Test',image_ids=ids,
               roi_partition='Fractional exact union m and1-m, each group normalized by its valid area; zero-support=None',
               mean=means,per_image=rows))
    arrays=dict(image_ids=np.asarray(ids),valid_coverage=np.concatenate(areas),roi_union=np.concatenate(unions))
    for name,values in patch_rows.items():
        d=np.concatenate(values)
        arrays['cosine_distance_'+name]=d
        arrays['normalized_l2_distance_'+name]=np.sqrt(np.maximum(2*d,0))
    np.savez_compressed(shared/'patch_drift_per_patch.npz',**arrays)
    print(json.dumps(means),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--base-run',required=True)
    p.add_argument('--shared',required=True);p.add_argument('--condition',choices=['P2','P3'])
    p.add_argument('--stage',choices=['train','drift'],default='train');p.add_argument('--p1-run');a=p.parse_args()
    if a.stage=='drift':drift(a.shared,a.base_run,a.p1_run)
    else:train(a.run,a.base_run,a.shared,a.condition)


if __name__=='__main__':main()
