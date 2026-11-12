"""P1 and the matched B0 prefix needed to recover its missing6k calibration state.

The existing B0/N1 source, checkpoints, and processes are never modified.
"""
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
from src.ivc.ft import enc_param_groups,ema_update,run_enc
from src.fgtrain.common import encoder,state_sha,freeze_export,write_json
from src.fgtrain.audit import target_audit
from src.fgtrain.train import modules,terms,diagnose
from src.fgtrain.gram import GramData,gram_loss,gradient_trace


def device_check(root):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    assert torch.cuda.device_count()==1
    props=torch.cuda.get_device_properties(0)
    write_json(Path(root)/'device_mapping.json',dict(physical_gpu=1,CUDA_VISIBLE_DEVICES='1',
               internal_device='cuda:0',name=props.name,uuid=str(getattr(props,'uuid','unavailable'))))


def batch0(base,device):
    ds=GramData(Path(base)/'teacher',spatial=False)
    excluded=set(json.loads((Path(base)/'teacher/manifest.json').read_text())['excluded_train_images'])
    ids=[int(i) for i in np.load(STREAM)[0] if int(i) not in excluded]
    return RC.to_dev(ds[ids],device)


def calibrate(root,base,checkpoint):
    root,base,checkpoint=map(Path,[root,base,checkpoint])
    device_check(root)
    path=root/'lambda_gram.json'
    if path.exists():return json.loads(path.read_text())
    torch.set_num_threads(2);torch.manual_seed(0)
    teacher=encoder(base/'initial_encoder.pt')
    student=encoder(base/'initial_encoder.pt',train=True)
    b=batch0(base,'cuda')
    with torch.no_grad():target,_=run_enc(teacher,b['full'])
    o=terms(b,student,None)
    identical,_,_=gram_loss(o['tok'],target,b['gram_area'])
    assert float(identical)<1e-8,float(identical)
    ck=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    assert ck['step']==6000 and ck['condition']=='B0' and ck['total_steps']==12000
    student.load_state_dict(ck['ema'],strict=True)
    o=terms(b,student,None);loss,_,stats=gram_loss(o['tok'],target,b['gram_area'])
    trace=gradient_trace(o['crop'].mean(),loss,student,1.)
    gc,gg=trace['crop']['encoder'],trace['gram']['encoder']
    write_json(root/'calibration_measurement.json',dict(checkpoint=str(checkpoint),step=6000,
               image_ids=b['image_ids'].tolist(),gram_loss=float(loss.detach()),gradient_trace=trace,coverage=stats))
    assert math.isfinite(gg) and gg>1e-8 and gc>1e-8,trace
    for k in ['crop','gram']:
        assert trace[k]['patch_embed']>1e-12
        assert all(trace[k][f'block_{i:02d}']>1e-12 for i in range(12))
    lam=.5*gc/gg
    result=dict(status='passed',lambda_gram=lam,ratio_target=.5,measured_ratio=lam*gg/gc,
                checkpoint=str(checkpoint),checkpoint_step=6000,checkpoint_kind='EMA',
                batch='COCO B0 stream0 with original exclusions',image_ids=b['image_ids'].tolist(),
                gram_loss=float(loss),identical_initial_gram_loss=float(identical),
                unweighted_gradient_trace=trace,
                gradient_trace=dict(trace,weighted_gram_norm=lam*gg,gram_crop_ratio=lam*gg/gc),
                coverage=stats,teacher_frozen=all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
                initial_sha=state_sha(teacher.state_dict()),calibration_only=True,P1_initialization='Original pretrained; not this calibration checkpoint')
    write_json(path,result);print(json.dumps(result),flush=True)
    return result


def train(root,base,replay=False):
    root,base=Path(root),Path(base)
    device_check(root)
    out=root/('calibration_B0_replay' if replay else 'P1');out.mkdir(exist_ok=True)
    total,stop=12000,6000 if replay else 12000
    condition='B0' if replay else 'P1'
    torch.set_num_threads(2);random.seed(0);np.random.seed(0);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    scope=json.loads((base/'official_split_exclusion.json').read_text())
    assert all(x==0 for x in scope['train_official_overlap'].values())
    assert all(x==0 for x in scope['dev_official_overlap'].values())
    write_json(out/'target_verification.json',target_audit(base))
    enc,_,_=modules(base/'initial_encoder.pt','B0')
    ema=copy.deepcopy(enc).eval().requires_grad_(False)
    teacher=None if replay else encoder(base/'initial_encoder.pt')
    teacher_sha=state_sha(teacher.state_dict()) if teacher else None
    versions=[p._version for p in teacher.parameters()] if teacher else None
    lam=0. if replay else json.loads((root/'lambda_gram.json').read_text())['lambda_gram']
    opt=torch.optim.AdamW(enc_param_groups(enc,5e-5,.75,.05),betas=(.9,.999))
    recovery=out/'recovery.pt';start=0
    if recovery.exists():
        ck=torch.load(recovery,map_location='cuda',weights_only=False)
        assert ck['condition']==condition and ck['total_steps']==total and ck['lambda_gram']==lam
        enc.load_state_dict(ck['enc']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['opt'])
        start=ck['step'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
    ds=GramData(base/'teacher',spatial=False);dev=GramData(base/'teacher','dev',spatial=False)
    excluded=json.loads((base/'teacher/manifest.json').read_text())['excluded_train_images']
    sampler=StreamSampler(np.load(STREAM)[:stop],0,1,exclude=excluded,start=start)
    dl=DataLoader(ds,sampler=sampler,batch_size=None,num_workers=4,pin_memory=False,prefetch_factor=2,persistent_workers=True)
    cfg=json.loads((base/'B0/config.json').read_text())
    cfg.update(condition=condition,lambda_gram=lam,gram_teacher=None if replay else 'Fixed Original pretrained DINOv3; never EMA updated',
               final_step=stop,total_schedule_steps=total,calibration_replay=replay,physical_gpu=1,
               gram='FP32 L2-normalized final-LN patch similarities; remove_neg=False; pair weight wi*wj; per-image normalized weighted MSE')
    write_json(out/'config.json',cfg)
    if start==0:diagnose(ema,None,None,dev,'cuda',0,out,'ema')
    def save(step):
        state=dict(enc=enc.state_dict(),ema=ema.state_dict(),opt=opt.state_dict(),step=step,condition=condition,
                   total_steps=total,lambda_gram=lam,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
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
            write_json(out/'initial_B0_loss_match.json',check)
            assert all(check.values()), 'Initial Crop/Global must match the recorded B0 batch'
        loss=o['crop'].mean()+.5*o['glob'].mean()
        drift=o['tok'].sum()*0.
        if teacher:
            with torch.no_grad():target,_=run_enc(teacher,b['full'])
            drift,_,stats=gram_loss(o['tok'],target,b['gram_area'])
            loss=loss+lam*drift
            if step in [0,1,2] or (step+1)%1000==0:
                trace=gradient_trace(o['crop'].mean(),drift,enc,lam)
                assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
                assert [p._version for p in teacher.parameters()]==versions
                with (out/'gradient_diagnostics.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step+1,gram_drift=float(drift.detach()),**trace))+'\n')
        assert torch.isfinite(loss)
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(enc.parameters(),1.,error_if_nonfinite=True))
        opt.step()
        mt=1-(1-.999)*(math.cos(math.pi*step/total)+1)/2
        ema_update(ema,enc,mt)
        rec=dict(loss=float(loss.detach()),crop=float(o['crop'].mean().detach()),glob=float(o['glob'].mean().detach()),
                 gram=float(drift.detach()),weighted_gram=lam*float(drift.detach()),encoder_grad=norm,
                 crop_cos=float((1-o['crop']).mean().detach()),cls_cos=float(o['cls_cos'].mean().detach()),
                 mean_cos=float(o['mean_cos'].mean().detach()))
        running.append(rec)
        if (step+1)%50==0:
            row=dict(step=step+1,seconds=round(time.time()-t0),**{k:float(np.mean([r[k] for r in running])) for k in rec})
            with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write_json(out/'progress.json',row);print(json.dumps(row),flush=True);running=[]
        if (step+1)%500==0:save(step+1)
        if (step+1)%1000==0:diagnose(ema,None,None,dev,'cuda',step+1,out,'ema')
    save(stop)
    if replay:
        # Final replay state is6000, but LR/EMA use the original12000-step schedule.
        write_json(out/'done.json',dict(status='done',step=6000,total_schedule_steps=12000,checkpoint=str(recovery),no_evaluation=True))
        return
    assert state_sha(teacher.state_dict())==teacher_sha
    sample=b['full'][:2] if 'b' in locals() else batch0(base,'cuda')['full'][:2]
    exported=freeze_export(ema,out/'export',sample,base/'initial_encoder.pt',step=12000)
    write_json(out/'done.json',dict(status='done',steps=12000,export=exported,lambda_gram=lam,
                                  gram_teacher_frozen=True,teacher_initial_sha=teacher_sha))


@torch.no_grad()
def drift(root,base):
    root,base=Path(root),Path(base)
    device_check(root)
    teacher=encoder(base/'initial_encoder.pt')
    encs=dict(Original=teacher,B0=encoder(base/'B0/export/encoder.pt'),P1=encoder(root/'P1/export/encoder.pt'))
    ds=GramData(base/'teacher','dev',spatial=False);rows={k:[] for k in encs};ids=list(ds.by)
    for i in range(0,len(ids),8):
        b=RC.to_dev(ds[ids[i:i+8]],'cuda');target,_=run_enc(teacher,b['full'])
        for name,enc in encs.items():
            tokens=target if name=='Original' else run_enc(enc,b['full'])[0]
            _,per,_=gram_loss(tokens,target,b['gram_area']);rows[name].extend(per.cpu().tolist())
    result=dict(split='Same excluded COCO dev126 images; no PACO Test',image_ids=ids,
                mean={k:float(np.mean(v)) for k,v in rows.items()},per_image=rows)
    write_json(root/'gram_drift.json',result);print(json.dumps(result['mean']),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--base-run',required=True)
    p.add_argument('--stage',choices=['replay','calibrate','train','drift'],required=True);p.add_argument('--checkpoint')
    a=p.parse_args()
    if a.stage=='calibrate':calibrate(a.run,a.base_run,a.checkpoint)
    elif a.stage=='drift':drift(a.run,a.base_run)
    else:train(a.run,a.base_run,replay=a.stage=='replay')


if __name__=='__main__':main()
