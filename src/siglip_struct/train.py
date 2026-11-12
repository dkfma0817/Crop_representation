"""P4 encoder or frozen-P12k ROI decoder; existing cosine objectives only."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.config import STREAM, FEAT
from src.msd.pretrain import eval_exclusions
from src.ivc.ft import cosd, ema_update
from src.siglip_pm.distill import atomic_save, groups, Slots, cell_w, pool
from src.siglip_pm.model import load_vit, checksum, MODEL_ID, REVISION, n_params
from src.siglip_extra.train import make_dev_batches, file_sha
from src.siglip_struct.data import TrainingDS, micro_batches, rectangle
from src.siglip_struct.model import build_p4, load_p8
from src.siglip_struct.decoder import ROIDecoder


def pool_rois(tokens, roi_img, weights):
    outputs, indices = [], []
    for i in range(len(tokens)):
        sel = torch.where(roi_img==i)[0]
        w = weights[sel].float()
        outputs.append((w@tokens[i].float())/w.sum(-1, keepdim=True).clamp_min(1e-12))
        indices.append(sel)
    order = torch.cat(indices)
    return torch.cat(outputs)[torch.argsort(order)]


def p4_terms(model, b, collect=False):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o = model.forward_all(b['full'], collect)
    a = pool_rois(o['coarse'], b['roi_img'], cell_w(b['box_c'], 24, 16))
    f = pool_rois(o['mem']['fine'], b['roi_img'], cell_w(b['box_c'], 96, 4))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        fused = model.fuse(a, dict(fine=f))
    return cosd(fused.float(), b['tpool']), .5*(cosd(o['cls'].float(), b['gembed'])+cosd(pool(o['coarse'].float(), b['fvalid']), b['gmean'])), o


def prepare_p8(encoder, b):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = encoder.forward_all(b['full'])
        o['fine_projected'] = encoder.Wd(o['mem']['fine'])
    return o


def decoder_terms(decoder, encoder, b):
    o = prepare_p8(encoder, b)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z = decoder(o, encoder, b['roi_img'], b['box_c'], b['valid_rect'])
    return cosd(z.float().mean(1), b['tpool']), z


@torch.no_grad()
def calibrate_p4(model, b):
    model.eval()
    history = []
    for rnd in range(3):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model.forward_all(b['full'][:8], True)
            f = torch.nn.functional.avg_pool2d(o['mem']['fine'].reshape(-1, 96, 96, 384).permute(0, 3, 1, 2), 4).flatten(2).transpose(1, 2)
            q = model.fuse(o['coarse'], dict(fine=f))
            ratio = float((q-o['coarse']).norm(dim=-1).mean()/o['coarse'].norm(dim=-1).mean())
        model.s_d.mul_(.05/ratio)
        for s, interaction in enumerate(model.inter):
            interaction.s_f2c.mul_(.05/model.diag[f'stage{s+1}_res_c'])
            interaction.s_c2f.mul_(.05/model.diag[f'stage{s+1}_res_f'])
        history.append(dict(round=rnd, ratio=ratio, **model.diag))
    model.train()
    return dict(rule='Original three-round5% residual norm rule', history=history,
                scales={n: float(v) for n,v in model.named_parameters() if v.ndim==0})


def init(a):
    torch.manual_seed(0)
    encoder = None
    if a.kind=='P4':
        model = build_p4('cuda').train()
    else:
        encoder = load_p8(a.p8, 'cuda')
        torch.manual_seed(0)
        model = ROIDecoder().to('cuda').train()
    return model, encoder


def optimizer(model, kind):
    if kind=='P4':
        return torch.optim.AdamW(groups(model), betas=(.9,.999))
    groups_ = []
    for decay in [True, False]:
        ps = [p for n,p in model.named_parameters() if (p.ndim>1 and n!='q_grid')==decay]
        groups_.append(dict(params=ps, base=3e-4, lr=3e-4, weight_decay=.05 if decay else 0.))
    return torch.optim.AdamW(groups_, betas=(.9,.999))


@torch.no_grad()
def diagnostics(a, model, encoder, batches, step):
    was = model.training
    model.eval()
    vals = []
    for batch in batches:
        for cpu in micro_batches(batch, a.microbatch):
            b = RC.to_dev(cpu, 'cuda')
            if a.kind=='P4':
                crop, glob, _ = p4_terms(model, b)
                vals.append((float((1-crop).sum()), len(crop), float(glob.sum()), len(glob)))
            else:
                crop, _ = decoder_terms(model, encoder, b)
                vals.append((float((1-crop).sum()), len(crop), 0., 0))
    if was:model.train()
    return dict(step=step, crop_cosine=sum(v[0] for v in vals)/sum(v[1] for v in vals),
                global_loss=sum(v[2] for v in vals)/max(1,sum(v[3] for v in vals)), split='same fixed48 COCO dev images')


def run(a):
    out = Path(a.out)
    if not a.smoke and (out/'done.json').exists():return
    model, encoder = init(a)
    opt = optimizer(model, a.kind)
    frozen_sum = checksum(encoder.parameters()) if encoder is not None else None
    original = load_vit('cuda').eval().requires_grad_(False)
    teacher_sum = checksum(original.parameters())
    init_encoder_sum = checksum(model.vit.parameters()) if a.kind=='P4' else frozen_sum
    if a.kind=='P4':assert init_encoder_sum==teacher_sum, 'P4 must start from the identical official pretrained encoder'
    ex = eval_exclusions()
    ds = TrainingDS(a.shared)
    first = next(iter(Slots(np.load(STREAM)[:1], 0, 1, ex['raev2_stream'])))
    cpu = ds[first]
    if a.kind=='P4':
        cal = out/'calib.json'
        if cal.exists():
            scales = json.loads(cal.read_text())['scales']
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in scales:p.copy_(torch.tensor(scales[name],device='cuda'))
        else:
            cal.write_text(json.dumps(calibrate_p4(model, RC.to_dev(cpu, 'cuda')), indent=2))
    if a.smoke:
        from src.siglip_extra.data import ExtraDS
        pixels=ExtraDS(a.shared,spatial=True)[first]
        n=int((pixels['roi_img']==0).sum())
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            t=original.forward_features(pixels['crops'][:n].to('cuda')).float()
        target=pool(t,pixels['local_teacher_weights'][:n].sum(1).to('cuda'))
        correspondence=float(torch.nn.functional.cosine_similarity(target,pixels['tpool'][:n].to('cuda'),dim=-1).min())
        assert correspondence>.9995,correspondence
        del pixels,t,target
        torch.cuda.reset_peak_memory_stats();started=time.time()
        opt.zero_grad(set_to_none=True);stats=[]
        for micro in micro_batches(cpu, a.microbatch):
            b = RC.to_dev(micro, 'cuda')
            if a.kind=='P4':
                cr, gl, o = p4_terms(model, b, True)
                assert o['coarse'].shape[1:]==(576,768) and o['mem']['fine'].shape[1:]==(9216,384)
                loss=cr.sum()/len(cpu['tpool'])+.5*gl.sum()/len(cpu['full'])
            else:
                cr,z=decoder_terms(model,encoder,b);assert z.shape[1:]==(16,768)
                loss=cr.sum()/len(cpu['tpool'])
            assert torch.isfinite(loss)
            loss.backward();stats.append(float(loss))
        bad=[n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not bad,bad
        module_norms={}
        for n,p in model.named_parameters():
            if p.grad is not None:
                key=n.split('.')[0];module_norms[key]=module_norms.get(key,0.)+float(p.grad.float().square().sum())
        assert all(v>0 for v in module_norms.values()),module_norms
        assert all(not p.requires_grad and p.grad is None for p in original.parameters())
        if encoder is not None:assert checksum(encoder.parameters())==frozen_sum and not any(p.grad is not None for p in encoder.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        torch.cuda.synchronize()
        info=dict(status='passed',kind=a.kind,effective_batch=len(cpu['full']),microbatch=a.microbatch,
                  loss=sum(stats),seconds=time.time()-started,peak_memory_GiB=torch.cuda.max_memory_allocated()/2**30,
                  module_gradient_norm={k:math.sqrt(v) for k,v in module_norms.items()},teacher_frozen=True,
                  encoder_frozen=a.kind=='Decode',teacher_checksum=teacher_sum,n_parameters=n_params(model),teacher_cache_cosine_min=correspondence)
        (out/'smoke.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
        return
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    start=0
    checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
    latest=sorted(checkpoints.glob('ckpt_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    if latest:
        ck=torch.load(latest[-1],map_location='cuda',weights_only=False)
        assert ck['architecture']==a.kind
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['opt']);start=ck['step']
    dev=make_dev_batches(SimpleNamespace(kind='Continue'),original,'cuda')
    dr=pd.read_parquet(FEAT+'/dev_rois.parquet',columns=['image_id','W','H']).drop_duplicates('image_id').set_index('image_id')
    for b in dev:
        b['valid_rect']=torch.tensor([rectangle(dr.loc[int(i),'W'],dr.loc[int(i),'H']) for i in b['image_ids'].cpu()],dtype=torch.float32,device='cuda')
    # Keep diagnostic inputs on CPU between diagnostics, including originals.
    dev=[{k:v.cpu() if torch.is_tensor(v) else v for k,v in b.items()} for b in dev]
    cfg=dict(architecture=a.kind,steps=a.steps,seed=0,input=384,teacher=MODEL_ID,revision=REVISION,teacher_checksum=teacher_sum,initial_encoder_checksum=init_encoder_sum,
             effective_batch=16,microbatch=a.microbatch,optimizer='original AdamW; warmup600/cosine12k; EMA.999->1',
             crop_loss='original coverage-weighted mean cosine',global_weight=.5 if a.kind=='P4' else 0.,
             encoder_frozen=a.kind=='Decode',p8_checkpoint=a.p8 if a.kind=='Decode' else None,
             p8_sha256=file_sha(a.p8) if a.kind=='Decode' else None,n_parameters=n_params(model),args=vars(a))
    (out/'config.json').write_text(json.dumps(cfg,indent=2));log=(out/'train_log.jsonl').open('a',buffering=1);dlog=(out/'diagnostics.jsonl').open('a',buffering=1)
    dl=DataLoader(ds,sampler=Slots(np.load(STREAM)[:a.steps],0,1,ex['raev2_stream'],start),batch_size=None,num_workers=a.workers,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    started=time.time();last=started;laststep=start
    dlog.write(json.dumps(diagnostics(a,model,encoder,dev,start))+'\n');torch.cuda.reset_peak_memory_stats()
    def save(step,final=False):
        meta=dict(architecture=a.kind,cond='P',step=step,backbone=MODEL_ID,revision=REVISION,encoder_checkpoint=a.p8 if encoder is not None else None)
        atomic_save(dict(**meta,model=model.state_dict(),ema=ema.state_dict(),opt=opt.state_dict()),checkpoints/f'ckpt_{step}.pt')
        if final:atomic_save(dict(**meta,kind='ema',model=ema.state_dict()),checkpoints/'ema_last.pt')
    for step,cpu in enumerate(dl,start):
        for g in opt.param_groups:g['lr']=RC.C.lr_at(step,g['base'],600,a.steps)
        opt.zero_grad(set_to_none=True);s_crop=0.;s_global=0.
        for micro in micro_batches(cpu,a.microbatch):
            b=RC.to_dev(micro,'cuda')
            if a.kind=='P4':
                cr,gl,_=p4_terms(model,b)
                loss=cr.sum()/len(cpu['tpool'])+.5*gl.sum()/len(cpu['full']);s_global+=float(gl.detach().sum())/len(cpu['full'])
            else:
                cr,_=decoder_terms(model,encoder,b);loss=cr.sum()/len(cpu['tpool'])
            assert torch.isfinite(loss),step
            loss.backward();s_crop+=float(cr.detach().sum())/len(cpu['tpool'])
        if step==start or (step+1)%500==0:
            bad=[n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()];assert not bad,bad
        if a.kind=='P4':
            gn=float(torch.nn.utils.clip_grad_norm_(model.vit.parameters(),1));gnew=float(torch.nn.utils.clip_grad_norm_([p for _,p in model.new_named()],1))
        else:gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));gnew=gn
        opt.step();ema_update(ema,model,1-(1-.999)*(math.cos(math.pi*step/a.steps)+1)/2)
        if (step+1)%50==0:
            now=time.time();record=dict(step=step+1,seconds=now-started,crop_loss=s_crop,crop_cosine=1-s_crop,global_loss=s_global,
                   total_loss=s_crop+.5*s_global,gradient_norm=gn,new_gradient_norm=gnew,steps_per_s=(step+1-laststep)/(now-last),peak_memory_GiB=torch.cuda.max_memory_allocated()/2**30)
            log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True);last=now;laststep=step+1
        if (step+1)%500==0:save(step+1)
        if (step+1)%3000==0:dlog.write(json.dumps(diagnostics(a,ema,encoder,dev,step+1))+'\n')
    save(a.steps,True)
    assert checksum(original.parameters())==teacher_sum
    if encoder is not None:assert checksum(encoder.parameters())==frozen_sum
    (out/'done.json').write_text(json.dumps(dict(status='done',steps=a.steps,seconds=time.time()-started,teacher_unchanged=True,encoder_unchanged=None if encoder is None else checksum(encoder.parameters())==frozen_sum),indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['P4','Decode'],required=True)
    for name in ['out','shared','p8']:ap.add_argument('--'+name,required=True)
    ap.add_argument('--steps',type=int,default=12000);ap.add_argument('--microbatch',type=int,default=4);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    from src.siglip_extra.resources import gpu_lease
    try:
        with gpu_lease(out.parent):run(a)
    except Exception:
        (out/'failed.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
