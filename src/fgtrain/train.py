"""B0/B1/B2: standard trainable DINOv3, auxiliary modules used only while training."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import src.readout.common as RC
from src.config import STREAM
from src.g1x.distill import pool, cosd, gnorm
from src.ivc.ft import enc_param_groups, ema_update, run_enc
from src.msd.common import StreamSampler
from src.fgtrain.common import RUN, encoder, freeze_export, state_sha, write_json
from src.fgtrain.data import Data
from src.fgtrain.model import FineDecoder, PixelDecoder


def modules(initial, condition, dev='cuda'):
    enc = encoder(initial, dev, train=condition != 'FrozenDecoder')
    torch.manual_seed(0)
    dec = FineDecoder().to(dev) if condition in ('B1','B2','FrozenDecoder') else None
    torch.manual_seed(0)
    pix = PixelDecoder().to(dev) if condition == 'B2' else None
    return enc, dec, pix


def terms(b, enc, dec, pix=None, rank=False, rank_details=False):
    tok, cls = run_enc(enc, b['full'])
    assert tok.shape[1:] == (576, 768)
    target, tc, tm = b['tpool'].detach(), b['g_cls'].detach(), b['g_mean'].detach()
    assert all(torch.isfinite(t).all() for t in (target, tc, tm))
    ri, wf, cc = b['roi_img'], b['wfull'], b['ccov']
    pooled = pool(tok[ri], wf)
    crop = cosd(pooled, target)
    gc, gm = cosd(cls, tc), cosd(pool(tok, b['fvalid']), tm)
    out = dict(crop=crop, glob=.5*(gc+gm), cls_cos=1-gc, mean_cos=1-gm, tok=tok)
    if rank:
        from src.fgtrain.ranking import ordering_loss
        out['rank'],out['rank_stats']=ordering_loss(pooled,target,b['ltrb'],b['image_ids'][ri],details=rank_details)
        out['target_pool']=target
    if dec is not None:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = dec(tok, ri, b['box_c'], b['qrel'], b['fvalid'])
        t = b['target'].detach()
        out['spatial'] = (cosd(z, t) * cc).sum(1) / cc.sum(1).clamp_min(1e-12)
        out['decoder_pool_cos'] = 1-cosd(pool(z, cc), target)
    if pix is not None:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = pix(tok).float()
        mean = tok.new_tensor([.485,.456,.406])[None,:,None,None]
        std = tok.new_tensor([.229,.224,.225])[None,:,None,None]
        rgb = b['full'] * std + mean
        target_rgb = F.unfold(rgb, 16, stride=16).transpose(1,2).detach()
        out['pixel'] = (pred-target_rgb).abs().mean((1,2))
        # Local edge error is a diagnostic; no edge term is added to the initial pilot loss.
        pr, tr = [x.reshape(-1,3,16,16) for x in (pred,target_rgb)]
        out['edge'] = .5*((pr[...,1:]-pr[...,:-1] - (tr[...,1:]-tr[...,:-1])).abs().mean() +
                         (pr[...,1:,:]-pr[...,:-1,:] - (tr[...,1:,:]-tr[...,:-1,:])).abs().mean())
    return out


def grouped_norms(named, grads):
    vals = {'encoder': [], 'patch_embed': [], **{f'block_{k:02d}': [] for k in range(12)}}
    for (name,p),g in zip(named, grads):
        if g is None: continue
        vals['encoder'].append(g)
        if name.startswith('patch_embed.'):
            vals['patch_embed'].append(g)
        elif name.startswith('blocks.'):
            vals[f'block_{int(name.split(".")[1]):02d}'].append(g)
    return {name: gnorm(gs) for name,gs in vals.items()}


def per_loss_grads(o, enc, dec):
    ep = [(n,p) for n,p in enc.named_parameters() if p.requires_grad]
    dp = list(dec.parameters()) if dec is not None else []
    result = {}
    for name in ['crop','spatial','glob','rank']:
        if name not in o: continue
        loss = o[name].mean()
        if not loss.requires_grad: continue
        gs = torch.autograd.grad(loss, [p for _,p in ep]+dp, retain_graph=True, allow_unused=True)
        result[name] = {**grouped_norms(ep,gs[:len(ep)]), 'decoder':gnorm(gs[len(ep):]), 'loss':float(loss.detach())}
    return result


def calibration(run, root):
    run, root = Path(run), Path(root)
    enc,dec,_ = modules(run/'initial_encoder.pt','B1')
    ds=Data(root,spatial=True)
    ex=json.loads((root/'manifest.json').read_text())['excluded_train_images']
    ids=[int(i) for i in np.load(STREAM)[0] if int(i) not in ex]
    b=RC.to_dev(ds[ids],torch.device('cuda'))
    o=terms(b,enc,dec)
    norms=per_loss_grads(o,enc,dec)
    gc,gs=norms['crop']['encoder'],norms['spatial']['encoder']
    assert math.isfinite(gs) and gs > 1e-10 and gc > 1e-10, norms
    lam=.5*gc/gs
    result=dict(lambda_spatial=lam, lambda_global=.5, lambda_pixel=1., target_ratio=.5,
                measured_weighted_ratio=lam*gs/gc, procedure='One initial SAME minibatch (stream step 0), no encoder/decoder update; fixed for all 12000 steps',
                gradients=norms, decoder_seed=0, initial_sha=json.loads((run/'initialisation.json').read_text())['state_sha256'],
                warmup_decoder_steps=0, image_ids=ids)
    write_json(run/'lambda_spatial.json',result)
    print(json.dumps(result),flush=True)
    return result


@torch.no_grad()
def diagnose(enc,dec,pix,ds,dev,step,out,which):
    enc_was=enc.training
    enc.eval()
    if dec:dec.eval()
    if pix:pix.eval()
    rows=[]
    for ids in [list(ds.by)[i:i+8] for i in range(0,len(ds.by),8)]:
        b=RC.to_dev(ds[ids],dev)
        o=terms(b,enc,dec,pix)
        for j,(iid,k) in enumerate(zip(b['image_ids'][b['roi_img']].tolist(),b['roi_k'].tolist())):
            rows.append(dict(image_id=iid,k=k,crop_cos=float(1-o['crop'][j]),
                             spatial_cos=float(1-o['spatial'][j]) if dec else None,
                             roi_token_area=float(b['roi_token_area'][j]),roi_intersected_cells=int(b['roi_token_cells'][j])))
        for key in ['cls_cos','mean_cos']:
            for j,v in enumerate(o[key].tolist()):
                # Store image metrics once alongside that image's first ROI row.
                rows[-len(o['crop']) + int((b['roi_img']==j).nonzero()[0])][key]=v
    import pandas as pd
    df=pd.DataFrame(rows)
    folder=Path(out)/'diagnostics';folder.mkdir(exist_ok=True)
    df.to_csv(folder/f'{which}_{step:06d}.csv',index=False)
    result=dict(step=step,model=which,**{k:float(df[k].mean()) for k in ['crop_cos','cls_cos','mean_cos']},
                spatial_cos=float(df.spatial_cos.mean()) if dec else None,
                n_images=len(ds.by),n_rois=len(df),roi_token_area_quantiles=df.roi_token_area.quantile([0,.25,.5,.75,1]).to_dict())
    if dec:
        result['spatial_by_observed_token_area']={str(interval):dict(rois=len(g),spatial_cos=float(g.spatial_cos.mean()))
            for interval,g in df.groupby(pd.cut(df.roi_token_area,[0,1,4,16,np.inf]),observed=True)}
    with (Path(out)/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    enc.train(enc_was)
    if dec:dec.train()
    if pix:pix.train()
    print(json.dumps(result),flush=True)
    return result


def train(a):
    rank,world=RC.ddp_setup();dev=torch.device('cuda')
    torch.set_num_threads(2)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    from src.fgtrain.audit import target_audit,await_spatial_cache
    scope=json.loads((Path(a.run)/'official_split_exclusion.json').read_text())
    assert all(v==0 for v in scope['train_official_overlap'].values()) and all(v==0 for v in scope['dev_official_overlap'].values())
    if a.condition in ('B1','B2','FrozenDecoder'):await_spatial_cache(a.run)
    write_json(out/'target_verification.json',target_audit(a.run))
    initial=Path(a.run)/'initial_encoder.pt'
    weights=json.loads((Path(a.run)/'lambda_spatial.json').read_text()) if a.condition in ('B1','B2','FrozenDecoder') else dict(lambda_spatial=0.,lambda_global=.5,lambda_pixel=1.)
    lam=weights['lambda_spatial']
    rank_config=json.loads(Path(a.rank_config).read_text()) if a.condition=='N1' else None
    lambda_rank=rank_config['lambda_rank'] if rank_config else 0.
    enc,dec,pix=modules(initial,a.condition)
    ema=copy.deepcopy(enc).eval().requires_grad_(False)
    dema=copy.deepcopy(dec).eval().requires_grad_(False) if dec else None
    pema=copy.deepcopy(pix).eval().requires_grad_(False) if pix else None
    eg=enc_param_groups(enc,5e-5,.75,.05) if a.condition!='FrozenDecoder' else []
    for g in eg:g['mod']='enc'
    groups=eg
    for label,mod in [('dec',dec),('pixel',pix)]:
        if mod is None:continue
        for decay in [True,False]:
            ps=[p for n,p in mod.named_parameters() if (p.ndim>1 and n!='q_grid') == decay]
            groups.append(dict(params=ps,lr=3e-4,base=3e-4,weight_decay=.05 if decay else 0.,mod=label))
    opt=torch.optim.AdamW(groups,betas=(.9,.999))
    start=0;recovery=out/'recovery.pt'
    if recovery.exists():
        ck=torch.load(recovery,map_location=dev,weights_only=False)
        assert ck['condition']==a.condition and ck['lambda_spatial']==lam and ck['total_steps']==a.steps
        assert ck.get('lambda_rank',0.)==lambda_rank
        enc.load_state_dict(ck['enc']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['opt'])
        if dec:dec.load_state_dict(ck['dec']);dema.load_state_dict(ck['dema'])
        if pix:pix.load_state_dict(ck['pix']);pema.load_state_dict(ck['pema'])
        start=ck['step'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        print(f'RESUME {a.condition} step {start}',flush=True)
    ds=Data(a.cache,spatial=dec is not None)
    dds=Data(a.cache,'dev',spatial=dec is not None)
    ex=json.loads((Path(a.cache)/'manifest.json').read_text())['excluded_train_images']
    stream=np.load(STREAM)[:a.steps]
    sampler=StreamSampler(stream,rank,world,exclude=ex,start=start)
    dl=DataLoader(ds,sampler=sampler,batch_size=None,num_workers=a.workers,pin_memory=False,
                  **(dict(prefetch_factor=2,persistent_workers=True) if a.workers else {}))
    info=dict(condition=a.condition,initial=str(initial),initial_sha=state_sha(torch.load(initial,map_location='cpu',weights_only=True)),
              steps=a.steps,seed=0,world=world,lambda_spatial=lam,lambda_global=.5,lambda_pixel=weights['lambda_pixel'] if pix else 0,
              encoder_lr=5e-5,llrd=.75,decoder_lr=3e-4,wd=.05,betas=[.9,.999],warmup=600,schedule='cosine',ema_m0=.999,
              global_images_per_step=16,rois_per_image=4,roi_rule='unchanged GT/random/fallback stream',cache=a.cache,
              trainable_encoder=a.condition!='FrozenDecoder',encoder_parameters=sum(p.numel() for p in enc.parameters()),
              decoder_parameters=sum(p.numel() for p in dec.parameters()) if dec else 0,selection='Final EMA at step 12000; no PACO/Test method selection')
    if rank_config:info.update(lambda_rank=lambda_rank,ranking_config=rank_config['ranking_config'],ranking_preflight=a.rank_config,attribute_labels_used=False)
    if rank==0:write_json(out/'config.json',info)
    # Diagnostics are single-GPU runs; the driver uses one GPU per condition.
    assert world==1,'Use independent conditions on two GPUs; stream remains the same 64 ROIs per global step'
    if start==0:diagnose(ema,dema,pema,dds,dev,0,out,'ema')
    t0=time.time();running=[]
    frozen_sha=state_sha(enc.state_dict()) if a.condition=='FrozenDecoder' else None
    def save(step):
        state=dict(enc=enc.state_dict(),ema=ema.state_dict(),dec=dec.state_dict() if dec else None,dema=dema.state_dict() if dema else None,
                   pix=pix.state_dict() if pix else None,pema=pema.state_dict() if pema else None,opt=opt.state_dict(),step=step,
                   condition=a.condition,lambda_spatial=lam,lambda_rank=lambda_rank,total_steps=a.steps,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        tmp=out/'recovery.tmp';torch.save(state,tmp);tmp.replace(recovery)
    for step,b in enumerate(dl,start=start):
        b=RC.to_dev(b,dev)
        for g in opt.param_groups:g['lr']=RC.C.lr_at(step,g['base'],600,a.steps)
        opt.zero_grad(set_to_none=True)
        o=terms(b,enc,dec,pix,rank=a.condition=='N1',rank_details=step in (0,1,2) or (step+1)%1000==0)
        loss=o['crop'].mean()+.5*o['glob'].mean()
        if dec:loss=loss+lam*o['spatial'].mean()
        if pix:loss=loss+weights['lambda_pixel']*o['pixel'].mean()
        if rank_config:loss=loss+lambda_rank*o['rank']
        if step in (0,1,2) or (step+1)%1000==0:
            gr=per_loss_grads(o,enc,dec)
            if a.condition!='FrozenDecoder':
                for name in ['crop']+(['spatial'] if dec else [])+(['rank'] if rank_config and o['rank_stats']['valid_pairs'] else []):
                    assert gr[name]['encoder']>1e-10 and gr[name]['patch_embed']>1e-12,gr
                    assert all(gr[name][f'block_{k:02d}']>1e-12 for k in range(12)),gr
            with (out/'per_loss_gradients.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,**gr))+'\n')
            if rank_config:
                with (out/'neighbor_diagnostics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,**o['rank_stats']))+'\n')
        assert torch.isfinite(loss),f'nonfinite loss step {step+1}'
        loss.backward()
        eparams=[p for p in enc.parameters() if p.requires_grad]
        en=float(torch.nn.utils.clip_grad_norm_(eparams,1.,error_if_nonfinite=True)) if eparams else 0.
        dn=float(torch.nn.utils.clip_grad_norm_(dec.parameters(),1.,error_if_nonfinite=True)) if dec else 0.
        pn=float(torch.nn.utils.clip_grad_norm_(pix.parameters(),1.,error_if_nonfinite=True)) if pix else 0.
        if step in (0,1,2) or (step+1)%100==0:
            eg=grouped_norms(list(enc.named_parameters()),[p.grad for p in enc.parameters()])
            with (out/'block_gradients.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,**eg))+'\n')
        opt.step()
        mt=1-(1-.999)*(math.cos(math.pi*step/a.steps)+1)/2
        if a.condition!='FrozenDecoder':ema_update(ema,enc,mt)
        if dec:ema_update(dema,dec,mt)
        if pix:ema_update(pema,pix,mt)
        rec=dict(loss=float(loss.detach()),crop=float(o['crop'].mean().detach()),glob=float(o['glob'].mean().detach()),
                 crop_cos=float((1-o['crop']).mean().detach()),cls_cos=float(o['cls_cos'].mean().detach()),patch_mean_cos=float(o['mean_cos'].mean().detach()),
                 encoder_grad=en,decoder_grad=dn,pixel_grad=pn)
        if dec:rec.update(spatial=float(o['spatial'].mean().detach()),spatial_cos=float((1-o['spatial']).mean().detach()))
        if pix:rec.update(pixel_l1=float(o['pixel'].mean().detach()),edge_error=float(o['edge'].detach()))
        if rank_config:rec.update(rank=float(o['rank'].detach()),weighted_rank=float(lambda_rank*o['rank'].detach()),
                                  **{'rank_'+k:float(v) for k,v in o['rank_stats'].items() if isinstance(v,(float,int))})
        running.append(rec)
        if (step+1)%50==0 or step+1==a.steps:
            record=dict(step=step+1,seconds=round(time.time()-t0),**{k:float(np.mean([r[k] for r in running])) for k in rec})
            with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(out/'progress.json',record);print(json.dumps(record),flush=True);running=[]
        if (step+1)%500==0:save(step+1)
        if (step+1)%1000==0 or step+1==a.steps:
            diagnose(ema,dema,pema,dds,dev,step+1,out,'ema')
    if a.condition=='FrozenDecoder':
        assert state_sha(enc.state_dict())==frozen_sha
        assert state_sha(ema.state_dict())==frozen_sha
    diagnose(enc,dec,pix,dds,dev,a.steps,out,'student')
    sample=b['full'][:2]
    export=freeze_export(ema,out/'export',sample,initial,step=a.steps)
    freeze_export(enc,out/'export_student',sample,initial,step=a.steps)
    torch.save(dict(dec=dema.state_dict() if dema else None,pixel=pema.state_dict() if pema else None,
                    step=a.steps,kind='training_auxiliaries'),out/'auxiliary_last.pt')
    save(a.steps)
    write_json(out/'done.json',dict(status='done',steps=a.steps,seconds=round(time.time()-t0),export=export,encoder_frozen=a.condition=='FrozenDecoder'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['B0','B1','B2','FrozenDecoder','N1'],default='B1')
    p.add_argument('--run',default=str(RUN));p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--workers',type=int,default=8);p.add_argument('--calibrate',action='store_true')
    p.add_argument('--rank-config',default='')
    a=p.parse_args()
    if a.calibrate:calibration(a.run,a.cache)
    else:train(a)

if __name__=='__main__':main()
