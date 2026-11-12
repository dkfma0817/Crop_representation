"""Independent P12k+E1/E24k (or unchanged-loss Continue control).
Official pretrained SigLIP2 remains the fixed teacher. Student/EMA start from the
same completed P12k EMA; optimizer and 4000-step warmup/cosine schedule restart.
"""
import argparse,copy,glob,hashlib,json,math,os,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import src.readout.common as RC
from src.config import STREAM,FEAT,COCO_TRAIN
from src.msd.pretrain import eval_exclusions
from src.ivc.ft import cosd,ema_update
from src.siglip_pm.model import load_hcf,load_vit,checksum,MODEL_ID,REVISION,n_params
from src.siglip_pm.distill import fwd,read_q,cell_w,pool,groups,Slots,atomic_save
from src.siglip_extra.data import ExtraDS
from src.siglip_extra.losses import contrastive,local_student,local_teacher,spatial


def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def initialize(a,dev):
 torch.manual_seed(0);model,ck=load_hcf(a.init,dev)
 assert ck['cond']=='P' and ck['step']==a.init_step and ck['kind']=='ema' and ck['backbone']==MODEL_ID
 if a.init_step>12000:assert ck.get('experiment')==a.kind,'Do not cross continuation branches'
 model.train().requires_grad_(True)
 teacher=load_vit(dev).eval().requires_grad_(False)
 return model,teacher

def terms(a,b,model,teacher,collect=False,local_target=None,report=True):
 o=fwd(model,b['full'],collect)
 A,Q=read_q(model,o,b['roi_img'],cell_w(b['box_c'],24,16),cell_w(b['box_c'],48,8))
 crop=cosd(Q,b['tpool']).mean();glob=.5*(cosd(o['cls'],b['gembed']).mean()+cosd(pool(o['coarse'],b['fvalid']),b['gmean']).mean())
 extra=Q.sum()*0;metrics={}
 if a.kind=='E1':
  extra,metrics=contrastive(Q,b['tpool'],b['category'],b['roi_image_id'],b['orig_boxes'],a.threshold,collect=report)
 elif a.kind=='E2':
  if local_target is None:local_target,_=local_teacher(teacher,b)
  qlocal,eligible=local_student(model,o,b);extra,metrics=spatial(qlocal,local_target,eligible,collect=report)
 return dict(crop=crop,global_=glob,extra=extra,Q=Q,coarse=A,metrics=metrics)

def move(b,dev):return RC.to_dev(b,dev)

def smoke(a,dev):
 model,teacher=initialize(a,dev);teacher_sum=checksum(teacher.parameters());ema=copy.deepcopy(model).eval().requires_grad_(False);opt=torch.optim.AdamW(groups(model),betas=(.9,.999))
 ds=ExtraDS(a.shared,spatial=a.kind=='E2');stream=np.load(STREAM);ex=set(eval_exclusions()['raev2_stream']);slots=[(c,int(i)) for c,i in enumerate(stream[a.stream_offset]) if int(i) not in ex]
 b=move(ds[(a.stream_offset,slots,len(slots))],dev);local_target=None;check={}
 if a.kind=='E2':
  local_target,whole=local_teacher(teacher,b)
  sim=F.cosine_similarity(whole,b['tpool'],dim=-1)
  assert float(sim.min())>.9995 and not local_target.requires_grad
  check=dict(online_teacher_vs_cache_cosine_min=float(sim.min()),local_target_shape=list(local_target.shape),padding_excluded=True)
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter();metrics=[];scales={};module_scales={}
 prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(wait=1,warmup=1,active=3,repeat=1),
      on_trace_ready=lambda p:p.export_chrome_trace(str(Path(a.out)/'smoke_trace.json')))
 prof.start()
 for k in range(5):
  t=terms(a,b,model,teacher,k==0,None if a.kind=='E2' else local_target,report=k in [0,4]);loss=t['crop']+.5*t['global_']+.1*t['extra']
  if k==0:
   params=list(model.parameters())
   for name,v in [('crop',t['crop']),('global',.5*t['global_']),('extra',.1*t['extra'])]:
    gr=torch.autograd.grad(v,params,retain_graph=True,allow_unused=True)
    scales[name]=float(torch.stack([g.detach().float().square().sum() for g in gr if g is not None]).sum().sqrt())
    named=list(model.named_parameters());module_scales[name]={}
    for prefix in ['vit.','fine_embed.','fine_blocks.','inter.','fine_norm.','Wd.','s_d']:
     values=[g.detach().float().square().sum() for (pn,_),g in zip(named,gr) if pn.startswith(prefix) and g is not None]
     module_scales[name][prefix]=float(torch.stack(values).sum().sqrt()) if values else 0.
  opt.zero_grad(set_to_none=True);loss.backward()
  if k==0:
   bad=[n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
   assert not bad,bad
  assert torch.isfinite(loss)
  torch.nn.utils.clip_grad_norm_(model.vit.parameters(),1);torch.nn.utils.clip_grad_norm_([p for _,p in model.new_named()],1)
  opt.step();ema_update(ema,model,.999);metrics.append(dict(loss=float(loss),crop_cosine=float(1-t['crop']),extra_loss=float(t['extra']),**t['metrics']))
  prof.step()
 prof.stop()
 Path(a.out,'profiler_summary.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
 torch.cuda.synchronize();elapsed=time.perf_counter()-t0
 assert teacher_sum==checksum(teacher.parameters()) and not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
 r=dict(status='passed',kind=a.kind,seed=0,init=a.init,init_sha256=file_sha(a.init),init_step=a.init_step,teacher_frozen=True,
        image_batch=len(slots),ROI_batch=len(b['tpool']),all_trainable_gradients_finite=True,gradient_norm_by_weighted_loss=scales,gradient_norm_per_module=module_scales,
        VRAM_peak_GiB=torch.cuda.max_memory_allocated()/2**30,steps_per_s=5/elapsed,metrics=metrics,teacher_correspondence=check)
 json.dump(r,open(Path(a.out)/'smoke.json','w'),indent=1);print(json.dumps(r),flush=True)

@torch.no_grad()
def make_dev_batches(a,teacher,dev):
 from src.siglip_pm.cocodata import CocoDS
 from src.siglip_pm.model import run_enc
 from src.siglip_pm.data import LB
 from src.siglip_extra.data import quadrants
 rois=pd.read_parquet(os.path.join(FEAT,'dev_rois.parquet'))
 ex=eval_exclusions();rois=rois[~rois.image_id.isin(ex['raev2_dev'])]
 ids=sorted(map(int,rois.image_id.unique()))[:48];ds=CocoDS(rois,COCO_TRAIN,384,16);batches=[]
 for lo in range(0,len(ids),16):
  ii=ids[lo:lo+16];b=move(ds[ii],dev)
  T,_=run_enc(teacher,b['crops']);ft,fg=run_enc(teacher,b['full'])
  b.update(tpool=pool(T,b['ccov']),gembed=fg,gmean=pool(ft,b['fvalid']),roi_image_id=b['image_ids'][b['roi_img']],orig_boxes=b['ltrb'].float(),
           category=torch.tensor([int(r.cat_id) for iid in ii for r in ds.by[iid].itertuples()],device=dev))
  if a.kind=='E2':
   weights=[]
   for bx in b['ltrb'].cpu().numpy():
    l,t,r,bot=bx;lb=LB(r-l,bot-t,384,16);rect=torch.tensor([[lb.ox,lb.oy,lb.ox+lb.nw,lb.oy+lb.nh]],device=dev,dtype=torch.float32)
    weights.append(cell_w(quadrants(rect)[0],24,16))
   w=torch.stack(weights);b['local_target']=torch.einsum('rtd,rkt->rkd',T,w)/w.sum(-1,keepdim=True).clamp_min(1e-12)
  del b['crops'];batches.append(b)
 return batches

@torch.no_grad()
def diagnostics(a,model,teacher,batches,dev,step):
 out=[];was=model.training;model.eval()
 for b in batches:
  t=terms(a,b,model,teacher,local_target=b.get('local_target'))
  out.append(dict(crop_cosine=float(1-t['crop']),coarse_cosine=float(1-cosd(t['coarse'],b['tpool']).mean()),global_loss=float(t['global_']),extra_loss=float(t['extra']),**t['metrics']))
 if was:model.train()
 keys=set().union(*(x.keys() for x in out));return dict(step=step,split='COCO dev first48 images (same fixed excluded-image rule as base)',**{k:float(np.mean([x[k] for x in out if x.get(k) is not None])) for k in keys if any(x.get(k) is not None for x in out)})

def train(a,dev):
 model,teacher=initialize(a,dev);teacher_sum=checksum(teacher.parameters());ema=copy.deepcopy(model).eval().requires_grad_(False)
 opt=torch.optim.AdamW(groups(model),betas=(.9,.999));start=0;out=Path(a.out);(out/'checkpoints').mkdir(exist_ok=True)
 latest=sorted((out/'checkpoints').glob('ckpt_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
 if latest:
  ck=torch.load(latest[-1],map_location=dev,weights_only=False);assert ck['init_sha256']==file_sha(a.init)
  model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['opt']);start=ck['additional_step']
 ds=ExtraDS(a.shared,spatial=a.kind=='E2');full_stream=np.load(STREAM);assert a.stream_offset+a.steps<=len(full_stream)
 stream=full_stream[:a.stream_offset+a.steps];exclude=eval_exclusions()['raev2_stream']
 dl=DataLoader(ds,sampler=Slots(stream,0,1,exclude,a.stream_offset+start),batch_size=None,num_workers=a.workers,pin_memory=True,persistent_workers=True,prefetch_factor=2)
 cfg=dict(kind=a.kind,label=a.label or f'P12k + {a.kind}_{a.steps//1000}k',seed=0,init=a.init,init_sha256=file_sha(a.init),init_step=a.init_step,additional_steps=a.steps,
          backbone=MODEL_ID,revision=REVISION,teacher_checksum=teacher_sum,crop_weight=1.,global_weight=.5,extra_weight=0 if a.kind=='Continue' else .1,extra_warmup=400,
          temperature=.07,negative_threshold=a.threshold if a.kind=='E1' else None,spatial_grid='2x2' if a.kind=='E2' else None,
          spatial_min_full_canvas_extent=32 if a.kind=='E2' else None,optimizer='same AdamW groups as base P; fresh optimizer; encoder top5e-5 LLRD.75; new3e-4; warmup600/cosine4000',
          images_per_step=16,world=1,ROI_stream=f'original stream rows[{a.stream_offset}:{a.stream_offset+a.steps}], same object/nested manifest, no flip',EMA='.999 to1; initialized from phase input EMA',n_parameters=n_params(model),args=vars(a))
 json.dump(cfg,open(out/'config.json','w'),indent=1);log=open(out/'train_log.jsonl','a',buffering=1);dlog=open(out/'diagnostics.jsonl','a',buffering=1)
 diag_batches=make_dev_batches(a,teacher,dev)
 t0=time.time();dlog.write(json.dumps(diagnostics(a,model,teacher,diag_batches,dev,start))+'\n');torch.cuda.reset_peak_memory_stats();last=time.time();laststep=start
 def save(step,final=False):
  common=dict(step=a.init_step+step,additional_step=step,cond='P',experiment=a.kind,s0=.1,backbone=MODEL_ID,revision=REVISION,init=a.init,init_sha256=cfg['init_sha256'])
  atomic_save(dict(**common,model=model.state_dict(),ema=ema.state_dict(),opt=opt.state_dict()),out/f'checkpoints/ckpt_{step}.pt')
  if final:atomic_save(dict(**common,kind='ema',model=ema.state_dict()),out/'checkpoints/ema_last.pt')
 for k,b in enumerate(dl):
  step=start+k;b=move(b,dev)
  for g in opt.param_groups:g['lr']=RC.C.lr_at(step,g['base'],600,a.steps)
  t=terms(a,b,model,teacher,collect=step==start,report=(step==start or (step+1)%50==0));weight=0. if a.kind=='Continue' else .1*min(1.,(step+1)/400)
  loss=t['crop']+.5*t['global_']+weight*t['extra'];assert torch.isfinite(loss),f'nonfinite loss at {step}'
  opt.zero_grad(set_to_none=True);loss.backward()
  if step<3 or (step+1)%500==0:
   bad=[n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()];assert not bad,bad
  ge=float(torch.nn.utils.clip_grad_norm_(model.vit.parameters(),1.));gn=float(torch.nn.utils.clip_grad_norm_([p for _,p in model.new_named()],1.));opt.step()
  momentum=1-(1-.999)*(math.cos(math.pi*step/a.steps)+1)/2;ema_update(ema,model,momentum)
  if (step+1)%50==0:
   now=time.time();r=dict(step=step+1,total_equivalent_steps=a.init_step+step+1,s=round(now-t0,2),crop_loss=float(t['crop']),crop_cosine=float(1-t['crop']),
                         global_loss=float(t['global_']),extra_loss=float(t['extra']),extra_weight=weight,total_loss=float(loss),gnorm_encoder=ge,gnorm_new=gn,
                         steps_per_s=(step+1-laststep)/(now-last),peak_memory_GiB=torch.cuda.max_memory_allocated()/2**30,**t['metrics'])
   log.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True);last=now;laststep=step+1
  if (step+1)%500==0:save(step+1)
  if (step+1)%1000==0:
   d=diagnostics(a,ema,teacher,diag_batches,dev,step+1);dlog.write(json.dumps(d)+'\n');print(json.dumps(dict(diagnostic=d)),flush=True)
 save(a.steps,True)
 assert checksum(teacher.parameters())==teacher_sum
 json.dump(dict(status='done',kind=a.kind,additional_steps=a.steps,init_step=a.init_step,seconds=time.time()-t0,teacher_unchanged=True,init_sha256=cfg['init_sha256']),open(out/'done.json','w'),indent=1)
 print('DONE',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['E1','E2','Continue'],required=True);ap.add_argument('--out',required=True);ap.add_argument('--shared',required=True);ap.add_argument('--init',required=True)
 ap.add_argument('--threshold',type=float,default=.95);ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--smoke',action='store_true')
 ap.add_argument('--init-step',type=int,default=12000);ap.add_argument('--stream-offset',type=int,default=0);ap.add_argument('--label',default='');a=ap.parse_args()
 assert 'RANK' not in os.environ;Path(a.out).mkdir(exist_ok=True,parents=True)
 try:(smoke if a.smoke else train)(a,torch.device('cuda'))
 except Exception:
  tb=traceback.format_exc();json.dump(dict(status='failed',error=tb),open(Path(a.out)/'failed.json','w'),indent=1);raise
if __name__=='__main__':main()
