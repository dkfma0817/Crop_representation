"""Optional existing R0 linear diagnostic, only run after the four main R2 evaluations.
Same image batches, flips, known-label BCE, z-score(4000 train images), heads and
10-epoch schedule for P/M/B0/C0. It does not modify representation checkpoints.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np,torch,yaml
from torch.utils.data import DataLoader
import src.siglip_pm.probe as BASE
from src.ivc.common import R0
from src.siglip_pm.bbdata import BBImageDS
import src.readout.common as RC

class MeanReader(R0):
 def __init__(self,key):
  super().__init__();self.key=key;self.fuse.layers=(key,)
 def forward(self,feats,b,ctx_mode='normal'):
  m=self.pooled(feats[self.key][b['img_index']],b['cov']);lv=b['lv']
  return dict(logits=self.heads((m-self.mu[lv])/self.sd[lv],lv),extra={})

@torch.no_grad()
def statistics(job,a,dev):
 out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
 if (out/'r0_stats.pt').exists():return
 df,Y=RC.load_data(a.exclude or None,a.limit,splits=('train',))['train']
 ids=np.array(sorted(df.image_id.unique()));nd=min(1000,max(1,len(ids)//10)) if a.limit else 1000
 hold=set(np.random.default_rng(0).choice(ids,nd,replace=False).tolist());keep=~df.image_id.isin(hold).values;df,Y=df[keep].reset_index(drop=True),Y[keep]
 ds=BBImageDS(df,Y,False,'c0' if job['cond']=='C0' else 'bil',384,16)
 idx=np.sort(np.random.default_rng(0).choice(len(ds),min(4000,len(ds)),replace=False))
 chunks=[idx[i:i+12].tolist() for i in range(0,len(idx),12)]
 dl=DataLoader(ds,sampler=chunks,batch_size=1,num_workers=a.workers,pin_memory=True,collate_fn=RC.collate_one)
 model=BASE.load_frozen(job,dev);key=job['readers'][0]['layers'][0]
 acc=torch.zeros(2,2,768,dtype=torch.float64,device=dev);cnt=torch.zeros(2,dtype=torch.float64,device=dev)
 for b in dl:
  b=RC.to_dev(b,dev);f=BASE.encode(model,b,[key]);z=R0.pooled(f[key][b['img_index']],b['cov']).double()
  for k in [0,1]:
   zz=z[b['lv']==k];acc[k,0]+=zz.sum(0);acc[k,1]+=zz.square().sum(0);cnt[k]+=len(zz)
 mu=acc[:,0]/cnt[:,None];var=(acc[:,1]-cnt[:,None]*mu.square())/(cnt[:,None]-1)
 torch.save(dict(mu=mu.float().cpu(),sd=(var.clamp_min(0).sqrt()+1e-6).float().cpu(),n=cnt.cpu(),n_images=len(idx)),out/'r0_stats.pt')
 del model;torch.cuda.empty_cache()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--out',required=True)
 ap.add_argument('--exclude',default='output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json');ap.add_argument('--limit',type=int,default=0)
 ap.add_argument('--epochs',type=int,default=0);ap.add_argument('--max_steps',type=int,default=0);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
 job=yaml.safe_load(open(a.job));dev=torch.device('cuda');torch.manual_seed(0)
 statistics(job,a,dev)
 def build(job,dev):
  out={};stats=torch.load(f'{a.out}/r0_stats.pt',map_location=dev,weights_only=False)
  for s in job['readers']:
   torch.manual_seed(0);r=MeanReader(s['layers'][0]).to(dev);r.set_stats(stats['mu'],stats['sd']);out[s['name']]=r
  return out
 BASE.build_readers=build
 BASE.run(job,a,0,1,dev)
if __name__=='__main__':main()
