"""Measured bf16 single-image latency and explicit dense/attention multiply-add count.
2 FLOPs/MAC; excludes LN/activations/softmax/bias and separately reports ROI fusion.
"""
import argparse,json,time,statistics
from pathlib import Path
import pandas as pd,torch
from src.siglip_pm.model import load_hcf,load_vit,n_params,SCALARS,MODEL_ID
from src.siglip_pm.data import LB
from PIL import Image

def costs(kind):
 N,D,F,NF=576,768,384,2304
 patch=N*D*3*16**2
 blocks=12*(12*N*D**2+2*N*N*D)
 # Native MAP head: single-query Q/out, all-token K/V, attention products, MLP4x.
 head=(2+2*N)*D**2+2*N*D+8*D**2
 total=patch+blocks+head
 if kind=='P':
  fine_patch=NF*F*3*8**2
  fine_blocks=4*(12*NF*F**2+2*NF*64*F)
  interactions=4*(N*F*D+NF*D*F)
  total+=fine_patch+fine_blocks+interactions
 fuse=(576*F*D if kind=='P' else 2*576*D*D if kind=='M' else 0)
 return dict(encoder_GFLOPs=2*total/1e9,ROI_fusion_GFLOPs_per_ROI=2*fuse/1e9,flop_convention='2 FLOPs/MAC; dense+conv+attention QK/AV; excludes normalization/softmax/activations/bias')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run);torch.manual_seed(0)
 manifest=pd.read_parquet('output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet');row=manifest.iloc[0]
 img=Image.open(row.path).convert('RGB');x=LB(*img.size,384,16).canvas(img)[None].to('cuda');rows=[]
 (run/'feature_examples').mkdir(exist_ok=True)
 for c in ['P','M','B0']:
  model=load_hcf(str(run/c/'checkpoints/ema_last.pt'),'cuda')[0] if c in ['P','M'] else load_vit('cuda').eval().requires_grad_(False)
  fn=model.forward_all if c in ['P','M'] else model.forward
  times=[]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for i in range(60):
    torch.cuda.synchronize();t=time.perf_counter();fn(x);torch.cuda.synchronize()
    if i>=10:times.append((time.perf_counter()-t)*1000)
  if c in ['P','M']:
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):o=model.forward_all(x)
   feature=dict(coarse=o['coarse'][0].float().cpu().reshape(24,24,768))
   for k,v in o['mem'].items():
    g=48 if o['grids'][k]==2 else 24;feature[k]=v[0].float().cpu().reshape(g,g,-1)
   assert sum(v.numel() for v in feature.values())==1327104
   torch.save(dict(image_id=int(row.image_id),image_path=row.path,backbone=MODEL_ID,features=feature),run/f'feature_examples/{c}.pt')
  scalars=SCALARS[c] if c in ['P','M'] else 576*768
  rows.append(dict(model=c,parameters=n_params(model),stored_scalars=scalars,feature_MiB_fp32=scalars*4/2**20,feature_MiB_fp16=scalars*2/2**20,
                   median_encode_ms=statistics.median(times),**costs(c)))
  del model;torch.cuda.empty_cache()
 pd.DataFrame(rows).to_csv(run/'results/efficiency.csv',index=False);json.dump(rows,open(run/'efficiency.json','w'),indent=1);print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
