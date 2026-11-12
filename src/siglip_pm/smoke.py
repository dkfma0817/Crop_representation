"""Real COCO forward/backward, native model parity, geometry and gradient gate."""
import argparse,json,time,os
import numpy as np,pandas as pd,torch
from PIL import Image
from transformers import SiglipVisionModel,SiglipImageProcessor
import src.readout.common as RC
from src.config import FEAT,COCO_TRAIN,STREAM
from src.siglip_pm.model import build_student,load_vit,MODEL_PATH,MODEL_ID,REVISION,checksum,run_enc,n_params
from src.siglip_pm.cocodata import CocoDS,roi_geom
from src.siglip_pm.data import LB,geometry
from src.siglip_pm.distill import fwd,read_q,cell_w,pool,groups
from src.ivc.ft import cosd
from src.siglip_pm.probe import roi_align

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cond',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 os.makedirs(a.out,exist_ok=True);torch.manual_seed(0);dev='cuda'
 ds=CocoDS(pd.read_parquet(f'{FEAT}/stream_rois.parquet'),COCO_TRAIN)
 ids=np.load(STREAM)[0,:8].tolist();b=RC.to_dev(ds[ids],dev)
 teacher=load_vit(dev).eval().requires_grad_(False);ts=checksum(teacher.parameters())
 native=SiglipVisionModel.from_pretrained(MODEL_PATH,attn_implementation='sdpa').to(dev).eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  ref=native(b['full'][:2]);t=teacher.forward_features(b['full'][:2]);g=teacher.forward_head(t)
  parity=dict(tokens_max_abs=float((ref.last_hidden_state-t).abs().max()),pooled_max_abs=float((ref.pooler_output-g).abs().max()))
 assert max(parity.values())==0,parity
 del native
 tp,_=run_enc(teacher,b['crops']);target=pool(tp,b['ccov']);ft,fg=run_enc(teacher,b['full'])
 # Check each original-pixel box maps to the exact same teacher crop and student ROI.
 for k,iid in enumerate(ids):
  img=Image.open(f'{COCO_TRAIN}/{iid:012d}.jpg').convert('RGB');W,H=img.size
  for r in torch.where(b['roi_img']==k)[0].tolist():
   bx=b['ltrb'][r].cpu().tolist();gg=roi_geom(W,H,bx,384,16)
   assert torch.equal(gg['crop_lb'].canvas(img.crop(tuple(bx))),b['crops'][r].cpu())
   assert np.allclose(gg['box_c'],b['box_c'][r].cpu().numpy())
   assert np.allclose(gg['ccov'],b['ccov'][r].cpu().numpy())
 assert (cell_w(b['box_c'],24,16)-b['wfull']).abs().max()<1e-3
 # ROIAlign linear-coordinate field must recover the full canvas sample coordinates.
 crop,full,grid=geometry(640,480,(80,40,480,360),384,16)
 gx,gy=torch.meshgrid(torch.arange(24,device=dev),torch.arange(24,device=dev),indexing='xy')
 coords=torch.stack([(gx+.5)*16,(gy+.5)*16],-1).reshape(1,576,2).float()
 rb=dict(img_index=torch.tensor([0],device=dev),bil_grid=torch.tensor(grid,device=dev)[None])
 sampled=roi_align(coords,1,rb);expected=(torch.tensor(grid,device=dev)+1)*192
 inside=(expected[:,0]>=8)&(expected[:,0]<=376)&(expected[:,1]>=8)&(expected[:,1]<=376)&torch.tensor(crop.coverage()>0,device=dev)
 err=float((sampled[0,inside]-expected[inside]).abs().max());assert err<1e-3
 # Official normalization parity on an already square image (letterbox has no padding).
 img=Image.open(f'{COCO_TRAIN}/{ids[0]:012d}.jpg').convert('RGB').resize((384,384))
 official=SiglipImageProcessor.from_pretrained(MODEL_PATH)(images=img,return_tensors='pt').pixel_values[0]
 assert torch.allclose(LB(384,384,384,16).canvas(img),official,atol=2e-7)
 m=build_student(a.cond,dev)
 with torch.no_grad():
  slow=fwd(m,b['full'],collect=True);fast=fwd(m,b['full'],collect=False)
  fast_error=max(float((slow['coarse']-fast['coarse']).abs().max()),max(float((slow['mem'][k]-fast['mem'][k]).abs().max()) for k in slow['mem']))
  assert fast_error==0,fast_error
 opt=torch.optim.AdamW(groups(m));m.train()
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();grads={};metrics=[]
 for step in range(8):
  o=fwd(m,b['full'],collect=(step==0))
  A,Q=read_q(m,o,b['roi_img'],cell_w(b['box_c'],24,16),cell_w(b['box_c'],48,8))
  lc=cosd(Q,target).mean();lg=.5*(cosd(o['cls'],fg).mean()+cosd(pool(o['coarse'],b['fvalid']),pool(ft,b['fvalid'])).mean());loss=lc+.5*lg
  assert torch.isfinite(loss)
  opt.zero_grad(set_to_none=True);loss.backward()
  if step==0:
   grads={n:None if p.grad is None else float(p.grad.float().norm()) for n,p in m.named_parameters()}
   bad=[n for n,p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.norm()==0]
   assert not bad,bad
  torch.nn.utils.clip_grad_norm_(m.parameters(),1)
  if step>0:opt.step()
  metrics.append(dict(loss=float(loss),crop_cosine=float(1-lc),global_loss=float(lg)))
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 assert ts==checksum(teacher.parameters()) and all(p.grad is None and not p.requires_grad for p in teacher.parameters())
 res=dict(status='passed',cond=a.cond,backbone=MODEL_ID,revision=REVISION,seed=0,teacher_frozen=True,native_parity=parity,fast_interaction_max_abs=fast_error,
          ROI_teacher_pixel_identity=True,roialign_coordinate_error_px=err,finite=True,gradients_all_trainable_parameters=grads,
          shapes=dict(coarse=list(o['coarse'].shape),memory={k:list(v.shape) for k,v in o['mem'].items()},global_embedding=list(o['cls'].shape)),
          n_params=n_params(m),n_params_vit=n_params(m.vit),n_params_new=n_params(m)-n_params(m.vit),
          batch_images=8,effective_training_batch_images=16,smoke_steps_per_s=8/elapsed,peak_memory_gb=torch.cuda.max_memory_allocated()/2**30,metrics=metrics)
 json.dump(res,open(f'{a.out}/smoke.json','w'),indent=1);print(json.dumps({k:v for k,v in res.items() if k!='gradients_all_trainable_parameters'}),flush=True)
if __name__=='__main__':main()
