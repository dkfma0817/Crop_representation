"""Short completed-model performance diagnostic; no checkpoints/model results saved.
Replays original forward/backward/AdamW/EMA/scalar logging on a prepared16-image
COCO batch, with profiled warmup. It does not modify running jobs or saved weights.
"""
import argparse,copy,gc,json,time
from pathlib import Path
import numpy as np,torch
import src.readout.common as RC
from src.config import STREAM
from src.msd.pretrain import eval_exclusions
from src.ivc.ft import ema_update
from src.siglip_pm.model import load_hcf
from src.siglip_pm.distill import TrainDS,terms,groups

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run)
 stream=np.load(STREAM);ex=set(eval_exclusions()['raev2_stream']);slots=[(c,int(i)) for c,i in enumerate(stream[0]) if int(i) not in ex]
 b=RC.to_dev(TrainDS(str(run/'shared'))[(0,slots,len(slots))],'cuda');results=[]
 for c in ['M','P']:
  path=run/c/'checkpoints/ema_last.pt';model,ck=load_hcf(str(path),'cuda');model.train().requires_grad_(True);ema=copy.deepcopy(model).eval().requires_grad_(False)
  opt=torch.optim.AdamW(groups(model),betas=(.9,.999));new=[p for _,p in model.new_named()];out=run/'profiles'/c;out.mkdir(exist_ok=True,parents=True)
  prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(wait=3,warmup=2,active=10,repeat=1),on_trace_ready=lambda p:p.export_chrome_trace(str(out/'trace.json')))
  prof.start();dur=[]
  for k in range(15):
   torch.cuda.synchronize();start=time.perf_counter()
   t=terms(b,model,collect=False);lr=t['roi'].mean();lg=t['glob'].mean();loss=lr+.5*lg
   opt.zero_grad(set_to_none=True);loss.backward()
   ge=float(torch.nn.utils.clip_grad_norm_(model.vit.parameters(),1.));gn=float(torch.nn.utils.clip_grad_norm_(new,1.));opt.step();ema_update(ema,model,.999)
   ns=t['nested'];lo=float(t['roi'][~ns].mean());ln=float(t['roi'][ns].mean()) if bool(ns.any()) else lo
   # The original loop extracts these scalars every step, including its ROI metrics.
   record=np.array([float(lr),lo,ln,float(t['coarse'].mean()),float(lg),ge,gn,1.])
   assert np.isfinite(record).all()
   torch.cuda.synchronize();dur.append(time.perf_counter()-start);prof.step()
  prof.stop();(out/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
  r=dict(model=c,batch_images=len(slots),batch_ROIs=len(b['tpool']),prepared_batch_median_step_ms=float(np.median(dur[5:])*1000),
         checkpoint=str(path),checkpoint_step=ck['step'],saved_weights_changed=False,
         limitation='Short scratch-copy profiling includes profiler overhead and explicit boundary synchronization; excludes DataLoader waiting, uses completedEMA weights and freshAdamW state after warmup. No new checkpoint or AP experiment.')
  json.dump(r,open(out/'benchmark.json','w'),indent=1);results.append(r);print(json.dumps(r),flush=True)
  del opt,ema,model,t,loss;gc.collect();torch.cuda.empty_cache()
 json.dump(results,open(run/'profiles/summary.json','w'),indent=1)
if __name__=='__main__':main()
