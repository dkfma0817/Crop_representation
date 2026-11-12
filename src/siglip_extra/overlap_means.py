"""Fit E1/E2 mean readers during CPU-only R2 official validation; same protocol.
No current experiment is interrupted. GPU leases serialize any later inference;
output leases and complete checks prevent duplicate fitting when the main queue
reaches the same stage. All completed stage records are real completed jobs.
"""
import argparse,concurrent.futures,json,os,subprocess,time,threading
from pathlib import Path
from src.siglip_pm.supervise import PY,ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--base',required=True);ap.add_argument('--models',nargs=2,default=['E1','E2']);ap.add_argument('--gate',choices=['val','test'],default='val');a=ap.parse_args();run=Path(a.run);deadline=json.load(open(Path(a.base)/'budget.json'))['deadline_unix']
 gate=[run/'state'/f'infer_{c}_{a.gate}.done.json' for c in ['E1','E2']]
 event=threading.Event()
 while not all(p.exists() for p in gate):
  if (run/'queue_failed.json').exists():raise RuntimeError('Main follow-up queue failed before validation gate')
  if time.time()>deadline:raise TimeoutError('Budget expired before frozen mean-reader overlap')
  event.wait(10)
 def one(c,g):
  marker=run/'state'/f'mean_probe_{c}.done.json'
  if marker.exists():return
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
  args=[PY,'-u','-m','src.siglip_extra.mean_probe','--job',str(run/f'scripts/mean_{c}.yaml'),'--out',str(run/f'mean/probe_{c}'),'--workers','8']
  t=time.time();print(json.dumps(dict(stage=f'early_mean_{c}',gpu=g,status='start',time=t)),flush=True)
  with open(run/f'logs/mean_probe_{c}.log','a') as f:rc=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,timeout=max(1,deadline-time.time())).returncode
  r=dict(stage=f'mean_probe_{c}',gpu=g,seconds=time.time()-t,returncode=rc,command=args,time=time.time(),scheduling='overlap CPU-only official R2 validation; exclusive per-GPU lease')
  json.dump(r,open(marker if rc==0 else run/'state'/f'mean_probe_{c}.failed.json','w'),indent=1);print(json.dumps(r),flush=True)
  if rc:raise RuntimeError(c+' early mean failed')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:list(p.map(lambda z:one(*z),[(a.models[0],2),(a.models[1],3)]))
 json.dump(dict(status='done',protocol_unchanged=True),open(run/f'early_means_{a.models[0]}_{a.models[1]}_done.json','w'),indent=1)
if __name__=='__main__':main()
