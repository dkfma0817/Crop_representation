"""After the priority follow-ups finish, complete original P/M/B0/C0 mean comparison.
Reuse exact completed P/C0 linear readers and predictions from the follow-up run;
train only the still-missing M/B0 linear probes. No representation learning here.
"""
import argparse,concurrent.futures,json,os,subprocess,time,threading
from pathlib import Path
import pandas as pd
from src.siglip_pm.supervise import PY,EPY,ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--extra',required=True);ap.add_argument('--wait-pid',type=int,default=0);a=ap.parse_args();base=Path(a.base);extra=Path(a.extra)
 if (base/'diagnostics_done.json').exists():print('Baseline diagnostics already complete.',flush=True);return
 deadline=json.load(open(base/'budget.json'))['deadline_unix'];logs=base/'logs';state=base/'state';mr=base/'mean';mr.mkdir(exist_ok=True)
 if a.wait_pid and not (extra/'done.json').exists():subprocess.run(['tail',f'--pid={a.wait_pid}','-f','/dev/null'],check=True,timeout=max(1,deadline-time.time()))
 assert json.load(open(extra/'done.json'))['status']=='done'
 if deadline-time.time()<3600:raise RuntimeError('Retain complete priority results; less than1h remains for optional baseline mean completion')
 def cmd(name,module,args,gpu=None,ep=False):
  marker=state/f'{name}.done.json'
  if marker.exists():return
  env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
  if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
  if ep:env['PYTHONPATH']=f'/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}'
  started=time.time();print(json.dumps(dict(stage=name,status='start',gpu=gpu,time=started)),flush=True)
  with open(logs/f'{name}.log','a') as f:rc=subprocess.run([EPY if ep else PY,'-u','-m',module]+list(map(str,args)),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time())).returncode
  r=dict(returncode=rc,seconds=time.time()-started);json.dump(r,open(marker if rc==0 else state/f'{name}.failed.json','w'),indent=1);print(json.dumps(dict(stage=name,**r)),flush=True)
  if rc:raise RuntimeError(name+' failed')
 def parallel(jobs,n=2):
  with concurrent.futures.ThreadPoolExecutor(max_workers=n) as p:return list(p.map(lambda f:f(),jobs))
 # Preserve the independently trained P/C0 mean probes; do not refit/reselect them.
 for c in ['P','C0']:
  for rel in [f'probe_{c}',f'eval_val/{c}_linear',f'eval_test/{c}_linear',f'preds/{c}_linear']:
   target=mr/rel;target.parent.mkdir(exist_ok=True,parents=True)
   if not target.exists():target.symlink_to((extra/'mean'/rel).resolve(),target_is_directory=True)
 def probe(c,g):return cmd(f'baseline_mean_probe_{c}','src.siglip_pm.mean_probe',['--job',base/f'scripts/mean_{c}.yaml','--out',mr/f'probe_{c}','--workers','8'],g)
 def infer(c,sp,g):
  args=['--run',mr,'--job',base/f'scripts/mean_{c}.yaml','--split',sp,'--workers','8']
  if sp=='test':args+=['--select',mr/'selection.json']
  return cmd(f'baseline_mean_infer_{c}_{sp}','src.siglip_pm.mean_infer',args,g)
 def official(c,sp,e):
  args=['--split',sp,'--pred',mr/f'preds/{c}_linear/ep{e}/probs_{sp}.npz','--out',mr/f'eval_{sp}/{c}_linear'/(f'ep{e}' if sp=='val' else 'test')]
  if sp=='test':args+=['--boot']
  return cmd(f'baseline_mean_official_{c}_{sp}_{e}','src.pacoattr.official_eval',args,ep=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=1) as cpu:
  sf=cpu.submit(parallel,[lambda c=c:cmd(f'roi_size_{c}','src.siglip_pm.size_diagnostic',['--run',base,'--model',c],ep=True) for c in ['P','M','B0','C0']],4)
  parallel([lambda:probe('M',2),lambda:probe('B0',3)])
  parallel([lambda:infer('M','val',2),lambda:infer('B0','val',3)])
  parallel([lambda c=c,e=e:official(c,'val',e) for c in ['M','B0'] for e in range(1,11)],8)
  cmd('baseline_mean_selection','src.siglip_pm.select_epoch',[mr])
  parallel([lambda:infer('M','test',2),lambda:infer('B0','test',3)])
  sel=json.load(open(mr/'selection.json'));parallel([lambda c=c:official(c,'test',sel[f'{c}_linear']['epoch']) for c in ['M','B0']],2)
  cmd('baseline_mean_bootstrap','src.siglip_extra.baseline_bootstrap',[base],ep=True)
  sf.result()
 D=pd.concat([pd.read_csv(base/f'results/roi_size_ap_{c}.csv') for c in ['P','M','B0','C0']],ignore_index=True);D.to_csv(base/'results/roi_size_ap.csv',index=False)
 rows=[]
 for c in ['B0','P','M']:
  for lv in ['Object','Part']:
   for name in D.size_bin.unique():
    x=D[(D.model=='C0')&(D.level==lv)&(D.size_bin==name)].iloc[0];y=D[(D.model==c)&(D.level==lv)&(D.size_bin==name)].iloc[0]
    rows.append(dict(comparison=f'C0 − {c}',level=lv,size_bin=name,Pattern=float(x.Pattern-y.Pattern),Material=float(x.Material-y.Material)))
 pd.DataFrame(rows).to_csv(base/'results/roi_size_crop_gaps.csv',index=False)
 cmd('baseline_diagnostic_report','src.siglip_pm.diagnostic_report',['--run',base])
 json.dump(dict(status='done',P_C0_mean_reused_from=str(extra/'mean'),M_B0_independently_trained=True,time=time.time()),open(base/'diagnostics_done.json','w'),indent=1)
if __name__=='__main__':main()
