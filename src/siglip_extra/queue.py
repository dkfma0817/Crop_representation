"""Wait for the running base supervisor to end normally, verify checkpoint/GPU gates,
then run independent E1/E2, matched R2/mean probes and (budget permitting) Continue.
No existing process is killed or modified. All work shares the original8h deadline.
"""
import argparse,concurrent.futures,fcntl,json,os,subprocess,threading,time
from pathlib import Path
from src.siglip_pm.supervise import PY,EPY,ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--run',required=True);ap.add_argument('--wait-pid',type=int,default=0);a=ap.parse_args();run=Path(a.run);base=Path(a.base)
 for d in ['logs','state','scripts','results']: (run/d).mkdir(exist_ok=True,parents=True)
 if (run/'done.json').exists():print('Follow-up experiments already complete.',flush=True);return
 lock=open(run/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 deadline=json.load(open(base/'budget.json'))['deadline_unix'];stop=threading.Event();active={}
 def command(name,module,args,gpu=None,official=False):
  done=run/'state'/f'{name}.done.json'
  if done.exists():return
  env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
  for k in ['RANK','LOCAL_RANK','WORLD_SIZE']:env.pop(k,None)
  if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
  if official:env['PYTHONPATH']=f'/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}'
  cmd=[EPY if official else PY,'-u','-m',module]+list(map(str,args));start=time.time();active[name]=dict(gpu=gpu,start=start)
  print(json.dumps(dict(event='start',stage=name,gpu=gpu,command=cmd,time=start)),flush=True)
  with open(run/'logs'/f'{name}.log','a') as f:
   try:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT,timeout=max(1,deadline-time.time())).returncode
   except subprocess.TimeoutExpired:rc=124
  rec=dict(stage=name,gpu=gpu,seconds=time.time()-start,returncode=rc,command=cmd,time=time.time());active.pop(name,None)
  json.dump(rec,open(done if rc==0 else run/'state'/f'{name}.failed.json','w'),indent=1);print(json.dumps(rec),flush=True)
  if rc:raise RuntimeError(f'{name} failed ({rc}): {run}/logs/{name}.log')
 def parallel(funcs,n=2):
  with concurrent.futures.ThreadPoolExecutor(max_workers=n) as p:return list(p.map(lambda f:f(),funcs))
 def meter():
  count=0
  with open(run/'logs/gpu_usage.csv','a',buffering=1) as f:
   if f.tell()==0:f.write('timestamp,index,uuid,utilization_percent,memory_used_MiB,power_W\n')
   while not stop.is_set():
    q=subprocess.run(['nvidia-smi','-i','2,3','--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
    for line in q.stdout.splitlines():f.write(f'{time.time():.3f},'+line+'\n')
    count+=1
    if count%4==0:
     latest={}
     for c in ['E1','E2','Continue']:
      lp=run/c/'train_log.jsonl'
      if lp.exists():
       with open(lp,'rb') as h:h.seek(max(0,lp.stat().st_size-8192));lines=h.read().splitlines()
       if lines:
        try:latest[c]=json.loads(lines[-1])
        except ValueError:pass
     print(json.dumps(dict(event='heartbeat',active=list(active),representation=latest,time=time.time())),flush=True)
    stop.wait(15)
 def idle_gate():
  g=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
  uuids={v.split(',')[1].strip() for v in g.splitlines()};assert all('H200' in v for v in g.splitlines())
  while True:
   lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
   busy=[v for v in lines if v.split(',')[0].strip() in uuids]
   if not busy:break
   print(json.dumps(dict(event='waiting_for_GPUs',processes=busy,time=time.time())),flush=True)
   if time.time()>deadline:raise TimeoutError('GPU budget deadline while waiting')
   stop.wait(15)
  json.dump(dict(passed=True,gpus=g,checked_at=time.time(),active_compute_processes=[]),open(run/'GPU_start_gate.json','w'),indent=1)
 def job(c,mean=False):return run/f'scripts/{"mean" if mean else "probe"}_{c}.yaml'
 def probe(c,g,mean=False):
  prefix='mean_' if mean else '';root=run/'mean' if mean else run
  return command(f'{prefix}probe_{c}','src.siglip_extra.mean_probe' if mean else 'src.siglip_extra.probe',['--job',job(c,mean),'--out',root/f'probe_{c}','--workers','8'],g)
 def infer(c,sp,g,mean=False):
  root=run/'mean' if mean else run;args=['--run',root,'--job',job(c,mean),'--split',sp,'--workers','8']
  if sp=='test':args+=['--select',root/'selection.json']
  return command(f'{"mean_" if mean else ""}infer_{c}_{sp}','src.siglip_extra.mean_infer' if mean else 'src.siglip_extra.infer',args,g)
 def official(c,sp,e,mean=False):
  root=run/'mean' if mean else run;reader='linear' if mean else 'r2';name=f'{c}_{reader}'
  args=['--split',sp,'--pred',root/f'preds/{name}/ep{e}/probs_{sp}.npz','--out',root/f'eval_{sp}/{name}'/(f'ep{e}' if sp=='val' else 'test')]
  if sp=='test':args+=['--boot']
  return command(f'{"mean_" if mean else ""}official_{c}_{sp}_{e}','src.pacoattr.official_eval',args,official=True)
 def evaluate(models,mean=False):
  for lo in range(0,len(models),2):parallel([lambda c=c,g=2+j:probe(c,g,mean) for j,c in enumerate(models[lo:lo+2])])
  for lo in range(0,len(models),2):parallel([lambda c=c,g=2+j:infer(c,'val',g,mean) for j,c in enumerate(models[lo:lo+2])])
  parallel([lambda c=c,e=e:official(c,'val',e,mean) for c in models for e in range(1,11)],8)
  root=run/'mean' if mean else run
  command(f'{"mean_" if mean else ""}select_{"_".join(models)}','src.siglip_pm.select_epoch',[root])
  selection=json.load(open(root/'selection.json'));key='linear' if mean else 'r2'
  for lo in range(0,len(models),2):parallel([lambda c=c,g=2+j:infer(c,'test',g,mean) for j,c in enumerate(models[lo:lo+2])])
  parallel([lambda c=c:official(c,'test',selection[f'{c}_{key}']['epoch'],mean) for c in models],4)
 def training(c,g,steps,threshold):
  args=['--kind',c,'--out',run/c,'--shared',base/'shared','--init',base/'P/checkpoints/ema_last.pt','--steps',steps,'--threshold',threshold,'--workers','8']
  command(f'smoke_{c}','src.siglip_extra.train',args+['--smoke'],g)
  command(f'train_{c}','src.siglip_extra.train',args,g)
 try:
  if a.wait_pid and not (base/'all_done.json').exists():
   print(json.dumps(dict(event='waiting_for_existing_pipeline',pid=a.wait_pid,base=str(base))),flush=True)
   subprocess.run(['tail',f'--pid={a.wait_pid}','-f','/dev/null'],check=True,timeout=max(1,deadline-time.time()))
  assert json.load(open(base/'all_done.json'))['status']=='done','Base work did not end normally'
  for c in ['P','M']:
   assert json.load(open(base/c/'done.json'))['steps']==12000
   assert json.load(open(base/f'probe_{c}/done.json'))['status']=='done'
  assert json.load(open(base/'common_checkpoint.json'))['common_step']==12000
  idle_gate();threading.Thread(target=meter,daemon=True).start()
  if deadline-time.time()<2.5*3600:raise RuntimeError('Insufficient remaining budget for both4k plus required evaluation; keep base results and resume explicitly')
  command('negative_calibration','src.siglip_extra.calibrate',['--run',run,'--base',base])
  threshold=json.load(open(run/'negative_calibration.json'))['threshold'];steps=4000
  parallel([lambda:training('E1',2,steps,threshold),lambda:training('E2',3,steps,threshold)])
  d1=json.load(open(run/'E1/done.json'));d2=json.load(open(run/'E2/done.json'))
  assert d1['init_sha256']==d2['init_sha256'] and d1['additional_steps']==d2['additional_steps']==steps
  json.dump(dict(source_checkpoint=str(base/'P/checkpoints/ema_last.pt'),source_sha256=d1['init_sha256'],source_step=12000,additional_common_steps=steps),open(run/'common_checkpoint.json','w'),indent=1)
  evaluate(['E1','E2'])
  # Reuse ONLY completed P/C0 seed0 R2 results and exact official matching packs.
  for c in ['P','C0']:
   dest=run/f'eval_test/{c}_r2';dest.parent.mkdir(exist_ok=True)
   if not dest.exists():dest.symlink_to((base/f'eval_test/{c}_r2').resolve(),target_is_directory=True)
  evaluate(['E1','E2','P','C0'],mean=True)
  command('bootstrap_primary','src.siglip_extra.bootstrap',['--run',run],official=True)
  command('mean_bootstrap_primary','src.siglip_extra.bootstrap',['--run',run,'--mean'],official=True)
  command('report_primary','src.siglip_extra.report',['--run',run,'--base',base])
  if deadline-time.time()>2*3600:
   idle_gate();training('Continue',2,steps,threshold)
   # Train the control's separate R2 and separate linear probe concurrently.
   parallel([lambda:probe('Continue',2),lambda:probe('Continue',3,True)])
   parallel([lambda:infer('Continue','val',2),lambda:infer('Continue','val',3,True)])
   parallel([lambda e=e,m=m:official('Continue','val',e,m) for e in range(1,11) for m in [False,True]],8)
   command('select_control','src.siglip_pm.select_epoch',[run]);command('mean_select_control','src.siglip_pm.select_epoch',[run/'mean'])
   parallel([lambda:infer('Continue','test',2),lambda:infer('Continue','test',3,True)])
   ss=json.load(open(run/'selection.json'))['Continue_r2']['epoch'];sm=json.load(open(run/'mean/selection.json'))['Continue_linear']['epoch']
   parallel([lambda:official('Continue','test',ss),lambda:official('Continue','test',sm,True)])
   command('bootstrap_control','src.siglip_extra.bootstrap',['--run',run,'--control'],official=True)
   command('mean_bootstrap_control','src.siglip_extra.bootstrap',['--run',run,'--control','--mean'],official=True)
   command('report_control','src.siglip_extra.report',['--run',run,'--base',base])
  json.dump(dict(status='done',time=time.time(),control_completed=(run/'Continue/done.json').exists()),open(run/'done.json','w'),indent=1)
 except Exception:
  import traceback
  json.dump(dict(status='failed',traceback=traceback.format_exc(),time=time.time()),open(run/'queue_failed.json','w'),indent=1);raise
 finally:stop.set()
if __name__=='__main__':main()
