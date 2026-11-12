"""Local process supervisor: separate single-GPU jobs, atomic completion records, resource meter.
No DDP. CLI can resume individual stages without overwriting completed stages.
"""
import argparse,concurrent.futures,csv,json,os,subprocess,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PY='/home/irteam/envs/dinov2_ord_py310/bin/python'
EPY='/home/irteam/envs/paco_eval/bin/python'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--wait-cache-pids',default='');ap.add_argument('--stage',choices=['train','eval','all'],default='all');a=ap.parse_args()
 run=Path(a.run);run.mkdir(exist_ok=True,parents=True);(run/'state').mkdir(exist_ok=True);(run/'logs').mkdir(exist_ok=True)
 if (run/f'{a.stage}_done.json').exists():print(f'{a.stage} stages already complete.',flush=True);return
 started=time.time();stop=threading.Event()
 budget=run/'budget.json'
 if not budget.exists():json.dump(dict(start_utc='2026-11-08T16:58:58Z',deadline_unix=__import__('datetime').datetime(2026,11,9,0,58,58,tzinfo=__import__('datetime').timezone.utc).timestamp(),max_wall_hours=8),open(budget,'w'),indent=1)
 deadline=json.load(open(budget))['deadline_unix']
 def meter():
  with open(run/'logs/gpu_usage.csv','a',buffering=1) as f:
   if f.tell()==0:f.write('timestamp,index,uuid,utilization_percent,memory_used_MiB,power_W\n')
   count=0
   while not stop.is_set():
    q=subprocess.run(['nvidia-smi','-i','2,3','--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
    for ln in q.stdout.splitlines():f.write(f'{time.time():.3f},'+ln+'\n')
    count+=1
    if count%4==0:
     latest={}
     for c in ['P','M']:
      lp=run/c/'logs/train_log.jsonl'
      if lp.exists():
       with open(lp,'rb') as h:
        h.seek(max(0,lp.stat().st_size-8192));lines=h.read().splitlines()
       if lines:
        try: latest[c]=json.loads(lines[-1])
        except ValueError: pass
     print(json.dumps(dict(event='heartbeat',time=time.time(),gpu=q.stdout.strip().splitlines(),representation=latest)),flush=True)
    stop.wait(15)
 threading.Thread(target=meter,daemon=True).start()
 def command(name,args,gpu=None,evalenv=False):
  done=run/'state'/f'{name}.done.json'
  if done.exists():return json.load(open(done))
  env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
  env.pop('RANK',None);env.pop('WORLD_SIZE',None);env.pop('LOCAL_RANK',None)
  if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
  if evalenv:env['PYTHONPATH']=f'/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}'
  t=time.time();print(json.dumps(dict(event='start',stage=name,gpu=gpu,command=args,time=t)),flush=True)
  with open(run/'logs'/f'{name}.log','a') as f:
   rc=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,timeout=max(1,deadline-time.time())).returncode
  r=dict(stage=name,gpu=gpu,seconds=time.time()-t,returncode=rc,command=args,time=time.time())
  json.dump(r,open(done if rc==0 else run/'state'/f'{name}.failed.json','w'),indent=1)
  print(json.dumps(r),flush=True)
  if rc:raise RuntimeError(f'{name} failed ({rc}): {run}/logs/{name}.log')
  return r
 def cmd(name,module,args,gpu=None,evalenv=False):return command(name,[EPY if evalenv else PY,'-u','-m',module]+args,gpu,evalenv)
 def parallel(jobs,workers=2):
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
   return list(pool.map(lambda f:f(),jobs))
 def train_one(c,gpu):
  out=str(run/c);shared=str(run/'shared')
  assert json.load(open(f'{out}/smoke.json'))['status']=='passed'
  cmd(f'calib_{c}','src.siglip_pm.distill',['--cond',c,'--out',out,'--shared',shared,'--calib'],gpu)
  return cmd(f'train_{c}','src.siglip_pm.distill',['--cond',c,'--out',out,'--shared',shared,'--steps','12000','--diag_every','3000','--log_every','100','--workers','8'],gpu)
 def job(c):return str(run/'scripts'/f'probe_{c}.yaml')
 def probe(c,gpu):return cmd(f'probe_{c}','src.siglip_pm.probe',['--job',job(c),'--out',str(run/f'probe_{c}'),'--workers','8'],gpu)
 def infer(c,sp,gpu):
  args=['--run',str(run),'--job',job(c),'--split',sp,'--workers','8']
  if sp=='test':args+=['--select',str(run/'selection.json')]
  return cmd(f'infer_{c}_{sp}','src.siglip_pm.infer',args,gpu)
 def off(c,sp,ep):
  name=f'{c}_r2';pred=run/'preds'/name/f'ep{ep}'/f'probs_{sp}.npz'
  out=run/f'eval_{sp}'/name/(f'ep{ep}' if sp=='val' else 'test')
  args=['--split',sp,'--pred',str(pred),'--out',str(out)]
  if sp=='test':args+=['--boot']
  return cmd(f'official_{c}_{sp}_{ep}','src.pacoattr.official_eval',args,evalenv=True)
 try:
  if a.wait_cache_pids:
   pids=a.wait_cache_pids.split(',')
   subprocess.run(['tail']+[f'--pid={p}' for p in pids]+['-f','/dev/null'],check=True,timeout=max(1,deadline-time.time()))
  if a.stage in ['train','all']:
   cmd('verify_teacher','src.siglip_pm.rois',['verify','--out',str(run/'shared')])
   parallel([lambda:train_one('P',2),lambda:train_one('M',3)])
   # Verify the exact same completed step before evaluating any model.
   common=min(json.load(open(run/c/'done.json'))['steps'] for c in ['P','M'])
   assert common==12000
   json.dump(dict(common_step=common,checkpoints={c:str(run/c/'checkpoints/ema_last.pt') for c in ['P','M']}),open(run/'common_checkpoint.json','w'),indent=1)
  if a.stage in ['eval','all']:
   assert (run/'common_checkpoint.json').exists()
   parallel([lambda:probe('P',2),lambda:probe('M',3)])
   parallel([lambda:infer('P','val',2),lambda:infer('M','val',3)])
   # Official CPU scoring overlaps baseline readout training after P/M training finishes.
   with concurrent.futures.ThreadPoolExecutor(max_workers=1) as cpu:
    fs=cpu.submit(parallel,[lambda c=c,e=e:off(c,'val',e) for c in ['P','M'] for e in range(1,11)],8)
    parallel([lambda:probe('B0',2),lambda:probe('C0',3)])
    fs.result()
   parallel([lambda:infer('B0','val',2),lambda:infer('C0','val',3)])
   parallel([lambda c=c,e=e:off(c,'val',e) for c in ['B0','C0'] for e in range(1,11)],8)
   cmd('selection','src.siglip_pm.select_epoch',[str(run)])
   for pair in [[('P',2),('M',3)],[('B0',2),('C0',3)]]:
    parallel([lambda c=c,g=g:infer(c,'test',g) for c,g in pair])
   sel=json.load(open(run/'selection.json'))
   parallel([lambda c=c:off(c,'test',sel[f'{c}_r2']['epoch']) for c in ['P','M','B0','C0']],4)
   cmd('bootstrap','src.siglip_pm.bootstrap',['--run',str(run)],evalenv=True)
   (run/'results').mkdir(exist_ok=True)
   cmd('efficiency','src.siglip_pm.efficiency',['--run',str(run)],2)
   cmd('report','src.siglip_pm.report',['--run',str(run)])
  json.dump(dict(status='done',seconds=time.time()-started),open(run/f'{a.stage}_done.json','w'),indent=1)
 finally:stop.set()
if __name__=='__main__':main()
