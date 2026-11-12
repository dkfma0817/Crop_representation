"""Two independent GPU workers; old jobs serialize through shared GPU leases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

from src.siglip_pm.supervise import PY,EPY,ROOT


def main():
    ap=argparse.ArgumentParser()
    for n in ['run','base']:ap.add_argument('--'+n,required=True)
    a=ap.parse_args();run,base=Path(a.run),Path(a.base)
    if (run/'done.json').exists():return
    lock=(run/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=threading.Event();active={};deadline=json.loads((run/'budget.json').read_text())['deadline_unix']
    def command(name,module,args,gpu=None,official=False):
        done=run/'state'/f'{name}.done.json'
        if done.exists():return
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        for k in ['RANK','LOCAL_RANK','WORLD_SIZE']:env.pop(k,None)
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        if official:env['PYTHONPATH']=f'/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}'
        cmd=[EPY if official else PY,'-u','-m',module]+list(map(str,args));started=time.time();active[name]=gpu
        print(json.dumps(dict(event='start',stage=name,gpu=gpu,command=cmd,time=started)),flush=True)
        with (run/'logs'/f'{name}.log').open('a') as f:
            try:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time())).returncode
            except subprocess.TimeoutExpired:rc=124
        record=dict(stage=name,gpu=gpu,returncode=rc,seconds=time.time()-started,time=time.time(),command=cmd);active.pop(name,None)
        (done if rc==0 else run/'state'/f'{name}.failed.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
        if rc:raise RuntimeError(f'{name} failed{rc}: logs/{name}.log')
    def meter():
        with (run/'logs/gpu_usage.csv').open('a',buffering=1) as f:
            if f.tell()==0:f.write('timestamp,index,uuid,utilization_percent,memory_used_MiB,power_W\n')
            count=0
            while not stop.is_set():
                q=subprocess.run(['nvidia-smi','-i','2,3','--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                for line in q.stdout.splitlines():f.write(f'{time.time():.3f},'+line+'\n')
                if count%4==0:print(json.dumps(dict(event='heartbeat',active=dict(active),time=time.time())),flush=True)
                count+=1;stop.wait(15)
    def training(c,g):
        args=['--kind',c,'--out',run/c,'--shared',base/'shared','--p8',base/'P/checkpoints/ema_last.pt','--steps',12000,'--microbatch',4,'--workers',8]
        command(f'smoke_{c}','src.siglip_struct.train',args+['--smoke'],g)
        command(f'train_{c}','src.siglip_struct.train',args,g)
    def worker(c,g):
        training(c,g)
        for mode in (['P4'] if c=='P4' else ['Native','Decode']):
            command(f'probe_{mode}','src.siglip_struct.readout',['probe','--job',run/f'scripts/probe_{mode}.yaml','--out',run/f'probe_{mode}','--workers',8],g)
            command(f'infer_{mode}_val','src.siglip_struct.readout',['infer','--job',run/f'scripts/probe_{mode}.yaml','--run',run,'--split','val','--workers',8],g)
        command(f'efficiency_{c}','src.siglip_struct.efficiency',['--run',run,'--kind',c,'--base',base],g)
    try:
        assert json.loads((run/'cpu_checks.json').read_text())['status']=='passed'
        threading.Thread(target=meter,daemon=True).start()
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(worker,'P4',2),pool.submit(worker,'Decode',3)]
            for f in fs:f.result()
        # GPU workers have already prepared all three independent reader VAL predictions.
        command('evaluation','src.siglip_struct.evaluate',['--run',run,'--base',base])
        command('gpu_report','src.siglip_pm.gpu_report',['--runs',run])
        (run/'done.json').write_text(json.dumps(dict(status='done',time=time.time(),steps=12000),indent=2))
    except Exception:
        (run/'queue_failed.json').write_text(json.dumps(dict(status='failed',time=time.time(),traceback=traceback.format_exc()),indent=2));raise
    finally:stop.set()


if __name__=='__main__':main()
