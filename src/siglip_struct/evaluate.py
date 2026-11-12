"""Unchanged official evaluator/selection, plus fixed-prediction bootstrap/size."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

from src.siglip_pm.supervise import PY,EPY,ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--base',required=True);a=ap.parse_args();run,base=Path(a.run),Path(a.base)
    deadline=json.loads((run/'budget.json').read_text())['deadline_unix']
    def cmd(name,module,args,gpu=None,official=False):
        done=run/'state'/f'{name}.done.json'
        if done.exists():return
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        if official:env['PYTHONPATH']=f'/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}'
        started=time.time();command=[EPY if official else PY,'-u','-m',module]+list(map(str,args));print(json.dumps(dict(stage=name,status='start',gpu=gpu,time=started)),flush=True)
        with (run/'logs'/f'{name}.log').open('a') as f:rc=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time())).returncode
        record=dict(stage=name,gpu=gpu,returncode=rc,seconds=time.time()-started,time=time.time());(done if rc==0 else run/'state'/f'{name}.failed.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
        if rc:raise RuntimeError(name)
    def parallel(jobs,n):
        with ThreadPoolExecutor(max_workers=n) as p:list(p.map(lambda f:f(),jobs))
    def official(c,split,ep):
        args=['--split',split,'--pred',run/f'preds/{c}_r2/ep{ep}/probs_{split}.npz','--out',run/f'eval_{split}/{c}_r2'/('test' if split=='test' else f'ep{ep}')]
        if split=='test':args+=['--boot']
        cmd(f'official_{c}_{split}_{ep}','src.pacoattr.official_eval',args,official=True)
    # These stage names may already be running in overlap_eval; stage locks
    # ensure fixed VAL/TEST predictions are evaluated exactly once.
    from src.siglip_struct.stages import Command
    cmd=Command(run)
    models=['P4','Native','Decode']
    parallel([lambda c=c,e=e:official(c,'val',e) for c in models for e in range(1,11)],8)
    cmd('selection','src.siglip_pm.select_epoch',[run]);selection=json.loads((run/'selection.json').read_text())
    for c in models:
        private=run/f'selection_{c}.json'
        if private.exists():assert json.loads(private.read_text())[f'{c}_r2']['epoch']==selection[f'{c}_r2']['epoch']
    def infer(c,g):
        cmd(f'infer_{c}_test','src.siglip_struct.readout',['infer','--run',run,'--job',run/f'scripts/probe_{c}.yaml','--split','test','--select',run/'selection.json','--workers',8],g)
    def gpu3():
        infer('Native',3);infer('Decode',3)
    parallel([lambda:infer('P4',2),gpu3],2)
    parallel([lambda c=c:official(c,'test',selection[f'{c}_r2']['epoch']) for c in models],3)
    for old,new in [('P','P8'),('M','M'),('B0','B0'),('C0','C0')]:
        target=run/f'eval_test/{new}_r2'
        if not target.exists():target.symlink_to((base/f'eval_test/{old}_r2').resolve(),target_is_directory=True)
    cmd('bootstrap','src.siglip_struct.bootstrap',['--run',run],official=True)
    parallel([lambda c=c:cmd(f'roi_size_{c}','src.siglip_struct.size',['--run',run,'--model',c],official=True) for c in models],3)
    cmd('report','src.siglip_struct.report',['--run',run,'--base',base])


if __name__=='__main__':main()
