"""Cross-process stage locks for safe early evaluation of completed readers."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from src.siglip_pm.supervise import PY,EPY,ROOT


@contextmanager
def cpu_slot(root):
    while True:
        held=None
        for i in range(4):
            f=(root/'state'/f'official_cpu_slot_{i}.lease').open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held=f;break
            except BlockingIOError:f.close()
        if held is not None:break
        time.sleep(.5)
    try:yield
    finally:fcntl.flock(held,fcntl.LOCK_UN);held.close()


class Command:
    def __init__(self,root):
        self.root=Path(root)
        self.deadline=json.loads((self.root/'budget.json').read_text())['deadline_unix']

    def __call__(self,name,module,args,gpu=None,official=False):
        root=self.root;done=root/'state'/f'{name}.done.json'
        with (root/'state'/f'{name}.command.lease').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if done.exists():return
            env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
            if official:env['PYTHONPATH']=f'/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}'
            command=[EPY if official else PY,'-u','-m',module]+list(map(str,args));started=time.time()
            print(json.dumps(dict(stage=name,status='start',gpu=gpu,time=started)),flush=True)
            def execute():
                with (root/'logs'/f'{name}.log').open('a') as log:
                    return subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,self.deadline-time.time())).returncode
            if official:
                with cpu_slot(root):rc=execute()
            else:rc=execute()
            record=dict(stage=name,gpu=gpu,returncode=rc,seconds=time.time()-started,time=time.time(),command=command)
            (done if rc==0 else root/'state'/f'{name}.failed.json').write_text(json.dumps(record,indent=2))
            print(json.dumps(record),flush=True)
            if rc:raise RuntimeError(name)
