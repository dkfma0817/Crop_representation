"""Give active pilot stages their GPUs; fill every other GPU with existing A/B work."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import time

from src.fgtrain.common import ROOT, RUN, write_json
from src.brixel_eval.finish import pidfd

PREFIX='fgtrain per-GPU priority:'
CTL=ROOT/'output/2026-11-09/bgjobs/control'


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN));a=p.parse_args()
    root=Path(a.run)
    lock=(root/'resource_manager.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    owner=read(root/'resource_ownership.json',{})
    prefix=read(root/'background_policy.json',{}).get('ownership_prefix',PREFIX)
    global_flag=CTL/'PAUSE_all'
    # Latest user instruction explicitly revokes this task's former global pause.
    if global_flag.exists() and global_flag.read_text()==owner.get('payload'):
        global_flag.unlink()
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    if libc.inotify_add_watch(fd,os.fsencode(root),0x8|0x80|0x100)<0:
        raise OSError(ctypes.get_errno(),'inotify_add_watch')
    launcher=read(root/'launch.json',{})['pid']
    try:parent_fd=pidfd(launcher)
    except ProcessLookupError:parent_fd=None
    last=None
    def stop(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while True:
            finished=(root/'done.json').exists() or (root/'failed.json').exists()
            active={} if finished else read(root/'active.json',{})
            if not finished and read(root/'background_policy.json',{}).get('include_final_test',False):
                active.update({'test/'+k:v for k,v in read(root/'test/active.json',{}).items()})
            occupied=set()
            stages={}
            for name,job in active.items():
                if not Path(f"/proc/{job['pid']}").exists():continue
                devices=[int(g) for g in job.get('gpus','').split(',') if g.strip()]
                for g in devices:occupied.add(g);stages.setdefault(g,[]).append(name)
            policy=read(root/'background_policy.json',{})
            allowed=set(policy.get('allow_background_with_stages',[]))
            # ADE's final validation uses single-image forwards. Restore backfill
            # once both final heads are saved, after its dense training finishes.
            if policy.get('ade_backfill_during_validation',False):
                if all((root/'ade'/f'head_{m}.pt').exists() for m in ['B0','B1']):
                    allowed.add('ade')
            shared={g for g in occupied if set(stages[g]).issubset(allowed)}
            reserved=occupied-shared
            preserved=[]
            for gpu in [0,1]:
                flag=CTL/f'PAUSE_gpu{gpu}'
                own=flag.exists() and flag.read_text().startswith(prefix)
                if gpu in reserved:
                    value=f'{prefix} GPU{gpu} used by '+','.join(stages[gpu])+'\n'
                    if not flag.exists() or own:
                        if not flag.exists() or flag.read_text()!=value:flag.write_text(value)
                    else:preserved.append(str(flag))
                elif own:flag.unlink()
            state=dict(pilot_gpus=sorted(occupied),reserved_gpus=sorted(reserved),shared_gpus=sorted(shared),AB_enabled_gpus=[g for g in [0,1] if g not in reserved],
                       pilot_stages=stages,preserved_user_flags=preserved,completed=finished)
            if state!=last:
                write_json(root/'gpu_allocation.json',dict(**state,utc=time.strftime('%FT%TZ',time.gmtime())))
                print(json.dumps(state),flush=True);last=state
            if finished:break
            ready,_,_=select.select([fd]+([parent_fd] if parent_fd is not None else []),[],[],5)
            if fd in ready:
                try:os.read(fd,65536)
                except BlockingIOError:pass
            if parent_fd is not None and parent_fd in ready:break
    finally:
        os.close(fd)
        if parent_fd is not None:os.close(parent_fd)
        for gpu in [0,1]:
            flag=CTL/f'PAUSE_gpu{gpu}'
            if flag.exists() and flag.read_text().startswith(prefix):flag.unlink()

if __name__=='__main__':main()
