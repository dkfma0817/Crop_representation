"""Cooperative GPU leases for independent frozen-reader jobs in this run only."""
from contextlib import contextmanager
from pathlib import Path
import argparse,fcntl,json,os,time

def root_from_argv(flag):
 ap=argparse.ArgumentParser(add_help=False);ap.add_argument(flag,required=True);a,_=ap.parse_known_args();p=Path(getattr(a,flag[2:].replace('-','_')))
 if flag=='--out':p=p.parent
 if p.name=='mean':p=p.parent
 return p

@contextmanager
def gpu_lease(root):
 root=Path(root);state=root/'state';state.mkdir(exist_ok=True,parents=True);gpu=os.environ.get('CUDA_VISIBLE_DEVICES','unassigned')
 assert gpu in ['2','3'],f'Expected one physical GPU2/3, got {gpu}'
 # Cross-run serialization for the already-authorized structural follow-up.
 # This changes scheduling only; running learners and experiment settings stay intact.
 shared=Path('/tmp')/f'crop_representation_{os.getuid()}_gpu_{gpu}.lease'
 priority=shared.with_suffix('.priority.json')
 # Finish the already-running GPU command, then admit the user's newly
 # prioritized encoder/decoder training before older, not-yet-started readers.
 while priority.exists():
  try:
   p=json.loads(priority.read_text());os.kill(p['owner_pid'],0)
   if (Path(p['run']).resolve()==root.resolve() or time.time()>=p['expires']
       or Path(p['completion']).exists() or Path(p['failure']).exists()):break
  except (OSError,ValueError,KeyError):break
  time.sleep(2)
 with open(shared,'a') as outer,open(state/f'gpu_{gpu}.lease','a') as f:
  fcntl.flock(outer,fcntl.LOCK_EX);fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:
   fcntl.flock(f,fcntl.LOCK_UN);fcntl.flock(outer,fcntl.LOCK_UN)

@contextmanager
def output_lease(out):
 out=Path(out);out.mkdir(exist_ok=True,parents=True)
 with open(out/'.training.lease','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)

def complete(out):
 p=Path(out)/'done.json'
 return p.exists() and json.load(open(p)).get('status')=='done'
