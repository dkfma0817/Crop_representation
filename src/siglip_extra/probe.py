"""Adapter only: the unchanged main R2 training loop for E1/E2/Continue P checkpoints."""
import src.siglip_pm.probe as BASE
ORIGINAL_LOAD=BASE.load_frozen

def load_frozen(job,dev):
 mapped=dict(job)
 if job['cond'] in ['E1','E2','Continue']:mapped['cond']='P'
 m=ORIGINAL_LOAD(mapped,dev);m.roi_mode=job['cond'];return m

if __name__=='__main__':
 from src.siglip_extra.resources import root_from_argv,gpu_lease,output_lease,complete
 import argparse
 ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--out',required=True);args,_=ap.parse_known_args()
 with output_lease(args.out):
  if complete(args.out):print('R2 reader already complete.',flush=True)
  else:
   with gpu_lease(root_from_argv('--out')):
    BASE.load_frozen=load_frozen;BASE.main()
