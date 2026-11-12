"""Same established mean/R0 protocol, with an adapter for the P-derived checkpoints."""
import src.siglip_pm.mean_probe as MEAN
from src.siglip_extra.probe import load_frozen
if __name__=='__main__':
 from src.siglip_extra.resources import root_from_argv,gpu_lease,output_lease,complete
 import argparse
 ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--out',required=True);args,_=ap.parse_known_args()
 with output_lease(args.out):
  if complete(args.out):print('Mean reader already complete.',flush=True)
  else:
   with gpu_lease(root_from_argv('--out')):
    MEAN.BASE.load_frozen=load_frozen;MEAN.main()
