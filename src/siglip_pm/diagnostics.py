"""Resume the final baseline diagnostics without refitting completed P/C0 probes."""
import argparse,sys
from pathlib import Path
from src.siglip_extra.finish_baselines import main as finish

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--extra',default='output/2026-11-08/siglip2_pm_extra_s0');ap.add_argument('--wait-pid',type=int,default=0);a=ap.parse_args()
 if (Path(a.run)/'diagnostics_done.json').exists():print('Baseline diagnostics already complete.');return
 sys.argv=['finish_baselines','--base',a.run,'--extra',a.extra,'--wait-pid',str(a.wait_pid)];finish()
if __name__=='__main__':main()
