"""GPU2/3 phase summaries from actual sampled utilization and stage timestamps."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def summarize(run):
 run=Path(run);fn=run/'logs/gpu_usage.csv'
 if not fn.exists():return []
 df=pd.read_csv(fn);rows=[]
 for f in sorted((run/'state').glob('*.done.json')):
  r=json.load(open(f));g=r.get('gpu');end=r.get('time');seconds=r.get('seconds')
  if g not in [2,3] or end is None or seconds is None:continue
  d=df[(df['index']==g)&(df.timestamp>=end-seconds)&(df.timestamp<=end)]
  rows.append(dict(stage=r.get('stage',f.stem),gpu=g,wall_seconds=seconds,GPU_hours=seconds/3600,samples=len(d),
                   utilization_mean_percent=float(d.utilization_percent.mean()) if len(d) else None,
                   utilization_p90_percent=float(d.utilization_percent.quantile(.9)) if len(d) else None,
                   peak_memory_used_MiB=float(d.memory_used_MiB.max()) if len(d) else None,power_mean_W=float(d.power_W.mean()) if len(d) else None))
 out=run/'results';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'gpu_phases.csv',index=False)
 json.dump(dict(actual_physical_GPUs=[2,3],sampling_seconds=15,phases=rows,note='NVML GPU utilization is sampled active time, not percentage of theoretical compute throughput; peak process/device memory differs from torch allocated memory.'),open(run/'gpu_summary.json','w'),indent=1)
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',required=True);a=ap.parse_args()
 for r in a.runs:print(r,len(summarize(r)),'GPU phases recorded')
if __name__=='__main__':main()
