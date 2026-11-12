"""Read-only Chrome trace summary: kernel activity and host CUDA synchronization.
Kernel busy fraction is time coverage, not FLOP utilization or theoretical GPU efficiency.
"""
import argparse,json
from pathlib import Path

def union_length(intervals,start,end):
 z=sorted((max(s,start),min(e,end)) for s,e in intervals if e>start and s<end);total=0.;right=start
 for s,e in z:
  if e>right:total+=e-max(s,right);right=e
 return total

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trace',required=True);a=ap.parse_args();p=Path(a.trace);j=json.load(open(p));events=j['traceEvents'];steps=[x for x in events if x.get('name','').startswith('ProfilerStep#') and 'dur' in x]
 if not steps:raise RuntimeError('No active profiler steps')
 start=min(x['ts'] for x in steps);end=max(x['ts']+x['dur'] for x in steps)
 kernels=[x for x in events if x.get('cat')=='kernel' and 'dur' in x];runtime=[x for x in events if x.get('cat')=='cuda_runtime' and 'dur' in x]
 if not kernels:raise RuntimeError('CUDA kernel events unavailable; cannot infer GPU activity from this trace')
 sync=[x for x in runtime if 'Synchronize' in x['name']];launch=[x for x in runtime if 'LaunchKernel' in x['name']]
 intervals=[(x['ts'],x['ts']+x['dur']) for x in kernels];busy=union_length(intervals,start,end)
 names={}
 for x in kernels:names[x['name']]=names.get(x['name'],0)+x['dur']
 result=dict(profiled_steps=len({x['name'] for x in steps}),profile_window_ms=(end-start)/1000,kernel_count=len(kernels),kernel_busy_ms=busy/1000,kernel_busy_time_percent=100*busy/(end-start),
             CUDA_host_synchronize_calls=len(sync),CUDA_host_synchronize_ms=sum(x['dur'] for x in sync)/1000,kernel_launch_CPU_ms=sum(x['dur'] for x in launch)/1000,
             top_kernel_time_ms=[dict(name=n,ms=t/1000) for n,t in sorted(names.items(),key=lambda z:-z[1])[:15]],
             caveat='Prepared-batch short smoke, profiler overhead included; host synchronization often waits for necessary GPU work and is not all removable overhead. This is not a direct profile of the earlier running M process.')
 json.dump(result,open(p.with_name('profile_analysis.json'),'w'),indent=1);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
