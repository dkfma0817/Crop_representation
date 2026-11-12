"""Native nvidia-smi stream: record utilization and request A/B backfill if low.

The unchanged foreground resource manager remains the sole pause-flag owner.
Low-use training stages are latched as shareable after two <50% samples, only
after two minutes and with >30GiB free. Reader/inference stages always share.
"""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess

from src.fgtrain.common import write_json
from src.fgtrain.p1 import read
from src.brixel_eval.finish import pidfd


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
    parent=pidfd(read(root/'launch.json')['pid'])
    proc=subprocess.Popen(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.free',
                           '--format=csv,noheader,nounits','--loop=15'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    buf=b'';low={0:0,1:0};latched=set()
    import datetime
    try:
        while not (root/'done.json').exists() and not (root/'failed.json').exists():
            ready,_,_=select.select([parent,proc.stdout],[],[])
            if parent in ready:break
            data=os.read(proc.stdout.fileno(),8192)
            if not data:break
            buf+=data
            while b'\n' in buf:
                line,buf=buf.split(b'\n',1)
                try:g,u,used,free=[int(v.strip()) for v in line.decode().split(',')]
                except ValueError:continue
                now=datetime.datetime.now(datetime.timezone.utc)
                active=read(root/'active.json') if (root/'active.json').exists() else {}
                low[g]=low.get(g,0)+1 if u<50 else 0
                for name,job in active.items():
                    if name not in ['train_P2','train_P3'] or str(g) not in job.get('gpus','').split(','):continue
                    age=(now-datetime.datetime.fromisoformat(job['start'].replace('Z','+00:00'))).total_seconds()
                    if low[g]>=2 and age>=120 and free>30720 and name not in latched:
                        policy=read(root/'background_policy.json')
                        policy['allow_background_with_stages']=sorted(set(policy['allow_background_with_stages'])|{name,name.replace('train_','reserve_')})
                        write_json(root/'background_policy.json',policy);latched.add(name)
                        print(f'{name}: utilization below50 twice; permit existing checkpoint-resumable A/B backfill',flush=True)
                with (root/'gpu_utilization.jsonl').open('a') as f:
                    f.write(json.dumps(dict(utc=now.isoformat(),physical_gpu=g,utilization=u,memory_used_mib=used,
                                           memory_free_mib=free,backfill_latched=sorted(latched)))+'\n')
    finally:
        proc.terminate();proc.wait();os.close(parent)


if __name__=='__main__':main()
