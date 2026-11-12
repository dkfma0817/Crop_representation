"""Operational GPU backfill monitor; never changes a U-series training setting."""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess

from src.brixel_eval.finish import pidfd
from src.fgtrain.common import write_json
from src.fgtrain.resources import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    root = Path(parser.parse_args().run).resolve()
    parent = pidfd(read(root / 'launch.json')['pid'])
    proc = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.free',
         '--format=csv,noheader,nounits', '--loop=15'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pending = b''
    window = {gpu: deque(maxlen=4) for gpu in (0, 1)}
    latched = set()
    try:
        while not (root / 'done.json').exists() and not (root / 'failed.json').exists():
            ready, _, _ = select.select([parent, proc.stdout], [], [])
            if parent in ready:
                break
            chunk = os.read(proc.stdout.fileno(), 8192)
            if not chunk:
                break
            pending += chunk
            while b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                try:
                    gpu, utilization, used, free = map(int, line.decode().split(','))
                except ValueError:
                    continue
                now = datetime.now(timezone.utc)
                window[gpu].append(utilization)
                average = sum(window[gpu]) / len(window[gpu])
                startup_grace = 60 if max(window[gpu]) < 10 else 120
                active = read(root / 'active.json', {})
                for name, job in active.items():
                    if str(gpu) not in job.get('gpus', '').split(','):
                        continue
                    age = (now - datetime.fromisoformat(job['start'].replace('Z', '+00:00'))).total_seconds()
                    if len(window[gpu]) == 4 and average < 80 and age >= startup_grace and free > 30720 and name not in latched:
                        policy = read(root / 'background_policy.json')
                        policy['allow_background_with_stages'] = sorted(
                            set(policy.get('allow_background_with_stages', [])) | {name})
                        write_json(root / 'background_policy.json', policy)
                        latched.add(name)
                        print(f'{name}: one-minute average {average:.1f}% below80; existing A/B may backfill', flush=True)
                with (root / 'gpu_utilization.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(utc=now.isoformat(), physical_gpu=gpu,
                        utilization=utilization, memory_used_mib=used, memory_free_mib=free,
                        rolling_average=average, rolling_samples=len(window[gpu]),
                        backfill_latched=sorted(latched))) + '\n')
    finally:
        proc.terminate()
        proc.wait()
        os.close(parent)


if __name__ == '__main__':
    main()
