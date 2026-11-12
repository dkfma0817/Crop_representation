"""A separate, bounded second 4k phase for E1/E2 and their matched control.

Previous training, epoch selections, TEST predictions and reports are read-only.
Each branch continues its own EMA, with a fresh matched 4k optimizer schedule.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

from src.siglip_pm.supervise import PY, EPY, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--parent', required=True)
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    base, parent, run = map(Path, [a.base, a.parent, a.run])
    for d in ['logs', 'state', 'scripts', 'results']:
        (run / d).mkdir(parents=True, exist_ok=True)
    if (run / 'done.json').exists():
        print('Continuation and evaluation already complete.', flush=True)
        return
    lock = (run / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = json.loads((run / 'budget.json').read_text())['deadline_unix']
    active, stop = {}, threading.Event()
    assert json.loads((parent / 'done.json').read_text())['control_completed']
    assert len({json.loads((parent / c / 'done.json').read_text())['init_sha256'] for c in ['E1', 'E2', 'Continue']}) == 1
    threshold = json.loads((parent / 'negative_calibration.json').read_text())['threshold']

    def command(name, module, args, gpu=None, official=False):
        marker = run / 'state' / f'{name}.done.json'
        if marker.exists():
            return
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
            env.pop(key, None)
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if official:
            env['PYTHONPATH'] = f'/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}'
        cmd = [EPY if official else PY, '-u', '-m', module] + list(map(str, args))
        started = time.time()
        active[name] = dict(gpu=gpu, started=started)
        print(json.dumps(dict(event='start', stage=name, command=cmd, gpu=gpu, time=started)), flush=True)
        with (run / 'logs' / f'{name}.log').open('a') as log:
            try:
                rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=max(1, deadline-time.time())).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        record = dict(stage=name, gpu=gpu, seconds=time.time()-started, time=time.time(), returncode=rc, command=cmd)
        active.pop(name, None)
        (marker if rc == 0 else marker.with_name(f'{name}.failed.json')).write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        if rc:
            raise RuntimeError(f'{name} exited {rc}; see logs/{name}.log')

    def parallel(jobs, n=2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda job: job(), jobs))

    def meter():
        with (run / 'logs/gpu_usage.csv').open('a', buffering=1) as log:
            if log.tell() == 0:
                log.write('timestamp,index,uuid,utilization_percent,memory_used_MiB,power_W\n')
            count = 0
            while not stop.is_set():
                q = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                for line in q.stdout.splitlines():
                    log.write(f'{time.time():.3f},' + line + '\n')
                if count % 4 == 0:
                    print(json.dumps(dict(event='heartbeat', active=list(active), time=time.time())), flush=True)
                count += 1
                stop.wait(15)

    def train(c, gpu):
        checkpoint = parent / c / 'checkpoints/ema_last.pt'
        args = ['--kind', c, '--out', run / c, '--shared', base / 'shared', '--init', checkpoint,
                '--init-step', 16000, '--steps', 4000, '--stream-offset', 4000,
                '--label', f'P12k + {c}_4k + {c}_4k (two restarted phases)',
                '--threshold', threshold, '--workers', 8]
        command(f'smoke_{c}', 'src.siglip_extra.train', args + ['--smoke'], gpu)
        command(f'train_{c}', 'src.siglip_extra.train', args, gpu)

    def job(c, mean=False):
        return run / 'scripts' / f'{"mean" if mean else "probe"}_{c}.yaml'

    def probe(c, gpu, mean=False):
        root = run / 'mean' if mean else run
        command(f'{"mean_" if mean else ""}probe_{c}',
                'src.siglip_extra.mean_probe' if mean else 'src.siglip_extra.probe',
                ['--job', job(c, mean), '--out', root / f'probe_{c}', '--workers', 8], gpu)

    def infer(c, split, gpu, mean=False):
        root = run / 'mean' if mean else run
        args = ['--run', root, '--job', job(c, mean), '--split', split, '--workers', 8]
        if split == 'test':
            args += ['--select', root / 'selection.json']
        command(f'{"mean_" if mean else ""}infer_{c}_{split}',
                'src.siglip_extra.mean_infer' if mean else 'src.siglip_extra.infer', args, gpu)

    def official(c, split, epoch, mean=False):
        root, key = (run / 'mean', 'linear') if mean else (run, 'r2')
        args = ['--split', split, '--pred', root / f'preds/{c}_{key}/ep{epoch}/probs_{split}.npz',
                '--out', root / f'eval_{split}/{c}_{key}' / (f'ep{epoch}' if split == 'val' else 'test')]
        if split == 'test':
            args += ['--boot']
        command(f'{"mean_" if mean else ""}official_{c}_{split}_{epoch}', 'src.pacoattr.official_eval', args, official=True)

    models = ['E1', 'E2', 'Continue']
    def evaluate(mean=False):
        for lo in range(0, len(models), 2):
            parallel([lambda c=c, g=2+j: probe(c, g, mean) for j, c in enumerate(models[lo:lo+2])])
        for lo in range(0, len(models), 2):
            parallel([lambda c=c, g=2+j: infer(c, 'val', g, mean) for j, c in enumerate(models[lo:lo+2])])
        parallel([lambda c=c, e=e: official(c, 'val', e, mean) for c in models for e in range(1, 11)], 8)
        root, key = (run / 'mean', 'linear') if mean else (run, 'r2')
        command(f'{"mean_" if mean else ""}selection', 'src.siglip_pm.select_epoch', [root])
        selected = json.loads((root / 'selection.json').read_text())
        for lo in range(0, len(models), 2):
            parallel([lambda c=c, g=2+j: infer(c, 'test', g, mean) for j, c in enumerate(models[lo:lo+2])])
        parallel([lambda c=c: official(c, 'test', selected[f'{c}_{key}']['epoch'], mean) for c in models], 3)

    try:
        # Refuse to occupy GPUs owned by another task. No existing process is stopped.
        uuids = {x.strip() for x in subprocess.check_output(['nvidia-smi', '-i', '2,3', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()}
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines()
        busy = [x for x in apps if x.split(',')[0].strip() in uuids]
        assert not busy, f'GPU2/3 occupied: {busy}'
        (run / 'GPU_start_gate.json').write_text(json.dumps(dict(passed=True, time=time.time(), GPU_UUIDs=sorted(uuids), active_processes=[]), indent=2))
        threading.Thread(target=meter, daemon=True).start()
        parallel([lambda: train('E1', 2), lambda: train('E2', 3)])
        train('Continue', 2)
        evaluate(False)
        evaluate(True)
        for mean in [False, True]:
            root = run / 'mean' if mean else run
            for c in ['P', 'C0']:
                target = root / f'eval_test/{c}_{"linear" if mean else "r2"}'
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.symlink_to(((parent / 'mean' if mean else parent) / f'eval_test/{c}_{"linear" if mean else "r2"}').resolve(), target_is_directory=True)
            command(f'{"mean_" if mean else ""}bootstrap', 'src.siglip_extra.bootstrap',
                    ['--run', run, '--control'] + (['--mean'] if mean else []), official=True)
        command('report', 'src.siglip_extra.continue_report', ['--base', base, '--parent', parent, '--run', run])
        command('gpu_report', 'src.siglip_pm.gpu_report', ['--runs', run])
        (run / 'done.json').write_text(json.dumps(dict(status='done', time=time.time(), steps_per_phase=4000, cumulative_after_P=8000), indent=2))
    except Exception:
        (run / 'queue_failed.json').write_text(json.dumps(dict(status='failed', traceback=traceback.format_exc(), time=time.time()), indent=2))
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
