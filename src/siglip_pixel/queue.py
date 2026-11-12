"""S1/S2 queue on GPU 0/1. Completed stages are skipped; PACO TEST is reached only after official VAL fixes the epoch.

  python -m src.siglip_pixel.queue --run <run> --base <P8 base run> --stage prep|smoke|full
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import traceback

import yaml

from src.siglip_pm.supervise import ROOT
from src.siglip_struct.stages import Command

GPU = {'S1': 0, 'S2': 1}
CONDS = ['S1', 'S2']
WEIGHT = {'S1': 0., 'S2': .5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--stage', required=True, choices=('prep', 'smoke', 'full'))
    a = ap.parse_args()
    run, base = Path(a.run).resolve(), Path(a.base).resolve()
    for d in ('logs', 'state', 'scripts', 'results'):
        (run / d).mkdir(parents=True, exist_ok=True)
    if not (run / 'budget.json').exists():
        (run / 'budget.json').write_text(json.dumps(dict(deadline_unix=time.time() + 24 * 3600), indent=1))
    cmd = Command(run)
    shared = str(base / 'shared')

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda f: f(), jobs))

    def train_args(c):
        return ['--condition', c, '--crop-weight', 1.0, '--centered-weight', WEIGHT[c], '--initial', run / 'initial.pt', '--stats', run / 'teacher_stats.npz', '--out', run / c, '--shared', shared, '--steps', 12000, '--workers', 8]

    if a.stage == 'prep':
        cmd('prepare', 'src.siglip_pixel.prepare', ['--run', run, '--p8', base / 'P', '--shared', shared], 0)
        return
    if a.stage == 'smoke':
        parallel([lambda c=c: cmd(f'smoke_{c}', 'src.siglip_pixel.train', train_args(c) + ['--smoke'], GPU[c]) for c in CONDS], 2)
        return

    def job(c, mean):
        return run / 'scripts' / f'{"mean" if mean else "probe"}_{c}.yaml'

    for c in CONDS:
        for mean in (False, True):
            spec = dict(name=f'probe_{c}', cond=c, seed=0, data_seed=0, epochs=10, paco_dev_images=1000, ckpt=str(run / c / 'checkpoints/ema_last.pt'),
                        readers=[dict(name='linear', kind='r0', layers=['Q'])] if mean else [dict(name='r2', kind='r2', layers=['Q'])])
            job(c, mean).write_text(yaml.safe_dump(spec, sort_keys=False))

    def root(mean):
        return run / 'mean' if mean else run

    def stem(mean):
        return ('mean_' if mean else '')

    def key(mean):
        return 'linear' if mean else 'r2'

    def probe(c, mean):
        cmd(f'{stem(mean)}probe_{c}', 'src.siglip_pixel.readout', ['mean' if mean else 'r2', '--job', job(c, mean), '--out', root(mean) / f'probe_{c}', '--workers', 8], GPU[c])

    def infer(c, split, mean):
        args = ['mean_infer' if mean else 'infer', '--run', root(mean), '--job', job(c, mean), '--split', split, '--workers', 8]
        if split == 'test':
            args += ['--select', root(mean) / 'selection.json']
        cmd(f'{stem(mean)}infer_{c}_{split}', 'src.siglip_pixel.readout', args, GPU[c])

    def official(c, split, ep, mean):
        out = root(mean) / f'eval_{split}/{c}_{key(mean)}' / (f'ep{ep}' if split == 'val' else 'test')
        args = ['--split', split, '--pred', root(mean) / f'preds/{c}_{key(mean)}/ep{ep}/probs_{split}.npz', '--out', out]
        if split == 'test':
            args += ['--boot']
        cmd(f'{stem(mean)}official_{c}_{split}_{ep}', 'src.pacoattr.official_eval', args, official=True)

    def evaluate(mean):
        (root(mean) / 'logs').mkdir(parents=True, exist_ok=True)
        parallel([lambda c=c: probe(c, mean) for c in CONDS], 2)
        parallel([lambda c=c: infer(c, 'val', mean) for c in CONDS], 2)
        parallel([lambda c=c, e=e: official(c, 'val', e, mean) for c in CONDS for e in range(1, 11)], 4)
        cmd(f'{stem(mean)}selection', 'src.siglip_pm.select_epoch', [root(mean)])
        selection = json.loads((root(mean) / 'selection.json').read_text())
        parallel([lambda c=c: infer(c, 'test', mean) for c in CONDS], 2)
        parallel([lambda c=c: official(c, 'test', selection[f'{c}_{key(mean)}']['epoch'], mean) for c in CONDS], 2)

    try:
        for c in CONDS:
            smoke = json.loads((run / c / 'smoke.json').read_text())
            assert smoke['status'] == 'passed', c
        if not (run / 'state' / 'train_S1.done.json').exists() or not (run / 'state' / 'train_S2.done.json').exists():
            uuids = {x.split(',')[1].strip() for x in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines() if x.split(',')[0].strip() in ('0', '1')}
            busy = [x for x in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines() if x.split(',')[0].strip() in uuids]
            assert not busy, f'GPU 0/1 occupied: {busy}'
            (run / 'GPU_start_gate.json').write_text(json.dumps(dict(passed=True, time=time.time(), active=busy), indent=1))
        parallel([lambda c=c: cmd(f'train_{c}', 'src.siglip_pixel.train', train_args(c), GPU[c]) for c in CONDS], 2)
        assert all(json.loads((run / c / 'done.json').read_text())['steps'] == 12000 for c in CONDS)
        cfg = {c: json.loads((run / c / 'config.json').read_text()) for c in CONDS}
        drop = {'args', 'condition', 'centered_weight'}
        assert {k: v for k, v in cfg['S1'].items() if k not in drop} == {k: v for k, v in cfg['S2'].items() if k not in drop}, 'S1/S2 configs differ beyond the centered weight'
        s1, s2 = [[json.loads(line) for line in (run / c / 'logs/objective_checks.jsonl').read_text().splitlines()] for c in CONDS]
        assert [r['step'] for r in s1] == [r['step'] for r in s2] and all(r['batch_sha'] == q['batch_sha'] for r, q in zip(s1, s2)), 'S1/S2 data streams differ'
        (run / 'parity_checks.json').write_text(json.dumps(dict(config_equal_except=sorted(drop), batch_sha_equal_at_steps=[r['step'] for r in s1], batch_sha=[r['batch_sha'] for r in s1]), indent=1))
        evaluate(False)
        evaluate(True)
        for mean in (False, True):
            for new, old in [('P8', 'P'), ('B0', 'B0'), ('C0', 'C0')]:
                source = (base / 'mean' if mean else base) / f'eval_test/{old}_{key(mean)}'
                target = root(mean) / f'eval_test/{new}_{key(mean)}'
                if not target.exists():
                    target.symlink_to(source, target_is_directory=True)
            cmd(f'{stem(mean)}bootstrap', 'src.siglip_pixel.bootstrap', ['--run', run] + (['--mean'] if mean else []), official=True)
        parallel([lambda c=c: cmd(f'roi_size_{c}', 'src.siglip_struct.size', ['--run', run, '--model', c], official=True) for c in CONDS], 2)
        (run / 'done.json').write_text(json.dumps(dict(status='done', time=time.time(), completed_utc=time.strftime('%FT%TZ', time.gmtime())), indent=1))
    except Exception:
        (run / 'queue_failed.json').write_text(json.dumps(dict(status='failed', traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == '__main__':
    print(ROOT, flush=True)
    main()
