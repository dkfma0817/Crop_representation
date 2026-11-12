"""M1 (GPU 2) / M2 (GPU 3) training and PACO evaluation. Completed stages are skipped; PACO TEST is reached only after official VAL fixes the epoch.

  python -m src.siglip_cue.queue --run <run> --base <P8 base run>
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import yaml

from src.siglip_pm.supervise import ROOT
from src.siglip_struct.stages import Command

GPU = {'M1': 2, 'M2': 3}
CONDS = ['M1', 'M2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', required=True)
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

    def job(c):
        return run / 'scripts' / f'probe_{c}.yaml'

    def official(c, split, ep):
        out = run / f'eval_{split}/{c}_r2' / (f'ep{ep}' if split == 'val' else 'test')
        args = ['--split', split, '--pred', run / f'preds/{c}_r2/ep{ep}/probs_{split}.npz', '--out', out]
        if split == 'test':
            args += ['--boot']
        cmd(f'official_{c}_{split}_{ep}', 'src.pacoattr.official_eval', args, official=True)

    def infer(c, split):
        args = ['infer', '--run', run, '--job', job(c), '--split', split, '--workers', 8]
        if split == 'test':
            args += ['--select', run / 'selection.json']
        cmd(f'infer_{c}_{split}', 'src.siglip_cue.readout', args, GPU[c])

    try:
        sanity = {c: json.loads((run / 'sanity' / f'sanity_{c}.json').read_text()) for c in CONDS}
        assert all(s['status'] == 'passed' for s in sanity.values()), 'sanity'
        assert sanity['M1']['batch_sha'] == sanity['M2']['batch_sha'], 'M1/M2 sanity batches differ'
        snapshot = run / 'source_snapshot/siglip_cue'
        if not snapshot.exists():
            shutil.copytree(ROOT / 'src/siglip_cue', snapshot, ignore=shutil.ignore_patterns('__pycache__'))
        if not all((run / 'state' / f'train_{c}.done.json').exists() for c in CONDS):
            uuids = {x.split(',')[1].strip() for x in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines() if x.split(',')[0].strip() in ('2', '3')}
            busy = [x for x in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines() if x.split(',')[0].strip() in uuids]
            assert not busy, f'GPU 2/3 occupied: {busy}'
            (run / 'GPU_start_gate.json').write_text(json.dumps(dict(passed=True, time=time.time(), active=busy), indent=1))

        def train_args(c):
            return ['--condition', c, '--initial', run / 'initial.pt', '--out', run / c, '--shared', shared, '--steps', 12000, '--workers', 12]
        parallel([lambda c=c: cmd(f'train_{c}', 'src.siglip_cue.train', train_args(c), GPU[c]) for c in CONDS], 2)
        assert all(json.loads((run / c / 'done.json').read_text())['steps'] == 12000 for c in CONDS)
        cfg = {c: json.loads((run / c / 'config.json').read_text()) for c in CONDS}
        drop = {'args', 'condition', 'n_params_total', 'n_params_new'}
        assert {k: v for k, v in cfg['M1'].items() if k not in drop} == {k: v for k, v in cfg['M2'].items() if k not in drop}, 'M1/M2 configs differ'
        m1, m2 = [[json.loads(line) for line in (run / c / 'logs/objective_checks.jsonl').read_text().splitlines()] for c in CONDS]
        assert [r['step'] for r in m1] == [r['step'] for r in m2] and all(r['batch_sha'] == q['batch_sha'] for r, q in zip(m1, m2)), 'M1/M2 data streams differ'
        (run / 'parity_checks.json').write_text(json.dumps(dict(config_equal_except=sorted(drop), batch_sha_equal_at_steps=[r['step'] for r in m1], batch_sha=[r['batch_sha'] for r in m1]), indent=1))
        models = ['initial=' + str(run / 'initial.pt'), 'P8=' + str(base / 'P/checkpoints/ema_last.pt'), 'M1=' + str(run / 'M1/checkpoints/ema_last.pt'), 'M2=' + str(run / 'M2/checkpoints/ema_last.pt') + ':M2']
        cmd('respond', 'src.siglip_cue.respond', ['--models'] + models + ['--out', run / 'response'], GPU['M1'])
        for c in CONDS:
            job(c).write_text(yaml.safe_dump(dict(name=f'probe_{c}', cond=c, seed=0, data_seed=0, epochs=10, paco_dev_images=1000, ckpt=str(run / c / 'checkpoints/ema_last.pt'), readers=[dict(name='r2', kind='r2', layers=['Q'])]), sort_keys=False))
        parallel([lambda c=c: cmd(f'probe_{c}', 'src.siglip_cue.readout', ['r2', '--job', job(c), '--out', run / f'probe_{c}', '--workers', 8], GPU[c]) for c in CONDS], 2)
        parallel([lambda c=c: infer(c, 'val') for c in CONDS], 2)
        parallel([lambda c=c, e=e: official(c, 'val', e) for c in CONDS for e in range(1, 11)], 4)
        cmd('selection', 'src.siglip_pm.select_epoch', [run])
        selection = json.loads((run / 'selection.json').read_text())
        parallel([lambda c=c: infer(c, 'test') for c in CONDS], 2)
        parallel([lambda c=c: official(c, 'test', selection[f'{c}_r2']['epoch']) for c in CONDS], 2)
        for new, old in [('P8', 'P'), ('B0', 'B0'), ('C0', 'C0')]:
            target = run / f'eval_test/{new}_r2'
            if not target.exists():
                target.symlink_to(base / f'eval_test/{old}_r2', target_is_directory=True)
        cmd('bootstrap', 'src.siglip_cue.bootstrap', ['--run', run], official=True)
        parallel([lambda c=c: cmd(f'roi_size_{c}', 'src.siglip_struct.size', ['--run', run, '--model', c], official=True) for c in CONDS], 2)
        cmd('efficiency', 'src.siglip_cue.efficiency', ['--run', run], GPU['M1'])
        (run / 'done.json').write_text(json.dumps(dict(status='done', time=time.time(), completed_utc=time.strftime('%FT%TZ', time.gmtime())), indent=1))
    except Exception:
        (run / 'queue_failed.json').write_text(json.dumps(dict(status='failed', traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == '__main__':
    print(ROOT, flush=True)
    main()
