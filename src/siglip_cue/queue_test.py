"""PACO Official Test of M1 / M2 at the epoch fixed by Official Val (user request, 2026-11-09). Runs only after the Val queue is done; nothing here feeds back into model or epoch choice.

  python -m src.siglip_cue.queue_test --run <run> --base <P8 base run>
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import traceback

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
    assert (run / 'done_val.json').exists(), 'the Val queue must be complete first'
    os.environ['SIGLIP_CUE_ALLOW_TEST'] = '1'
    cmd = Command(run)
    selection = json.loads((run / 'selection.json').read_text())

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda f: f(), jobs))

    def official(c):
        ep = selection[f'{c}_r2']['epoch']
        cmd(f'official_{c}_test_{ep}', 'src.pacoattr.official_eval', ['--split', 'test', '--pred', run / f'preds/{c}_r2/ep{ep}/probs_test.npz', '--out', run / f'eval_test/{c}_r2/test', '--boot'], official=True)

    try:
        parallel([lambda c=c: cmd(f'infer_{c}_test', 'src.siglip_cue.readout', ['infer', '--run', run, '--job', run / f'scripts/probe_{c}.yaml', '--split', 'test', '--select', run / 'selection.json', '--workers', 8], GPU[c]) for c in CONDS], 2)
        parallel([lambda c=c: official(c) for c in CONDS], 2)
        for new, old in [('P8', 'P'), ('B0', 'B0'), ('C0', 'C0')]:
            target = run / f'eval_test/{new}_r2'
            if not target.exists():
                target.symlink_to(base / f'eval_test/{old}_r2', target_is_directory=True)
        cmd('test_bootstrap', 'src.siglip_cue.bootstrap', ['--run', run], official=True)
        parallel([lambda c=c: cmd(f'test_roi_size_{c}', 'src.siglip_struct.size', ['--run', run, '--model', c], official=True) for c in CONDS], 2)
        (run / 'done_test.json').write_text(json.dumps(dict(status='done', protocol='PACO Official Test — GT-ROI / GT-Category Oracle Attribute AP', epochs_from='Official Val selection.json', selection={k: v['epoch'] for k, v in selection.items()}, completed_utc=time.strftime('%FT%TZ', time.gmtime())), indent=1))
    except Exception:
        (run / 'queue_test_failed.json').write_text(json.dumps(dict(status='failed', traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == '__main__':
    print(ROOT, flush=True)
    main()
