"""M1 / M2 evaluation under the pilot protocol (user, 2026-11-09): PACO Official Val only, GT-ROI / GT-Category oracle attribute AP. PACO Official Test is never run here.

R2 complete -> Official Val every epoch -> best epoch by (obj + part) / 2 -> Val comparison -> Val bootstrap -> Val ROI-size AP -> costs. Completed stages are skipped.

  python -m src.siglip_cue.queue_val --run <run> --base <P8 base run>
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback

from src.siglip_pm.supervise import ROOT
from src.siglip_struct.stages import Command

GPU = {'M1': 2, 'M2': 3}
CONDS = ['M1', 'M2']
BASELINES = {'P8': 'P', 'B0': 'B0', 'C0': 'C0'}


def no_test_artifacts(run):
    found = [str(p) for p in list(run.glob('preds/*/*/probs_test.npz')) + list(run.glob('eval_test/*'))]
    assert not found, f'PACO Test artifacts exist: {found}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', required=True)
    a = ap.parse_args()
    run, base = Path(a.run).resolve(), Path(a.base).resolve()
    no_test_artifacts(run)
    cmd = Command(run)

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda f: f(), jobs))

    def job(c):
        return run / 'scripts' / f'probe_{c}.yaml'

    def official(c, e):
        cmd(f'official_{c}_val_{e}', 'src.pacoattr.official_eval', ['--split', 'val', '--pred', run / f'preds/{c}_r2/ep{e}/probs_val.npz', '--out', run / f'eval_val/{c}_r2/ep{e}'], official=True)

    try:
        parallel([lambda c=c: cmd(f'probe_{c}', 'src.siglip_cue.readout', ['r2', '--job', job(c), '--out', run / f'probe_{c}', '--workers', 8], GPU[c]) for c in CONDS], 2)
        parallel([lambda c=c: cmd(f'infer_{c}_val', 'src.siglip_cue.readout', ['infer', '--run', run, '--job', job(c), '--split', 'val', '--workers', 8], GPU[c]) for c in CONDS], 2)
        parallel([lambda c=c, e=e: official(c, e) for c in CONDS for e in range(1, 11)], 4)
        cmd('selection', 'src.siglip_pm.select_epoch', [run])
        selection = json.loads((run / 'selection.json').read_text())
        base_selection = json.loads((base / 'selection.json').read_text())
        pred = {f'{c}': run / f'preds/{c}_r2/ep{selection[f"{c}_r2"]["epoch"]}/probs_val.npz' for c in CONDS}
        for new, old in BASELINES.items():
            pred[new] = base / f'preds/{old}_r2/ep{base_selection[f"{old}_r2"]["epoch"]}/probs_val.npz'
        (run / 'selection_val_all.json').write_text(json.dumps(dict(rule='max (Official Val GT-ROI obj AP + opart AP) / 2; earlier epoch on ties', **{f'{c}_r2': selection[f'{c}_r2'] for c in CONDS}, **{f'{new}_r2': base_selection[f'{old}_r2'] for new, old in BASELINES.items()}, predictions={k: str(v) for k, v in pred.items()}), indent=1))
        parallel([lambda c=c: cmd(f'val_boot_official_{c}', 'src.pacoattr.official_eval', ['--split', 'val', '--pred', pred[c], '--out', run / f'eval_val_boot/{c}_r2/val', '--boot'], official=True) for c in pred], 4)
        cmd('val_bootstrap', 'src.siglip_cue.bootstrap_val', ['--run', run], official=True)
        parallel([lambda c=c: cmd(f'val_roi_size_{c}', 'src.siglip_cue.size_val', ['--model', c, '--pred', pred[c], '--out', run / f'results/val_gt_roi_oracle/roi_size_{c}.csv'], official=True) for c in pred], 4)
        cmd('efficiency', 'src.siglip_cue.efficiency', ['--run', run], GPU['M1'])
        no_test_artifacts(run)
        (run / 'done_val.json').write_text(json.dumps(dict(status='done', protocol='PACO Official Val — GT-ROI / GT-Category Oracle Attribute AP', paco_test_run=False, time=time.time(), completed_utc=time.strftime('%FT%TZ', time.gmtime())), indent=1))
    except Exception:
        (run / 'queue_val_failed.json').write_text(json.dumps(dict(status='failed', traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == '__main__':
    print(ROOT, flush=True)
    main()
