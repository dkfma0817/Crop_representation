"""S1/S2 pilot on PACO official Val only (GT box + GT category oracle attribute AP). Never starts a Test stage or opens Test results.

Waits for the two Mean probes started by the paused queue, stops that queue, then: Mean Val inference and epoch selection -> official Val
with bootstrap packs at the fixed best epochs (R2 and Mean; P8, S1, S2, B0, C0) -> paired bootstrap -> ROI sizes -> tables. Results go to <run>/val_pilot.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import time
import traceback

from src.siglip_struct.stages import Command

GPU = {'S1': 0, 'S2': 1}
CONDS = ['S1', 'S2']
MODELS = ['P8', 'S1', 'S2', 'B0', 'C0']
SIZE_MODELS = ['P8', 'S1', 'S2', 'C0']
OLD = {'P8': 'P', 'B0': 'B0', 'C0': 'C0'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--paused-queue', type=int, required=True)
    a = ap.parse_args()
    run, base = Path(a.run).resolve(), Path(a.base).resolve()
    root = run / 'val_pilot'
    for d in ('logs', 'state', 'results', 'mean'):
        (root / d).mkdir(parents=True, exist_ok=True)
    if not (root / 'budget.json').exists():
        (root / 'budget.json').write_text(json.dumps(dict(deadline_unix=time.time() + 24 * 3600), indent=1))
    cmd = Command(root)

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda f: f(), jobs))

    try:
        while not all((run / 'mean' / f'probe_{c}/done.json').exists() for c in CONDS):
            bad = [c for c in CONDS if (run / 'mean' / f'probe_{c}/failed.json').exists()]
            assert not bad, f'Mean probe failed: {bad}'
            time.sleep(20)
        marker = root / 'old_queue_stopped.json'
        if not marker.exists():
            cmdline = Path(f'/proc/{a.paused_queue}/cmdline').read_bytes().decode().replace('\0', ' ') if Path(f'/proc/{a.paused_queue}').exists() else ''
            if 'src.siglip_pixel.queue' in cmdline:
                os.kill(a.paused_queue, signal.SIGKILL)
            marker.write_text(json.dumps(dict(pid=a.paused_queue, had_cmdline=cmdline, stopped_unix=time.time(), reason='Pilot protocol changed to official Val; Mean Test and later stages never started'), indent=1))
        mean_root = run / 'mean'
        parallel([lambda c=c: cmd(f'mean_infer_{c}_val', 'src.siglip_pixel.readout', ['mean_infer', '--run', mean_root, '--job', run / f'scripts/mean_{c}.yaml', '--split', 'val', '--workers', 8], GPU[c]) for c in CONDS], 2)

        def official_all(c, e):
            cmd(f'mean_official_{c}_val_{e}', 'src.pacoattr.official_eval', ['--split', 'val', '--pred', mean_root / f'preds/{c}_linear/ep{e}/probs_val.npz', '--out', mean_root / f'eval_val/{c}_linear/ep{e}'], official=True)

        parallel([lambda c=c, e=e: official_all(c, e) for c in CONDS for e in range(1, 11)], 4)
        cmd('mean_selection', 'src.siglip_pm.select_epoch', [mean_root])
        sel = dict(r2={**{f'{c}_r2': json.loads((run / 'selection.json').read_text())[f'{c}_r2'] for c in CONDS}, **{f'{n}_r2': json.loads((base / 'selection.json').read_text())[f'{o}_r2'] for n, o in OLD.items()}},
                   linear={**{f'{c}_linear': json.loads((mean_root / 'selection.json').read_text())[f'{c}_linear'] for c in CONDS}, **{f'{n}_linear': json.loads((base / 'mean/selection.json').read_text())[f'{o}_linear'] for n, o in OLD.items()}})
        (root / 'selected_epochs.json').write_text(json.dumps(sel, indent=1))

        def pred(key, model, epoch):
            if model in CONDS:
                r = run if key == 'r2' else mean_root
                return r / f'preds/{model}_{key}/ep{epoch}/probs_val.npz'
            r = base if key == 'r2' else base / 'mean'
            return r / f'preds/{OLD[model]}_{key}/ep{epoch}/probs_val.npz'

        def eval_boot(key, model):
            epoch = sel[key][f'{model}_{key}']['epoch']
            r = root if key == 'r2' else root / 'mean'
            cmd(f'official_boot_{key}_{model}', 'src.pacoattr.official_eval', ['--split', 'val', '--pred', pred(key, model, epoch), '--out', r / f'eval_boot/{model}_{key}/val', '--boot'], official=True)

        parallel([lambda k=k, m=m: eval_boot(k, m) for k in ('r2', 'linear') for m in MODELS], 4)
        for key, r in (('r2', root), ('linear', root / 'mean')):
            cmd(f'bootstrap_{key}', 'src.siglip_pixel.val_bootstrap', ['--root', r, '--key', key], official=True)

        def size(key, model):
            epoch = sel[key][f'{model}_{key}']['epoch']
            cmd(f'size_{key}_{model}', 'src.siglip_pixel.val_size', ['--pred', pred(key, model, epoch), '--model', model, '--out', root / f'results/roi_size_{key}_{model}.csv'], official=True)

        parallel([lambda k=k, m=m: size(k, m) for k in ('r2', 'linear') for m in SIZE_MODELS], 4)
        cmd('report', 'src.siglip_pixel.val_report', ['--run', run, '--base', base])
        result = dict(status='done', protocol='PACO official Val GT-ROI oracle attribute AP; no Test stage after the protocol change', completed_utc=time.strftime('%FT%TZ', time.gmtime()), time=time.time())
        (root / 'done.json').write_text(json.dumps(result, indent=1))
        (run / 'done.json').write_text(json.dumps(result, indent=1))
    except Exception:
        (run / 'queue_failed.json').write_text(json.dumps(dict(status='failed', traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == '__main__':
    main()
