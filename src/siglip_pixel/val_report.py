"""S1/S2 pilot tables on PACO official Val (GT-ROI / GT-category oracle attribute AP). Reads no Test file.

  python -m src.siglip_pixel.val_report --run <run> --base <P8 base run>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.siglip_extra.report import ATTRS, TYPES, flat
from src.siglip_pm.report import md

MODELS = ['P8', 'S1', 'S2', 'B0', 'C0']
PAIRS = [('S1', 'P8'), ('S2', 'P8'), ('S2', 'S1'), ('C0', 'P8'), ('C0', 'S1'), ('C0', 'S2'), ('S1', 'B0'), ('S2', 'B0')]
GT = Path('/home/irteam/datasets/paco/paco_lvis_v1_val.json')


def load(path):
    return json.loads(Path(path).read_text())


def table(root, key):
    rows = []
    for c in MODELS:
        h = load(root / f'eval_boot/{c}_{key}/val/official_results.json')['hierarchical']
        r = flat(c, h)
        r['Overall AP'] = (r['Object AP'] + r['Part AP']) / 2
        rows.append(r)
    columns = ['Model', 'Overall AP', 'Object AP', 'Part AP'] + [f'{lv} {n}' for lv in ('Obj', 'Part') for _, n in TYPES]
    return pd.DataFrame(rows)[columns]


def boot_key(col):
    if col == 'Object AP':
        return 'obj'
    if col == 'Part AP':
        return 'part'
    lv, n = col.split(' ', 1)
    return ('obj_' if lv == 'Obj' else 'part_') + dict((b, a) for a, b in TYPES)[n]


def contrasts(df, boot, key):
    T = df.set_index('Model')
    rows = []
    for x, y in PAIRS:
        for col in df.columns[1:]:
            X, Y = boot['boot'][f'{x}_{key}'], boot['boot'][f'{y}_{key}']
            z = (np.array(X['obj']) + np.array(X['part']) - np.array(Y['obj']) - np.array(Y['part'])) / 2 if col == 'Overall AP' else np.array(X[boot_key(col)]) - np.array(Y[boot_key(col)])
            lo, hi = np.percentile(z, [2.5, 97.5])
            rows.append(dict(comparison=f'{x} − {y}', metric=col, delta=float(T.loc[x, col] - T.loc[y, col]), ci95_low=float(lo), ci95_high=float(hi), excludes_zero=bool(lo > 0 or hi < 0), B=boot['B']))
    return pd.DataFrame(rows)


def attributes(root, key):
    gt = load(GT)
    by_id = {v['id']: v['name'] for v in gt['attributes']}
    categories = {v['id']: v['name'] for v in gt['categories']}
    per = {c: load(root / f'eval_boot/{c}_{key}/val/official_results.json')['hierarchical'] for c in MODELS}
    rows = []
    for t, label in TYPES:
        for i in gt['attr_type_to_attr_idxs'][t]:
            for level, part, field in [('Object', False, 'obj-attr-AP-per-class'), ('Part', True, 'obj-attr-part-AP-per-class')]:
                pos = [a for a in gt['annotations'] if i in a.get('attribute_ids', []) and (':' in categories[a['category_id']]) == part]
                row = dict(level=level, type=label, attribute=by_id[i], positive_ROIs=len(pos))
                for c in MODELS:
                    row[c] = per[c][field].get(by_id[i], float('nan'))
                rows.append(row)
    df = pd.DataFrame(rows)
    for x, y in [('S1', 'P8'), ('S2', 'P8'), ('S2', 'S1'), ('C0', 'S1')]:
        df[f'{x}−{y}'] = df[x] - df[y]
    return df


def sizes(root, key):
    data = pd.concat([pd.read_csv(root / f'results/roi_size_{key}_{m}.csv') for m in ('P8', 'S1', 'S2', 'C0')], ignore_index=True)
    rows = []
    for x, y in [('S1', 'P8'), ('S2', 'P8'), ('S2', 'S1'), ('C0', 'S1'), ('C0', 'S2')]:
        for lv in ('Object', 'Part'):
            for name in data.size_bin.unique():
                a = data[(data.model == x) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                b = data[(data.model == y) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                rows.append(dict(comparison=f'{x} − {y}', level=lv, size_bin=name, roi_count=int(a.roi_count), **{m: float(a[m] - b[m]) for m in ('Color', 'Pattern', 'Material', 'Transparency')}))
    return data, pd.DataFrame(rows)


def costs(run, base):
    rows = []
    state = None
    for c, log, done in [('P8', base / 'P/logs/train_log.jsonl', base / 'P/done.json'), ('S1', run / 'S1/logs/train.jsonl', run / 'S1/done.json'), ('S2', run / 'S2/logs/train.jsonl', run / 'S2/done.json')]:
        rec = pd.read_json(log, lines=True)
        d = load(done)
        step = float(1 / rec.steps_per_s.median()) if 'steps_per_s' in rec else float(rec.step_seconds[1:].median())
        ckpt = base / 'P/checkpoints/ema_last.pt' if c == 'P8' else run / c / 'checkpoints/ema_last.pt'
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
        n = sum(v.numel() for v in sd.values())
        if state is None:
            state = sorted(sd)
        rows.append(dict(model=c, train_seconds=d['time_s'], median_step_seconds=step, peak_train_vram_gib=float(rec.mem_gb.max()), encoder_parameters=n, same_state_dict_keys_as_P8=sorted(sd) == state,
                         aux_head_parameters=0 if c == 'P8' else load(run / c / 'config.json')['head_params']))
    df = pd.DataFrame(rows)
    df['train_time_vs_P8'] = df.train_seconds / df.train_seconds.iloc[0]
    df['step_time_vs_P8'] = df.median_step_seconds / df.median_step_seconds.iloc[0]
    p = pd.read_csv(base / 'results/efficiency.csv').query('model=="P"').iloc[0]
    inference = dict(note='The PixelDiT head is training-only and is not in the S1/S2 checkpoints; the encoder architecture equals P8, so these P8 values apply.', parameters=int(p.parameters), median_encode_ms=float(p.median_encode_ms), encoder_GFLOPs=float(p.encoder_GFLOPs), stored_scalars=int(p.stored_scalars))
    return df, inference


def loss_summary(run):
    out = {}
    for c in ('S1', 'S2'):
        rec = pd.read_json(run / c / 'logs/train.jsonl', lines=True)
        checks = [json.loads(s) for s in (run / c / 'logs/objective_checks.jsonl').read_text().splitlines()]
        diag = pd.read_json(run / c / 'logs/diag.jsonl', lines=True).to_dict('records')
        out[c] = dict(first=rec.iloc[0].to_dict(), last=rec.iloc[-1].to_dict(), checks=checks, dev_diag=diag)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', required=True)
    a = ap.parse_args()
    run, base = Path(a.run), Path(a.base)
    root = run / 'val_pilot'
    out = root / 'results'
    results, lines = {}, []
    for key, r, name in (('r2', root, 'R2'), ('linear', root / 'mean', 'Mean + Linear')):
        t = table(r, key)
        c = contrasts(t, load(r / 'boot_val.json'), key)
        attr = attributes(r, key)
        size_data, size_delta = sizes(root, key)
        t.to_csv(out / f'val_ap_{key}.csv', index=False)
        c.to_csv(out / f'val_contrasts_{key}.csv', index=False)
        attr.to_csv(out / f'val_attribute_ap_{key}.csv', index=False)
        size_data.to_csv(out / f'val_roi_size_ap_{key}.csv', index=False)
        size_delta.to_csv(out / f'val_roi_size_contrasts_{key}.csv', index=False)
        focus = attr[(attr.level == 'Object') & attr.attribute.isin(ATTRS)][['attribute', 'type', 'positive_ROIs', 'P8', 'S1', 'S2', 'B0', 'C0', 'S1−P8', 'S2−P8', 'S2−S1']]
        results[key] = dict(ap=t.to_dict('records'), contrasts=c.to_dict('records'), attributes=attr.to_dict('records'), roi_size=size_data.to_dict('records'), roi_size_contrasts=size_delta.to_dict('records'))
        lines += [f'## {name}: official Val AP', md(t), '', '## Contrasts (paired image bootstrap B=1000, 95% CI)', md(c.drop(columns=['B'])), '', '## Six Object attributes (point estimates)', md(focus), '',
                  '## ROI size', md(size_data), '', md(size_delta[size_delta.comparison.isin(['S1 − P8', 'S2 − P8', 'S2 − S1'])]), '']
    cost, inference = costs(run, base)
    cost.to_csv(out / 'cost_training.csv', index=False)
    results.update(cost_training=cost.to_dict('records'), cost_inference=inference, losses=loss_summary(run), selected_epochs=load(root / 'selected_epochs.json'))
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=float))
    lines += ['## Cost', md(cost), '', json.dumps(inference, indent=1)]
    (out / 'tables.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
