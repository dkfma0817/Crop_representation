"""Tables and CSVs for the S1/S2 report, from fixed official predictions and training logs (run after the queue is done).

  python -m src.siglip_pixel.report --run <run> --base <P8 base run>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.siglip_extra.report import TYPES, flat
from src.siglip_pm.report import md

MODELS = ['P8', 'S1', 'S2', 'B0', 'C0']
PAIRS = [('S1', 'P8'), ('S2', 'P8'), ('S2', 'S1'), ('C0', 'S1'), ('C0', 'S2'), ('C0', 'P8'), ('S1', 'B0'), ('S2', 'B0')]
GT = Path('/home/irteam/datasets/paco/paco_lvis_v1_test.json')


def load(path):
    return json.loads(Path(path).read_text())


def table(root, key):
    rows = []
    for c in MODELS:
        h = load(root / f'eval_test/{c}_{key}/test/official_results.json')['hierarchical']
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
            if col == 'Overall AP':
                z = (np.array(boot['boot'][f'{x}_{key}']['obj']) + np.array(boot['boot'][f'{x}_{key}']['part']) - np.array(boot['boot'][f'{y}_{key}']['obj']) - np.array(boot['boot'][f'{y}_{key}']['part'])) / 2
            else:
                bk = boot_key(col)
                z = np.array(boot['boot'][f'{x}_{key}'][bk]) - np.array(boot['boot'][f'{y}_{key}'][bk])
            lo, hi = np.percentile(z, [2.5, 97.5])
            rows.append(dict(comparison=f'{x} − {y}', metric=col, delta=float(T.loc[x, col] - T.loc[y, col]), ci95_low=float(lo), ci95_high=float(hi), B=boot['B'], excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def attribute_table(run):
    gt = load(GT)
    attrs = {v['name']: v for v in gt['attributes']}
    categories = {v['id']: v['name'] for v in gt['categories']}
    per = {c: load(run / f'eval_test/{c}_r2/test/official_results.json')['hierarchical'] for c in MODELS}
    by_id = {v['id']: v['name'] for v in gt['attributes']}
    groups = {t: [by_id[i] for i in gt['attr_type_to_attr_idxs'][t]] for t, _ in TYPES}
    rows = []
    for t, label in TYPES:
        for name in groups[t]:
            for level, part, field in [('Object', False, 'obj-attr-AP-per-class'), ('Part', True, 'obj-attr-part-AP-per-class')]:
                pos = [v for v in gt['annotations'] if attrs[name]['id'] in v.get('attribute_ids', []) and (':' in categories[v['category_id']]) == part]
                row = dict(level=level, type=label, attribute=name, positive_ROIs=len(pos))
                for c in MODELS:
                    row[c] = per[c][field].get(name, float('nan'))
                rows.append(row)
    df = pd.DataFrame(rows)
    df['S1−P8'] = df.S1 - df.P8
    df['S2−P8'] = df.S2 - df.P8
    df['S2−S1'] = df.S2 - df.S1
    df['C0−S2'] = df.C0 - df.S2
    return df


def sizes(run, base):
    frame = pd.read_csv(base / 'results/roi_size_ap.csv').replace({'model': {'P': 'P8'}})
    new = [pd.read_csv(run / f'results/roi_size_ap_{c}.csv') for c in ('S1', 'S2')]
    data = pd.concat([frame[frame.model.isin(['P8', 'C0', 'B0'])]] + new, ignore_index=True)
    rows = []
    for x, y in [('S1', 'P8'), ('S2', 'P8'), ('S2', 'S1'), ('C0', 'S1'), ('C0', 'S2')]:
        for lv in ('Object', 'Part'):
            for name in data.size_bin.unique():
                a = data[(data.model == x) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                b = data[(data.model == y) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                rows.append(dict(comparison=f'{x} − {y}', level=lv, size_bin=name, roi_count=int(a.roi_count), **{m: float(a[m] - b[m]) for m in ('Color', 'Pattern', 'Material', 'Transparency')}))
    return data, pd.DataFrame(rows)


def training(run, base):
    out = {}
    for c, log, diag, check in [('P8', base / 'P/logs/train_log.jsonl', base / 'P/logs/diag.jsonl', None),
                                ('S1', run / 'S1/logs/train.jsonl', run / 'S1/logs/diag.jsonl', run / 'S1/logs/objective_checks.jsonl'),
                                ('S2', run / 'S2/logs/train.jsonl', run / 'S2/logs/diag.jsonl', run / 'S2/logs/objective_checks.jsonl')]:
        rec = pd.read_json(log, lines=True)
        d = pd.read_json(diag, lines=True)
        entry = dict(log=rec.to_dict('records'), diag=d.to_dict('records'))
        if check is not None:
            entry['checks'] = [json.loads(s) for s in check.read_text().splitlines()]
            entry['config'] = load(check.parents[1] / 'config.json')
            entry['done'] = load(check.parents[1] / 'done.json')
        out[c] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', required=True)
    a = ap.parse_args()
    run, base = Path(a.run), Path(a.base)
    out = run / 'results'
    out.mkdir(exist_ok=True)
    ap_table = table(run, 'r2')
    ap_table.to_csv(out / 'paco_ap.csv', index=False)
    boot = load(run / 'boot_test.json')
    pair = contrasts(ap_table, boot, 'r2')
    pair.to_csv(out / 'contrasts.csv', index=False)
    mean_table = table(run / 'mean', 'linear')
    mean_table.to_csv(out / 'mean_ap.csv', index=False)
    mean_boot = load(run / 'mean/boot_test.json')
    mean_pair = contrasts(mean_table, mean_boot, 'linear')
    mean_pair.to_csv(out / 'mean_contrasts.csv', index=False)
    attrs = attribute_table(run)
    attrs.to_csv(out / 'attribute_ap.csv', index=False)
    size_data, size_delta = sizes(run, base)
    size_data.to_csv(out / 'roi_size_ap.csv', index=False)
    size_delta.to_csv(out / 'roi_size_contrasts.csv', index=False)
    train = training(run, base)
    (out / 'training_summary.json').write_text(json.dumps(train, default=float))
    selection = load(run / 'selection.json')
    selection['P8_r2'] = load(base / 'selection.json')['P_r2']
    (out / 'selection_all.json').write_text(json.dumps(selection, indent=1))
    lines = ['## R2 Official Test', md(ap_table), '', '## Contrasts (R2, paired image bootstrap B=1000, 95% CI)', md(pair.drop(columns=['B'])), '',
             '## Mean + Linear Probe (auxiliary)', md(mean_table), '', md(mean_pair.drop(columns=['B'])), '', '## Epoch selection', json.dumps(selection, indent=1), '']
    (out / 'tables.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
