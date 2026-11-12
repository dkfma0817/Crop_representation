"""Tables and CSVs for the M1/M2 report, from fixed official predictions and training logs (run after the queue is done).

  python -m src.siglip_cue.report --run <run> --base <P8 base run>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.siglip_extra.report import TYPES, flat
from src.siglip_pm.report import md

MODELS = ['B0', 'P8', 'M1', 'M2', 'C0']
PAIRS = [('M1', 'M2'), ('M1', 'P8'), ('M2', 'P8'), ('C0', 'P8'), ('C0', 'M1'), ('C0', 'M2'), ('M1', 'B0'), ('M2', 'B0')]
PROTOCOL = 'PACO Official Test — GT-ROI / GT-Category Oracle Attribute AP (epoch fixed on Official Val)'
FOCUS = ['Obj Pattern', 'Obj Material', 'Part Pattern', 'Part Material']


def load(path):
    return json.loads(Path(path).read_text())


def table(run):
    rows = []
    for c in MODELS:
        r = flat(c, load(run / f'eval_test/{c}_r2/test/official_results.json')['hierarchical'])
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


def contrasts(df, boot):
    T = df.set_index('Model')
    rows = []
    for x, y in PAIRS:
        for col in df.columns[1:]:
            bx, by = boot['boot'][f'{x}_r2'], boot['boot'][f'{y}_r2']
            if col == 'Overall AP':
                z = (np.array(bx['obj']) + np.array(bx['part']) - np.array(by['obj']) - np.array(by['part'])) / 2
            else:
                z = np.array(bx[boot_key(col)]) - np.array(by[boot_key(col)])
            lo, hi = np.percentile(z, [2.5, 97.5])
            rows.append(dict(comparison=f'{x} − {y}', metric=col, delta=float(T.loc[x, col] - T.loc[y, col]), ci95_low=float(lo), ci95_high=float(hi), B=boot['B'], excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def sizes(run, base):
    frame = pd.read_csv(base / 'results/roi_size_ap.csv').replace({'model': {'P': 'P8'}})
    new = [pd.read_csv(run / f'results/roi_size_ap_{c}.csv') for c in ('M1', 'M2')]
    data = pd.concat([frame[frame.model.isin(['P8', 'C0', 'B0'])]] + new, ignore_index=True)
    rows = []
    for x, y in [('M1', 'M2'), ('M1', 'P8'), ('M2', 'P8'), ('C0', 'M1'), ('C0', 'M2')]:
        for lv in ('Object', 'Part'):
            for name in data.size_bin.unique():
                a = data[(data.model == x) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                b = data[(data.model == y) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                rows.append(dict(comparison=f'{x} − {y}', level=lv, size_bin=name, roi_count=int(a.roi_count), **{m: float(a[m] - b[m]) for m in ('Color', 'Pattern', 'Material', 'Transparency')}))
    return data, pd.DataFrame(rows)


def responses(run):
    frames = []
    for name in ('initial', 'P8', 'M1', 'M2'):
        t = pd.read_csv(run / f'response/{name}.csv')
        t.insert(0, 'model', name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', required=True)
    a = ap.parse_args()
    run, base = Path(a.run), Path(a.base)
    out = run / 'results/test_gt_roi_oracle'
    out.mkdir(parents=True, exist_ok=True)
    ap_table = table(run)
    ap_table.to_csv(out / 'paco_ap.csv', index=False)
    boot = load(run / 'boot_test.json')
    pair = contrasts(ap_table, boot)
    pair.to_csv(out / 'contrasts.csv', index=False)
    size_data, size_delta = sizes(run, base)
    size_data.to_csv(out / 'roi_size_ap.csv', index=False)
    size_delta.to_csv(out / 'roi_size_contrasts.csv', index=False)
    resp = responses(run)
    resp.to_csv(out / 'response_summary.csv', index=False)
    selection = load(run / 'selection.json')
    selection['P8_r2'] = load(base / 'selection.json')['P_r2']
    (out / 'selection_all.json').write_text(json.dumps(selection, indent=1))
    focus = pair[pair.metric.isin(FOCUS + ['Overall AP', 'Object AP', 'Part AP'])]
    lines = [f'# {PROTOCOL}', '', '## R2', md(ap_table), '', '## Contrasts (paired image bootstrap B=1000, 95% CI)', md(focus.drop(columns=['B'])), '', '## Held-out COCO response', md(resp[resp.group == 'all']), '', '## Epoch selection', json.dumps(selection, indent=1), '']
    (out / 'tables.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
