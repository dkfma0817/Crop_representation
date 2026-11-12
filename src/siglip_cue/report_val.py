"""CSV / JSON tables of the M1/M2 pilot under PACO Official Val — GT-ROI / GT-Category Oracle Attribute AP. Reads fixed Val predictions and logs only.

  python -m src.siglip_cue.report_val --run <run> --base <P8 base run>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.siglip_extra.report import TYPES, flat
from src.siglip_pm.report import md

PROTOCOL = 'PACO Official Val — GT-ROI / GT-Category Oracle Attribute AP'
MODELS = ['B0', 'P8', 'M1', 'M2', 'C0']
PAIRS = [('M1', 'M2'), ('M1', 'P8'), ('M2', 'P8'), ('C0', 'M1'), ('C0', 'M2'), ('C0', 'P8'), ('M1', 'B0'), ('M2', 'B0')]
KEY_ATTRS = ['fabric', 'paper', 'ceramic', 'striped', 'perforated', 'text', 'wood', 'metal', 'plastic', 'glass', 'woven', 'logo']
GT = Path('/home/irteam/datasets/paco/paco_lvis_v1_val.json')


def load(path):
    return json.loads(Path(path).read_text())


def table(run):
    rows = []
    for c in MODELS:
        r = flat(c, load(run / f'eval_val_boot/{c}_r2/val/official_results.json')['hierarchical'])
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


def ci(z):
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    lo, hi = np.percentile(z, [2.5, 97.5])
    return float(lo), float(hi), len(z)


def contrasts(df, boot):
    T = df.set_index('Model')
    rows = []
    for x, y in PAIRS:
        bx, by = boot['boot'][f'{x}_r2'], boot['boot'][f'{y}_r2']
        for col in df.columns[1:]:
            if col == 'Overall AP':
                z = (np.array(bx['obj']) + np.array(bx['part']) - np.array(by['obj']) - np.array(by['part'])) / 2
            else:
                z = np.array(bx[boot_key(col)], dtype=float) - np.array(by[boot_key(col)], dtype=float)
            lo, hi, n = ci(z)
            rows.append(dict(protocol=PROTOCOL, comparison=f'{x} − {y}', metric=col, delta=float(T.loc[x, col] - T.loc[y, col]), ci95_low=lo, ci95_high=hi, B=n, excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def attributes(run, boot):
    gt = load(GT)
    idmap = {a['id']: a['name'] for a in gt['attributes']}
    kind = {idmap[i]: t for t, ids in gt['attr_type_to_attr_idxs'].items() for i in ids}
    cats = {c['id']: c['name'] for c in gt['categories']}
    positives = {('Object', a): 0 for a in idmap.values()} | {('Part', a): 0 for a in idmap.values()}
    for ann in gt['annotations']:
        lv = 'Part' if ':' in cats[ann['category_id']] else 'Object'
        for i in ann.get('attribute_ids', []):
            positives[(lv, idmap[i])] += 1
    per = {c: load(run / f'eval_val_boot/{c}_r2/val/official_results.json')['hierarchical'] for c in MODELS}
    rows, key_rows = [], []
    for level, field, prefix in [('Object', 'obj-attr-AP-per-class', 'obj_attr::'), ('Part', 'obj-attr-part-AP-per-class', 'part_attr::')]:
        for name in sorted(idmap.values()):
            row = dict(protocol=PROTOCOL, level=level, type=kind.get(name, ''), attribute=name, positive_ROIs=positives[(level, name)], **{c: per[c][field].get(name, float('nan')) for c in MODELS})
            rows.append(row)
            if name in KEY_ATTRS and prefix + name in boot['boot']['M1_r2']:
                entry = dict(protocol=PROTOCOL, level=level, attribute=name, positive_ROIs=row['positive_ROIs'], **{c: row[c] for c in MODELS})
                for x, y in [('M1', 'M2'), ('M1', 'P8'), ('M2', 'P8'), ('C0', 'M1'), ('C0', 'M2')]:
                    z = np.array(boot['boot'][f'{x}_r2'][prefix + name], dtype=float) - np.array(boot['boot'][f'{y}_r2'][prefix + name], dtype=float)
                    lo, hi, _ = ci(z)
                    entry.update({f'{x}−{y}': row[x] - row[y], f'{x}−{y}_lo': lo, f'{x}−{y}_hi': hi})
                key_rows.append(entry)
    return pd.DataFrame(rows), pd.DataFrame(key_rows)


def sizes(out):
    data = pd.concat([pd.read_csv(out / f'roi_size_{c}.csv') for c in MODELS], ignore_index=True)
    rows = []
    for x, y in [('M1', 'M2'), ('M1', 'P8'), ('M2', 'P8'), ('C0', 'M1'), ('C0', 'M2'), ('C0', 'P8')]:
        for lv in ('Object', 'Part'):
            for name in data.size_bin.unique():
                a = data[(data.model == x) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                b = data[(data.model == y) & (data.level == lv) & (data.size_bin == name)].iloc[0]
                rows.append(dict(protocol=PROTOCOL, comparison=f'{x} − {y}', level=lv, size_bin=name, roi_count=int(a.roi_count), **{m: float(a[m] - b[m]) for m in ('AP', 'Color', 'Pattern', 'Material', 'Transparency')}))
    data.insert(0, 'protocol', PROTOCOL)
    return data, pd.DataFrame(rows)


def curves(run):
    rows = []
    for c in ('M1', 'M2'):
        for e in range(1, 11):
            h = load(run / f'eval_val/{c}_r2/ep{e}/official_results.json')['hierarchical']
            rows.append(dict(protocol=PROTOCOL, model=c, epoch=e, obj=h['obj-attr-AP'], part=h['obj-attr-part-AP'], mean=(h['obj-attr-AP'] + h['obj-attr-part-AP']) / 2))
    return pd.DataFrame(rows)


def responses(run):
    frames = []
    for name in ('initial', 'P8', 'M1', 'M2'):
        t = pd.read_csv(run / f'response/{name}.csv')
        t.insert(0, 'model', name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def training(run):
    out = {}
    for c in ('M1', 'M2'):
        cfg = load(run / c / 'config.json')
        done = load(run / c / 'done.json')
        log = pd.read_json(run / c / 'logs/train.jsonl', lines=True)
        checks = [json.loads(s) for s in (run / c / 'logs/objective_checks.jsonl').read_text().splitlines()]
        diag = [json.loads(s) for s in (run / c / 'logs/diag.jsonl').read_text().splitlines()]
        out[c] = dict(data_budget=cfg['data_budget'], lambda_cue=cfg['lambda_cue'], tau=cfg['tau'], n_params_total=cfg['n_params_total'], n_params_new=cfg['n_params_new'], time_s=done['time_s'], peak_gib=done['peak_gib'], median_step_seconds=float(log.step_seconds.median()), last_log=log.iloc[-1].to_dict(), objective_checks=checks, diag=diag)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', required=True)
    a = ap.parse_args()
    run, base = Path(a.run), Path(a.base)
    out = run / 'results/val_gt_roi_oracle'
    out.mkdir(parents=True, exist_ok=True)
    ap_table = table(run)
    ap_table.insert(0, 'protocol', PROTOCOL)
    ap_table.to_csv(out / 'paco_ap_val.csv', index=False)
    plain = ap_table.drop(columns=['protocol'])
    boot = load(run / 'boot_val.json')
    pair = contrasts(plain, boot)
    pair.to_csv(out / 'contrasts_val.csv', index=False)
    attrs, key = attributes(run, boot)
    attrs.to_csv(out / 'attribute_ap_val.csv', index=False)
    key.to_csv(out / 'key_attributes_val.csv', index=False)
    size_data, size_delta = sizes(out)
    size_data.to_csv(out / 'roi_size_ap_val.csv', index=False)
    size_delta.to_csv(out / 'roi_size_contrasts_val.csv', index=False)
    curve = curves(run)
    curve.to_csv(out / 'val_epoch_curves.csv', index=False)
    resp = responses(run)
    resp.to_csv(out / 'response_summary.csv', index=False)
    train = training(run)
    (out / 'training_summary.json').write_text(json.dumps(train, indent=1, default=float))
    selection = load(run / 'selection_val_all.json')
    summary = dict(protocol=PROTOCOL, selection=selection, ap={r['Model']: {k: v for k, v in r.items() if k != 'Model'} for r in plain.to_dict('records')}, contrasts=pair.to_dict('records'))
    (out / 'summary_val.json').write_text(json.dumps(summary, indent=1, default=float))
    lines = [f'# {PROTOCOL}', '', '## AP', md(plain), '', '## Contrasts', md(pair[pair.metric.isin(['Overall AP', 'Object AP', 'Part AP', 'Obj Pattern', 'Obj Material', 'Part Pattern', 'Part Material', 'Obj Color', 'Part Color', 'Obj Transparency', 'Part Transparency'])].drop(columns=['protocol', 'B'])), '', '## Key attributes', md(key.drop(columns=['protocol'])), '', '## Epoch curves', md(curve.drop(columns=['protocol'])), '', '## ROI size contrasts', md(size_delta.drop(columns=['protocol'])), '']
    (out / 'tables.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
