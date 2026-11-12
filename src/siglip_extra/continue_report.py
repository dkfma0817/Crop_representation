"""Report the predeclared second 4k phase; never select hyperparameters on TEST."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.siglip_extra.report import flat, comparisons, ATTRS
from src.siglip_pm.report import md


def main():
    ap = argparse.ArgumentParser()
    for name in ['base', 'parent', 'run']:
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args()
    base, parent, run = map(Path, [a.base, a.parent, a.run])
    out = run / 'results'
    out.mkdir(exist_ok=True)
    load = lambda p: json.loads(p.read_text())
    tables, deltas, focus = {}, {}, []
    gt = load(Path('/home/irteam/datasets/paco/paco_lvis_v1_test.json'))
    attrs = {v['name']: v['id'] for v in gt['attributes']}
    categories = {v['id']: v['name'] for v in gt['categories']}
    for mean in [False, True]:
        root, key, reader = (run / 'mean', 'linear', 'Mean linear') if mean else (run, 'r2', 'R2')
        rows = []
        for c in ['P', 'E1', 'E2', 'Continue', 'C0']:
            h = load(root / f'eval_test/{c}_{key}/test/official_results.json')['hierarchical']
            rows.append(flat(c, h))
            if c in ['E1', 'E2', 'Continue']:
                old_root = parent / 'mean' if mean else parent
                old = load(old_root / f'eval_test/{c}_{key}/test/official_results.json')['hierarchical']
                rows.append(flat(c + '_4k', old))
            for attr in ATTRS:
                for level, part, field in [('Object', False, 'obj-attr-AP-per-class'), ('Part', True, 'obj-attr-part-AP-per-class')]:
                    positives = [v for v in gt['annotations'] if attrs[attr] in v.get('attribute_ids', []) and (':' in categories[v['category_id']]) == part]
                    focus.append(dict(model=c, reader=reader, level=level, attribute=attr,
                                      AP=h[field].get(attr, float('nan')), positive_ROIs=len(positives),
                                      positive_categories=len({v['category_id'] for v in positives})))
        table = pd.DataFrame(rows)
        table.to_csv(out / f'{key}_ap.csv', index=False)
        boot = load(root / 'boot_test.json')
        contrast = comparisons(table.query('Model in ["P", "E1", "E2", "Continue", "C0"]'), boot, key)
        # Incremental 4k->8k point differences. Bootstrap CIs here would require
        # jointly packing both stages; never substitute unpaired CI arithmetic.
        indexed = table.set_index('Model')
        additions = []
        for c in ['E1', 'E2', 'Continue']:
            for metric in table.columns[1:]:
                additions.append(dict(comparison=f'{c} second phase − {c}_4k', metric=metric,
                                      delta=float(indexed.loc[c, metric]-indexed.loc[c+'_4k', metric]),
                                      ci95_low=np.nan, ci95_high=np.nan, B=0))
        contrast = pd.concat([contrast, pd.DataFrame(additions)], ignore_index=True)
        contrast.to_csv(out / f'{key}_contrasts.csv', index=False)
        tables[key], deltas[key] = table, contrast
    F = pd.DataFrame(focus)
    F.to_csv(out / 'focus_attributes.csv', index=False)
    training = {c: dict(config=load(run / c / 'config.json'), done=load(run / c / 'done.json'),
                       diagnostics=[json.loads(x) for x in (run / c / 'diagnostics.jsonl').read_text().splitlines()]) for c in ['E1', 'E2', 'Continue']}
    data = dict(status='complete', label='P12k + branch4k + branch4k; two restarted phases; not from-scratch20k',
                R2=tables['r2'].to_dict('records'), mean_probe=tables['linear'].to_dict('records'),
                R2_contrasts=json.loads(deltas['r2'].to_json(orient='records')), mean_contrasts=json.loads(deltas['linear'].to_json(orient='records')),
                focus_attributes=json.loads(F.to_json(orient='records')), training=training,
                caution='Second-phase training length fixed before execution; TEST diagnostic only, no hyperparameter selection. Single seed0.')
    (run / 'results.json').write_text(json.dumps(data, indent=2, ensure_ascii=False))
    text = ['# SigLIP2 E1/E2/Control: 두 번째 추가4k 결과', '',
            'E1/E2/Continue는 각자 완료된 첫4k EMA에서 다시4k를 학습했습니다. 동일한 원래P12k에 뿌리를 두지만 이번 phase 입력 checkpoint는 branch마다 다릅니다. 표의 E1/E2/Continue는 누적추가8k, `_4k`는 보존된 이전 결과입니다. 처음부터20k 학습한 모델이 아닙니다.', '',
            '세 branch 모두 fresh AdamW, 동일한600-step LR warmup/cosine4k, EMA .999→1, 원래COCO stream rows4000:8000과 동일한 ROI를 사용했습니다. 새 loss가 있는 branch는 기존과 같은400-step weight warmup을 반복합니다. 가중치·threshold·decoder·seed를 바꾸지 않았습니다.', '',
            '## R2 Official TEST AP', '', md(tables['r2']), '',
            '## Mean Linear Probe Official TEST AP', '', md(tables['linear']), '',
            '## R2 contrasts', '', md(deltas['r2'][deltas['r2'].metric.isin(['Object AP', 'Part AP', 'Obj Material', 'Obj Pattern'])]), '',
            '## Mean contrasts', '', md(deltas['linear'][deltas['linear'].metric.isin(['Obj Material', 'Obj Pattern', 'Part Material', 'Part Pattern'])]), '',
            '이전4k 대비 증분 차이는 점추정치만 보고합니다. 현 phase 모델 간 paired image95% CI는1000회 bootstrap으로 계산했습니다.', '',
            '## Object 속성 (탐색적)', '', md(F[(F.reader=='R2') & (F.level=='Object')]), '',
            '## 해석 원칙', '',
            '동일 누적 학습량의 Continue를 넘는 Material/Pattern 개선 여부가 핵심입니다. Loss/cosine 감소만으로 성공을 판단하지 않습니다. C0 자체의 AP와 한계를 함께 보며 단일seed의 CI는 학습seed 변동을 포함하지 않습니다. 개별 TEST 속성이나 이번 결과로 hyperparameter를 고르지 않습니다.', '',
            '## 재개', '', f'`bash {run}/scripts/resume.sh`', '',
            '원래 P/M 및 첫4k 결과는 변경하지 않았습니다. 새 budget만 사용했습니다. checkpoint는500step마다 optimizer/EMA와 함께 저장합니다.', '']
    (run / 'REPORT.md').write_text('\n'.join(text))
    p = run / 'README.md'
    original = p.read_text().split('<!-- RESULTS -->')[0]
    p.write_text(original + '\n<!-- RESULTS -->\n' + '\n'.join(text))
    print(md(tables['r2']), flush=True)


if __name__ == '__main__':
    main()
