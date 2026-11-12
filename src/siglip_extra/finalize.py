"""Consolidate completed experiments without changing models or evaluations."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.siglip_pm.report import md


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--extra', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    base, extra, out = map(Path, [args.base, args.extra, args.out])
    out.mkdir(parents=True, exist_ok=True)
    load = lambda p: json.loads(p.read_text())
    assert load(base / 'all_done.json')['status'] == 'done'
    assert load(base / 'diagnostics_done.json')['status'] == 'done'
    assert load(extra / 'done.json')['control_completed']
    r2 = pd.concat([pd.read_csv(base / 'results/paco_ap.csv'),
                    pd.read_csv(extra / 'results/r2_ap.csv').query('Model in ["E1", "E2", "Continue"]')], ignore_index=True)
    means = pd.concat([pd.read_csv(base / 'results/mean_linear_ap.csv').drop(columns='Epoch'),
                       pd.read_csv(extra / 'results/mean_probe_ap.csv').query('Model in ["E1", "E2", "Continue"]')], ignore_index=True)
    r2.to_csv(out / 'r2_ap.csv', index=False)
    means.to_csv(out / 'mean_probe_ap.csv', index=False)
    rc = pd.concat([pd.read_csv(base / 'results/contrasts.csv'),
                    pd.read_csv(extra / 'results/r2_contrasts.csv')], ignore_index=True).drop_duplicates(['comparison', 'metric'])
    mc = pd.read_csv(extra / 'results/mean_probe_contrasts.csv')
    boot = load(base / 'mean/boot_test.json')
    table = means.set_index('Model')
    mapping = {'Object AP': 'obj', 'Part AP': 'part'}
    for level in ['Obj', 'Part']:
        for label, field in [('Color', 'color'), ('Pattern', 'pattern_marking'), ('Material', 'material'), ('Transparency', 'transparency')]:
            mapping[level + ' ' + label] = ('obj_' if level == 'Obj' else 'part_') + field
    additions = []
    for x, y in [('C0', 'B0'), ('P', 'M'), ('P', 'B0'), ('M', 'B0')]:
        for metric, field in mapping.items():
            values = np.array(boot['boot'][x + '_linear'][field]) - np.array(boot['boot'][y + '_linear'][field])
            lo, hi = np.percentile(values, [2.5, 97.5])
            additions.append(dict(comparison=f'{x} − {y}', metric=metric,
                                  delta=float(table.loc[x, metric] - table.loc[y, metric]),
                                  ci95_low=float(lo), ci95_high=float(hi), B=boot['B']))
    mc = pd.concat([mc, pd.DataFrame(additions)], ignore_index=True)
    rc.to_csv(out / 'r2_contrasts.csv', index=False)
    mc.to_csv(out / 'mean_probe_contrasts.csv', index=False)
    focus = pd.concat([
        pd.read_csv(extra / 'results/focus_attributes.csv'),
        pd.read_csv(base / 'results/focus_attributes.csv').query('model in ["M", "B0"]').assign(reader='R2'),
    ], ignore_index=True)
    focus.to_csv(out / 'focus_attributes.csv', index=False)
    checkpoints = []
    for root, models in [(base, ['P', 'M']), (extra, ['E1', 'E2', 'Continue'])]:
        for model in models:
            d = load(root / model / 'done.json')
            assert d['teacher_unchanged']
            assert d.get('steps', d.get('additional_steps')) == (12000 if root == base else 4000)
            p = root / model / 'checkpoints/ema_last.pt'
            checkpoints.append(dict(model=model, path=str(p.resolve()), bytes=p.stat().st_size,
                                    sha256=sha256(p), completion=d))
    source = load(extra / 'source_checkpoint.json')
    assert checkpoints[0]['sha256'] == source['sha256']
    assert {c['completion']['init_sha256'] for c in checkpoints[2:]} == {source['sha256']}
    files = sorted(Path('src/siglip_pm').glob('*.py')) + sorted(Path('src/siglip_extra').glob('*.py'))
    provenance = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      note='Post-run source manifest; original run provenance and checkpoints are preserved.',
                      files={str(p): sha256(p) for p in files}, checkpoints=checkpoints)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    result = dict(status='all_requested_stages_complete', physical_GPUs=[2, 3],
                  base=str(base.resolve()), extra=str(extra.resolve()),
                  R2=r2.to_dict('records'), mean_probe=means.to_dict('records'),
                  R2_contrasts=rc.to_dict('records'), mean_contrasts=mc.to_dict('records'),
                  focus_attributes=focus.to_dict('records'), checkpoints=checkpoints,
                  continuation='No extra training started: completed plan; awaiting additional budget/scope.',
                  original_deadline_KST='2026-11-09 09:58:58+09:00')
    (out / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    important = rc[rc.comparison.isin(['C0 − B0', 'P − M', 'C0 − P', 'E1 − Continue', 'E2 − Continue']) & rc.metric.isin(['Object AP', 'Part AP', 'Obj Material', 'Obj Pattern'])]
    key_mean = mc[mc.comparison.isin(['C0 − P', 'C0 − E1', 'C0 − E2', 'E1 − Continue', 'E2 − Continue']) & mc.metric.isin(['Obj Material', 'Obj Pattern', 'Part Material', 'Part Pattern'])]
    focus_obj = focus[(focus.reader == 'R2') & (focus.level == 'Object')]
    focus_table = focus_obj.pivot(index=['attribute', 'positive_ROIs', 'positive_categories'], columns='model', values='AP').reset_index()
    focus_table = focus_table[['attribute', 'positive_ROIs', 'positive_categories', 'P', 'E1', 'E2', 'Continue', 'C0']]
    gpu = []
    for root in [base, extra]:
        gpu.extend([v for v in load(root / 'gpu_summary.json')['phases'] if v['stage'].startswith('train_')])
    lines = [
        '# SigLIP2 P/M 및 후속 E1/E2 완료 결과', '',
        '기존 요청 범위는 모두 정상 완료했습니다. 기본 P/M 및 B0/C0 평가는 11월9일03:46 KST, E1/E2·Continue와 Mean Probe 평가는06:43 KST, 나머지 M/B0 Mean Probe·ROI 크기 진단은07:20 KST에 끝났습니다. 확인 시점 GPU2/3에는 실행 중인 작업이 없습니다.', '',
        '## 핵심 결론', '',
        '- P는 M보다 Object +0.99 / Part +0.90 AP이지만 두 paired95% CI 모두0을 포함합니다. 이번 seed0만으로 구조 우위를 확정할 수 없습니다.',
        '- C0−P는 Object +4.99, Object Pattern +8.67, Object Material +3.86 AP입니다. Mean Probe에서도 Material +5.60 [2.39,7.39], Pattern +9.92 [6.32,12.92] AP의 Crop 우위가 남습니다.',
        '- E1/E2는 이번4k 조건에서 Material/Pattern 개선 목표를 충족하지 못했습니다. Object Material은 R2와 Mean Probe 모두 원래P보다 낮고, E2 Pattern도 낮습니다. E1의 Mean Pattern +0.44 AP는 CI[-1.27,2.17]로 불확실합니다.',
        '- 같은 추가4k Control보다 E1/E2의 R2 Object Material/Pattern 점추정치가 모두 낮습니다. 다만 관련 CI는0을 포함하므로 확정적인 악화 또는 차이를 주장하지 않습니다. 새 loss의 이점은 입증되지 않았습니다.',
        '- E1이 E2보다 R2 Material +0.89 / Pattern +1.63 AP입니다. E2가 Pattern에서 더 좋다는 가설은 지지되지 않았고, 두 차이의 CI도0을 포함합니다.', '',
        '## R2: Official TEST AP', '', md(r2), '',
        '## Mean-pooled Linear Probe: Official TEST AP', '', md(means), '',
        'Mean Probe는 동일한 train image 통계·선형 head·학습 예산·VAL 선택 규칙을 사용합니다. R2와 Mean Probe 수치는 서로 다른 reader의 결과입니다.', '',
        '## 주요 R2 비교와 paired image95% CI', '', md(important), '',
        '## 주요 Mean 비교와 paired image95% CI', '', md(key_mean), '',
        'Bootstrap은 이미지 단위1000회입니다. CI는 데이터 표본 변동을 나타내며 seed 간 학습 변동과 다중 비교 보정은 포함하지 않습니다.', '',
        '## Object 속성 진단', '', md(focus_table), '',
        'Ceramic은 P32.29, C032.58로 Teacher 자체의 우위가 작습니다. E2의 +1.22 AP도 탐색적 점추정치입니다. Fabric은 E2가P보다 +0.50이지만 C0보다6.58 AP 낮습니다. E1/E2 모두 Striped·Perforated에서P보다 낮아 Crop 격차를 줄이지 못했습니다. Text는 각각+2.39/+1.73이지만 다른 Pattern 속성의 개선으로 일반화할 수 없습니다.', '',
        'Color 변화는 E1−P −0.04, E2−P +0.04로 작고, Part AP는 각각−0.43/−0.63입니다. Part 악화를 확정할 CI 근거는 없지만 성능을 유지하며 Material/Pattern을 개선했다는 결과도 아닙니다.', '',
        '## 학습 및 GPU 사용', '', md(pd.DataFrame(gpu)[['stage', 'gpu', 'wall_seconds', 'utilization_mean_percent', 'peak_memory_used_MiB']]), '',
        'P12k18분10초, M12k15분09초입니다. E1/E2/Continue는 모두 동일P12k EMA SHA256에서 추가4k를 독립 학습했고 각각6분03초/10분51초/5분55초입니다. 위표는 프로세스 시작·종료 비용을 포함한 stage wall time이므로 순수 train 기록과 몇 초 차이가 있습니다.', '',
        md(pd.read_csv(extra / 'results/training_summary.csv')), '',
        '기본P/M의 COCO dev teacher cosine은 각각0.7327→0.9258 /0.7330→0.9245입니다. 추가 실험의 고정48-image dev에서는 E1이0.9260→0.9002로 하락했고 E2/Continue는약0.9259를 유지했습니다. 서로 다른 dev 집합의 숫자를 직접 뺄셈하지 않습니다.', '',
        'E1의 InfoNCE는2.111→1.400으로 감소했지만 평균 cosine이 악화됐고 Material AP 개선으로 이어지지 않았습니다. 같은 category 전체 후보 retrieval은0.7874→0.8117로 개선됐지만, teacher-similarity 필터 후 retrieval은0.9649→0.9316으로 낮아져 모든 판별 지표가 좋아진 것은 아닙니다. E2 Local cosine은0.9105→0.9145로 소폭 개선됐지만 Pattern AP 개선으로 이어지지 않았습니다.', '',
        '모든 teacher unchanged 검증과 smoke/gradient/finite 검사가 통과했고, 실패 stage 기록은 없습니다. GPU2=P/E1/Continue, GPU3=M/E2의 단일GPU 학습이며 DDP를 사용하지 않았습니다.', '',
        'P/M 저장 feature 원소 수는동일1,327,104이지만 parameter는103,008,393 /94,360,322, encoder GFLOPs는152.78 /112.13으로 다릅니다. 전체image feature extraction 측정치는8.29 /5.56ms입니다. normalization/activation 등을 제외한 FLOPs convention은기본보고서에 명시돼 있습니다.', '',
        '## ROI 크기 및 DINOv3 해석', '',
        '작은 Object ROI(384 canvas에서16×16 token 면적 기준4–16개)의 C0−P Pattern 격차는17.92 AP로 큽니다. 크기별 category 구성 차이가 있으므로 크기의 인과 효과로 해석하지 않습니다. Part Material은 여러 크기 구간에서P가Crop보다 좋아 Teacher 우위가 보편적이지 않습니다.',
        'SigLIP2 P/M은G1 선행 학습 없이 같은 공식 pretrained encoder에서 시작했습니다. G1을 사용한 기존DINOv3의 backbone-only 재현이 아닙니다. 공통점은Crop 대비Pattern/Fabric 격차이며, 차이점은SigLIP2 Ceramic에서Crop 우위가 거의 없다는 점입니다. DINOv3 Mean M2의5.46 AP와 이번 official AP를 직접 비교하지 않습니다.', '',
        '## 완료 범위와 재개', '',
        '요청한P/M 공통12k, E1/E2·Control 추가4k, 독립R2·Mean Probe, official VAL 선택/Test, paired bootstrap, 속성별positive 수, ROI 크기 진단이 완료됐습니다. 기존 결과와 checkpoint를 보존했습니다. 추가 학습은 별도 예산·범위를 확인한 뒤 새 출력 폴더에서 진행해야 합니다.', '',
        '`bash output/2026-11-08/siglip2_pm_extra_s0/scripts/resume.sh`는 완료 stage를 건너뛰는 재개 명령입니다. 이것은 추가4k를 새로 시작하는 명령이 아닙니다. Encoder별 checkpoint와 source SHA256은이폴더 provenance.json 및원래 run폴더에 있습니다.', '',
        '다음 검증의 핵심 가설은 높은 평균 cosine이 Crop의 속성 판별 정보를 보존하기에 충분한가입니다. 현재결과는충분하지않음을시사하지만, reader 학습 변동·추가학습량·ROI크기별 분포를 통제한 추가 검증이 필요합니다. 이미확인한Test 속성을이용해 loss hyperparameter를 고르지 않습니다.', '',
        '## 원본 산출물', '',
        f'- 기본 보고서: `{base}/README.md`',
        f'- 후속 보고서: `{extra}/README.md`',
        f'- 후속 Mean CI 그림: `{extra}/figures/mean_relative_ap.png`',
        f'- Object 속성 그림: `{extra}/figures/focus_attributes.png`',
        '- 이 폴더의CSV/JSON은기존결과를모은것이며 모델·예측·epoch선택을변경하지않았습니다.', '',
    ]
    (out / 'README.md').write_text('\n'.join(lines))
    print('Saved consolidated results:', out)


if __name__ == '__main__':
    main()
