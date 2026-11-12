"""Measured U1/U2 gradient statistics and the final standalone experiment report."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.fgtrain.common import ROOT, write_json
from src.fgtrain.p1 import read
from src.fgtrain.n1 import metrics


DESTINATION = ROOT / 'results/u1_u2_gradient_conflict'
COLORS = dict(U1='#246caa', U2='#84679b', Original='#162e4e', B0='#b86770', P1='#6c98af', P2='#367c85', P3='#9f9fbd')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.grid':True, 'grid.alpha':.17,
                     'axes.titlelocation':'left', 'axes.titleweight':'bold', 'svg.fonttype':'none',
                     'svg.hashsalt':'u12-gradients'})


def lines(path):
    if not path.exists():
        return []
    records = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[row['step']] = row
    return [records[key] for key in sorted(records)]


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, mean=None, quantiles=None)
    return dict(n=len(values), mean=float(values.mean()),
                quantiles=np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist())


def summarize(root, condition):
    rows = lines(root / condition / 'gradient_log.jsonl')
    if not rows:
        return dict(steps_logged=0), rows
    result = dict(steps_logged=len(rows), last_step=rows[-1]['step'],
                  conflict_frequency=float(np.mean([r['conflict'] for r in rows])),
                  projection_frequency=float(np.mean([r['projection_applied'] for r in rows])),
                  nonfinite_steps=int(sum(r['nonfinite'] for r in rows)), phases={})
    for name, low, high in [('early', 1, 4000), ('middle', 4001, 8000), ('late', 8001, 12000)]:
        phase = [r for r in rows if low <= r['step'] <= high]
        if phase:
            result['phases'][name] = dict(n=len(phase), conflict_frequency=float(np.mean([r['conflict'] for r in phase])),
                                          projection_frequency=float(np.mean([r['projection_applied'] for r in phase])))
    conflicted = [r for r in rows if r['conflict']]
    result['rho_on_conflict'] = distribution([r['rho'] for r in conflicted])
    result['rho_on_projection'] = distribution([r['rho'] for r in rows if r['projection_applied']])
    for key in ['lambda_eff','lambda_eff_ratio','crop_patch_cosine','crop_norm','global_norm','patch_norm',
                'composite_norm','clip_input_norm','patch_drift']:
        result[key] = distribution([r[key] for r in rows if r[key] is not None])
    blocks = {}
    for record in lines(root / condition / 'block_gradients.jsonl'):
        for name, data in record['blocks'].items():
            if name == 'composite':
                continue
            value = data['cosine']
            if value is not None:
                blocks.setdefault(name, []).append(value)
    result['block_cosine'] = {name:distribution(values) for name,values in blocks.items()}
    cosines = np.array([r['crop_patch_cosine'] if r['crop_patch_cosine'] is not None else np.nan for r in rows])
    drift = np.array([r['patch_drift'] for r in rows])
    valid = np.isfinite(cosines)
    result['cosine_drift_pearson'] = float(np.corrcoef(cosines[valid],drift[valid])[0,1]) if valid.sum()>2 and np.std(cosines[valid])>0 and np.std(drift[valid])>0 else None
    return result, rows


def esc(value):
    return html.escape(str(value), quote=True)


def value(number, digits=3):
    return '진행 중' if number is None else f'{number:.{digits}f}'


def delta(number):
    return '진행 중' if number is None else f'{number:+.3f}'


def table(headers, rows):
    heading = ''.join(f'<th>{esc(x)}</th>' for x in headers)
    body = ''.join('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in row)+'</tr>' for row in rows)
    return f'<div class="scroll"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'


def details(title, text):
    return f'<details><summary>{esc(title)}</summary>{text}</details>'


def figure(fig, name, caption):
    path = DESTINATION / f'{name}.svg'
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/svg+xml;base64,{encoded}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'


def gradient_figures(logs, summaries):
    if not any(logs.values()):
        return '<p>학습 Gradient 로그를 수집 중입니다.</p>'
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.1), layout='constrained')
    for name, records in logs.items():
        if not records:
            continue
        chunks = np.array_split(np.arange(len(records)), min(48, len(records)))
        step = [np.mean([records[i]['step'] for i in c]) for c in chunks]
        for ax, key in zip(axes, ['conflict','rho','lambda_eff_ratio']):
            data = [np.mean([records[i][key] for i in c]) for c in chunks]
            ax.plot(step, data, label=name, color=COLORS[name])
    for ax, label in zip(axes, ['Conflict frequency','Crop norm ratio rho','U1-equivalent lambda / base lambda']):
        ax.set(xlabel='Optimizer step', ylabel=label)
        ax.legend(frameon=False)
    output = figure(fig, 'gradient_timeline', '전체 Step 로그를 최대48개 연속 구간으로 평균한 추이. U2의 lambda_eff는 U1을 적용했을 때의 계산값입니다.')
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.1), layout='constrained')
    for name, rows in logs.items():
        if not rows:
            continue
        selected = [r for r in rows if r['projection_applied']]
        for ax, key, subset in [(axes[0], 'crop_patch_cosine', rows), (axes[1], 'rho', selected), (axes[2], 'lambda_eff_ratio', rows)]:
            vals = [r[key] for r in subset if r[key] is not None]
            if vals:
                ax.hist(vals, bins=30, density=True, histtype='step', color=COLORS[name], label=name, linewidth=1.8)
    for ax, title in zip(axes, ['Crop–Patch cosine','rho on projected steps','U1-equivalent lambda ratio']):
        ax.set(xlabel=title, ylabel='Density')
        ax.legend(frameon=False)
    output += figure(fig, 'gradient_distributions', '실제 Gradient 분포. rho는 Projection이 적용된 Step만 포함합니다.')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), layout='constrained')
    for name, rows in logs.items():
        if not rows:
            continue
        pick = np.linspace(0, len(rows)-1, min(240, len(rows)), dtype=int)
        axes[0].plot([rows[i]['step'] for i in pick], [rows[i]['composite_norm'] for i in pick], color=COLORS[name], label=name, alpha=.85)
        axes[1].scatter([rows[i]['crop_patch_cosine'] for i in pick], [rows[i]['patch_drift'] for i in pick], label=name, color=COLORS[name], s=9, alpha=.5)
    axes[0].set(xlabel='Step', ylabel='Composite gradient norm before clip')
    axes[1].set(xlabel='Crop–Patch cosine', ylabel='Uniform patch drift')
    for ax in axes:
        ax.legend(frameon=False)
    output += figure(fig, 'norm_and_drift', '동일 Step의 합성 Gradient Norm과 Patch Drift. 시간에 따른 학습 상태 변화가 섞여 있으므로 상관을 원인으로 해석하지 않습니다.')
    names = sorted(set().union(*(set(s.get('block_cosine',{})) for s in summaries.values())))
    if names:
        fig, ax = plt.subplots(figsize=(5.2, 5.1), layout='constrained')
        matrix = np.array([[summaries[n].get('block_cosine',{}).get(k,{}).get('mean',np.nan) for n in ('U1','U2')] for k in names])
        image = ax.imshow(matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax.set(yticks=range(len(names)), yticklabels=names, xticks=[0,1], xticklabels=['U1','U2'])
        ax.grid(False)
        for y in range(len(names)):
            for x in (0,1):
                if np.isfinite(matrix[y,x]):
                    ax.text(x,y,f'{matrix[y,x]:.2f}',ha='center',va='center',fontsize=9)
        fig.colorbar(image, ax=ax, label='Mean Crop–Patch gradient cosine')
        output += figure(fig, 'block_cosines', '50 Step 간격의 Block별 Cosine 평균. Projection은 Encoder 전체에 한 번 적용합니다.')
    return output


def score_rows(root, protocol):
    p23 = Path(protocol['p23_run'])
    baseline = read(p23 / 'results.json')['metrics']
    rows = {('Original' if key == 'Frozen' else key): dict(baseline[key]) for key in ('Frozen','B0','P1','P2','P3')}
    ade = read(root / 'ade/done.json')['results'] if (root / 'ade/done.json').exists() else {}
    for name in ('U1','U2'):
        rows[name] = {}
        if (root / name / 'selection.json').exists():
            selected = read(root / name / 'selection.json')
            path = root / name / 'official' / f"ep{selected['epoch']}" / 'official_results.json'
            official = read(path)
            assert official['split'] == 'val' and official['n_eval_images'] == 2410 and official['alignment_mismatch'] == 0
            rows[name] = dict(metrics(path), epoch=selected['epoch'])
        if name in ade:
            rows[name]['miou'] = ade[name]['mIoU']
    return rows


CSS = '''
:root{color-scheme:light;--ink:#162e4e;--blue:#245f95;--line:#d9e5f0}*{box-sizing:border-box}body{margin:0;background:white;color:var(--ink);font:15px/1.68 "Gothic A1",Inter,system-ui,"Malgun Gothic",sans-serif}.wrap{max-width:1240px;margin:auto;padding:0 26px}header{border-top:7px solid #102a48;background:linear-gradient(135deg,#eef6ff,white);border-bottom:1px solid var(--line)}header .wrap{padding-top:38px;padding-bottom:24px}.kicker{color:var(--blue);letter-spacing:.12em;text-transform:uppercase;font-weight:750;font-size:.78rem}h1{font-size:clamp(2rem,5vw,3.2rem);line-height:1.12;letter-spacing:-.04em;margin:.35em 0}h2{font-size:1.55rem;line-height:1.25;padding-bottom:10px;border-bottom:2px solid #bfd8ee}h3{font-size:1.13rem;margin:22px 0 6px}p{margin:.5em 0 1em}nav{display:flex;gap:10px 18px;flex-wrap:wrap;font-size:.88rem}a{color:#1c6096;text-underline-offset:3px}section{padding:19px 0;scroll-margin-top:16px}.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:10px 0 15px}table{width:100%;border-collapse:collapse;font-size:.84rem;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;border-bottom:1px solid #e7eef6;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eaf4fe}tbody tr:nth-child(even){background:#f7faff}tbody tr:hover{background:#eaf3fc}.box{border-left:4px solid #367aaf;background:#eef6ff;border-radius:0 8px 8px 0;padding:12px 16px;margin:15px 0}.equation{background:#f6f9fc;padding:12px;font-family:ui-monospace,monospace;overflow-wrap:anywhere}figure{border:1px solid var(--line);border-radius:9px;padding:11px;margin:13px 0;break-inside:avoid}img{display:block;width:100%;max-height:530px;object-fit:contain}figcaption,.muted{color:#50687f;font-size:.86rem}figcaption{text-align:center;margin-top:6px}details{border:1px solid var(--line);border-radius:7px;padding:9px 12px;margin:11px 0}summary{font-weight:650;cursor:pointer}pre{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f9fc;padding:12px}.footer{border-top:1px solid var(--line);padding:15px 0;color:#50687f}@media(max-width:720px){.wrap{padding:0 15px}}@media print{nav{display:none}.wrap{max-width:100%;padding:0}figure{break-inside:avoid}}
'''


def report(root):
    root = Path(root)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    protocol = read(root / 'training_protocol.json')
    preflight = read(root / 'preflight.json') if (root / 'preflight.json').exists() else {}
    units = read(root / 'unit_vectors.json') if (root / 'unit_vectors.json').exists() else {}
    rows = score_rows(root, protocol)
    summaries, logs = {}, {}
    for name in ('U1','U2'):
        summaries[name], logs[name] = summarize(root, name)
    exports = {name:read(root / name / 'export/export_verification.json') for name in ('U1','U2') if (root / name / 'export/export_verification.json').exists()}
    differences = {name:{key:v - rows['P2'][key] for key,v in rows[name].items() if key in rows['P2'] and key not in ('epoch',)} for name in ('U1','U2')}
    data = dict(protocol=protocol, preflight=preflight, unit_tests=units, metrics=rows,
                U_minus_P2_pp=differences, gradients=summaries, exports=exports,
                status='done' if all('miou' in rows[n] and 'overall' in rows[n] for n in ('U1','U2')) else 'running')
    write_json(DESTINATION / 'results.json', data)
    write_json(root / 'results.json', data)
    columns = ['overall','object','part','obj_pattern','part_pattern','obj_color','part_color','obj_material','part_material','miou']
    (DESTINATION / 'combined_results.tsv').write_text('Method\t'+'\t'.join(columns)+'\n'+'\n'.join(name+'\t'+'\t'.join(str(r.get(k,'')) for k in columns) for name,r in rows.items())+'\n')
    scores = table(['Method','Overall AP','Object AP','Part AP','Obj Pattern','Part Pattern','Obj Color','Part Color','Obj Material','Part Material','ADE mIoU'],
                   [(n,*[value(r.get(k)) for k in columns]) for n,r in rows.items()])
    delta_table = table(['Method − P2','Overall','Obj Pattern','Part Pattern','ADE mIoU'],
                        [(n,*[delta(differences[n].get(k)) for k in ('overall','obj_pattern','part_pattern','miou')]) for n in ('U1','U2')])
    pre_rows = []
    for r in preflight.get('records', []):
        q = r['U1']
        pre_rows.append((r['snapshot'],r['batch_index'],value(q['crop_norm'],4),value(q['global_norm'],4),value(q['patch_norm'],4),
                         value(q['crop_patch_cosine'],4),q['projection_applied'],value(q['rho'],4),value(q['lambda_eff'],5),value(q['lambda_eff_ratio'],3),
                         value(100*r['P2_BF16_sum_error']['relative_l2'],3)))
    pre_table = table(['Checkpoint','Stream Batch','Crop norm','Global norm','Patch norm','Cosine','Projection','rho','lambda_eff','lambda_eff / base','BF16 sum ΔL2 %'], pre_rows) if pre_rows else '<p>Preflight 측정 중입니다.</p>'
    frequency_rows, distribution_rows, norm_rows, block_rows = [], [], [], []
    for name, s in summaries.items():
        if not s['steps_logged']:
            continue
        phase = [s['phases'].get(k,{}).get('conflict_frequency') for k in ('early','middle','late')]
        frequency_rows.append((name,s['steps_logged'],value(100*s['conflict_frequency'],2),value(100*s['projection_frequency'],2),
                               *[value(100*x,2) if x is not None else '진행 중' for x in phase],s['nonfinite_steps']))
        for key in ('rho_on_conflict','rho_on_projection','lambda_eff','lambda_eff_ratio','crop_patch_cosine','composite_norm'):
            q = s[key]
            quantiles = q['quantiles'] or [None] * 7
            distribution_rows.append((name,key,q['n'],value(q['mean'],5),*[value(x,5) for x in quantiles]))
        norm_rows.append((name,*[value(s[k]['mean'],5) for k in ('crop_norm','global_norm','patch_norm','composite_norm','patch_drift')],value(s['cosine_drift_pearson'],4)))
        block_rows.extend((name,k,q['n'],value(q['mean'],4),*[value(x,4) for x in q['quantiles']]) for k,q in s['block_cosine'].items())
    frequency_table = table(['조건','측정 Step','Conflict %','Projection %','초반 1–4k %','중반 4–8k %','후반 8–12k %','Non-finite'],frequency_rows)
    dist_table = table(['조건','지표','n','평균','Min','P05','P25','중앙값','P75','P95','Max'],distribution_rows)
    norm_table = table(['조건','Crop mean','Global mean','Patch mean','Composite mean','Patch drift mean','Cosine–Drift Pearson'],norm_rows)
    block_table = details('Block별 Cosine 평균 및 분위수',table(['조건','Group','측정 수','평균','Min','P05','P25','중앙값','P75','P95','Max'],block_rows))
    export_table = table(['조건','Strict','Parameters','Patch Shape','Forward Equality','Module Tree','추가 모듈','Encoding / images','EMA SHA256'],
                         [(n,x['strict_loading'],x['parameter_count'],str(x['patch_shape']),x['bit_equal_forward'],x['identical_module_tree'],str(x['additional_modules']),f"{x['encoder_canvases']} / {x['input_images']}",x['state_sha256']) for n,x in exports.items()])
    scatter = ''
    if any(rows[n] for n in ('U1','U2')):
        fig, ax = plt.subplots(figsize=(7.3,4.1), layout='constrained')
        for name, point in rows.items():
            if 'miou' not in point or 'overall' not in point:
                continue
            ax.scatter(point['overall'],point['miou'],s=48 if name in ('U1','U2') else 32,color=COLORS[name])
            ax.annotate(name,(point['overall'],point['miou']),xytext=(4,6),textcoords='offset points',fontsize=9)
        ax.set(xlabel='PACO Official Val Overall AP',ylabel='ADE20K Official Val mIoU',title='PACO–ADE20K trade-off')
        ax.margins(.15)
        scatter = figure(fig,'paco_ade_scatter','각 조건의 최종 EMA와 Val 선택 R2, 최종40k Linear Head 점수. 모두 Seed0 점추정입니다.')
    interpretation = '<p>두 조건의 PACO Val과 ADE20K 결과를 기다리고 있습니다.</p>'
    if data['status'] == 'done':
        u1, u2, p2 = rows['U1'], rows['U2'], rows['P2']
        difference = {key:u1[key]-u2[key] for key in ('overall','miou','obj_pattern','part_pattern')}
        interpretation = f"<p>U1−U2는 Overall {difference['overall']:+.6f}pp, ADE {difference['miou']:+.6f}pp, Object Pattern {difference['obj_pattern']:+.6f}pp, Part Pattern {difference['part_pattern']:+.6f}pp입니다.</p>"
        if u1['overall']>=max(p2['overall'],u2['overall']) and u1['miou']>=max(p2['miou'],u2['miou']):
            interpretation += '<p>이 Seed0 점추정에서는 U1이 P2와 U2의 Overall·ADE 두 축을 유지하거나 높였습니다. 방향 조절이 Crop Norm 축소 이상의 효과를 낼 가능성과 일치합니다.</p>'
        elif all(rows[n]['overall']<=p2['overall'] and rows[n]['miou']<=p2['miou'] for n in ('U1','U2')):
            interpretation += '<p>U1과 U2 모두 P2의 Overall·ADE 두 축보다 높지 않았습니다. 이번 고정 Projection 규칙으로 두 성능의 양립 목표를 달성했다고 판단할 수 없습니다.</p>'
        for name in ('U1','U2'):
            if rows[name]['miou']>p2['miou'] and rows[name]['overall']<p2['overall']:
                interpretation += f'<p>{name}은 P2보다 ADE가 높고 PACO Overall이 낮았습니다. 보존 강화와 Crop 학습 억제 사이의 Trade-off와 일치합니다.</p>'
        interpretation += '<p>차이의 크기와 각 Pattern 변화를 함께 봅니다. 단일 Seed에서의 작은 점수 차이는 안정적인 개선을 입증하지 못합니다. U1/U2 비교는 일반적인 Adaptive Preservation Weighting과의 우열을 검증하지 않습니다.</p>'
    unit_info = details('Unit Test와 실제 Preflight 상세 JSON','<pre>'+esc(json.dumps(dict(unit=units,preflight=preflight),indent=2,ensure_ascii=False))+'</pre>')
    nav = '<nav>'+''.join(f'<a href="#{key}">{label}</a>' for key,label in [('question','1. 연구 질문'),('method','2. Method'),('gradients','3. Gradients'),('paco','4. PACO'),('ade','5. ADE'),('interpretation','6. 해석'),('implication','7. Method Implication')])+'</nav>'
    pre_frequency = value(100*preflight['P2_conflict_frequency'],2) if 'P2_conflict_frequency' in preflight else '측정 중'
    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>U1/U2 Preservation-aware Crop Gradient Projection</title><style>{CSS}</style></head><body>
<header><div class="wrap"><div class="kicker">DINOv3-B/16 · Seed 0 · U-series</div><h1>Preservation-aware<br>Crop Gradient Projection</h1><p>Crop Distillation과 Patch Preservation의 충돌을 조절하면 PACO–ADE20K 양립이 개선되는지 비교합니다.</p>{nav}</div></header><main class="wrap">
<section id="question"><h2>1. Research Question</h2><p>P2는 B0 대비 ADE20K를 {rows['P2']['miou']-rows['B0']['miou']:.3f}pp 회복했고, Original 대비 {rows['P2']['miou']-rows['Original']['miou']:.3f}pp의 격차가 남았습니다. 이번에는 P2의 동일 Loss·Teacher·Architecture에서 Crop Gradient 결합 규칙을 바꾼 U1과 U2를 비교합니다.</p></section>
<section id="method"><h2>2. Method &amp; Verification</h2><div class="equation">g_P2 = gc + 0.5 gg + λ gp<br>gc_proj = gc − min(0, gc·gp / ||gp||²) gp<br>g_U1 = gc_proj + 0.5 gg + λ gp = gc + 0.5 gg + λ_eff gp<br>λ_eff = λ + max(0, −gc·gp / ||gp||²)<br>rho = ||gc_proj|| / (||gc|| + ε)<br>g_U2 = rho gc + 0.5 gg + λ gp</div>
<p>λ={protocol['lambda_preserve']:.12f}. ||gp|| ≤ {protocol['numerical']['gp_norm_min']:.0e} 또는 ||gc|| ≤ {protocol['numerical']['gc_norm_min']:.0e}일 때 Projection을 건너뜁니다. 충돌이 없으면 rho=1, λ_eff=λ입니다. Encoder 전체를 고정된 Parameter Ordering의 한 벡터로 취급합니다. 세 Loss의 Gradient를 같은 배치에서 각각 계산하고 FP32, unscaled 상태로 결합합니다. 계수 계산은 Autograd Graph 밖에서 수행합니다.</p>
<p>두 실험은 각각 world1, Accumulation1이며 동일16-image 전역 Stream과 기존 제외 정책을 사용합니다. Rank별 Local Projection이나 All-reduce는 없습니다. BF16 Forward, FP32 파라미터·Gradient, AdamW,600-step Warmup+Cosine, LLRD0.75, WD0.05, Clip1, EMA를 P2와 맞췄습니다. 독립 Original 초기화에서 12k 학습하고 Final EMA를 평가합니다.</p>
<div class="box"><p>U1은 PCGrad 계열의 비대칭 변형이며 Gradient 수준에서 Adaptive Preservation Weighting과 같습니다. U2는 Projected Crop Gradient의 Norm을 맞추는 대조군입니다. 합성 Gradient Norm과 AdamW 실제 Parameter Update Norm은 맞추지 않습니다. Momentum과 Weight Decay도 실제 Update에 영향을 줍니다.</p></div>
<p>FP32 Gradient 합산과 Projection 식의 Unit Test, U2 Norm Matching, 충돌 없음, Zero/Tiny Preservation, Non-finite 거부, Patch Embedding·12block·최종 Norm Gradient를 검증했습니다. 실제 BF16 경로의 분리 VJP 합산은 Total Loss 단일 VJP와 반올림 차이가 있어, 사전 고정한 상대 L2 2% 기준과 실제 측정값을 아래 표에 기록합니다. 정규화 오류 확인용 전체 FP32 기준 계산은 TF32와 Fused Attention을 끈 Math Reference를 사용합니다. 학습은 P2의 BF16 경로를 사용합니다.</p>{unit_info}</section>
<section id="gradients"><h2>3. Gradient Conflict Analysis</h2><p>P2의 기존 최종12k EMA를 고정 Stream 배치8개에서 측정한 Conflict Frequency는 {pre_frequency}%입니다. P2 6k 가중치는 남아 있지 않아, 사전 진단은 최종 EMA로 수행했습니다. 이 표본 빈도를 P2 전체12k 학습의 빈도로 일반화하지 않습니다.</p>{pre_table}
<h3>U1 / U2 전체 Step 기록</h3>{frequency_table}<p class="muted">초반1–4000, 중반4001–8000, 후반8001–12000입니다. Conflict는 dot&lt;0, Projection은 여기에 Norm 기준을 통과한 경우입니다. 모든 수치는 확보한 Step 수에 대해 계산합니다.</p>{gradient_figures(logs,summaries)}{dist_table}{norm_table}{block_table}
<p>U2의 λ_eff는 같은 Gradient에서 U1을 적용했을 때의 등가 계수입니다. U2의 실제 Preservation 계수는 P2의 λ입니다. Raw Log에는 lambda_applied도 저장합니다. Crop–Patch Conflict는 학습 목표의 충돌을 나타내며, Semantic Information 손상의 직접 측정값은 아닙니다.</p></section>
<section id="paco"><h2>4. PACO Official Validation</h2><p>Official Val2410, GT Box+GT Category Oracle Attribute AP, 동일 bilinear ROIAlign24와 R2 Seed0/Init7,10epochs를 사용합니다. Final12k EMA를 고정하고 Reader Epoch은 Official Object/Part AP 평균 최대값으로 선택합니다.</p>{scores}{delta_table}</section>
<section id="ade"><h2>5. ADE20K Frozen Linear Probe</h2><p>Train20210/Val2000,150classes, Frozen 최종 LN Encoder, 독립 Linear768→150 Head를 사용합니다. 기존 src.g1eval.ade의 Seed0,40k,world2/globalbatch16,동일 Augmentation·Preprocessing,최종 Iteration 평가를 재사용합니다. PACO와 ADE 두 평가를 U1/U2 모두 수행합니다.</p>{scatter}</section>
<section id="interpretation"><h2>6. Joint Interpretation</h2>{interpretation}<p>U1은 전체 Crop–Patch Dot Product가 음수일 때 Patch Gradient 방향의 보존 계수를 늘리는 Update와 같습니다. 따라서 성능 변화는 방향 Projection과 등가 Adaptive Weighting을 함께 고려해 해석합니다. 동일 Norm의 Crop Control U2는 방향 변화의 영향을 조사하지만 전체 Update를 통제한 비교는 아닙니다.</p></section>
<section id="implication"><h2>7. Method Implication</h2><p>관측된 사실은 실제 Conflict 빈도와 Gradient 분포, 두 공식 평가의 점수입니다. Conflict 조절과 성능 변화가 함께 나타나는지는 이 고정 파일럿에서 판단합니다. Semantic-preserving Update가 실제 Semantic 손상을 막는지는 별도 검증이 필요한 가설입니다. 현재 Patch Preservation Objective의 충돌을 줄인 결과만으로 그 효과를 입증하지 않습니다.</p>
<h3>Standard Encoder Export</h3>{export_table}<p>Strict 표준 클래스 Loading,85,641,216 Parameters,동일 Module Tree,576×768 Patch,Forward Equality와 이미지당 Encoding1회를 검사합니다. Projection과 Norm Scaling은 Optimizer Update 계산에만 사용합니다.</p>
{details('Config·Source Hash·Dataset Manifest','<pre>'+esc(json.dumps(dict(protocol=protocol,source=read(root/'source_manifest.json'),data=read(root/'data_manifest.json')),indent=2,ensure_ascii=False))+'</pre>')}
<p><a href="results.json">측정 JSON</a> · <a href="combined_results.tsv">결과 TSV</a> · <a href="../../{root.relative_to(ROOT)}/U1/gradient_log.jsonl">U1 Step Log</a> · <a href="../../{root.relative_to(ROOT)}/U2/gradient_log.jsonl">U2 Step Log</a></p></section>
</main><footer class="footer"><div class="wrap">U1/U2 Gradient Conflict Pilot · Local HTML</div></footer></body></html>'''
    (DESTINATION / 'report.html').write_text(document, encoding='utf-8')
    (root / 'README.md').write_text('# U1/U2 Gradient Conflict Pilot\n\nReport: '+str(DESTINATION / 'report.html')+'\n\nStatus: '+data['status']+'\n')
