"""Measured-only, standalone report for the two fixed partial projections."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.fgtrain.common import ROOT, write_json
from src.fgtrain.n1 import metrics
from src.fgtrain.p1 import read
from src.fgtrain.partial_rules import ALPHAS
from src.fgtrain.update_report import CSS, esc, table, value, delta, lines, distribution, summarize


DESTINATION = ROOT / 'results/u3_u4_partial_projection'
COLORS = dict(Original='#17324f',B0='#a58a91',P1='#a3aeb8',P2='#327a8b',
              U1='#3b6197',U2='#9293a2',U3='#198c95',U4='#a96f3a')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
    'svg.hashsalt':'u34-partial','axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.17})


def figure(fig,name,caption):
    path=DESTINATION/f'{name}.svg'
    fig.savefig(path,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    uri='data:image/svg+xml;base64,'+base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="{uri}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'


def scores(root,protocol):
    old=read(Path(protocol['prior_run'])/'results.json')['metrics']
    result={n:dict(old[n]) for n in ('Original','B0','P1','P2','U1','U2')}
    ade=read(root/'ade/done.json')['results'] if (root/'ade/done.json').exists() else {}
    for name in ALPHAS:
        result[name]={}
        if (root/name/'selection.json').exists():
            selection=read(root/name/'selection.json')
            path=root/name/'official'/f"ep{selection['epoch']}"/'official_results.json'
            raw=read(path)
            assert raw['split']=='val' and raw['n_eval_images']==2410 and raw['alignment_mismatch']==0
            result[name]=dict(metrics(path),reader_epoch=selection['epoch'])
        if name in ade:
            result[name]['miou']=ade[name]['mIoU']
    return result


def gradient_data(root,protocol):
    prior=Path(protocol['prior_run'])
    summaries,logs={},{}
    for name,location in [('U3',root),('U4',root),('U1',prior),('U2',prior)]:
        summaries[name],logs[name]=summarize(location,name)
        for label,lo,hi in [('early',1,4000),('middle',4001,8000),('late',8001,12000)]:
            records=[r for r in logs[name] if lo<=r['step']<=hi]
            if not records:
                continue
            phase=summaries[name]['phases'][label]
            for key in ('crop_patch_cosine','rho','lambda_eff','patch_drift'):
                phase[key]=distribution([r[key] for r in records if r[key] is not None])
    p2=lines(Path(protocol['p2_run'])/'P2/train_log.jsonl')
    previous=0
    weighted=0.
    for row in p2:
        weighted+=(row['step']-previous)*row['preserve']
        previous=row['step']
    p2_drift=weighted/previous if previous else None
    old=read(prior/'preflight.json')
    summaries['P2']=dict(alpha=0.,rho=1.,lambda_eff=protocol['lambda_preserve'],
        patch_drift_mean_from_existing_window_logs=p2_drift,
        conflict_frequency=old['P2_conflict_frequency'],
        conflict_frequency_scope=f"Final P2 EMA diagnosis, {old['P2_final_batch_count']} fixed mini-batches; not full training",
        training_log_steps=previous)
    return summaries,logs


def gradient_plots(logs):
    fig,axes=plt.subplots(2,3,figsize=(11.6,6.1),layout='constrained')
    keys=['conflict','crop_patch_cosine','rho','lambda_eff_ratio','patch_drift','composite_norm']
    titles=['Conflict frequency','Crop-Patch gradient cosine','Partial crop norm ratio rho',
            'Effective lambda / fixed lambda','Uniform patch feature drift','Composite gradient norm']
    for name in ('U3','U4','U1','U2'):
        rows=logs[name]
        if not rows:
            continue
        chunks=np.array_split(np.arange(len(rows)),min(60,len(rows)))
        steps=[np.mean([rows[i]['step'] for i in c]) for c in chunks]
        for ax,key in zip(axes.flat,keys):
            values=[np.mean([rows[i][key] for i in c if rows[i][key] is not None]) for c in chunks]
            ax.plot(steps,values,color=COLORS[name],label=name,alpha=.88)
    for ax,title in zip(axes.flat,titles):
        ax.set(xlabel='Optimizer step',ylabel=title)
        ax.legend(frameon=False,fontsize=8)
    return figure(fig,'gradient_timelines','Step별 Raw Log를 최대60개 연속 구간으로 평균. U2 lambda_eff는 가상 U1 등가 계수이며 실제 적용 계수는 고정 lambda입니다.')


def dominates(a,b):
    return a['overall']>=b['overall'] and a['miou']>=b['miou'] and (a['overall']>b['overall'] or a['miou']>b['miou'])


def pareto_info(rows):
    ready={n:rows[n] for n in ('P2','U3','U4','U1','U2') if 'overall' in rows[n] and 'miou' in rows[n]}
    frontier=[n for n in ready if not any(dominates(v,ready[n]) for k,v in ready.items() if k!=n)]
    gains={n:[r for r in ('P2','U1','U2') if dominates(ready[n],ready[r])] for n in ALPHAS if n in ready}
    return dict(available=list(ready),frontier=frontier,new_point_dominates_references=gains,
                definition='Exact unrounded point estimates of Overall AP and mIoU; no statistical or tolerance-based equivalence')


def tradeoff_plot(rows,pareto):
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,key,title in zip(axes,('overall','obj_pattern','part_pattern'),
                            ('PACO Overall AP','Object Pattern AP','Part Pattern AP')):
        for n in pareto['available']:
            r=rows[n]
            ax.scatter(r[key],r['miou'],s=67,color=COLORS[n],zorder=3)
            ax.annotate(n,(r[key],r['miou']),xytext=(5,5),textcoords='offset points',fontsize=9)
        ax.axhline(rows['Original']['miou'],color='#9baabc',lw=.8,ls=':',label='Original ADE')
        ax.set(xlabel=title,ylabel='ADE20K mIoU')
    front=sorted(pareto['frontier'],key=lambda n:rows[n]['overall'])
    axes[0].plot([rows[n]['overall'] for n in front],[rows[n]['miou'] for n in front],
                 color='#96aabe',ls='--',lw=1.3,zorder=1)
    return figure(fig,'paco_ade_pareto','좌측 점선은 P2/U3/U4/U1/U2의 Overall-ADE 점추정 Pareto Frontier입니다. 우측 Pattern 비교에는 이 Frontier를 적용하지 않습니다. 미완료 조건은 표시하지 않습니다.')


def strength_plot(summaries):
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.1),layout='constrained')
    entries=[('P2',0.),('U3',.25),('U4',.5),('U1',1.)]
    for n,a in entries:
        s=summaries[n]
        vals=([s['rho'],s['lambda_eff'],s['patch_drift_mean_from_existing_window_logs']] if n=='P2' else
              [s.get('rho_on_projection',{}).get('mean'),s.get('lambda_eff',{}).get('mean'),s.get('patch_drift',{}).get('mean')])
        for ax,v in zip(axes,vals):
            if v is not None:
                ax.scatter(a,v,color=COLORS[n],s=48)
                ax.annotate(n,(a,v),xytext=(3,5),textcoords='offset points',fontsize=8)
    for ax,title in zip(axes,['Mean rho on projected steps','Mean effective lambda','Mean training patch drift']):
        ax.set(xlabel='Fixed alpha',ylabel=title,xticks=[0,.25,.5,1])
    return figure(fig,'projection_strength','각 점은 별도의 학습 궤적입니다. alpha에 대한 단조성을 가정하지 않습니다. P2 Drift는 기존50-step Window Mean의 Step 가중 평균입니다.')


def report(root):
    root=Path(root).resolve()
    DESTINATION.mkdir(parents=True,exist_ok=True)
    protocol=read(root/'training_protocol.json')
    rows=scores(root,protocol)
    summaries,logs=gradient_data(root,protocol)
    pareto=pareto_info(rows)
    complete=all('miou' in rows[n] and 'overall' in rows[n] for n in ALPHAS)
    differences={n:{ref:{k:rows[n][k]-rows[ref][k] for k in
        ('overall','object','part','obj_pattern','part_pattern','obj_color','part_color','obj_material','part_material','miou')
        if k in rows[n]} for ref in ('P2','U1','U2','Original')} for n in ALPHAS}
    exports={n:read(root/n/'export/export_verification.json') for n in ALPHAS
             if (root/n/'export/export_verification.json').exists()}
    configs={n:read(root/n/'config.json') for n in ALPHAS if (root/n/'config.json').exists()}
    preflight=read(root/'preflight.json') if (root/'preflight.json').exists() else {}
    unit=read(root/'partial_unit_tests.json') if (root/'partial_unit_tests.json').exists() else {}
    data=dict(status='done' if complete else 'running',protocol=protocol,metrics=rows,
        gradients=summaries,delta_pp=differences,pareto=pareto,preflight=preflight,
        unit_tests=unit,exports=exports,training_configs=configs,
        matched_training=read(root/'matched_training_verification.json') if (root/'matched_training_verification.json').exists() else None)
    write_json(root/'results.json',data)
    write_json(DESTINATION/'results.json',data)
    columns=['overall','object','part','obj_pattern','part_pattern','obj_color','part_color','obj_material','part_material','miou']
    order=['Original','B0','P1','P2','U3','U4','U1','U2']
    (DESTINATION/'combined_results.tsv').write_text('Method\t'+'\t'.join(columns)+'\n'+'\n'.join(
        n+'\t'+'\t'.join(str(rows[n].get(k,'')) for k in columns) for n in order)+'\n')
    score_table=table(['Method','Overall AP','Object AP','Part AP','Obj Pattern','Part Pattern',
                       'Obj Color','Part Color','Obj Material','Part Material','ADE mIoU'],
                     [(n,*[value(rows[n].get(k)) for k in columns]) for n in order])
    compact=table(['Method','alpha','Overall AP','Obj Pattern','Part Pattern','ADE mIoU'],
        [(n,{'Original':'—','P2':'0','U3':'.25','U4':'.50','U1':'1','U2':'Norm control'}[n],
          *[value(rows[n].get(k)) for k in ('overall','obj_pattern','part_pattern','miou')])
         for n in ('Original','P2','U3','U4','U1','U2')])
    delta_table=table(['Method','Reference','Overall Δpp','Obj Pattern Δpp','Part Pattern Δpp','ADE Δpp'],
        [(n,ref,*[delta(differences[n][ref].get(k)) for k in ('overall','obj_pattern','part_pattern','miou')])
         for n in ALPHAS for ref in ('P2','U1','U2')])
    grad_rows=[('P2','0',value(100*summaries['P2']['conflict_frequency'],2)+'% (8-batch diagnosis)',
        '1.000',value(protocol['lambda_preserve'],6),value(summaries['P2']['patch_drift_mean_from_existing_window_logs'],6))]
    for n in ('U3','U4','U1','U2'):
        s=summaries[n]
        grad_rows.append((n,{'U3':'.25','U4':'.50','U1':'1','U2':'Norm control'}[n],
            value(100*s['conflict_frequency'],2)+'%' if s.get('steps_logged') else '진행 중',
            value(s.get('rho_on_projection',{}).get('mean'),4),
            value(s.get('lambda_eff',{}).get('mean'),6),value(s.get('patch_drift',{}).get('mean'),6)))
    grad_table=table(['Method','alpha','Conflict frequency','Mean rho | projected','Mean lambda_eff','Mean patch drift'],grad_rows)
    phase_table=table(['Method','Phase','Steps','Conflict %','Mean cosine','Mean rho','Mean lambda_eff','Patch drift'],
        [(n,p,s['n'],value(100*s['conflict_frequency'],2),value(s['crop_patch_cosine']['mean'],4),
          value(s['rho']['mean'],4),value(s['lambda_eff']['mean'],4),value(s['patch_drift']['mean'],6))
         for n in ('U3','U4','U1','U2') for p,s in summaries[n].get('phases',{}).items()])
    distribution_rows=[]
    for n in ('U3','U4','U1','U2'):
        for key in ('rho_on_projection','lambda_eff','crop_patch_cosine','composite_norm','patch_drift'):
            d=summaries[n].get(key)
            if d and d['n']:
                q=d['quantiles']
                distribution_rows.append((n,key,d['n'],value(d['mean'],5),value(q[1],5),value(q[3],5),value(q[5],5)))
    distributions=table(['Method','Metric','n','Mean','P05','Median','P95'],distribution_rows)
    block_names=sorted(set().union(*(set(summaries[n].get('block_cosine',{})) for n in ('U3','U4','U1','U2'))))
    blocks=table(['Encoder group','U3 cosine','U4 cosine','U1 cosine','U2 cosine'],
        [(b,*[value(summaries[n].get('block_cosine',{}).get(b,{}).get('mean'),4)
               for n in ('U3','U4','U1','U2')]) for b in block_names])
    if complete:
        summary=''.join(f'<p><strong>{n} (alpha={ALPHAS[n]:.2f})</strong>: P2 대비 Overall {delta(differences[n]["P2"]["overall"])}pp, '
            f'Object Pattern {delta(differences[n]["P2"]["obj_pattern"])}pp, Part Pattern {delta(differences[n]["P2"]["part_pattern"])}pp, '
            f'ADE {delta(differences[n]["P2"]["miou"])}pp.</p>' for n in ALPHAS)
        dominating=pareto['new_point_dominates_references']
        if any(dominating.values()):
            summary+='<p>Overall-ADE 점추정에서 '+', '.join(n+' → '+','.join(refs) for n,refs in dominating.items() if refs)+': 기존 조건을 지배하는 점이 관측됐습니다. Pattern과 작은 차이는 별도로 판단해야 합니다.</p>'
        else:
            summary+='<p>U3/U4 중 P2/U1/U2를 Overall과 ADE 양쪽에서 지배하는 점은 관측되지 않았습니다.</p>'
        if dominates(rows['U3'],rows['U4']):
            choice='U3가 U4보다 Overall-ADE 점추정에서 우세합니다. P2 대비 Delta와 Pattern 손실도 함께 확인해야 합니다.'
        elif dominates(rows['U4'],rows['U3']):
            choice='U4가 U3보다 Overall-ADE 점추정에서 우세합니다. P2 대비 Delta와 Pattern 손실도 함께 확인해야 합니다.'
        else:
            choice='U3/U4 사이에도 명확한 Overall-ADE 지배 관계가 없어, 단일 우수 조건으로 결론 내릴 수 없습니다.'
    else:
        summary='<p>두 조건의 학습·공식 평가를 진행 중입니다. 측정되지 않은 수치는 비워 두며, PACO 점수와 무관하게 ADE20K를 모두 수행합니다.</p>'
        choice='모든 PACO/ADE 평가가 완료된 뒤 Intermediate Trade-off와 Pareto 비교를 판단합니다.'
    pre_rows=[]
    for r in preflight.get('records',[]):
        for a in ('0.25','0.5'):
            c=r['partial_cases'][a]
            pre_rows.append((r['snapshot'],a,value(r['BF16_sum_error']['relative_l2']*100,3),
                             f"{c['direct_formula_error']['relative_l2']:.3e}",
                             f"{c['adaptive_lambda_error']['relative_l2']:.3e}"))
    checks=table(['Snapshot','alpha','BF16 sum relative L2 %','Partial formula error','Adaptive lambda error'],pre_rows)
    export_table=table(['Method','Strict loading','Parameters','Patch output','Forward equality','Additional modules'],
        [(n,e['strict_loading'],e['parameter_count'],'576×768',e['bit_equal_forward'],e['additional_modules']) for n,e in exports.items()])
    artifacts={name:str(root/name) for name in ('source_manifest.json','data_manifest.json','training_protocol.json',
        'preflight.json','partial_unit_tests.json','protected_inputs.json','matched_training_verification.json')}
    write_json(DESTINATION/'artifact_index.json',artifacts)
    reproduction=esc(json.dumps(dict(protocol=protocol,configs=configs,source=read(root/'source_manifest.json'),
        data=read(root/'data_manifest.json'),artifacts=artifacts),indent=2,ensure_ascii=False))
    nav=' · '.join(f'<a href="#{i}">{label}</a>' for i,label in [('summary','1. 요약'),('method','2. Method'),
        ('gradient','3. Gradients'),('paco','4. PACO'),('ade','5. ADE'),('tradeoff','6. Trade-off'),('conclusion','7. 결론')])
    document=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>U3/U4 — Partial Conflict Projection</title><style>{CSS}</style></head><body>
<header><div class="wrap"><div class="kicker">DINOv3-B/16 · Seed 0 · Fixed partial projection</div><h1>Partial Conflict Projection</h1><p>Crop 학습의 자유도와 Dense Feature 보존 사이에 유용한 중간 균형점이 있는가?</p><nav>{nav}</nav></div></header><main class="wrap">
<section id="summary"><h2>1. Executive Summary</h2><div class="box">{summary}</div>{compact}<p>동일 Seed0의 점추정 비교이며, 작은 차이를 안정적인 개선 또는 통계적 동등성으로 간주하지 않습니다. P2/U1/U2는 기존 완료 결과를 재사용합니다.</p></section>
<section id="method"><h2>2. Method &amp; Verification</h2><div class="equation">g_P2 = gc + 0.5 gg + lambda gp<br>gc_partial = gc - alpha min(0, gc·gp / ||gp||²) gp<br>g_Ualpha = gc_partial + 0.5 gg + lambda gp<br>lambda_eff = lambda + alpha max(0, -gc·gp / ||gp||²)<br>g_Ualpha = gc + 0.5 gg + lambda_eff gp</div>
<p>lambda={protocol['lambda_preserve']:.14f}. U3 alpha=0.25, U4 alpha=0.50은 사전 고정입니다. 기존 U1 함수를 수정하지 않고 전체 투영 결과와 gc 사이를 FP32에서 보간합니다. Encoder 전체 Parameter를 고정 순서의 한 벡터로 취급합니다. ||gp||≤1e-7 또는 ||gc||≤1e-12이면 건너뜁니다. 계수는 no_grad 상수입니다.</p>
<p>이 Update는 Adaptive Preservation Weighting과 동등하며 독창적인 알고리즘 또는 엄밀한 Safe Update로 주장하지 않습니다. Global/Preservation Gradient는 변경하지 않습니다. AdamW Momentum·Weight Decay와 Gradient Clip1도 실제 Parameter Update에 영향을 줍니다.</p>
<p>Endpoint alpha0=P2, alpha1=U1은 같은 입력 Gradient에서 Bit Equality로 검증했습니다. 고정 alpha의 직접식·Adaptive Lambda 동등성, Non-conflict, Zero/Tiny, Non-finite 거부를 검증했습니다. 실제 Original/P2 Encoder의 동일 COCO Batch에서도 검사했습니다.</p>{checks}
<p>기존 BF16 Forward/FP32 Gradient 분리 VJP를 유지합니다. 단일 Total Loss Backward와 BF16 반올림 차이는 기존 상대 L2 2% 기준으로 기록합니다. 전체 FP32 합산 기준 검증은 Source Hash가 일치하는 기존 U1의 결과를 재사용합니다. 두 학습은 world1/Accumulation1/명목 Global Batch16이며, 기존 제외 정책 적용 후 동일 Image/ROI Stream을 사용합니다.</p></section>
<section id="gradient"><h2>3. Gradient Analysis</h2>{grad_table}<p class="muted">Mean rho는 Projection이 실제 적용된 Step에서 계산합니다. P2의 Conflict는 최종 EMA의 고정8개 배치 진단이며 전체12k 빈도가 아닙니다. P2 Drift는 기존 학습 로그의 Step 가중 평균, U1/U3/U4는 각 Step의 동일 Uniform Patch Loss 평균입니다. U2의 lambda_eff는 가상 등가 계수입니다.</p>
{gradient_plots(logs)}{strength_plot(summaries)}{phase_table}<p class="muted">초반1–4000 / 중반4001–8000 / 후반8001–12000. 각 Phase rho는 Skip을 포함한 전체 Step 평균입니다.</p>{distributions}{blocks}<p>Block별 Cosine은 진단용이며 Projection은 전체 Encoder에 한 번만 적용합니다. 서로 다른 학습 궤적의 결과이므로 alpha 증가에 대한 단조성은 가정하지 않습니다. Gradient Conflict와 Semantic Information 손실을 동일시하지 않습니다.</p></section>
<section id="paco"><h2>4. PACO Official Validation</h2><p>Official Val2410, GT Box+GT Category Oracle Attribute AP, 동일 bilinear ROIAlign24×24와 R2 Seed0/Init7,10epoch 학습. Final12k EMA를 고정하고 Official Object/Part 평균 최대 Reader Epoch을 선택합니다. 이 실행에서는 Test를 평가하지 않습니다.</p>{score_table}{delta_table}</section>
<section id="ade"><h2>5. ADE20K Frozen Linear Probe</h2><p>기존 src.g1eval.ade를 수정하지 않습니다. Train20210 / Val2000,150classes, Frozen 최종 LN Patch Feature, 독립 Linear768→150 Head, Seed0,40k iterations,world2/globalbatch16,동일 Augmentation·Preprocessing·최종 Iteration 평가를 유지합니다. PACO 성능으로 ADE 평가를 생략하지 않습니다.</p>{compact}</section>
<section id="tradeoff"><h2>6. PACO–ADE20K Trade-off</h2>{tradeoff_plot(rows,pareto)}<p>Overall-ADE Frontier: {esc(', '.join(pareto['frontier']))}. 표의 실제 값을 반올림하기 전 비교한 산술적 Frontier이며, 통계적 우월성이나 모든 Attribute의 개선을 뜻하지 않습니다. 손실 강도·Feature Drift 변화는 Gradient Table과 함께 판단합니다.</p></section>
<section id="conclusion"><h2>7. Conclusion &amp; Reproducibility</h2><h3>관측된 결과</h3>{summary}<p>{choice}</p><h3>해석과 남은 가설</h3><p>유효한 중간 균형점의 근거는 PACO 판별력과 ADE가 함께 유지·개선되는 실제 비교입니다. ADE만 오르고 Pattern이 떨어지면 Preservation–Adaptation Trade-off입니다. Feature Drift 감소만으로 Semantic Information 보존을 입증하지 않습니다. 유리한 점이 보이더라도 추가 Seed에서의 재현과 클래스·경계별 오류 검증이 필요합니다. 추가 alpha, Loss 또는 Method 실험은 자동 실행하지 않습니다.</p>
<h3>Standard Encoder Export</h3>{export_table}<p>85,641,216 Parameters,Original과 동일 Module Tree,Strict Loading,576×768 Patch,Bit-equal Forward,추가 모듈 없음,Full384 이미지당 Encoding1회를 검사합니다. Teacher와 Gradient 처리는 Training-only입니다.</p>
<details><summary>Config · Source · Dataset · Artifact Index</summary><pre>{reproduction}</pre></details>
<p><a href="results.json">전체 측정 JSON</a> · <a href="combined_results.tsv">결과 TSV</a> · <a href="artifact_index.json">Artifact Index</a> · <a href="../../{root.relative_to(ROOT)}/U3/gradient_log.jsonl">U3 Gradient Log</a> · <a href="../../{root.relative_to(ROOT)}/U4/gradient_log.jsonl">U4 Gradient Log</a></p></section>
</main><footer class="footer"><div class="wrap">Fixed alpha0.25 / alpha0.50 Pilot · Standard DINOv3-only inference · Offline HTML</div></footer></body></html>'''
    (DESTINATION/'report.html').write_text(document,encoding='utf-8')
    (root/'README.md').write_text('# U3/U4 Partial Conflict Projection\n\nStatus: '+data['status']+
        '\n\nReport: '+str(DESTINATION/'report.html')+'\n\nExactly U3/U4; no Test, sweeps or automatic follow-up training.\n')
