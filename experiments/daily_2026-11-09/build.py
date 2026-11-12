"""Build an offline, single-file Korean HTML research report from data.json."""
import base64,html,json,os,re
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
esc=lambda x:html.escape(str(x))

CONTROL={
 'hcf_official_val':'같은 causal checkpoint와 같은 공식 evaluator. 순정 DINOv3·G1-pool·C0 참조를 함께 둡니다.',
 'hier_h0h1_s0':'기존 P384. H0와 H1은 구조·학습을 같게 두고 detail 입력 픽셀만 바꿉니다.',
 'pixeldit_cond_p1p2_s0':'기존 64-token memory의 PixelDiT Hybrid. 초기 encoder·crop/flow/global loss·12k 학습을 고정합니다.',
 'pcond_p1p2_s0':'비슷한 dense memory / spatial AdaLN을 사용한 별도 구현입니다. 위 PixelDiT P1/P2와 checkpoint·결과를 합치지 않습니다.',
 'g1_upscaled_teacher_s0':'P384 crop 정렬 기준선. 같은 초기화·12k recipe에서 공간 teacher loss만 추가합니다.',
 'g1_native768_s0':'G1U. Student 384와 구조·loss를 고정하고 teacher의 실제 RGB 입력만 바꿉니다.',
 'coarse_fine_ablation_s0':'A0 = 기존 양방향 4-stage P384 재사용. 세 구조의 공유 초기 tensor와 학습 recipe를 맞췄습니다.',
 'fine_bottleneck_diag_s0':'D1은 C0 원본 crop과 저해상도 RGB crop을 비교. D2는 같은 frozen A1의 coarse-only / fine-only / fused를 비교합니다.',
 'dinov3_crop_aware_pretrain':'동일 설정으로 새로 학습한 step-0 probe와 기존 FrozenFull을 모두 참조합니다. 두 probe의 변동을 학습 효과와 구분합니다.',
 'siglip-structure':'P4↔P8, Native↔P8, Decode↔Native. 구조마다 달라지는 계산량과 token 수를 함께 봅니다.',
 'siglip-extra-phase2':'같은 추가 4k를 수행한 Continue. E1/E2/Continue는 각자의 이전 16k checkpoint에서 출발합니다.',
 'siglip-localcue':'공식 pretrained 기반 P8 12k. M1은 편집 반응 loss, M2−M1은 native Detail 추가를 비교합니다.',
 'siglip-pixeldit':'같은 초기화·12k optimizer step의 P8. S2−S1은 centered 항 추가 비교이며, 별도의 matched Continue arm은 없습니다.',
 'siglip-completed-compilation':'기존 P/M, Frozen B0/C0, 첫 추가 4k를 모은 표입니다. 오늘의 두 번째 4k와 구분합니다.',
}
TAKE={
 'hcf_official_val':'공식 Test에서도 768의 이득은 남습니다. 다만 C0보다 Object는 낮고 Part는 높아, Overall만으로 Crop 격차를 판단하면 안 됩니다.',
 'prior-generative':'Pure 과제는 직접 crop 정렬을 일관되게 대체하지 못했습니다. Hybrid의 작은 점추정 차이는 이 표만으로 우위가 확정되지 않습니다.',
 'hcf-pixel-followup':'같은 6k control 대비 Val M2 +0.35 [−0.67,+1.64]. PixelDiT 추가 이득은 확인되지 않았습니다.',
 'hier_h0h1_s0':'H1 Test Overall +0.67 [−0.13,+1.19] vs P384. 원본 Detail 입력의 AP 이득을 확인하지 못했습니다.',
 'pixeldit_cond_p1p2_s0':'조건 memory를 더 쓰게 만들었지만 Test AP는 개선되지 않았습니다. Hybrid 42.07 → P1 41.42 / P2 41.43.',
 'pcond_p1p2_s0':'P1/P2 표현 학습 12k는 끝났지만 probe는 미완료입니다. 최종 AP를 보고할 수 없습니다.',
 'g1_upscaled_teacher_s0':'G1U +0.40 [−0.42,+1.12], G1R −0.22 [−0.87,+0.64] vs P384. 공간 teacher의 이득 전달은 불확실합니다.',
 'g1_native768_s0':'Test 점추정 G1N 41.84 vs G1U 42.01. 실제 원본 768 teacher가 더 좋다는 결과는 아닙니다. CI 후처리 중입니다.',
 'coarse_fine_ablation_s0':'A0/A1/A2 Overall 차이는 불확실합니다. A1에서 fine 추가는 +1.35 [0.54,2.04]로 관찰됩니다.',
 'fine_bottleneck_diag_s0':'D2 Test 점추정 Fine-only 37.92 < Coarse 40.65 < Fused 42.00. Fine 기여 norm은 0이 아닙니다. D1/D2 후처리 중입니다.',
 'dinov3_crop_aware_pretrain':'완료 평가 70k의 Val M2는 46.37. 기존 FrozenFull 45.59, C0 47.86이며 최종 100k 결과가 아닙니다.',
 'siglip-structure':'P4·Native의 이득은 확인되지 않았습니다. Decode−Native Object −2.13 [−3.20,−1.06]로 이 decoder는 성능이 낮습니다.',
 'siglip-extra-phase2':'E1/E2의 같은 학습량 Continue 대비 개선 CI가 0을 포함합니다. 학습량을 늘려도 목표한 Material/Pattern 개선은 확인되지 않았습니다.',
 'siglip-localcue':'편집 반응 cosine은 좋아졌지만 AP는 개선되지 않았습니다. M2 Object Material은 P8 대비 −5.46 [−9.90,−0.49]. Val 결과입니다.',
 'siglip-pixeldit':'Test Overall P8 40.54 → S1 39.61 / S2 39.60. 두 PixelDiT 조건 모두 P8보다 낮고 centered의 추가 이득도 없습니다.',
 'siglip-completed-compilation':'기본 P8의 Crop 대비 Object Pattern·Material 격차는 Mean Probe에서도 남습니다. 신규 encoder 실험이 아니라 배경 집계입니다.',
 'roi-pair-difference':'크기 축소는 성공 pair에도 있습니다. 난도를 맞춘 A/B의 차이 CI는 0을 포함해, Relative Distillation의 근거는 아직 약합니다.',
}
SKETCH={
 'hcf_official_val':'구조 추가 없이 기존 checkpoint를 공식 Val/Test로 다시 평가했습니다. 384/768 입력과 centered loss를 분리해 봅니다.',
 'prior-generative':'MAE: 가린 RGB 복원 · JEPA: EMA crop feature 예측 · PixelDiT: ROI 조건부 RGB flow. Pure와 crop-loss 결합을 나눴습니다.',
 'hcf-pixel-followup':'기존 768-centered + 학습용 PixelDiT decoder. Encoder와 6k 예산은 control과 같습니다.',
 'hier_h0h1_s0':'P384 + 640 Detail branch. H0는 384 픽셀 확대, H1은 원본→640. 같은 구조에서 새 픽셀의 효과를 구분합니다.',
 'pixeldit_cond_p1p2_s0':'Hybrid 8×8 memory → P1 24×24 dense memory → P2 위치별 AdaLN. 학습 후 decoder는 모두 제거합니다.',
 'pcond_p1p2_s0':'독립 구현의 dense cross-attention P1 / spatial AdaLN P2. 같은 이름의 완료 실험과 구분합니다.',
 'g1_upscaled_teacher_s0':'Frozen teacher에 Full384를 768로 확대해 입력. 48×48 공간 feature를 Student에 증류하며 G1R은 ROI 위치를 더 가중합니다.',
 'g1_native768_s0':'G1U의 단순 확대 teacher를 실제 원본→768 teacher로 바꿨습니다. Student 입력은 384 그대로입니다.',
 'coarse_fine_ablation_s0':'A0: Coarse↔Fine · A1: Coarse→Fine만 · A2: 중간 interaction 없음. 마지막 ROI fusion은 유지합니다.',
 'fine_bottleneck_diag_s0':'D1: 원본 crop vs Full384 RGB에서 다시 자른 crop. D2: frozen encoder의 Fine-only readout.',
 'dinov3_crop_aware_pretrain':'Fine/decoder 없이 순정 DINOv3 후반 4 block만 장기 학습. Pooled+3×3 spatial crop loss와 global/outside-ROI 보존을 사용합니다.',
 'siglip-structure':'P4: Fine stride 8→4 · Native: ROI 보간 제거 · Decode: frozen P8 위 16-query ROI decoder.',
 'siglip-extra-phase2':'E1: ROI contrastive · E2: crop 2×2 quadrant alignment · Continue: 기존 loss 유지. 각 branch에서 4k를 더 학습합니다.',
 'siglip-localcue':'M1: 질감 편집 전후 Δfeature를 teacher와 정렬. M2: 여기에 native RGB Conv Detail 경로 추가.',
 'siglip-pixeldit':'S1: crop + pixel flow + global · S2: centered loss 0.5 추가. 공식 pretrained 기반 공통 초기화에서 각각 12k.',
 'siglip-completed-compilation':'기존 P/M과 첫 E1/E2·Continue 4k, Mean Probe 결과를 모았습니다. 오늘의 후속 4k와 중복 계산하지 않습니다.',
 'roi-pair-difference':'기존 feature·Mean/R2 score를 재사용. L2-normalized Δfeature의 크기·방향과 기존 선형 head 투영을 비교했습니다.',
}

def rich(x):
    if x is None or x=={} or x==[]:return '<p class="muted">해당 없음</p>'
    if isinstance(x,dict):return ''.join(f'<p><b>{esc(k)}</b> {rich_inline(v)}</p>' for k,v in x.items())
    if isinstance(x,list):return '<ul>'+''.join(f'<li>{rich_inline(v)}</li>' for v in x)+'</ul>'
    return '<p>'+esc(x)+'</p>'
def rich_inline(x):return esc(x) if not isinstance(x,(dict,list)) else esc(json.dumps(x,ensure_ascii=False))
def source_links(paths):
    links=[]
    for s in paths:
        p=Path(s);p=p if p.is_absolute() else ROOT/p
        if not p.exists():continue
        label=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        links.append(f'<a href="{esc(os.path.relpath(p,OUT))}" title="{esc(p)}">{esc(label)}</a>')
    return '<div class="source-links">'+''.join(links)+'</div>'

def status(e):
    if e['scope']=='context':return ('bg','배경 · 오늘 집계/평가')
    if e['id']=='dinov3_crop_aware_pretrain':return ('live','장기학습 / 평가 진행 중')
    if e['id']=='pcond_p1p2_s0':return ('partial','표현학습 완료 · probe 미완료')
    if e['id'] in ['g1_native768_s0','fine_bottleneck_diag_s0']:return ('partial','점추정 / 일부 진단 완료')
    if e['id']=='siglip-localcue':return ('partial','공식 Val 완료 · Test 없음')
    return ('done','완료')

def table_html(table):
    labels=['모델','Overall','Object','Part','Obj Pattern','Obj Material','ep']
    keys=['name','overall','object','part','pattern','material','epoch']
    rows=[]
    for r in table['rows']:
        cells=[]
        for k in keys:
            v=r.get(k)
            cells.append(esc(v) if k=='name' else ('—' if v is None else f'{int(v)}' if k=='epoch' else f'{v:.2f}'))
        rows.append('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{v}</{"th" if i==0 else "td"}>' for i,v in enumerate(cells))+'</tr>')
    return f'<div class="result-title"><span class="protocol {esc(table["protocol"])}">{esc(table["label"])}</span></div><div class="table-scroll"><table><thead><tr>'+''.join(f'<th>{x}</th>' for x in labels)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'

def card(e,i):
    c,label=status(e);title=e['title'];id=e['id']
    variants=e.get('variants')
    variant_html=''
    if variants:
        if isinstance(variants,dict):variant_html=rich(variants)
        else:variant_html=''.join('<div class="variant"><b>'+esc(v.get('name',''))+'</b>'+rich(v.get('change',''))+(rich('학습: '+v['learned']) if 'learned' in v else '')+'</div>' for v in variants)
    progress=''
    if id=='dinov3_crop_aware_pretrain':
        step=e.get('progress_step',0);progress=f'<div class="progress" aria-label="학습 step"><span style="width:{min(step/100000*100,100):.1f}%"></span></div><p class="small">수집된 상태: {step:,} / 100,000 step · AP 표는 완료된 평가만 표시</p>'
    details='<div class="spec-grid">'+''.join(f'<div><h4>{k}</h4>{rich(v)}</div>' for k,v in [('대조군 · 무엇을 고정했나',CONTROL.get(id,e.get('control'))),('시작 checkpoint',e.get('initialization','기존 frozen encoder / 개별 결과 출처 참조')),('구조 변경',e.get('architecture_change',e.get('architecture'))),('학습 변경',e.get('training_change')),('예산 · 학습량',e.get('budget')),('평가 방법',e.get('protocol','각 표의 split·지표를 확인. 새 학습 없이 기존 결과를 집계했습니다.'))])+'</div>'
    details+=('<h4>조건별 구조</h4>'+variant_html) if variant_html else ''
    details+=''.join(table_html(t) for t in e.get('tables',[]) if t['rows'])
    findings=e.get('findings',[]);findings=[findings] if isinstance(findings,str) else findings
    details+='<h4>관찰한 결과</h4>'+rich(findings)
    if e.get('limitations'):details+='<div class="limit"><b>해석 범위</b>'+rich(e['limitations'])+'</div>'
    details+='<h4>원자료</h4>'+source_links(e.get('sources',[]))
    return f'''<article class="experiment" id="exp-{esc(id)}" data-family="{esc(e['family'])}" data-group="{esc(e['group'])}" data-status="{c}">
 <div class="exp-top"><span class="exp-index">{i:02d}</span><span class="family">{esc(e['family'])}</span><span class="status {c}">{label}</span></div>
 <h3>{esc(title)}</h3><p class="question">{esc(e.get('question',''))}</p>
 <div class="change"><span>무엇을 바꿨나</span><p>{esc(SKETCH[id])}</p></div>
 <div class="takeaway"><span>현재 판단</span><p>{esc(TAKE[id])}</p></div>{progress}
 <details><summary>구조 · 학습 · 결과 자세히 <span aria-hidden="true">＋</span></summary><div class="detail-body">{details}</div></details>
 </article>'''

CSS='''
:root{--ink:#182d38;--muted:#61717b;--line:#dce3e4;--paper:#f5f6f2;--white:#fff;--teal:#007e78;--navy:#233b50;--gold:#b27525;--red:#ad4f56}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:86px}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,Pretendard,"Noto Sans KR","Apple SD Gothic Neo",system-ui,sans-serif;line-height:1.75;font-size:15px}a{color:var(--teal);text-underline-offset:3px}button,input,select{font:inherit}button,select{cursor:pointer}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #c79738;outline-offset:3px}button{color:inherit}main{max-width:1220px;padding:0 38px 70px;margin:auto}.masthead{max-width:1220px;margin:auto;padding:25px 38px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line)}.brand{font-size:12px;letter-spacing:.13em;font-weight:800}.brand i{display:inline-block;width:9px;height:9px;background:var(--teal);margin-right:10px}.masthead .small{font-size:12px}.nav-wrap{position:sticky;top:0;z-index:20;background:rgba(245,246,242,.97);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}nav{max-width:1220px;margin:auto;padding:13px 38px;display:flex;gap:26px;overflow:auto;white-space:nowrap}nav a{text-decoration:none;color:var(--muted);font-size:13px;font-weight:650}nav a:hover{color:var(--teal)}.hero{padding:56px 0 35px;display:grid;grid-template-columns:1.5fr .8fr;gap:50px}.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--teal);font-weight:800;margin:0 0 14px}h1{font-size:clamp(32px,4vw,53px);line-height:1.25;font-weight:750;letter-spacing:-.055em;margin:0 0 23px}h1 span{font-weight:430}.lead{color:var(--muted);font-size:16px;max-width:650px;margin:0}.meta-box{border-left:3px solid var(--teal);padding:5px 0 4px 22px;align-self:center}.meta-box b{font-size:36px;letter-spacing:-.04em;display:block;line-height:1.1}.meta-box p{margin:7px 0;color:var(--muted);font-size:13px}.actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:21px}.btn{border:1px solid var(--line);border-radius:5px;padding:8px 13px;background:var(--white);font-size:12px;font-weight:650;text-decoration:none;display:inline-block}.btn.primary{background:var(--ink);color:white;border-color:var(--ink)}section{padding-top:40px;margin-bottom:12px}.section-head{display:flex;gap:18px;align-items:baseline;margin-bottom:20px}.section-no{color:var(--teal);font-size:12px;font-weight:750;font-variant-numeric:tabular-nums}h2{font-size:25px;letter-spacing:-.04em;margin:0}h3{font-size:19px;line-height:1.5;letter-spacing:-.035em;margin:12px 0}h4{font-size:13px;margin:19px 0 7px}p{margin:8px 0}.muted,.small{color:var(--muted)}.small{font-size:12px}.notice{padding:18px 21px;background:#e7efeb;border-left:3px solid var(--teal);font-size:13px}.notice strong{color:#075e59}.insights{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0;border:1px solid var(--line);background:var(--white)}.insight{padding:24px;min-width:0}.insight:nth-child(odd){border-right:1px solid var(--line)}.insight:nth-child(-n+2){border-bottom:1px solid var(--line)}.insight h3{margin:0 0 8px;font-size:17px}.insight p{margin:0;color:var(--muted);font-size:14px}.insight .tag{font-size:11px;color:var(--teal);font-weight:750;display:block;margin-bottom:8px}.box{border:1px solid var(--line);background:var(--white);padding:25px}.flow{display:flex;gap:22px;align-items:stretch;margin:22px 0}.flow>div{background:var(--white);border:1px solid var(--line);padding:17px;flex:1;position:relative;min-width:0}.flow>div:not(:last-child):after{content:'→';position:absolute;right:-18px;top:38%;color:var(--teal)}.flow strong{display:block;font-size:14px}.flow small{display:block;color:var(--muted);font-size:12px;margin-top:6px}.flow .accent{border-top:3px solid var(--teal)}.flow .ref{border-top:3px solid var(--navy)}.legend-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.legend-grid p{font-size:13px;color:var(--muted)}.legend-grid strong{color:var(--ink)}.toolbar{display:flex;align-items:end;gap:14px;flex-wrap:wrap;margin:20px 0}.toolbar label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:6px}.toolbar input,.toolbar select{border:1px solid #c7d3d6;background:white;border-radius:5px;padding:9px 11px;min-height:42px;color:var(--ink)}.toolbar input{width:260px;max-width:100%}.toolbar label.grow{flex:1}.toolbar label.grow input{width:100%}.seg{display:flex;gap:5px;flex-wrap:wrap}.seg button{border:1px solid var(--line);background:white;padding:8px 13px;border-radius:4px;font-size:12px}.seg button[aria-pressed=true]{background:var(--teal);border-color:var(--teal);color:white}.chart-scroll{overflow-x:auto;max-width:100%}.chart{width:100%;display:block;max-height:560px}.chart text{font-family:inherit}.benchmark-caption{font-size:12px;padding:12px 15px;background:#f0f3f2;margin:0 0 15px}.table-scroll{overflow:auto;border:1px solid var(--line);margin:10px 0 16px;max-width:100%}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;background:white}th,td{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}thead th{background:#edf1ef;font-size:11px;color:#4e636d;font-weight:650}th:first-child,td:first-child{text-align:left}tbody th{font-weight:550}tbody tr:last-child>*{border-bottom:0}tbody tr.reference{background:#edf1f7}tbody tr.highlight{background:#e8f2ee}tbody tr:hover{background:#f2f5ef}.protocol{display:inline-block;font-size:11px;font-weight:750;padding:4px 8px;border:1px solid #b8c9cf;border-radius:4px;color:var(--navy);background:#eff3f6}.protocol.project_val{background:#fff3de;color:#8b5c14;border-color:#d9bd85}.protocol.official_val{background:#e9f2ed;color:#1b6b56}.protocol.official_test_mean{background:#f0ecf5;color:#685475}.result-title{margin-top:21px}.filters{border:1px solid var(--line);background:#edf1ed;padding:18px 20px}.filter-bottom{display:flex;justify-content:space-between;align-items:center;gap:20px}.experiment-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:19px;margin-top:20px;align-items:start}.experiment{background:white;border:1px solid var(--line);padding:23px;min-width:0;scroll-margin-top:90px}.exp-top{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.exp-index{font-size:11px;font-weight:800;color:#89999d;font-variant-numeric:tabular-nums}.family{font-size:11px;font-weight:700;color:var(--navy);padding-right:3px}.status{font-size:10px;padding:3px 7px;border-radius:3px;white-space:normal;line-height:1.6}.status.done{background:#e5f0e9;color:#236b47}.status.partial{background:#fff1d9;color:#8d631e}.status.live{background:#e8eef9;color:#345c98}.status.bg{background:#edf0f1;color:#65747c}.question{font-size:13px;color:#64737b}.change{margin-top:19px;border-top:1px solid var(--line);padding-top:13px}.change>span,.takeaway>span{font-size:10px;letter-spacing:.04em;font-weight:750;color:var(--teal)}.change p,.takeaway p{font-size:13px;line-height:1.8;margin:5px 0}.takeaway{margin-top:13px;border-left:2px solid var(--teal);padding:0 0 0 12px}.takeaway>span{color:var(--ink)}details{border-top:1px solid var(--line);margin-top:20px}summary{list-style:none;font-size:12px;font-weight:700;cursor:pointer;padding:13px 0 0;display:flex;justify-content:space-between;gap:10px}summary::-webkit-details-marker{display:none}details[open]>summary span{transform:rotate(45deg)}.detail-body{margin-top:14px;font-size:12px}.detail-body p,.detail-body li{font-size:12px;line-height:1.85;overflow-wrap:anywhere}.detail-body ul{padding-left:18px}.spec-grid{display:grid;grid-template-columns:1fr;gap:0}.spec-grid h4{color:var(--teal);margin-top:13px}.variant{border-left:2px solid #b5caca;padding-left:12px;margin:14px 0}.limit{background:#f5f2e9;padding:13px;margin:16px 0}.limit ul{margin-bottom:0}.source-links{display:flex;flex-direction:column;gap:7px}.source-links a{font-size:10px;overflow-wrap:anywhere;color:#5a747d}.progress{height:5px;background:#e9edf1;margin-top:15px;border-radius:4px;overflow:hidden}.progress span{height:100%;display:block;background:#4b77b9}.question-map{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin-top:16px}.question-map button{text-align:left;border:1px solid var(--line);border-top:3px solid var(--teal);padding:18px 14px;background:white;line-height:1.7}.question-map button b{display:block;font-size:14px}.question-map button small{color:var(--muted);font-size:11px;display:block;margin-top:6px}.empty{padding:30px;border:1px dashed var(--line);color:var(--muted);text-align:center}.architecture-grid{display:grid;grid-template-columns:1fr 1fr;gap:19px}.architecture-grid .box h3{margin-top:0}.pixel-flow{display:grid;grid-template-columns:1fr;gap:8px;font-size:12px}.pixel-flow>div{padding:11px 13px;border:1px solid var(--line);background:#f7f9f6}.pixel-flow b{display:block;color:var(--teal)}.pixel-flow small{display:block;color:var(--muted)}.interaction-row{display:flex;gap:10px;margin:18px 0}.interaction{flex:1;text-align:center;border:1px solid var(--line);background:#f6f8f4;padding:14px 5px}.interaction b{font-size:12px}.interaction .arrows{color:var(--teal);font-size:24px;line-height:1.3;margin:5px 0}.interaction small{display:block;font-size:10px;color:var(--muted)}.diagnosis-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.diagnosis-grid h3{margin-top:0}.big-number{font-size:37px;line-height:1.3;letter-spacing:-.04em;color:var(--teal);font-weight:700}.diagnosis-grid p{font-size:13px}.figure{margin:18px 0 0;border:1px solid var(--line);background:white;padding:15px}.figure img{display:block;width:100%;height:auto}.figure figcaption{font-size:11px;color:var(--muted);margin-top:10px}.conclusion{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0;border:1px solid var(--line);background:white}.conclusion>div{padding:23px}.conclusion>div+div{border-left:1px solid var(--line)}.conclusion h3{font-size:16px;margin-top:0}.conclusion p{font-size:13px;color:var(--muted)}.source-table a{font-size:11px;white-space:normal;word-break:break-all}.source-table td:first-child{max-width:650px}.page-footer{border-top:1px solid var(--line);margin-top:45px;padding-top:23px;display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:11px}.jump-top{color:var(--muted)}[hidden]{display:none!important}
@media(min-width:1050px){.experiment-grid .experiment:has(details[open]){grid-column:1/-1}.experiment-grid .experiment:has(details[open]) .spec-grid{grid-template-columns:1fr 1fr;gap:8px 30px}}
@media(max-width:760px){body{font-size:14px}main{padding:0 19px 45px}.masthead{padding:18px 19px;gap:15px}.masthead .small{font-size:10px}.brand{font-size:10px;letter-spacing:.07em}nav{padding:12px 19px;gap:21px}.hero{grid-template-columns:1fr;gap:26px;padding-top:35px}.meta-box{display:flex;gap:20px;align-items:center}.meta-box b{font-size:30px}.hero h1{font-size:36px}.insights,.experiment-grid,.architecture-grid,.diagnosis-grid,.legend-grid,.conclusion{grid-template-columns:1fr}.insight:nth-child(odd){border-right:0}.insight:not(:last-child){border-bottom:1px solid var(--line)}.flow{flex-direction:column;gap:18px}.flow>div:not(:last-child):after{content:'↓';right:50%;top:auto;bottom:-22px}.question-map{grid-template-columns:1fr 1fr}.filter-bottom{align-items:start;flex-direction:column;gap:8px}.toolbar{align-items:stretch;gap:10px}.toolbar label{width:100%}.toolbar select{max-width:100%;width:100%}.experiment{padding:19px}.conclusion>div+div{border-left:0;border-top:1px solid var(--line)}.box{padding:20px}.section-head{gap:12px}h2{font-size:22px}.page-footer{flex-direction:column;gap:8px}.chart{max-height:none;min-width:720px}th,td{padding:10px 11px}.source-table td{font-size:10px}.interaction-row{gap:6px}.interaction{padding:12px 3px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:white;font-size:11px}main{max-width:none;padding:0}.nav-wrap,.actions,.filters,.question-map,.toolbar,.jump-top{display:none!important}.masthead{padding:0 0 15px}.hero{padding:25px 0}.hero h1{font-size:30px}section{padding-top:24px}.experiment-grid,.insights,.architecture-grid,.diagnosis-grid,.legend-grid,.conclusion{display:block}.experiment,.box,.insight{margin-bottom:15px;break-inside:avoid}.experiment[hidden]{display:block!important}details{display:block}.table-scroll{overflow:visible}table{font-size:9px}th,td{padding:6px}.source-links a{font-size:8px}.figure{break-inside:avoid}.conclusion>div+div{border:0}.chart{max-height:400px}.source-table{font-size:8px}a{color:inherit}.page-footer{margin-top:20px}}
'''

TEMPLATE='''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="2026년 11월 9일 Crop Representation 실험: 목적, 대조군, 구조와 학습 변경, 완료된 결과와 남은 질문"><title>2026.11.09 · Crop Representation 실험 기록</title><style>__CSS__</style></head>
<body id="top"><header class="masthead"><div class="brand"><i></i>CROP REPRESENTATION / RESEARCH LOG</div><div class="small">2026.11.09 · Seed 0 · 로컬 보고서</div></header>
<div class="nav-wrap"><nav aria-label="보고서 목차"><a href="#findings">핵심 관찰</a><a href="#baselines">기준선 · 성능</a><a href="#experiments">실험별 변경</a><a href="#architectures">구조 비교</a><a href="#diagnosis">Pair 진단</a><a href="#decisions">판단 · 한계</a><a href="#sources">출처</a></nav></div>
<main>
<div class="hero"><div><p class="eyebrow">DAILY EXPERIMENT REVIEW</p><h1>Crop Representation<br><span>오늘의 실험 기록</span></h1><p class="lead">무엇을 확인하려 했고, 어떤 입력·구조·학습 목표를 바꿨는가.<br>완료된 결과와 진행 중인 작업을 같은 기준으로 정리했습니다.</p><div class="actions"><button class="btn primary" id="download-json">결과 JSON 저장</button><a class="btn" href="experiments.csv" download>실험 목록 CSV</a><button class="btn" id="print-report">인쇄 / PDF</button></div></div><aside class="meta-box"><div><b>__COUNT__</b><p>실험·진단·배경 집계 묶음</p></div><div><p>DINOv3 / SigLIP2</p><p>정리 기준: 2026-11-09 실행 폴더와<br>오늘 완료된 평가. 전날 학습은 별도 표시.</p><p id="snapshot" class="small">집계 __SNAPSHOT__ UTC</p></div></aside></div>
<div class="notice"><strong>숫자를 읽기 전에</strong> &nbsp;공식 Test AP, 공식 Val AP, project M2 Val을 분리했습니다. ‘초기 encoder 0 step’은 <b>G1-pool을 이미 학습한 뒤 이번 단계에서 0</b>이라는 뜻입니다. C0는 고정된 원본 encoder의 crop 재인코딩 참조이며, 모든 조건의 절대 상한을 보장하지 않습니다.</div>

<section id="findings"><div class="section-head"><span class="section-no">01</span><h2>오늘 확인한 것</h2></div><div class="insights">__INSIGHTS__</div><p class="small">점추정, 신뢰구간, 진단 지표를 구분했습니다. 단일 학습 seed의 이미지 bootstrap CI는 학습 seed 변동을 포함하지 않습니다.</p></section>

<section id="baselines"><div class="section-head"><span class="section-no">02</span><h2>기준선부터 다시 맞추기</h2></div>
<div class="flow" aria-label="DINOv3 학습 계보"><div class="ref"><strong>순정 DINOv3</strong><small>공식 pretrained 가중치<br>우리 encoder 추가 학습 0<br>PACO reader는 별도 학습</small></div><div><strong>G1-pool</strong><small>COCO crop 증류 12k<br>Fine 경로 없음<br>학습된 출발점</small></div><div><strong>P 초기화</strong><small>G1-pool + 새 Fine/Interaction<br>Residual scale 보정<br><b>이번 단계만 0 step</b></small></div><div class="accent"><strong>P384 / P768</strong><small>P 구조 전체를 12k 추가 학습<br>crop + 0.5 global<br>centered 조건은 항 추가</small></div></div>
<div class="legend-grid"><div class="box"><strong>C0 / Crop oracle의 정확한 뜻</strong><p>GT ROI를 실제 RGB에서 자르고 384로 확대해 <b>고정된 순정 DINOv3</b>로 다시 인코딩합니다. Student와 학습·입력·문맥이 달라 AP의 수학적 상한은 아닙니다.</p></div><div class="box"><strong>‘아무것도 안 한 성능’과 다른 것</strong><p>Frozen encoder에도 PACO 정답으로 학습한 Mean Linear / R2가 붙습니다. GT ROI가 주어진 속성 평가이며 zero-shot이나 object detection 결과가 아닙니다. SigLIP2 P8은 G1 없이 공식 pretrained에서 시작합니다.</p></div></div>
<div class="box" style="margin-top:19px"><div class="toolbar"><label class="grow">평가 프로토콜<select id="benchmark-protocol"><option value="official_test">공식 Test AP · R2 (기본)</option><option value="official_val">공식 Val AP · R2 (epoch 선택용)</option><option value="project_val_r2">project M2 Val · R2 (dev M0 선택)</option><option value="project_val_mean">project M2 Val · Mean + Linear (dev M0 선택)</option></select></label><label>그래프 지표<select id="benchmark-metric"><option value="overall">Overall</option><option value="object">Object</option><option value="part">Part</option><option value="pattern">Object Pattern</option><option value="material">Object Material</option></select></label></div><p class="benchmark-caption" id="benchmark-caption"></p><div class="chart-scroll"><svg id="benchmark-chart" class="chart" role="img" aria-label="DINOv3 기준선과 대표 조건 성능 비교" viewBox="0 0 1020 510"></svg></div><div class="table-scroll"><table id="benchmark-table"><thead><tr><th>모델</th><th>Overall</th><th>Object</th><th>Part</th><th>Obj Pattern</th><th>Obj Material</th><th>ep</th></tr></thead><tbody></tbody></table></div><p class="small">공식 AP의 Overall은 (Object + Part) / 2입니다. 프로토콜을 바꾸면 비교 모집단·평가 구현·선택된 reader epoch이 바뀝니다. 같은 그래프에 섞어 그리지 않았습니다. 이 대표 표의 공식 Test 차이는 별도 CI가 없어 점추정으로 읽습니다.</p><p class="small"><b>이번 단계 12k의 의미:</b> 공식 Test Overall은 P 초기화 41.65 → P384 41.61로 점추정이 비슷합니다. 이미 G1-pool을 거친 출발점과 순정 pretrained를 혼동하지 않아야 합니다. 768 조건의 이득에는 입력·token 수·연산량의 변화도 포함됩니다.</p></div>
</section>

<section id="experiments"><div class="section-head"><span class="section-no">03</span><h2>질문별로 보는 실험 변경</h2></div><p class="muted">각 묶음에서 질문 → 바꾼 것 → 현재 판단을 먼저 보고, 펼치면 대조군·초기화·학습·전체 결과를 확인할 수 있습니다.</p>
<div class="question-map"><button type="button" data-quick-group="input"><b>입력 · Teacher 해상도</b><small>새 픽셀인가, 격자만 촘촘한가?<br>768 / H0·H1 / G1U·G1N</small></button><button type="button" data-quick-group="objective"><b>학습 목표 · 조건 전달</b><small>무엇을 예측하게 해야 하는가?<br>MAE·JEPA·PixelDiT / local cue</small></button><button type="button" data-quick-group="structure"><b>Encoder · Readout 구조</b><small>Fine과 interaction이 필요한가?<br>A0·A1·A2 / P4·Native·Decode</small></button><button type="button" data-quick-group="diagnostic"><b>Feature · 병목 진단</b><small>차이를 잃었나, 방향이 다른가?<br>ROI pair / Fine-only / RGB 진단</small></button></div>
<div class="filters"><div class="toolbar" style="margin:0 0 13px"><label class="grow">실험 검색<input id="experiment-search" type="search" placeholder="예: PixelDiT, G1N, Material, Fine"></label><label>질문 축<select id="group-filter"><option value="all">전체</option><option value="input">입력 · Teacher</option><option value="objective">학습 목표</option><option value="structure">구조 · Readout</option><option value="diagnostic">진단</option><option value="context">기존 결과 집계</option></select></label><label>완료 상태<select id="status-filter"><option value="all">완료 + 진행 중 + 배경</option><option value="done">완료된 묶음</option><option value="partial">일부 완료 / 후처리</option><option value="live">장기학습 진행 중</option><option value="bg">배경 / 이전 학습</option></select></label></div><div class="filter-bottom"><div class="seg" role="group" aria-label="Backbone 선택"><button data-family-filter="all" aria-pressed="true">전체</button><button data-family-filter="DINOv3" aria-pressed="false">DINOv3</button><button data-family-filter="SigLIP2" aria-pressed="false">SigLIP2</button></div><div><span id="visible-count" class="small" aria-live="polite"></span> <button class="btn" id="reset-filters">필터 초기화</button></div></div></div>
<div class="experiment-grid" id="experiment-grid">__CARDS__</div><div id="empty-results" class="empty" hidden>조건에 맞는 실험이 없습니다. 검색어나 필터를 바꿔주세요.</div>
</section>

<section id="architectures"><div class="section-head"><span class="section-no">04</span><h2>구조가 실제로 달라진 지점</h2></div><div class="architecture-grid"><div class="box"><h3>PixelDiT: 복원 조건을 더 쓰게 만들기</h3><div class="pixel-flow"><div><b>전체 이미지 → Coarse–Fine encoder</b><small>ROI grid 24×24×768을 읽습니다. Encoder는 전체 이미지당 공유합니다.</small></div><div><b>기존 Hybrid: 64-token memory</b><small>8×8 pooling → ROI cross-attention. Noisy crop + 시간 t와 함께 decoder에 입력합니다.</small></div><div><b>P1: 576-token memory</b><small>24×24를 그대로 cross-attention에 넣어 pooling 손실을 줄입니다.</small></div><div><b>P2: 위치별 AdaLN</b><small>각 ROI 위치의 feature와 t가 해당 patch의 scale·shift·gate를 조절합니다. ROI cross-attention을 제거합니다.</small></div></div><p class="small">Patch DiT 2 block(width384) + local pixel block(width64). v = noise − RGB를 예측하며 VAE는 없습니다. Decoder는 학습 후 제거합니다. 조건 사용 증가가 속성 AP 개선을 뜻하지는 않습니다.</p></div><div class="box"><h3>Interaction: 어느 방향이 필요한가</h3><div class="interaction-row"><div class="interaction"><b>A0</b><div>Coarse</div><div class="arrows">↕</div><div>Fine</div><small>양방향 4-stage</small></div><div class="interaction"><b>A1</b><div>Coarse</div><div class="arrows">↓</div><div>Fine</div><small>문맥 전달만 유지</small></div><div class="interaction"><b>A2</b><div>Coarse</div><div class="arrows">⋮</div><div>Fine</div><small>중간 교환 없음</small></div></div><p>세 조건 모두 마지막 ROI readout에서는 coarse와 fine을 합칩니다. Coarse-only probe도 따로 학습해 Fine의 추가 이득을 비교했습니다.</p><p class="small">A1의 fused 성능은 A0보다 낮아졌다고 확인되지 않았고, A1에서 fused−coarse +1.35 [0.54,2.04]가 관찰됐습니다. 반복 횟수 4의 최적성은 이번 실험으로 판단하지 않습니다.</p><h4>같은 ‘P1/P2’ 이름 주의</h4><p class="small">완료된 <code>pixeldit_cond_p1p2_s0</code>와 미완료인 <code>pcond_p1p2_s0</code>는 독립 실행입니다. P2 decoder parameter 수도 다릅니다. 결과를 합치지 않았습니다.</p></div></div></section>

<section id="diagnosis"><div class="section-head"><span class="section-no">05</span><h2>ROI pair에서 새로 확인한 것</h2></div><div class="diagnosis-grid"><div class="box"><div class="big-number">129,328</div><h3>공통 Val positive–negative pair</h3><p>같은 category·attribute의 명시적 양성/음성만 사용했습니다. Crop만 맞는 A와 둘 다 맞는 B를 비교하되 category·양쪽 ROI 크기·Crop margin을 통제했습니다.</p><p><strong>Magnitude ratio는 A와 B가 비슷합니다.</strong> 가까운 Crop margin으로 매칭한 뒤 direction 차이도 95% CI가 0을 포함했습니다.</p><p class="small">11,109 ROI / 2,051 images. 동일 이미지 pair는 0개. 양 끝 image 의존성을 반영한 vertex bootstrap 1,000회. 추가 학습 없음.</p></div><div class="box"><h3>Relative Distillation의 근거가 나왔나?</h3><p>개별 ROI cosine이 높아도 difference cosine은 더 낮을 수 있습니다. 다만 이 사실만으로 A에만 특유한 정보 소실이나 AP 실패의 원인을 입증하지 못했습니다.</p><p>동일 P384 A 집합에서 P768이 Material 약41%, Pattern 약45%를 회복했습니다. 입력·sampling 개선의 근거와 속성 판별 축의 문제를 분리해 봐야 합니다.</p><p class="small">새 Method 학습이나 새 probe fit은 수행하지 않았습니다. 관찰·가능한 해석·미검증 가설을 원 보고서에서 구분했습니다.</p><a href="../../results/roi_pair_feature_difference_2026-11-09/REPORT.md">ROI Pair 상세 Markdown 보고서 ↗</a></div></div><details><summary>난도 통제 및 크기별 feature difference 그래프 보기 <span aria-hidden="true">＋</span></summary><figure class="figure"><img src="__FIG_ADJUST__" alt="Magnitude와 direction의 A-B 차이를 난도 통제 전후 비교한 95% 신뢰구간 그래프" loading="lazy"><figcaption>난도 통제 전후 A−B. Pair 평균과 macro 통제 집합은 모집단이 다르며 close matching에는 Object A의 약77%가 남습니다.</figcaption></figure><figure class="figure"><img src="__FIG_SIZE__" alt="384 canvas 기준 동일 ROI 크기 bin에서 P384 P768 centered의 feature difference를 비교한 그래프" loading="lazy"><figcaption>기존 ranking 분석을 반복한 것이 아니라 새 feature difference 지표만 크기로 나눴습니다. 작은 ROI에서 방향 보존이 더 낮고 768의 개선 폭이 큽니다.</figcaption></figure></details></section>

<section id="decisions"><div class="section-head"><span class="section-no">06</span><h2>관찰과 결론의 범위</h2></div><div class="conclusion"><div><h3>관찰한 사실</h3><p>768 입력의 기하·AP 개선, 생성 조건 의존성과 AP의 불일치, A0/A1/A2의 불확실한 Overall 차이, 난도 매칭 후 약해지는 pair geometry 차이를 확인했습니다.</p></div><div><h3>가능한 해석</h3><p>입력 세부 정보, feature가 담는 내용, 속성 판별 방향이 함께 작용할 수 있습니다. 큰 cosine이나 낮은 복원 loss 하나로 속성 정보를 평가하기 어렵습니다.</p></div><div><h3>아직 증명하지 못한 것</h3><p>Relative Distillation의 AP 이득, Fine 경로만이 병목이라는 주장, 더 높은 생성 조건 의존성이 더 좋은 표현을 만든다는 주장, 긴 학습의 최종 개선은 확정하지 않았습니다.</p></div></div><div class="notice" style="margin-top:18px"><strong>진행 중인 결과는 최종 성능이 아닙니다.</strong> Crop-aware 100k, G1N 후처리, D1/D2 진단, 별도 Pcond probe의 상태는 집계 시점 스냅샷입니다. 없는 Test·CI를 추정해 채우지 않았습니다. 이 HTML 정리 과정에서는 새 학습을 실행하지 않았습니다.</div></section>

<section id="sources"><div class="section-head"><span class="section-no">07</span><h2>출처와 재현 가능한 스냅샷</h2></div><p class="small">기존 README의 “scheduled / no Test”가 오래된 경우 실제 done·selection·결과 파일을 우선했습니다. 전날 학습 결과를 오늘 새 학습으로 세지 않았습니다. 원자료는 변경하지 않았고 경로·mtime·SHA256을 data.json에 기록했습니다.</p><details><summary>사용한 원자료 __SOURCE_COUNT__개 보기 <span aria-hidden="true">＋</span></summary><div class="table-scroll"><table class="source-table"><thead><tr><th>원자료</th><th>집계 시 파일 크기</th><th>수정 시각 UTC</th></tr></thead><tbody>__SOURCES__</tbody></table></div></details><p class="small">원자료 링크는 현재 작업 폴더 기준입니다. HTML의 본문·데이터·그림은 내장되어 인터넷 연결 없이 열 수 있습니다. 진행 중인 원자료는 이후 바뀔 수 있으며 이 보고서는 자동 갱신하지 않습니다.</p></section>
<footer class="page-footer"><div>Crop Representation · 2026.11.09<br>Val / Test와 평가 지표를 구분한 연구 기록</div><div>__SNAPSHOT__ UTC<br><a class="jump-top" href="#top">맨 위로 ↑</a></div></footer>
</main><script id="report-data" type="application/json">__DATA__</script><script>__JS__</script></body></html>'''

JS='''
'use strict';
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id);
let selectedFamily='all';
const fmt=(x,d=2)=>x==null?'—':Number(x).toFixed(d);
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const descriptions={
 official_test:'공식 PACO Test AP · R2 · 각 조건의 공식 Val (Object+Part)/2 최대 epoch을 고정해 평가. C0와 Frozen/G1은 기존 공식 참조를 재사용합니다. 아래 차이는 점추정이며 이 표에 CI가 없습니다.',
 official_val:'공식 PACO Val AP · R2 · reader epoch 선택에 사용한 데이터입니다. Test와 같은 모델이어도 split이 다르며 project M2 Val과도 직접 비교하지 않습니다.',
 project_val_r2:'PACO Val · project M2 AP · R2 · train 내부 dev 1,000장의 M0로 epoch 선택. 공식 evaluator 결과가 아닙니다. 순정 native/bilinear readout도 구분합니다.',
 project_val_mean:'PACO Val · project M2 AP · Mean + Linear · train 내부 dev M0 선택. Encoder는 고정했지만 선형 probe는 PACO 정답으로 학습했습니다.'
};
function benchmark(){
 const protocol=$('benchmark-protocol').value,metric=$('benchmark-metric').value,rows=DATA.benchmarks[protocol];
 $('benchmark-caption').textContent=descriptions[protocol];
 $('benchmark-table').querySelector('tbody').innerHTML=rows.map(r=>'<tr class="'+(r.name.startsWith('C0')?'reference':r.name.includes('P768')?'highlight':'')+'"><th scope="row">'+escapeHTML(r.name)+'</th>'+['overall','object','part','pattern','material'].map(k=>'<td>'+fmt(r[k])+'</td>').join('')+'<td>'+fmt(r.epoch,0)+'</td></tr>').join('');
 const w=1020,h=rows.length*37+46,left=270,right=62,plot=w-left-right,max=Math.max(60,Math.ceil(Math.max(...rows.map(r=>r[metric]||0))/10)*10);
 $('benchmark-chart').setAttribute('viewBox','0 0 '+w+' '+h);
 let s='<title>'+escapeHTML(descriptions[protocol])+'</title><desc>지표 '+metric+'. '+rows.map(r=>r.name+' '+fmt(r[metric])).join('; ')+'</desc>';
 for(let v=0;v<=max;v+=10){let x=left+plot*v/max;s+='<line x1="'+x+'" x2="'+x+'" y1="5" y2="'+(h-35)+'" stroke="#e0e7e5"/><text x="'+x+'" y="'+(h-12)+'" text-anchor="middle" fill="#6b7c85" font-size="12">'+v+'</text>';}
 rows.forEach((r,i)=>{let y=i*37+10,value=r[metric]||0,color=r.name.startsWith('C0')?'#233b50':r.name.includes('中心')||r.name.includes('중심화')?'#b27525':r.name.includes('순정')?'#8c9ba1':'#007e78';s+='<text x="'+(left-14)+'" y="'+(y+17)+'" text-anchor="end" font-size="13" fill="#263e49">'+escapeHTML(r.name)+'</text><rect x="'+left+'" y="'+y+'" width="'+(plot*value/max)+'" height="24" rx="2" fill="'+color+'"/><text x="'+(left+plot*value/max+9)+'" y="'+(y+17)+'" font-size="13" font-weight="650" fill="#263e49">'+fmt(value)+'</text>';});
 $('benchmark-chart').innerHTML=s;
}
function filter(){
 const query=$('experiment-search').value.toLowerCase().trim(),group=$('group-filter').value,status=$('status-filter').value;
 let n=0;document.querySelectorAll('.experiment').forEach(card=>{let show=(selectedFamily==='all'||card.dataset.family===selectedFamily)&&(group==='all'||card.dataset.group===group)&&(status==='all'||card.dataset.status===status)&&(!query||card.textContent.toLowerCase().includes(query)||card.id.toLowerCase().includes(query));card.hidden=!show;if(show)n++;});
 $('visible-count').textContent=n+' / '+DATA.experiments.length+' 묶음';$('empty-results').hidden=n>0;
 document.querySelectorAll('[data-family-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.familyFilter===selectedFamily)));
}
function reset(){selectedFamily='all';$('experiment-search').value='';$('group-filter').value='all';$('status-filter').value='all';filter();}
$('benchmark-protocol').addEventListener('change',benchmark);$('benchmark-metric').addEventListener('change',benchmark);
['experiment-search','group-filter','status-filter'].forEach(id=>$(id).addEventListener(id==='experiment-search'?'input':'change',filter));
document.querySelectorAll('[data-family-filter]').forEach(b=>b.addEventListener('click',()=>{selectedFamily=b.dataset.familyFilter;filter();}));
document.querySelectorAll('[data-quick-group]').forEach(b=>b.addEventListener('click',()=>{$('group-filter').value=b.dataset.quickGroup;filter();$('experiment-grid').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth',block:'start'});}));
$('reset-filters').addEventListener('click',reset);
$('download-json').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(DATA,null,2)],{type:'application/json;charset=utf-8'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='crop_representation_2026-11-09.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});
let beforePrint=[];
window.addEventListener('beforeprint',()=>{beforePrint=Array.from(document.querySelectorAll('details')).map(d=>[d,d.open]);beforePrint.forEach(([d])=>d.open=true);});
window.addEventListener('afterprint',()=>beforePrint.forEach(([d,open])=>d.open=open));
$('print-report').addEventListener('click',()=>window.print());
benchmark();filter();
window.REPORT_READY={experiments:DATA.experiments.length,protocol:$('benchmark-protocol').value,snapshot:DATA.snapshot_utc};
'''

def main():
    d=json.loads((OUT/'data.json').read_text())
    cards=''.join(card(e,i+1) for i,e in enumerate(d['experiments']))
    insights=''.join(f'<article class="insight"><span class="tag">OBSERVATION {i+1:02d}</span><h3>{esc(x["title"])}</h3><p>{esc(x["detail"])}</p></article>' for i,x in enumerate(d['headline_facts']))
    sources=''.join(f'<tr><td><a href="{esc(s["href"])}">{esc(s["path"].replace(str(ROOT)+"/",""))}</a></td><td>{s["bytes"]:,} B</td><td>{esc(s["modified_utc"][:19].replace("T"," "))}</td></tr>' for s in d['source_manifest'])
    def png(name):return 'data:image/png;base64,'+base64.b64encode((OUT/'assets'/name).read_bytes()).decode()
    content=TEMPLATE
    fontdata=base64.b64encode((OUT/'assets/ReportKorean.woff').read_bytes()).decode()
    fontcss='@font-face{font-family:ReportKorean;src:url(data:font/woff;base64,'+fontdata+') format("woff");font-weight:400;font-style:normal;font-display:swap}'
    reportcss=fontcss+CSS.replace('font-family:Inter,Pretendard','font-family:ReportKorean,Inter,Pretendard')
    fields={'CSS':reportcss,'COUNT':str(len(d['experiments'])),'SNAPSHOT':d['snapshot_utc'][:19].replace('T',' '),'INSIGHTS':insights,'CARDS':cards,'SOURCES':sources,'SOURCE_COUNT':str(len(d['source_manifest'])),'DATA':json.dumps(d,ensure_ascii=False,allow_nan=False).replace('</','<\\/'),'JS':JS,'FIG_ADJUST':png('02_adjustment_sensitivity.png'),'FIG_SIZE':png('03_feature_difference_by_size.png')}
    for k,v in fields.items():content=content.replace('__'+k+'__',v)
    license_text=(OUT/'assets/FONT_LICENSE.txt').read_text().replace('</','<\\/')
    content=content.replace('</body>','<script id="font-license" type="text/plain">'+license_text+'</script></body>')
    assert not re.search(r'__[A-Z_]+__',content)
    (OUT/'index.html').write_text(content)
    (OUT/'report.js').write_text(JS)
    print('Built',OUT/'index.html',f'({len(content.encode()):,} bytes)')

if __name__=='__main__':main()
