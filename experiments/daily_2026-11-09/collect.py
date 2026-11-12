"""Collect existing experiment artifacts into a dated, read-only report snapshot."""
import csv,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/home/irteam/Crop_representation')
OUT=Path(__file__).resolve().parent
SG=Path('/home/irteam/sg/crop-rep')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def number(x):
    try:return float(x)
    except (ValueError,TypeError):return None

ALIASES={
 'name':['Model','model','name','cell','method'],
 'overall':['Overall AP','Overall','overall','M2 Overall'],
 'object':['Object AP','Object','obj','object','M2 Object'],
 'part':['Part AP','Part','part','M2 Part'],
 'pattern':['Obj Pattern','obj_pat','obj_pattern_marking','object/pattern_marking'],
 'material':['Obj Material','obj_mat','obj_material','object/material'],
 'epoch':['epoch','ep','reader_epoch'],
}
def norm_rows(obj):
    if isinstance(obj,dict):obj=[dict(name=k,**v) for k,v in obj.items() if isinstance(v,dict)]
    if not isinstance(obj,list):return []
    out=[]
    for r in obj:
        if not isinstance(r,dict):continue
        x={}
        for key,aliases in ALIASES.items():
            v=next((r[a] for a in aliases if a in r),None)
            x[key]=str(v) if key=='name' and v is not None else number(v)
        if x['object'] is None and x['overall'] is None:continue
        if x['overall'] is None and x['object'] is not None and x['part'] is not None:x['overall']=(x['object']+x['part'])/2
        if x['name'] is None:continue
        out.append(x)
    return out

def off(path,name,epoch=None):
    h=read(path)['hierarchical']
    return dict(name=name,overall=(h['obj-attr-AP']+h['obj-attr-part-AP'])/2,object=h['obj-attr-AP'],part=h['obj-attr-part-AP'],pattern=h['obj-attrs']['pattern_marking'],material=h['obj-attrs']['material'],epoch=epoch,source=str(path))

def main():
    dino=read(OUT/'dinov3_audit.json');sig=read(OUT/'siglip_audit.json')
    ex=[]
    groups={'hcf_official_val':'input','hier_h0h1_s0':'input','pixeldit_cond_p1p2_s0':'objective','pcond_p1p2_s0':'objective','g1_upscaled_teacher_s0':'input','g1_native768_s0':'input','coarse_fine_ablation_s0':'structure','fine_bottleneck_diag_s0':'diagnostic','dinov3_crop_aware_pretrain':'objective','siglip-structure':'structure','siglip-extra-phase2':'objective','siglip-localcue':'objective','siglip-pixeldit':'objective','siglip-completed-compilation':'context'}
    for audit,family in [(dino,'DINOv3'),(sig,'SigLIP2')]:
        for e in audit['experiments']:
            e=dict(e);e['family']=family;e['group']=groups[e['id']]
            e['scope']='today' if e['group']!='context' else 'context'
            raw=e.get('results',{});tables=[]
            if isinstance(raw,list):tables.append(dict(label=e.get('result_split','Official Test'),protocol='official_val' if e['id']=='siglip-localcue' else 'official_test',rows=norm_rows(raw)))
            elif isinstance(raw,dict):
                for key in ['official_test_r2','test_r2','test','official_val_r2','val_r2','val']:
                    if key in raw:
                        rows=norm_rows(raw[key])
                        if rows:tables.append(dict(label='Official Test · R2' if 'test' in key else 'Official Val · R2',protocol='official_test' if 'test' in key else 'official_val',rows=rows))
                if e['id']=='fine_bottleneck_diag_s0':
                    fast=raw['D2_fast']
                    for split in ['test','val']:
                        rows=norm_rows(fast.get(split,{}))
                        if rows:tables.append(dict(label=f'D2 Official {split.title()} · R2 · 점추정',protocol='official_'+split,rows=rows))
            for key,label,protocol in [('mean_results','Official Test · Mean + Linear','official_test_mean'),('val_results','Official Val · R2','official_val')]:
                if key in e:
                    rows=norm_rows(e[key])
                    if rows:tables.append(dict(label=label,protocol=protocol,rows=rows))
            e['tables']=tables
            if e['id']=='dinov3_crop_aware_pretrain':
                status=read(ROOT/'output/2026-11-09/dinov3_crop_aware_pretrain/control/status.json')
                e['live_status']=status
                p=ROOT/'output/2026-11-09/dinov3_crop_aware_pretrain/evals/summary.csv'
                rows=list(csv.DictReader(p.open()));e['tables']=[dict(label='PACO Val · project M2 · R2 / 완료된 평가만',protocol='project_val',rows=norm_rows(rows))]
                e['progress_step']=status.get('step');e['progress_target']=100000
                e['sources'].append(str(p))
                e['results']['live_status']=status
                e['status']='complete_training_evaluation_pending' if status.get('done') else 'running'
            ex.append(e)
    # Current official evaluator on exact causal checkpoints; use full precision JSON.
    run=ROOT/'output/2026-11-09/hcf_official_val';sel=read(run/'selection.json')
    names={'r384_initial':'P 초기화 · 이번 단계 0','r384_cosine':'P384 · crop 정렬','r384_centered':'P384 · 중심화 추가','r768_cosine':'P768 · crop 정렬','r768_centered':'P768 · 중심화 추가','followup_control':'768 · 같은 양 6k 연장','followup_pixel':'768 · PixelDiT 6k 추가'}
    bench=[];officialval=[]
    pilot=ROOT/'output/2026-11-08/hcf_pilot';oldsel=read(pilot/'selection.json')
    for key,name in [('orig_r2','순정 DINOv3 · full image'),('g1pool_r2','G1-pool · crop 증류 12k')]:
        bench.append(off(pilot/f'eval_test/{key}/test/official_results.json',name,oldsel[key]['epoch']))
        officialval.append(off(pilot/f'eval_val/{key}/ep{oldsel[key]["epoch"]}/official_results.json',name,oldsel[key]['epoch']))
    for key,name in names.items():
        bench.append(off(run/f'eval_test/{key}/test/official_results.json',name,sel[key]['epoch']))
        officialval.append(off(run/f'eval_val/{key}/ep{sel[key]["epoch"]}/official_results.json',name,sel[key]['epoch']))
    for target,split in [(bench,'test'),(officialval,'val')]:
        ep=oldsel['c0_r2']['epoch'];path=pilot/(f'eval_test/c0_r2/test/official_results.json' if split=='test' else f'eval_val/c0_r2/ep{ep}/official_results.json')
        target.append(off(path,'C0 · Crop 재인코딩 참조',ep))
    prior=read(SG/'results/hcf_material_causal_2026-11-08/report.json')
    valR2=[];valMean=[];extra_sources=[]
    for folder,label in [(ROOT/'output/2026-11-06/ivc_v3/probe_bil','순정 DINOv3 · bilinear ROI'),(ROOT/'output/2026-11-06/g1_cross_distill/probe_frozen_full','순정 DINOv3 · native token'),(ROOT/'output/2026-11-06/g1_cross_distill/probe_G1pool_full','G1-pool · crop 증류 12k')]:
        for reader,table in [('r2',valR2),('r0',valMean)]:
            path=folder/reader
            dev=[json.loads(s) for s in (path/'curves_dev.jsonl').read_text().splitlines()]
            ep=max((x for x in dev if x['epoch']>=1),key=lambda x:(x['dev']['M0']['overall'],-x['epoch']))['epoch']
            val=next(json.loads(s) for s in (path/'curves.jsonl').read_text().splitlines() if json.loads(s)['epoch']==ep)
            table.extend(norm_rows([dict(name=label,epoch=ep,**val['val']['M2'])]))
            extra_sources.extend([str(path/'curves_dev.jsonl'),str(path/'curves.jsonl')])
    for key,name in names.items():
        if key.startswith('followup'):continue
        c=prior['cells'][key]
        for reader,table in [('r2fused',valR2),('r0',valMean)]:
            x=c['readers'][reader];table.extend(norm_rows([dict(name=name,epoch=x['epoch'],**x['metrics'])]))
    for table,reader in [(valR2,'r2'),(valMean,'mean')]:
        p=read(ROOT/'results/roi_pair_feature_difference_2026-11-09/provenance.json')['conditions']
        c=p['C0'][reader];table.extend(norm_rows([dict(name='C0 · Crop 재인코딩 참조',epoch=c['epoch'],**c['val_M2'])]))
    for e in ex:
        if e['id']=='hcf_official_val':
            e['tables']=[dict(label='Official Test · R2 · 기준선과 같은 평가',protocol='official_test',rows=bench),dict(label='Official Val · R2 · epoch 선택에 사용',protocol='official_val',rows=officialval)]
    # Methods completed on Nov8 are background; their official reevaluation was today.
    method_names={'r384_initial':'P 초기화','r384_cosine':'Crop 정렬','r384_mae':'MAE pure','r384_mae_hybrid':'MAE + crop','r384_jepa':'JEPA pure','r384_jepa_hybrid':'JEPA + crop','r384_pixeldit':'PixelDiT pure','r384_pixeldit_hybrid':'PixelDiT + crop'}
    methodtable=[off(run/f'eval_test/{k}/test/official_results.json',v,sel[k]['epoch']) for k,v in method_names.items()]
    ex.insert(1,dict(id='prior-generative',title='MAE · JEPA · PixelDiT: 무엇을 예측하게 할 것인가',family='DINOv3',group='objective',scope='context',status='background_training_today_test',question='평균 crop feature만 맞추는 대신 픽셀 또는 공간 feature 예측을 시키면 속성 표현이 더 좋아지는가?',control='공통 P 초기화에서 직접 crop 정렬 12k. Pure는 crop loss를 빼고, hybrid는 유지합니다.',architecture_change='MAE: masked full image→ROI decoder→crop RGB. JEPA: 같은 masked context→EMA crop feature. PixelDiT: unmasked full image→ROI memory + noisy crop→RGB flow velocity.',training_change='COCO 12k ×16 images, 공통 encoder 초기화·stream·EMA. Pure=L_method+0.5 global, Hybrid=Pure+crop. 평가에서 보조 decoder를 제거합니다.',initialization='이미 G1-pool 증류를 거친 Coarse + 새 Fine/Interaction의 공통 초기 상태. 순정 DINOv3 시작점이 아닙니다.',budget='표현 학습은 11/8 완료; 오늘은 공식 Val/Test 재평가. 새 12k 학습으로 중복 계산하지 않음.',findings=['공식 Test에서 PixelDiT hybrid는 Overall 42.07, crop 정렬 41.61입니다. 점추정의 소폭 차이이며 이 재평가 표에는 CI가 없습니다. Pure 과제가 crop 정렬을 일관되게 대체했다는 결과는 없습니다.'],limitations=['소형 ROI용 변형이며 원논문 재현이 아닙니다. 생성 decoder의 loss 감소를 encoder 속성 AP 개선으로 해석하지 않습니다.'],tables=[dict(label='오늘 재평가 · Official Test / R2',protocol='official_test',rows=methodtable)],sources=[str(run/'table_test.csv'),str(ROOT/'output/2026-11-08/hcf_material_causal/architecture_plan.json'),str(ROOT/'output/2026-11-08/hcf_material_causal/architecture_source/src/hcf/crop_objectives.py')]))
    follow=read(ROOT/'output/2026-11-09/hcf_followup_2h/report.json')
    followrows=[dict(name={'initial':'P768-centered 시작점','control':'같은 양 6k 연장','pixel':'PixelDiT 보조 6k'}[k],**v['r2fused']['metrics']) for k,v in follow['results'].items()]
    ex.insert(2,dict(id='hcf-pixel-followup',family='DINOv3',group='objective',scope='today',status='complete_val_and_test',title='768 모델에 PixelDiT를 추가해도 도움이 되는가',question='이미 좋은 P768-centered에서 픽셀 복원 보조 과제만 추가하면, 같은 양의 기존 학습 연장보다 더 좋아지는가?',control='동일 P768-centered checkpoint에서 6,000 step 연장한 control. 학습량과 시작점이 같습니다.',architecture_change='기존 encoder 유지. 8×8 ROI memory를 조건으로 Patch DiT 2 block(width384) + local pixel transformer(width64)를 추가합니다. 평가 때 decoder는 제거합니다.',training_change='두 arm 모두 crop +0.5 centered +0.5 global 유지. Pixel arm만 flow MSE 추가. Encoder LR 0.2배, warmup200, 동일 6k stream.',initialization='P768-centered 12k final EMA. 384 PixelDiT pure/hybrid의 초기화와 다릅니다.',budget='6k 추가 ×16 images. 새 probe 10 epoch. 기존 12k와 별도 continuation.',findings=['Val project M2 Overall은 control49.67, Pixel50.02: +0.35 [−0.67,+1.64]로 이득이 불확실합니다. 오늘 공식 Test 점추정도 생성됐으며, 같은 arm의 Object47.12/Part45.21입니다.'],limitations=['Val project M2의 CI를 공식 Test 숫자에 붙이지 않습니다. 두 프로토콜의 reader epoch 선택도 다릅니다.'],tables=[dict(label='Val · project M2 / R2 · B=1,000',protocol='project_val',rows=norm_rows(followrows)),dict(label='Official Test / R2 · 점추정',protocol='official_test',rows=[r for r in bench if '6k' in r['name']])],sources=[str(ROOT/'output/2026-11-09/hcf_followup_2h/report.json'),str(ROOT/'output/2026-11-09/hcf_followup_2h/plan.json'),str(run/'table_test.csv')]))
    pairroot=ROOT/'results/roi_pair_feature_difference_2026-11-09';pair=read(pairroot/'summary.json')
    ex.append(dict(id='roi-pair-difference',family='DINOv3',group='diagnostic',scope='today',status='complete_diagnostic',title='Crop만 맞는 pair에서 feature 차이가 사라졌는가',question='ROI 간 차이의 크기를 잃었는지, 차이는 남지만 판별 방향이 달라졌는지 구분합니다.',control='같은 category·attribute의 명시적 양성/음성 Val pair를 네 모델에서 공통 사용. A=Crop만 정답, B=둘 다 정답, C=Student만 정답.',architecture_change='새 구조 없음. C0/P384/P768/P768-centered의 기존 mean feature를 비교합니다.',training_change='추가 학습 없음. 기존 Mean/R2 score exact replay; L2-normalized pair difference, 기존 head 방향 투영, category·크기·Crop margin 통제.',initialization='기존 checkpoint를 그대로 고정해 추론. Mean feature cache는 새로 추출하되 encoder/reader는 학습하지 않았습니다.',budget='129,328 pair, 11,109 ROI, 2,051 Val images. 양 끝 image 의존성을 반영한 vertex bootstrap B1,000, seed0.',findings=['A에서 차이 크기가 작아도 B에서도 비슷합니다. Crop margin을 가깝게 매칭하면 Object Material·Pattern의 magnitude/direction A−B CI가 0을 포함합니다.','같은 P384 실패 pair에서 P768이 Material 약41%, Pattern 약45%를 회복합니다. Relative Distillation을 새 Method로 제안할 원인·효과 근거는 아직 부족합니다.'],limitations=['기존 ranking은 Test 집계라 Val pair ID가 없습니다. 선택 원리를 재사용해 Val 공통 pair를 구성했습니다.','동일 이미지의 유효 반대-label pair는 0개. Fixed pair graph와 match에 조건부인 bootstrap이며 학습 seed 변동은 포함하지 않습니다.'],tables=[],sources=[str(pairroot/'REPORT.md'),str(pairroot/'summary.json'),str(pairroot/'validation.json')]))
    # Read-only source snapshot manifest, including original report inputs.
    sources=[]
    for e in ex:
        for value in e.get('sources',[]):
            p=Path(value)
            if not p.is_absolute():p=ROOT/p
            if p.is_file() and str(p) not in [s['path'] for s in sources]:
                st=p.stat();sources.append(dict(path=str(p),href=os.path.relpath(p,OUT),bytes=st.st_size,modified_utc=datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),sha256=sha(p)))
    for value in extra_sources:
        p=Path(value)
        if str(p) not in [s['path'] for s in sources]:sources.append(dict(path=str(p),href=os.path.relpath(p,OUT),bytes=p.stat().st_size,modified_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),sha256=sha(p)))
    for rows in [bench,officialval]:
        for r in rows:
            p=Path(r['source'])
            if str(p) not in [s['path'] for s in sources]:sources.append(dict(path=str(p),href=os.path.relpath(p,OUT),bytes=p.stat().st_size,modified_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),sha256=sha(p)))
    snapshot=datetime.now(timezone.utc).isoformat(timespec='seconds')
    result=dict(date='2026-11-09',snapshot_utc=snapshot,title='Crop Representation · 오늘의 실험 기록',scope='2026-11-09 run folders and today-completed evaluations, plus explicitly marked prior context and ROI pair diagnosis. Static snapshot, no new model training for this report.',experiments=ex,benchmarks={'official_test':bench,'official_val':officialval,'project_val_r2':valR2,'project_val_mean':valMean},source_manifest=sources,roi_pair=pair,headline_facts=[dict(title='768 입력의 효과',detail='Official Test Overall P384 41.61 → P768 46.04. 입력·token 수·연산량이 함께 변한 결과입니다.'),dict(title='생성 과제의 추가 이득',detail='조건 의존성을 높인 DINO P1/P2는 AP 개선이 없고, SigLIP2 PixelDiT S1/S2는 Test Overall이 하락했습니다.'),dict(title='양방향 interaction의 필요성',detail='A0/A1/A2 Overall 차이는 불확실합니다. Coarse→Fine만 둔 A1에서는 Fine 추가 +1.35 [0.54,2.04]가 관찰됐습니다.'),dict(title='오류 pair의 원인',detail='크기 축소는 성공 pair에도 있습니다. 난도를 맞춘 뒤 차이 방향의 오류 특이성은 확인하지 못했습니다.')])
    (OUT/'data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    with (OUT/'experiments.csv').open('w',newline='') as f:
        keys=['id','title','family','group','scope','status','question','control','architecture_change','training_change','initialization','budget']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for e in ex:writer.writerow({k:json.dumps(e.get(k),ensure_ascii=False) if isinstance(e.get(k),(list,dict)) else e.get(k,'') for k in keys})
    print('Collected',len(ex),'experiment families;',len(sources),'source files;',snapshot)

if __name__=='__main__':main()
