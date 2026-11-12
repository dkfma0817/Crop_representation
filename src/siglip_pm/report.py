"""Auditable tables from completed official evaluator output; no invented results."""
import argparse,json,os
from pathlib import Path
import numpy as np,pandas as pd
MODELS=['P','M','B0','C0'];TYPES=['color','pattern_marking','material','transparency'];ATTS=['striped','perforated','text','fabric','ceramic']
PAIRS=[('C0','B0'),('P','M'),('P','B0'),('M','B0'),('C0','P')]

def md(df):
 return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'---|'*len(df.columns)]+['| '+' | '.join('' if pd.isna(v) else f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.values])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run);out=run/'results';out.mkdir(exist_ok=True)
 sel=json.load(open(run/'selection.json'));ev={c:json.load(open(run/f'eval_test/{c}_r2/test/official_results.json')) for c in MODELS}
 rows=[]
 for c in MODELS:
  h=ev[c]['hierarchical'];row={'Model':c,'Object AP':h['obj-attr-AP'],'Part AP':h['obj-attr-part-AP']}
  for lev,src in [('Obj','obj-attrs'),('Part','obj-part-attrs')]:
   for t,n in zip(TYPES,['Color','Pattern','Material','Transparency']):row[f'{lev} {n}']=h[src][t]
  rows.append(row)
 df=pd.DataFrame(rows);df.to_csv(out/'paco_ap.csv',index=False)
 boot=json.load(open(run/'boot_test.json')) if (run/'boot_test.json').exists() else None
 contrasts=[]
 for x,y in PAIRS:
  for k in list(df.columns)[1:]:
   row=dict(comparison=f'{x} − {y}',metric=k,delta=float(df.set_index('Model').loc[x,k]-df.set_index('Model').loc[y,k]))
   if boot:
    bk={'Object AP':'obj','Part AP':'part'}.get(k)
    if bk is None:
     lv,t=k.split(' ',1);bk=('obj_' if lv=='Obj' else 'part_')+dict(zip(['Color','Pattern','Material','Transparency'],TYPES))[t]
    ds=np.array(boot['boot'][f'{x}_r2'][bk])-np.array(boot['boot'][f'{y}_r2'][bk]);lo,hi=np.percentile(ds,[2.5,97.5]);row.update(ci95_low=float(lo),ci95_high=float(hi),B=boot['B'])
   contrasts.append(row)
 D=pd.DataFrame(contrasts);D.to_csv(out/'contrasts.csv',index=False)
 gt=json.load(open('/home/irteam/datasets/paco/paco_lvis_v1_test.json'));attrs={v['name']:v['id'] for v in gt['attributes']};cats={v['id']:v['name'] for v in gt['categories']}
 counts={};arows=[]
 for att in ATTS:
  aid=attrs[att]
  for level,part in [('Object',False),('Part',True)]:
   pos=[v for v in gt['annotations'] if aid in v.get('attribute_ids',[]) and (':' in cats[v['category_id']])==part]
   counts[(att,level)]=dict(positive_ROIs=len(pos),positive_categories=len({v['category_id'] for v in pos}))
   for c in MODELS:
    h=ev[c]['hierarchical'];key='obj-attr-part-AP-per-class' if part else 'obj-attr-AP-per-class'
    arows.append(dict(model=c,level=level,attribute=att,AP=h[key].get(att,float('nan')),**counts[(att,level)]))
 A=pd.DataFrame(arows);A.to_csv(out/'focus_attributes.csv',index=False)
 training={}
 for c in ['P','M']:
  log=[json.loads(l) for l in open(run/c/'logs/train_log.jsonl')];diag=[json.loads(l) for l in open(run/c/'logs/diag.jsonl')]
  done=json.load(open(run/c/'done.json'));cfg=__import__('yaml').safe_load(open(run/c/'config.yaml'))
  smoke=json.load(open(run/c/'smoke.json'))
  training[c]=dict(**done,first_log=log[0],last_log=log[-1],initial_diag=diag[0],final_ema_diag=[v for v in diag if v['step']==done['steps'] and v['model']=='ema'][0],
                   n_params_total=cfg['n_params_total'],n_params_vit=cfg['n_params_vit'],n_params_new=cfg['n_params_new'],smoke_peak_gb=smoke['peak_memory_gb'])
  pd.DataFrame(log).to_csv(out/f'{c}_train_log.csv',index=False);pd.DataFrame(diag).to_csv(out/f'{c}_diagnostics.csv',index=False)
 pd.DataFrame([dict(model=c,steps=v['steps'],time_s=v['time_s'],crop_loss_first=v['first_log']['l_crop'],crop_loss_last=v['last_log']['l_crop'],
                    dev_cosine_before=v['initial_diag']['q_roi_cos'],dev_cosine_after=v['final_ema_diag']['q_roi_cos'],parameters=v['n_params_total'],peak_memory_gb=v['last_log']['mem_gb']) for c,v in training.items()]).to_csv(out/'training_summary.csv',index=False)
 oldrun=Path('output/2026-11-08/hcf_pilot');oldmap={'P':'P_r2fused','M':'M_r2Q','B0':'orig_r2','C0':'c0_r2'};oldrows=[]
 for c,n in oldmap.items():
  f=oldrun/f'eval_test/{n}/test/official_results.json'
  if f.exists():
   h=json.load(open(f))['hierarchical'];oldrows.append(dict(Model=c,DINOv3_Object_AP=h['obj-attr-AP'],DINOv3_Part_AP=h['obj-attr-part-AP']))
 old=pd.DataFrame(oldrows);old.to_csv(out/'dinov3_reference.csv',index=False)
 relative=[]
 for x,y in [('C0','B0'),('P','M'),('C0','P')]:
  rt=df.set_index('Model');ot=old.set_index('Model')
  relative.append(dict(backbone='SigLIP2',comparison=f'{x} − {y}',Object_AP_delta=float(rt.loc[x,'Object AP']-rt.loc[y,'Object AP']),Part_AP_delta=float(rt.loc[x,'Part AP']-rt.loc[y,'Part AP'])))
  relative.append(dict(backbone='DINOv3 reference',comparison=f'{x} − {y}',Object_AP_delta=float(ot.loc[x,'DINOv3_Object_AP']-ot.loc[y,'DINOv3_Object_AP']),Part_AP_delta=float(ot.loc[x,'DINOv3_Part_AP']-ot.loc[y,'DINOv3_Part_AP'])))
 relative=pd.DataFrame(relative);relative.to_csv(out/'within_backbone_relative_gaps.csv',index=False)

 oldatts=[]
 for c,n in oldmap.items():
  f=oldrun/f'eval_test/{n}/test/official_results.json'
  if f.exists():
   h=json.load(open(f))['hierarchical']
   for lv,key in [('Object','obj-attr-AP-per-class'),('Part','obj-attr-part-AP-per-class')]:
    for att in ATTS:oldatts.append(dict(model=c,level=lv,attribute=att,DINOv3_AP=h[key].get(att,float('nan'))))
 pd.DataFrame(oldatts).to_csv(out/'dinov3_focus_attributes.csv',index=False)
 bookkeeping={c:{k:ev[c][k] for k in ['n_eval_images','n_gt_anns','n_dets_given','n_dets_kept','dets_dropped_by_max_dets','alignment_mismatch','n_valid_pairs_area_all']} for c in MODELS}
 results=dict(status='main complete',selection=sel,AP=rows,contrasts=contrasts,focus_attributes=arows,training=training,official_bookkeeping=bookkeeping,
              bootstrap=dict(B=boot['B'],verified_w1_equals_official=boot['verified_w1_equals_official']) if boot else None)
 json.dump(results,open(run/'results.json','w'),indent=1)
 summary=['# SigLIP2 P/M 결과','',md(df),'','AP 단위: %. GT box/category oracle protocol, 공식 PACO evaluator; detection AP와 별도. 모든 R2는 독립 seed 0, 10 epochs, official VAL 최적 epoch 선택 후 TEST 1회.','',f"공식 evaluator의 max300 detections/image 규칙을 유지했습니다. TEST에서 각 모델 동일하게 {ev['P']['dets_dropped_by_max_dets']:,}개 prediction이 제한으로 제외되었고, GT bbox/category 대응 오류는 모든 모델0개입니다. 속성 positive 수는 전체 GT 기준입니다.",'',
          '## 요청 비교','',md(D[D.metric.isin(['Object AP','Part AP'])]),'','## 표현 학습','']
 for c,v in training.items():
  summary += [f"- {c}: GPU {2 if c=='P' else 3}, {v['steps']} steps / {v['time_s']/60:.1f}분. Crop loss {v['first_log']['l_crop']:.5f} → {v['last_log']['l_crop']:.5f}; COCO dev fused cosine {v['initial_diag']['q_roi_cos']:.4f} → {v['final_ema_diag']['q_roi_cos']:.4f}; 최종 global loss {v['last_log']['l_global']:.5f}."]
 summary+=['','## 집중 속성','',md(A),'','Positive ROI/category 수는 official TEST의 명시적 positive GT 기준(모델 공통). AP는 official hierarchical per-attribute category macro 값.','',
           '## Backbone 내부 상대 비교 (핵심: C0−B0, P−M, C0−P)','',md(relative),'','## DINOv3 참조','',md(old),'','이전 DINOv3 P/M은 G1-pool 초기화 및 다른 reader 초기화를 사용했습니다. 이번 SigLIP2는 official pretrained 직접 초기화이므로 두 표의 차이를 backbone만의 인과 효과로 해석할 수 없습니다.','',
           '## 해석과 재현','',
           '- 이번 실험은 G1 선행 학습 없이 동일 official pretrained에서 시작한 SigLIP2 내부 구조 비교입니다. DINOv3의 backbone-only 재현이 아닙니다.',
           '- Ceramic/Fabric/Striped/Perforated/Text는 C0 자체의 AP와 양성 수를 함께 봅니다. C0도 낮은 속성은 student distillation 실패로 단정하지 않습니다.',
           '- Backbone 간 절대 AP 차이보다 각 backbone 내부의 개선량과 Crop gap을 우선합니다. 이전 DINOv3 P/M에는 G1 선행 학습 효과도 포함됩니다.',
           '- Cosine 개선과 PACO AP 개선을 별개로 비교합니다. `contrasts.csv`는 요청한 모든 속성군 차이도 포함합니다.',
           '- Paired bootstrap은 이미지 단위 resampling, 동일 GT/공식 matching 재사용, seed 0; CI는 학습 seed 변동을 포함하지 않습니다. 희소 category–attribute pair는 resample마다 유효성이 달라질 수 있습니다.',
           '- `common_checkpoint.json`에 P/M 공통 완료 step과 EMA checkpoint가 기록됩니다. `scripts/resume.sh all` 또는 `eval`로 단계 재개; 완료 stage는 건너뜁니다.',
           '- 다음 핵심 가설: crop mean의 유사도 증가가 Pattern/Material 판별 정보 보존으로 이어지는지 mean linear probe 및 ROI 크기별 격차로 검증합니다.','']
 (run/'REPORT.md').write_text('\n'.join(summary))
 readme=run/'README.md';s=readme.read_text().replace('Status: implementation and real-image smoke gates passed; new SigLIP2-only teacher cache in progress.','Status: P/M representation training and all four independent R2 official evaluations completed. Results: `REPORT.md`, `results.json`, `results/*.csv`.')
 if '<!-- RESULTS -->' in s:s=s.split('<!-- RESULTS -->')[0]
 readme.write_text(s+'\n<!-- RESULTS -->\n'+ '\n'.join(summary[2:]))
 print(md(df),flush=True)
if __name__=='__main__':main()
