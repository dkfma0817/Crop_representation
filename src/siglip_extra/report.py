"""Separate follow-up report; only complete, fixed-hyperparameter official results."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd
from src.siglip_pm.report import md
from src.siglip_pm.supervise import EPY,ROOT
ATTRS=['ceramic','fabric','paper','striped','perforated','text']
TYPES=[('color','Color'),('pattern_marking','Pattern'),('material','Material'),('transparency','Transparency')]

def flat(c,h):
 r={'Model':c,'Object AP':h['obj-attr-AP'],'Part AP':h['obj-attr-part-AP']}
 for lv,src in [('Obj','obj-attrs'),('Part','obj-part-attrs')]:
  for t,n in TYPES:r[f'{lv} {n}']=h[src][t]
 return r

def comparisons(df,boot,key):
 models=set(df.Model);pairs=[('E1','P'),('E2','P'),('E1','E2'),('C0','P'),('C0','E1'),('C0','E2')]
 if 'Continue' in models:pairs += [('Continue','P'),('E1','Continue'),('E2','Continue')]
 out=[];T=df.set_index('Model')
 for x,y in pairs:
  for col in list(df.columns)[1:]:
   r=dict(comparison=f'{x} − {y}',metric=col,delta=float(T.loc[x,col]-T.loc[y,col]))
   if boot:
    bk={'Object AP':'obj','Part AP':'part'}.get(col)
    if bk is None:
     lv,n=col.split(' ',1);bk=('obj_' if lv=='Obj' else 'part_')+dict((n,t) for t,n in TYPES)[n]
    if f'{x}_{key}' in boot['boot'] and f'{y}_{key}' in boot['boot']:
     z=np.array(boot['boot'][f'{x}_{key}'][bk])-np.array(boot['boot'][f'{y}_{key}'][bk]);lo,hi=np.percentile(z,[2.5,97.5]);r.update(ci95_low=float(lo),ci95_high=float(hi),B=boot['B'])
   out.append(r)
 return pd.DataFrame(out)

def ensure_mean_boot(run,models,deadline):
 # The waiting supervisor may have started before this optional mean-bootstrap
 # enhancement was written. Reuse the SAME fixed predictions; no model/epoch changes.
 if deadline-time.time()<1200:return
 env=os.environ.copy();env.update(PYTHONPATH=f'/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 from concurrent.futures import ThreadPoolExecutor
 def pack(c):
  out=run/f'mean/eval_test/{c}_linear/test'
  if (out/'boot_pack.pkl').exists():return
  ep=json.load(open(run/'mean/selection.json'))[f'{c}_linear']['epoch']
  with open(run/f'logs/mean_boot_pack_{c}.log','w') as f:
   subprocess.run([EPY,'-u','-m','src.siglip_extra.pack_mean_boot','--pred',str(run/f'mean/preds/{c}_linear/ep{ep}/probs_test.npz'),'--reference',str(out/'official_results.json'),'--out',str(out/'boot_pack.pkl')],stdout=f,stderr=subprocess.STDOUT,env=env,check=True,timeout=max(1,deadline-time.time()))
 with ThreadPoolExecutor(max_workers=4) as p:list(p.map(pack,models))
 path=run/'mean/boot_test.json';need=True
 if path.exists():need=set(models)-{k.removesuffix('_linear') for k in json.load(open(path))['boot']}
 if need:
  args=[EPY,'-u','-m','src.siglip_extra.bootstrap','--run',str(run),'--mean']+(['--control'] if 'Continue' in models else [])
  with open(run/'logs/mean_bootstrap.log','a') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.time()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--base',required=True);a=ap.parse_args();run=Path(a.run);base=Path(a.base);out=run/'results';out.mkdir(exist_ok=True)
 models=['P','E1','E2','C0']+(['Continue'] if (run/'eval_test/Continue_r2/test/official_results.json').exists() else [])
 E={c:json.load(open(run/f'eval_test/{c}_r2/test/official_results.json')) for c in models};R=pd.DataFrame([flat(c,E[c]['hierarchical']) for c in models]);R.to_csv(out/'r2_ap.csv',index=False)
 deadline=json.load(open(base/'budget.json'))['deadline_unix'];ensure_mean_boot(run,models,deadline)
 mean={c:json.load(open(run/f'mean/eval_test/{c}_linear/test/official_results.json')) for c in models};L=pd.DataFrame([flat(c,mean[c]['hierarchical']) for c in models]);L.to_csv(out/'mean_probe_ap.csv',index=False)
 boot=json.load(open(run/'boot_test.json')) if (run/'boot_test.json').exists() else None
 mb=json.load(open(run/'mean/boot_test.json')) if (run/'mean/boot_test.json').exists() else None
 D=comparisons(R,boot,'r2');D.to_csv(out/'r2_contrasts.csv',index=False);LD=comparisons(L,mb,'linear');LD.to_csv(out/'mean_probe_contrasts.csv',index=False)
 gt=json.load(open('/home/irteam/datasets/paco/paco_lvis_v1_test.json'));att={v['name']:v['id'] for v in gt['attributes']};cat={v['id']:v['name'] for v in gt['categories']};rows=[]
 for attr in ATTRS:
  for level,ispart,key in [('Object',False,'obj-attr-AP-per-class'),('Part',True,'obj-attr-part-AP-per-class')]:
   positives=[v for v in gt['annotations'] if att[attr] in v.get('attribute_ids',[]) and (':' in cat[v['category_id']])==ispart]
   counts=dict(positive_ROIs=len(positives),positive_categories=len({v['category_id'] for v in positives}))
   for c in models:
    for reader,ev in [('R2',E),('Mean linear',mean)]:rows.append(dict(model=c,reader=reader,level=level,attribute=attr,AP=ev[c]['hierarchical'][key].get(attr,float('nan')),**counts))
 A=pd.DataFrame(rows);A.to_csv(out/'focus_attributes.csv',index=False)
 train={};stats=[]
 for c in ['E1','E2']+(['Continue'] if (run/'Continue/done.json').exists() else []):
  cfg=json.load(open(run/c/'config.json'));done=json.load(open(run/c/'done.json'));logs=[json.loads(x) for x in open(run/c/'train_log.jsonl')];diags=[json.loads(x) for x in open(run/c/'diagnostics.jsonl')]
  train[c]=dict(config=cfg,done=done,initial=diags[0],final=diags[-1],first_log=logs[0],last_log=logs[-1],smoke=json.load(open(run/c/'smoke.json')))
  pd.DataFrame(logs).to_csv(out/f'{c}_train.csv',index=False);pd.DataFrame(diags).to_csv(out/f'{c}_dev_diagnostics.csv',index=False)
  v=train[c];stats.append(dict(experiment=cfg['label'],source_sha256=cfg['init_sha256'],additional_steps=done['additional_steps'],minutes=done['seconds']/60,
     crop_cosine_before=v['initial']['crop_cosine'],crop_cosine_after=v['final']['crop_cosine'],coarse_cosine_before=v['initial']['coarse_cosine'],coarse_cosine_after=v['final']['coarse_cosine'],
     extra_loss_before=v['initial']['extra_loss'],extra_loss_after=v['final']['extra_loss'],peak_VRAM_GiB=max(x['peak_memory_GiB'] for x in logs),median_steps_per_s=float(np.median([x['steps_per_s'] for x in logs]))))
 T=pd.DataFrame(stats);T.to_csv(out/'training_summary.csv',index=False)
 assert len(set(v['config']['init_sha256'] for v in train.values()))==1
 # Objective-specific interpretation uses only the one planned final TEST evaluation.
 rt=R.set_index('Model');lt=L.set_index('Model');answers=[]
 for title,table,col,x,y in [
  ('E1 vs E2 Object Material (R2)',rt,'Obj Material','E1','E2'),('E2 vs E1 Object Pattern (R2)',rt,'Obj Pattern','E2','E1'),
  ('E1 Object Material vs P (mean)',lt,'Obj Material','E1','P'),('E2 Object Pattern vs P (mean)',lt,'Obj Pattern','E2','P'),
  ('E1 Part Material vs P (mean)',lt,'Part Material','E1','P'),('E2 Part Pattern vs P (mean)',lt,'Part Pattern','E2','P')]:answers.append(dict(question=title,delta_AP=float(table.loc[x,col]-table.loc[y,col])))
 focus=A[(A.reader=='R2')&(A.level=='Object')].pivot(index='attribute',columns='model',values='AP')
 for attr in ATTRS:
  answers.append(dict(question=f'{attr}: E1−P / E2−P / C0−E1 / C0−E2 (Object R2)',E1_minus_P=float(focus.loc[attr,'E1']-focus.loc[attr,'P']),E2_minus_P=float(focus.loc[attr,'E2']-focus.loc[attr,'P']),C0_minus_E1=float(focus.loc[attr,'C0']-focus.loc[attr,'E1']),C0_minus_E2=float(focus.loc[attr,'C0']-focus.loc[attr,'E2'])))
 sacrifice=[]
 for c in ['E1','E2']:
  sacrifice.append(dict(Model=c,Obj_Color_minus_P=float(rt.loc[c,'Obj Color']-rt.loc['P','Obj Color']),Part_AP_minus_P=float(rt.loc[c,'Part AP']-rt.loc['P','Part AP']),Part_Color_minus_P=float(rt.loc[c,'Part Color']-rt.loc['P','Part Color'])))
 if 'Continue' in models:
  for c in ['E1','E2']:
   answers.append(dict(question=f'{c} vs matched Continue: Object Material / Pattern',R2_Material=float(rt.loc[c,'Obj Material']-rt.loc['Continue','Obj Material']),R2_Pattern=float(rt.loc[c,'Obj Pattern']-rt.loc['Continue','Obj Pattern']),Mean_Material=float(lt.loc[c,'Obj Material']-lt.loc['Continue','Obj Material']),Mean_Pattern=float(lt.loc[c,'Obj Pattern']-lt.loc['Continue','Obj Pattern'])))
 results=dict(status='complete',base=str(base),models=models,source_common_sha256=next(iter(train.values()))['config']['init_sha256'],R2=R.to_dict('records'),mean_probe=L.to_dict('records'),R2_contrasts=D.to_dict('records'),mean_contrasts=LD.to_dict('records'),
              focus_attributes=json.loads(A.to_json(orient='records')),training=train,answers=answers,tradeoffs=sacrifice,control_completed='Continue' in models,
              protocol='same seed0 independently initialized10-epoch readers; official VAL mean object/part selects; TEST fixed once; no label-based mining; focus attributes exploratory; DINOv3 M2 not numerically compared')
 json.dump(results,open(run/'results.json','w'),indent=1)
 report=['# P12k 이후 E1/E2 추가 학습 결과','',
   '모든 후속 모델은 동일한 완료 SigLIP2 P12k EMA에서 독립 시작했습니다. 표의 E1/E2/Continue는 각각 추가4k 단계이며 처음부터16k 학습한 모델이 아닙니다.','',
   '## Frozen encoder + 독립 R2 (Official TEST AP)','',md(R),'',f"기본 평가와 동일한 공식 max300 detections/image 규칙을 유지했습니다. 모든 모델에서 같은 {E['P']['dets_dropped_by_max_dets']:,}개 prediction이 제외됩니다. Positive 수는 전체 official GT 기준입니다.",'','## Mean feature linear probe (Official TEST AP)','',md(L),'',
   'Mean probe는 같은 coverage-weighted ROI mean, level별4000-image z-score, 기존 R0 linear heads, flip/batch/AdamW/schedule/10epochs/seed0, official VAL 선택 규칙을 사용했습니다.','',
   '## 주요 Mean 비교 및 paired image95% CI','',md(LD[LD.metric.isin(['Obj Material','Obj Pattern','Part Material','Part Pattern'])]),'',
   '## R2 비교 및 paired image95% CI','',md(D[D.metric.isin(['Object AP','Part AP','Obj Material','Obj Pattern'])]),'',
   '## Object 속성 (탐색적)','',md(A[(A.reader=='R2')&(A.level=='Object')]),'',
   'Positive ROI/category 수는 official TEST의 명시적 positive 기준입니다. 속성별 TEST 결과를 재확인하여 hyperparameter를 선택하지 않았습니다. 전체 object/part 및 Mean 속성표는CSV에 있습니다.','',
   '## 학습 및 source checkpoint','',md(T),'','source SHA256: `'+results['source_common_sha256']+'`','',
   '## 질문별 수치','',json.dumps(answers,indent=2,ensure_ascii=False),'','## Color / Part 변화','',md(pd.DataFrame(sacrifice)),'',
   '## 해석','',
   '- Contrastive/local loss 또는 teacher cosine의 개선만으로 성공을 판단하지 않습니다. COCO dev diagnostics, Mean Material/Pattern, R2 AP와 control 대비 차이를 함께 봅니다.',
   '- P 대비 양의 변화도 Continue보다 나쁘면 새로운 loss의 추가 학습량 대비 이점을 지지하지 않습니다. CI는 paired image resampling1000회이며 training seed 변동은 포함하지 않습니다.',
   '- DINOv3에서 관찰한 Mean M2 Material gap5.46은 평가 지표가 다르므로 본 공식 GT-ROI AP와 직접 수치 비교하지 않습니다.',
   '- E2는 crop re-encoding과 full-image의 관찰 scale이 달라 위치별 teacher/student feature의 정확한 동일성을 보장하지 않습니다. 실제 stride8 fine 해상도 기준 tiny ROI mask와 valid-content coverage를 적용했습니다.',
   '- 새 encoder/decoder는 없으며 P 구조는 동일합니다.','',
   '## 재개','',
   '`bash output/2026-11-08/siglip2_pm_extra_s0/scripts/resume.sh`는 완료 stage를 건너뜁니다. 원래8h deadline을 유지하므로 새 budget이 필요한 재실행은 명시적으로 deadline을 갱신해야 합니다.','']
 (run/'REPORT.md').write_text('\n'.join(report))
 p=run/'README.md';s=p.read_text();marker='<!-- RESULTS -->'
 if marker in s:s=s.split(marker)[0]
 p.write_text(s.replace('Status: queued behind the complete original P/M + B0/C0 R2 evaluation pipeline. Existing jobs will finish normally; no process is killed, and their training code is unchanged.','Status: requested E1/E2 representation learning, independent R2, essential mean probes and official evaluation completed.')+'\n'+marker+'\n'+'\n'.join(report))
 print(md(R)+'\n\nMean probe:\n'+md(L),flush=True)
if __name__=='__main__':main()
