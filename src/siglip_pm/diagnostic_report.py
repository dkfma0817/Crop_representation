import argparse,json
from pathlib import Path
import pandas as pd
from src.siglip_pm.report import md

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run);mr=run/'mean';sel=json.load(open(mr/'selection.json'));rows=[]
 for c in ['C0','B0','P','M']:
  h=json.load(open(mr/f'eval_test/{c}_linear/test/official_results.json'))['hierarchical']
  r={'Model':c,'Epoch':sel[f'{c}_linear']['epoch'],'Object AP':h['obj-attr-AP'],'Part AP':h['obj-attr-part-AP']}
  for lv,src in [('Obj','obj-attrs'),('Part','obj-part-attrs')]:
   for t,n in [('color','Color'),('pattern_marking','Pattern'),('material','Material'),('transparency','Transparency')]:r[f'{lv} {n}']=h[src][t]
  rows.append(r)
 D=pd.DataFrame(rows);D.to_csv(run/'results/mean_linear_ap.csv',index=False);diff=[]
 for x,y in [('C0','B0'),('P','M'),('P','B0'),('M','B0'),('C0','P')]:
  for k in list(D.columns)[2:]:diff.append(dict(comparison=f'{x} − {y}',metric=k,delta=float(D.set_index('Model').loc[x,k]-D.set_index('Model').loc[y,k])))
 pd.DataFrame(diff).to_csv(run/'results/mean_linear_contrasts.csv',index=False)
 old=json.load(open(run/'results.json'));sim=[]
 for c in ['P','M']:
  for when,key in [('Before','initial_diag'),('After EMA12k','final_ema_diag')]:
   v=old['training'][c][key];sim.append(dict(model=c,checkpoint=when,fused_teacher_cosine=v['q_roi_cos'],coarse_teacher_cosine=v['coarse_roi_cos'],image_embedding_cosine=v['image_embed_cos'],spatial_mean_cosine=v['mean_cos']))
 S=pd.DataFrame(sim);S.to_csv(run/'results/crop_similarity.csv',index=False)
 summary=['','## 추가 진단 — Mean feature linear probe','',md(D),'','기존 R0 구현(coverage-weighted ROI mean → level별 z-score → 선형 heads)을 사용했습니다. 통계는 각 표현의 같은 4,000 train images에서 계산하고, 네 조건 모두 동일한 10 epochs, flip/batch/optimizer/scheduler/seed0 및 official VAL 선택 규칙을 사용했습니다.','',
          '## Crop teacher similarity (COCO dev, 468 images)','',md(S),'','Cosine 변화는 속성 AP와 별도 지표입니다.','',
          '## ROI 크기별 Crop gap','',md(pd.read_csv(run/'results/roi_size_crop_gaps.csv')),'','ROI 크기는 GT bbox를 longest-side384 canvas에 매핑한 면적/16²로 정의했습니다. fresh in-memory official GT/DT의 area field만 bin 정의에 맞추고, 모든 GT box/category와 prediction을 유지하여 공식 matching 및 hierarchical AP를 계산했습니다. tiny<4, small4–16, medium16–64, large≥64 coarse-token area.','']
 for f in ['README.md','REPORT.md']:
  p=run/f;s=p.read_text();marker='<!-- DIAGNOSTICS -->'
  if marker in s:s=s.split(marker)[0]
  p.write_text(s+'\n'+marker+'\n'+'\n'.join(summary))
 json.dump(dict(status='done',mean_AP=rows,mean_contrasts=diff,similarity=sim),open(run/'diagnostic_results.json','w'),indent=1);print(md(D),flush=True)
if __name__=='__main__':main()
