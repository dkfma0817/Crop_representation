"""Optional official matching/AP stratified by GT ROI box area on the 384 canvas.
Only area fields are changed in fresh in-memory GT/DT objects to define bins;
boxes, attributes, matching and all model predictions stay unchanged.
"""
import argparse,json,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from paco.evaluation.paco_eval_api import PACO,PACOEval,PACOResults
from paco.evaluation.utils.paco_utils import heirachrichal_APs
from src.pacoattr.official_eval import make_dets,ANN
BINS=[('tiny_lt4_tokens',0,32**2),('small_4to16_tokens',32**2,64**2),('medium_16to64_tokens',64**2,128**2),('large_ge64_tokens',128**2,1e12)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--model',choices=['P','M','B0','C0']);a=ap.parse_args();run=Path(a.run)
 M=pd.read_parquet('output/2026-11-07/paco_oracle_attr/data/manifest_test.parquet');sel=json.load(open(run/'selection.json'));rows=[]
 for c in ([a.model] if a.model else ['P','M','B0','C0']):
  ep=sel[f'{c}_r2']['epoch'];z=np.load(run/f'preds/{c}_r2/ep{ep}/probs_test.npz');assert np.array_equal(z['ann_id'],M.ann_id.values)
  gt=PACO(ANN.format('test'));dt=PACOResults(gt,make_dets(M,z['probs']),max_dets=300)
  wh={im['id']:max(im['width'],im['height']) for im in gt.dataset['images']}
  for dataset in [gt,dt]:
   for ann in dataset.dataset['annotations']:
    b=ann['bbox'];ann['area']=b[2]*b[3]*(384/wh[ann['image_id']])**2
  E=PACOEval(gt,dt,'bbox',attr_ap_type='usual');E.params.area_rng=[[lo,hi] for _,lo,hi in BINS];E.params.area_rng_lbl=[n for n,_,_ in BINS]
  E.evaluate();E.evaluate_joint_attr();E.accumulate_joint_attr()
  for i,(name,lo,hi) in enumerate(BINS):
   ev=SimpleNamespace(lvis_gt=gt,lvis_dt=dt,eval_joint_attr={'precision':E.eval_joint_attr['precision'][:,:,:,i:i+1]})
   h=heirachrichal_APs(ev)
   for lv,key in [('Object','obj-attrs'),('Part','obj-part-attrs')]:
    part=lv=='Part';cats={v['id']:v['name'] for v in gt.dataset['categories']}
    count=sum(lo<=ann['area']<=hi and (':' in cats[ann['category_id']])==part for ann in gt.dataset['annotations'])
    rows.append(dict(model=c,level=lv,size_bin=name,roi_count=count,Color=h[key]['color'],Pattern=h[key]['pattern_marking'],Material=h[key]['material'],Transparency=h[key]['transparency']))
 if a.model:
  pd.DataFrame(rows).to_csv(run/f'results/roi_size_ap_{a.model}.csv',index=False)
  json.dump(dict(status='done',model=a.model,bins=BINS,rows=rows),open(run/f'roi_size_diagnostic_{a.model}.json','w'),indent=1)
  print(json.dumps(dict(model=a.model,status='done')),flush=True);return
 pd.DataFrame(rows).to_csv(run/'results/roi_size_ap.csv',index=False)
 d=pd.DataFrame(rows);delta=[]
 for student in ['B0','P','M']:
  for lv in ['Object','Part']:
   for name,_,_ in BINS:
    x=d[(d.model=='C0')&(d.level==lv)&(d.size_bin==name)].iloc[0];y=d[(d.model==student)&(d.level==lv)&(d.size_bin==name)].iloc[0]
    delta.append(dict(comparison=f'C0 − {student}',level=lv,size_bin=name,Pattern=float(x.Pattern-y.Pattern),Material=float(x.Material-y.Material)))
 pd.DataFrame(delta).to_csv(run/'results/roi_size_crop_gaps.csv',index=False)
 json.dump(dict(status='done',size='GT bbox area mapped to 384 longest-side canvas; unit=16x16 patch area',bins=BINS,rows=rows),open(run/'roi_size_diagnostic.json','w'),indent=1)
 print(json.dumps(delta),flush=True)
if __name__=='__main__':main()
