import json
from pathlib import Path
import numpy as np,pandas as pd
r=Path('output/2026-11-08/hcf_pilot');root=Path('output/2026-11-07/paco_oracle_attr/data')
J=json.loads(Path('/home/irteam/datasets/paco/paco_lvis_v1_test.json').read_text());M=pd.read_parquet(root/'manifest_test.parquet');Y=np.load(root/'labels_test.npy')
sel=json.loads((r/'selection.json').read_text());conds=['c0_r2','P_r2fused','g1pool_r2'];scores={}
for c in conds:
 with np.load(r/'preds'/c/f"ep{sel[c]['epoch']}"/'probs_test.npz') as z:
  assert np.array_equal(z['ann_id'],M.ann_id.values);scores[c]=z['probs']
off=json.loads((r/'eval_test/P_r2fused/test/official_results.json').read_text());dropped=off['dropped_gt_ann_ids'];assert len(dropped)==off['dets_dropped_by_max_dets'];keep=~M.ann_id.isin(dropped).values
attrs={a['id']:a['name'] for a in J['attributes']};catname={c['id']:c['name'] for c in J['categories']};pairs=set(off['hierarchical']['per-obj-part-attr-AP']);ids=[i for i in J['attr_type_to_attr_idxs']['material'] if attrs[i]!='other(material)']
# Short side measured on the 384-letterboxed full-image input, before patch embedding.
scale=384/np.maximum(M.W.values,M.H.values);short=np.minimum(M.r.values-M.l.values,M.b.values-M.t.values)*scale
rng=np.random.default_rng(0);units=[];bins=[0,16,32,64,float('inf')]
for (level,cat),index in M[keep].groupby(['level','category_id']).groups.items():
 index=np.asarray(list(index))
 for a in ids:
  if catname[cat]+':'+attrs[a] not in pairs:continue
  pos=index[Y[index,a]==1];neg=index[Y[index,a]==0]
  if len(pos)<5 or len(neg)<20:continue
  count=min(4096,len(pos)*len(neg))
  if len(pos)*len(neg)<=4096:pi=np.repeat(pos,len(neg));ni=np.tile(neg,len(pos))
  else:pi=rng.choice(pos,count);ni=rng.choice(neg,count)
  cmp={c:np.sign(scores[c][pi,a]-scores[c][ni,a]) for c in conds};C=cmp['c0_r2'];P=cmp['P_r2fused']
  row=dict(level=level,category=catname[cat],attribute=attrs[a],n_positive=len(pos),n_negative=len(neg),sampled_pairs=len(pi),rank_crop=float((C>0).mean()+.5*(C==0).mean()),rank_fused=float((P>0).mean()+.5*(P==0).mean()),rank_distilled=float((cmp['g1pool_r2']>0).mean()+.5*(cmp['g1pool_r2']==0).mean()),crop_only=float(((C>0)&(P<0)).mean()),fused_only=float(((P>0)&(C<0)).mean()),tie_any=float(((C==0)|(P==0)).mean()),bins=[])
  for lo,hi in zip(bins[:-1],bins[1:]):
   mask=(short[pi]>=lo)&(short[pi]<hi)
   if mask.sum()<20:continue
   v={k:float((cmp[c][mask]>0).mean()+.5*(cmp[c][mask]==0).mean()) for k,c in [('crop','c0_r2'),('fused','P_r2fused'),('distilled','g1pool_r2')]}
   row['bins'].append(dict(lo=lo,hi=None if np.isinf(hi) else hi,n_pairs=int(mask.sum()),**v))
  units.append(row)
summary=[]
for level in ['object','part']:
 U=[u for u in units if u['level']==level]
 def macro(U,key):
  a={}
  for u in U:a.setdefault(u['attribute'],[]).append(u[key])
  return float(np.mean([np.mean(v) for v in a.values()])) if a else None
 row=dict(level=level,n_units=len(U),n_attributes=len(set(u['attribute'] for u in U)),n_sampled_pairs=sum(u['sampled_pairs'] for u in U),**{k:macro(U,k) for k in ['rank_crop','rank_fused','rank_distilled','crop_only','fused_only','tie_any']},sizes=[])
 for lo,hi in zip(bins[:-1],bins[1:]):
  E=[dict(attribute=u['attribute'],**b) for u in U for b in u['bins'] if b['lo']==lo]
  row['sizes'].append(dict(lo=lo,hi=None if np.isinf(hi) else hi,n_units=len(E),n_pairs=sum(e['n_pairs'] for e in E),**{k:macro(E,k) for k in ['crop','fused','distilled']}))
 summary.append(row)
print(json.dumps(summary,indent=1))
for level in ['object','part']:
 print('TOP CATEGORY GAPS',level)
 U=[u for u in units if u['level']==level and u['n_positive']>=20 and u['n_negative']>=40]
 for u in sorted(U,key=lambda x:x['rank_fused']-x['rank_crop'])[:8]:print(u['category'],u['attribute'],u['n_positive'],u['n_negative'],'crop %.1f fused %.1f distilled %.1f'%tuple(100*u[k] for k in ['rank_crop','rank_fused','rank_distilled']))
out=dict(protocol='Exploratory within-category positive-vs-negative attribute ranking. Fixed seed 0, up to 4096 pairs per category-attribute unit; minimum 5 positives and 20 negatives. Macro over categories then attributes. Ties count 0.5 in ranking accuracy. Crop-only/fused-only exclude tied pairs. Positive-ROI short side bins are diagnostic, with differing unit composition; not a causal scale test. This metric is not official PACO AP.',summary=summary,units=units)
p=Path('results/hcf_oracle_gap_2026-11-08/material_ranking_audit.json');p.write_text(json.dumps(out,indent=1));print('SAVED',p)
