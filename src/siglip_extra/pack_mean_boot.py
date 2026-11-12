"""Recover only area-all official matching for bootstrap from fixed predictions.
The full official TEST result already exists. Run the UNMODIFIED official matching
and accumulation for its identical area-all slice, validate every hierarchical
and per-pair AP against that reference, and save the existing official pack format.
No model/epoch choice or prediction changes. Falls back to full official run if needed.
"""
import argparse,copy,hashlib,json,pickle,time
from pathlib import Path
import numpy as np,pandas as pd
from paco.evaluation.paco_eval_api import PACO,PACOEval,PACOResults
from paco.evaluation.utils.paco_utils import heirachrichal_APs
from src.pacoattr.official_eval import ANN,make_dets,pack_boot

def same(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
 return bool(np.isclose(float(a),float(b),rtol=0,atol=1e-9,equal_nan=True))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pred',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--out',required=True);ap.add_argument('--compare-pack',default='');a=ap.parse_args();t0=time.time()
 M=pd.read_parquet('output/2026-11-07/paco_oracle_attr/data/manifest_test.parquet');z=np.load(a.pred);assert np.array_equal(z['ann_id'],M.ann_id.values)
 dets=make_dets(M,z['probs'].astype(np.float64));reference=json.load(open(a.reference))
 def evaluator(all_areas=False):
  gt=PACO(ANN.format('test'));dt=PACOResults(gt,copy.deepcopy(dets),max_dets=300);E=PACOEval(gt,dt,'bbox',attr_ap_type='usual')
  if all_areas:E.run()
  else:
   E.params.area_rng=[E.params.area_rng[0]];E.params.area_rng_lbl=['all'];E.evaluate();E.evaluate_joint_attr();E.accumulate_joint_attr()
  return E
 E=evaluator();h=heirachrichal_APs(E);fallback=False
 if not same(h,reference['hierarchical']):
  del E;E=evaluator(True);h=heirachrichal_APs(E);fallback=True
 assert same(h,reference['hierarchical']),'Fixed prediction matching does not reproduce the completed full official result'
 pack=pack_boot(E);proof={}
 if a.compare_pack:
  other=pickle.load(open(a.compare_pack,'rb'))
  for k in ['K','I','img_ids','joint_attr_ids']:assert pack[k]==other[k],k
  assert np.array_equal(pack['precision'],other['precision'])
  assert len(pack['entries'])==len(other['entries'])
  for x,y in zip(pack['entries'],other['entries']):
   assert x[:2]==y[:2] and all(np.array_equal(xx,yy) for xx,yy in zip(x[2:],y[2:]))
  proof=dict(exact_full_official_precision_equal=True,exact_every_matching_entry_equal=True)
 out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True);tmp=out.with_suffix(out.suffix+'.tmp');pickle.dump(pack,open(tmp,'wb'),protocol=4);tmp.replace(out)
 result=dict(status='passed',full_reference=a.reference,fixed_predictions=a.pred,every_hierarchical_and_per_pair_AP_matches_full=True,
             method='Official PACOEval unchanged; same area-all slice only; original pack_boot unchanged',fallback_full_evaluation=fallback,seconds=time.time()-t0,**proof)
 json.dump(result,open(out.with_suffix('.provenance.json'),'w'),indent=1);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
