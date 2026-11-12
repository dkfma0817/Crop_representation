"""COCO-only category mapping and teacher similarity calibration; no GPU/PACO reads."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from src.config import FEAT
from src.msd.pretrain import eval_exclusions

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--base',required=True);a=ap.parse_args();run=Path(a.run);run.mkdir(exist_ok=True,parents=True)
 if (run/'negative_calibration.json').exists():print('Existing fixed calibration retained');return
 shared=Path(a.base)/'shared';ids=np.load(shared/'image_ids.npy');df=pd.read_parquet(f'{FEAT}/stream_rois.parquet').set_index(['image_id','k'])
 cat=df.reindex(pd.MultiIndex.from_product([ids,range(4)],names=['image_id','k']))['cat_id'].to_numpy(dtype=np.int64).reshape(-1,4)
 np.save(run/'object_categories.npy',cat)
 teacher=np.load(shared/'t_object.npy',mmap_mode='r').reshape(-1,768);cats=cat.reshape(-1);known=np.where((cats>=0)&~np.repeat(np.isin(ids,eval_exclusions()['raev2_stream']),4))[0]
 idx=np.sort(np.random.default_rng(0).choice(known,min(4096,len(known)),replace=False));x=np.array(teacher[idx]);x=x/np.linalg.norm(x,axis=1,keepdims=True);sim=x@x.T
 off=~np.eye(len(x),dtype=bool);same=(cats[idx,None]==cats[idx][None,:])&off;sv=sim[same];av=sim[off]
 q90=float(np.quantile(sv,.9));threshold=min(.95,q90)
 bins=np.linspace(-1,1,81);rows=[]
 for name,v in [('all_other_roi',av),('same_category',sv),('different_category',sim[off&~same])]:
  hist,_=np.histogram(v,bins);rows += [dict(group=name,lo=float(bins[i]),hi=float(bins[i+1]),count=int(k)) for i,k in enumerate(hist)]
 pd.DataFrame(rows).to_csv(run/'teacher_similarity_distribution.csv',index=False)
 cfg=dict(status='fixed_before_training',seed=0,sampled_known_COCO_ROIs=len(idx),same_category_pairs=int(same.sum()),
          all_pair_quantiles={str(q):float(np.quantile(av,q)) for q in [.1,.5,.9,.95,.99]},same_category_quantiles={str(q):float(np.quantile(sv,q)) for q in [.1,.5,.9,.95,.99]},
          threshold=threshold,rule='min(0.95, COCO same-category teacher cosine 90th percentile); fixed, no PACO labels/results',
          same_category_excluded_fraction=float((sv>=threshold).mean()),all_pair_excluded_fraction=float((av>=threshold).mean()),
          temperature=.07,loss_weight=.1,loss_warmup_steps=400,negative_source='all trusted in-batch teacher ROIs; includes known same-category ROIs when present; unknown/random/nested category=-1')
 json.dump(cfg,open(run/'negative_calibration.json','w'),indent=1);print(json.dumps(cfg),flush=True)
if __name__=='__main__':main()
