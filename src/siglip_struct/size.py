"""Same official GT-area stratification as the completed P8 comparison."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from paco.evaluation.paco_eval_api import PACO,PACOEval,PACOResults
from paco.evaluation.utils.paco_utils import heirachrichal_APs
from src.pacoattr.official_eval import make_dets,ANN
from src.siglip_pm.size_diagnostic import BINS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--model',required=True);ap.add_argument('--select',default='');a=ap.parse_args();run=Path(a.run);c=a.model
    M=pd.read_parquet('output/2026-11-07/paco_oracle_attr/data/manifest_test.parquet');selected=json.loads((Path(a.select) if a.select else run/'selection.json').read_text());ep=selected[f'{c}_r2']['epoch']
    z=np.load(run/f'preds/{c}_r2/ep{ep}/probs_test.npz');assert np.array_equal(z['ann_id'],M.ann_id.values)
    gt=PACO(ANN.format('test'));dt=PACOResults(gt,make_dets(M,z['probs']),max_dets=300)
    wh={im['id']:max(im['width'],im['height']) for im in gt.dataset['images']}
    for dataset in [gt,dt]:
        for ann in dataset.dataset['annotations']:
            box=ann['bbox'];ann['area']=box[2]*box[3]*(384/wh[ann['image_id']])**2
    E=PACOEval(gt,dt,'bbox',attr_ap_type='usual');E.params.area_rng=[[lo,hi] for _,lo,hi in BINS];E.params.area_rng_lbl=[n for n,_,_ in BINS]
    E.evaluate();E.evaluate_joint_attr();E.accumulate_joint_attr();rows=[]
    categories={v['id']:v['name'] for v in gt.dataset['categories']}
    for i,(name,lo,hi) in enumerate(BINS):
        ev=SimpleNamespace(lvis_gt=gt,lvis_dt=dt,eval_joint_attr={'precision':E.eval_joint_attr['precision'][:,:,:,i:i+1]});h=heirachrichal_APs(ev)
        for lv,key in [('Object','obj-attrs'),('Part','obj-part-attrs')]:
            count=sum(lo<=ann['area']<=hi and (':' in categories[ann['category_id']])==(lv=='Part') for ann in gt.dataset['annotations'])
            rows.append(dict(model=c,level=lv,size_bin=name,roi_count=count,Color=h[key]['color'],Pattern=h[key]['pattern_marking'],Material=h[key]['material'],Transparency=h[key]['transparency']))
    pd.DataFrame(rows).to_csv(run/f'results/roi_size_ap_{c}.csv',index=False)


if __name__=='__main__':main()
