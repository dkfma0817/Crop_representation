"""ROI-size AP on PACO official Val: the src.siglip_struct.size protocol (GT-box area on the 384 canvas, official matching) with the split set to val."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from paco.evaluation.paco_eval_api import PACO, PACOEval, PACOResults
from paco.evaluation.utils.paco_utils import heirachrichal_APs

from src.pacoattr.official_eval import ANN, make_dets
from src.siglip_pm.size_diagnostic import BINS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    M = pd.read_parquet('output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet')
    z = np.load(a.pred)
    assert np.array_equal(z['ann_id'], M.ann_id.values)
    gt = PACO(ANN.format('val'))
    dt = PACOResults(gt, make_dets(M, z['probs']), max_dets=300)
    wh = {im['id']: max(im['width'], im['height']) for im in gt.dataset['images']}
    for dataset in [gt, dt]:
        for ann in dataset.dataset['annotations']:
            box = ann['bbox']
            ann['area'] = box[2] * box[3] * (384 / wh[ann['image_id']]) ** 2
    E = PACOEval(gt, dt, 'bbox', attr_ap_type='usual')
    E.params.area_rng = [[lo, hi] for _, lo, hi in BINS]
    E.params.area_rng_lbl = [n for n, _, _ in BINS]
    E.evaluate()
    E.evaluate_joint_attr()
    E.accumulate_joint_attr()
    rows = []
    categories = {v['id']: v['name'] for v in gt.dataset['categories']}
    for i, (name, lo, hi) in enumerate(BINS):
        ev = SimpleNamespace(lvis_gt=gt, lvis_dt=dt, eval_joint_attr={'precision': E.eval_joint_attr['precision'][:, :, :, i:i + 1]})
        h = heirachrichal_APs(ev)
        for lv, key in [('Object', 'obj-attrs'), ('Part', 'obj-part-attrs')]:
            count = sum(lo <= ann['area'] <= hi and (':' in categories[ann['category_id']]) == (lv == 'Part') for ann in gt.dataset['annotations'])
            rows.append(dict(model=a.model, level=lv, size_bin=name, roi_count=count, Color=h[key]['color'], Pattern=h[key]['pattern_marking'], Material=h[key]['material'], Transparency=h[key]['transparency']))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
