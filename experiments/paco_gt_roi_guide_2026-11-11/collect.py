"""Read existing PACO predictions and annotations; build a small, verified report snapshot. CPU only."""
import base64
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / 'output/2026-11-07/paco_oracle_attr/data'
ANN = Path('/home/irteam/datasets/paco/paco_lvis_v1_val.json')


def safe(x):
    if isinstance(x, dict):
        return {k: safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def main():
    J = json.loads(ANN.read_text())
    M = pd.read_parquet(DATA / 'manifest_val.parquet')
    Y = np.load(DATA / 'labels_val.npy')
    names = [a['name'] for a in sorted(J['attributes'], key=lambda a: a['id'])]
    type_of = {a: t for t, aa in J['attr_type_to_attr_idxs'].items() for a in aa}
    gt = {a['id']: a for a in J['annotations']}
    source_paths = [ANN, DATA / 'manifest_val.parquet', DATA / 'labels_val.npy']
    pred_paths = {
        'c0': ROOT / 'output/2026-11-07/paco_oracle_attr/preds/c0_crop/probs_val.npz',
        'p768': ROOT / 'output/2026-11-09/hcf_official_val/preds/r768_cosine/ep5/probs_val.npz',
    }
    predictions = {}
    checks = {}
    for key, path in pred_paths.items():
        z = np.load(path)
        check = {k: bool(np.array_equal(z[k], M[k].values)) for k in ['ann_id', 'image_id', 'category_id']}
        assert all(check.values()), check
        assert z['probs'].shape == Y.shape == (31919, 59)
        assert np.isfinite(z['probs']).all()
        err = float(np.max(np.abs(z['probs'] - 1 / (1 + np.exp(-z['logits'].astype(np.float64))))))
        assert err < 1e-6
        checks[key] = dict(**check, sigmoid_max_error=err, n_rows=len(M))
        predictions[key] = {k: z[k] for k in ['probs', 'logits']}
        source_paths.append(path)

    examples = []
    csv_rows = []
    selected = [1695, 1383557, 1383559, 27612, 44487, 54056]
    for ann_id in selected:
        idx = int(np.flatnonzero(M.ann_id.values == ann_id)[0])
        row = M.iloc[idx]
        a = gt[ann_id]
        assert int(row.image_id) == a['image_id'] and int(row.category_id) == a['category_id']
        label = np.full(59, -1, np.int8)
        for ty, attrs in J['attr_type_to_attr_idxs'].items():
            if not a['unknown_' + ty]:
                label[attrs] = 0
        for attr in a['attribute_ids']:
            if label[attr] == 0:
                label[attr] = 1
        assert np.array_equal(label, Y[idx])
        siblings = M[(M.image_id == row.image_id) & (M.obj_ann_id == row.obj_ann_id)]
        example = dict(ann_id=ann_id, image_id=int(row.image_id), category_id=int(row.category_id),
                       category=row.category, level=row.level, W=int(row.W), H=int(row.H),
                       box=[int(row[k]) for k in ['l', 't', 'r', 'b']], bbox_gt=a['bbox'],
                       label=Y[idx].astype(int).tolist(), image_key=str(int(row.image_id)),
                       related=[dict(ann_id=int(r.ann_id), category=r.category, level=r.level,
                                     bbox=[int(getattr(r, k)) for k in ['l', 't', 'r', 'b']])
                                for r in siblings.itertuples()],
                       predictions={k: {v: predictions[k][v][idx].astype(float).tolist()
                                        for v in ['logits', 'probs']} for k in predictions})
        examples.append(example)
        for attr in range(59):
            csv_rows.append(dict(ann_id=ann_id, image_id=int(row.image_id), category=row.category, level=row.level,
                                 attribute=names[attr], attr_type=type_of[attr], label=int(Y[idx, attr]),
                                 c0_logit=float(predictions['c0']['logits'][idx, attr]),
                                 c0_prob=float(predictions['c0']['probs'][idx, attr]),
                                 p768_logit=float(predictions['p768']['logits'][idx, attr]),
                                 p768_prob=float(predictions['p768']['probs'][idx, attr])))

    images = {}
    for ex in examples:
        if ex['image_key'] not in images:
            path = Path(M.loc[M.ann_id == ex['ann_id'], 'path'].iloc[0])
            images[ex['image_key']] = 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()
            source_paths.append(path)

    results = []
    for split in ['val', 'test']:
        for key, label, path in [
            ('c0', 'C0 Crop + R2 · ep6', ROOT / f'output/2026-11-07/paco_oracle_attr/eval/c0_crop/{split}/official_results.json'),
            ('p768', 'P768 Cosine + R2 · ep5', ROOT / f'output/2026-11-09/hcf_official_val/eval_{split}/r768_cosine/' / ('ep5' if split == 'val' else 'test') / 'official_results.json'),
        ]:
            z = json.loads(path.read_text())
            h = z['hierarchical']
            results.append(dict(key=key, name=label, split=split, object=h['obj-attr-AP'],
                                part=h['obj-attr-part-AP'], overall=(h['obj-attr-AP'] + h['obj-attr-part-AP']) / 2,
                                object_types=h['obj-attrs'], part_types=h['obj-part-attrs'],
                                flat_joint_ap=100*z['joint_attr_summary']['AP'],
                                n_images=z['n_eval_images'], n_given=z['n_dets_given'], n_kept=z['n_dets_kept'],
                                n_valid=z['n_valid_pairs_area_all'], dropped=z['dets_dropped_by_max_dets'],
                                source=str(path.relative_to(ROOT))))
            source_paths.append(path)
    source_paths += [ROOT / p for p in [
        'src/ivc/bbdata.py', 'src/ivc/common.py', 'src/readout/models.py', 'src/readout/common.py',
        'src/pacoattr/audit.py', 'src/pacoattr/infer.py', 'src/pacoattr/official_eval.py',
        'src/hcf/probe.py', 'src/hcf/model.py', 'src/ivc/train_bb.py',
        'output/2026-11-09/hcf_official_val/selection.json',
        'output/2026-11-09/hcf_official_val/scripts/infer_val.py',
        'third_party/research/2026-10-28/paco_crop_finetune/code/eval_tables.py',
        'third_party/research/2026-10-24/dinov2_oracle_region_decoder/paco_lp_official_metric/code/pom_common.py',
        'third_party/research/2026-10-24/dinov2_oracle_region_decoder/paco_lp_official_metric/code/aggregate.py',
        'third_party/research/2026-10-26/multilevel_relation_decoder/code/eval_mlr.py',
        'third_party/research/2026-10-30/paco_region_reader/code/common_rr.py',
    ]]
    source_paths += [Path('/home/irteam/datasets/lvis_src/paco/paco/evaluation') / p for p in [
        'paco_eval_api/eval.py', 'paco_eval_api/results.py', 'utils/paco_utils.py']]
    provenance = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
                  for p in dict.fromkeys(source_paths)]
    snapshot = dict(title='PACO GT ROI Attribute 평가 가이드', date='2026-11-11',
                    scope='Existing DINOv3 PACO pipelines, saved Val/Test predictions. No new training or inference.',
                    names=names, types=J['attr_type_to_attr_idxs'], examples=examples, images=images,
                    checks=checks, results=results, sources=provenance,
                    example_selection='Fixed hand-selected illustration IDs: chair + back + seat, bicycle, cup, handbag. Not a representative accuracy sample.',
                    model_notes=dict(c0='Original DINOv3-B/16 crop384 + independent R2 reader ep6, dev-M0 selection from the Nov6 G1 experiment.',
                                     p768='r768_cosine, final EMA encoder + independent R2 reader ep5; selected by official Val mean(Object,Part).'))
    (OUT / 'data.json').write_text(json.dumps(safe(snapshot), ensure_ascii=False, indent=2))
    with (OUT / 'example_predictions.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    (OUT / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(json.dumps(dict(examples=len(examples), images=len(images), csv_rows=len(csv_rows), checks=checks), indent=2))


if __name__ == '__main__':
    main()
