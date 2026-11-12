"""Aggregate frozen ADE predictions and matched Train patch pairs."""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata, spearmanr
import timm
import torch

from infer import OUT, ROOT, RUN, NAMES, SEED, ade, sources
from mask_audit import P23, sha
from src.fgtrain.common import write_json
from src.fgtrain.n1 import metrics as paco_metrics

MAP_SOURCE = 'https://github.com/facebookresearch/Mask2Former/blob/main/datasets/ade20k_instance_catid_mapping.txt'
PANOPTIC_SOURCE = 'https://github.com/facebookresearch/Mask2Former/blob/main/datasets/prepare_ade20k_pan_seg.py'
SCENE_SOURCE = 'https://github.com/CSAILVision/sceneparsing/blob/master/README.md'
REGIMES = ('interior', 'mixed')


def read(path):
    return json.loads(Path(path).read_text())


def official_scores():
    paths = sources(P23)
    paths['P3'] = (P23/'P3/export/encoder.pt', P23/'ade', 'P3')
    folders = dict(Original=RUN/'Frozen', B0=RUN/'B0', P1=paths['P1'][1].parent/'P1', P2=P23/'P2', P3=P23/'P3')
    configs = [read(directory/'config.json') for _, directory, _ in paths.values()]
    assert all(q['cfg'] == configs[0]['cfg'] and q['world'] == 2 and q['n_train'] == 20210 and q['n_val'] == 2000 for q in configs)
    refs, official = {}, {}
    for name, (ckpt, directory, key) in paths.items():
        epoch = read(folders[name]/'selection.json')['epoch']
        paco = folders[name]/'official'/f'ep{epoch}'/'official_results.json'
        check = read(paco)
        assert check['split'] == 'val' and check['n_eval_images'] == 2410 and check['alignment_mismatch'] == 0
        ade_done = directory/'done.json'
        done = read(ade_done)
        assert done['iters'] == 40000 and done['unchanged'][key]
        head = directory/f'head_{key}.pt'
        old = directory/'val_inter_union.npz'
        with np.load(old) as archive:
            reference = ade.miou(archive[f'I_{key}'], archive[f'U_{key}'])
        assert abs(reference-done['results'][key]['mIoU']) < 1e-10
        official[name] = dict(**paco_metrics(paco), ade=reference, paco_reader_epoch=epoch)
        refs[name] = dict(encoder=str(ckpt.relative_to(ROOT)), sha256=sha(ckpt), head=str(head.relative_to(ROOT)), head_sha256=sha(head),
                          checksum=read(directory/'config.json')['checksum'][key], ade_json=str(ade_done.relative_to(ROOT)),
                          ade_inter_union=str(old.relative_to(ROOT)), paco_json=str(paco.relative_to(ROOT)))
    return official, refs


def auc(positive, negative):
    n, m = len(positive), len(negative)
    rank = rankdata(np.r_[positive, negative])
    return (rank[:n].sum()-n*(n+1)/2)/(n*m)


def cosine_pairs(features, a, b):
    return (features[a]*features[b]).sum(-1).T


def semantic_pairs(train):
    feats = train['features']
    meta = train['metadata']
    pools = [{c: np.flatnonzero((meta[:,0] == r) & (meta[:,1] == c)) for c in range(ade.NC)} for r in range(2)]
    eligible = [c for c in range(ade.NC) if min(len(pools[r][c]) for r in (0,1)) >= 8]
    rng = np.random.default_rng(SEED+2)
    records, distributions = {}, {}
    bins = np.linspace(-.25, 1., 51)
    for r, region in enumerate(REGIMES):
        by_model = [[] for _ in NAMES]
        positive, negative = [], []
        for c in eligible:
            pool = pools[r][c]
            anchor = rng.choice(pool, size=256, replace=True)
            other = rng.choice([x for x in eligible if x != c], len(anchor))
            pos = np.array([rng.choice(pool[meta[pool,2] != meta[a,2]]) for a in anchor])
            neg = np.array([rng.choice(pools[r][o][meta[pools[r][o],2] != meta[a,2]]) for a,o in zip(anchor,other)])
            assert np.all(meta[anchor,2] != meta[pos,2]) and np.all(meta[anchor,2] != meta[neg,2])
            pp, nn = cosine_pairs(feats, anchor, pos), cosine_pairs(feats, anchor, neg)
            positive.append(pp)
            negative.append(nn)
            for k in range(len(NAMES)):
                by_model[k].append(dict(class_id=c, n_images=len(pool), within=float(pp[k].mean()), between=float(nn[k].mean()),
                                        separation=float(pp[k].mean()-nn[k].mean()), auc=float(auc(pp[k], nn[k]))))
        p = np.concatenate(positive, axis=1)
        q = np.concatenate(negative, axis=1)
        records[region] = {}
        distributions[region] = {'edges': bins.tolist()}
        for k, name in enumerate(NAMES):
            rows = by_model[k]
            records[region][name] = dict(classes=rows, **{x:float(np.mean([row[x] for row in rows])) for x in ['within','between','separation','auc']})
            distributions[region][name] = {key: (np.histogram(a.clip(bins[0], bins[-1]), bins)[0]/len(a)).tolist() for key,a in [('within',p[k]),('between',q[k])]}
    return records, distributions, pools, eligible


def hard_pair_geometry(feats, meta, pools, a, b):
    result = {}
    rng = np.random.default_rng(SEED+3+150*a+b)
    for r, region in enumerate(REGIMES):
        pa, pb = pools[r][a], pools[r][b]
        if min(len(pa),len(pb)) < 8:
            continue
        positive, negative = [], []
        for x,y in ((pa,pb),(pb,pa)):
            anchor = rng.choice(x, 256, replace=True)
            pos = np.array([rng.choice(x[meta[x,2] != meta[i,2]]) for i in anchor])
            neg = np.array([rng.choice(y[meta[y,2] != meta[i,2]]) for i in anchor])
            positive.append(cosine_pairs(feats, anchor, pos))
            negative.append(cosine_pairs(feats, anchor, neg))
        positive = np.concatenate(positive, axis=1)
        negative = np.concatenate(negative, axis=1)
        result[region] = {name:float(auc(positive[k],negative[k])) for k,name in enumerate(NAMES)}
    return result


def class_iou(confusion):
    inter = np.diagonal(confusion, axis1=-2, axis2=-1)
    union = confusion.sum(-2)+confusion.sum(-1)-inter
    return np.divide(100*inter, union, out=np.full(union.shape, np.nan), where=union>0)


def finite_list(array):
    return [float(x) if np.isfinite(x) else None for x in array]


def val_presence_counts():
    cache = OUT/'val_gt_image_counts.npy'
    if cache.exists():
        return np.load(cache)

    def present(item):
        with Image.open(item[1]) as label:
            return np.bincount(np.asarray(label).ravel(), minlength=151)[1:151] > 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = np.sum(list(pool.map(present, ade.listing('val'))), axis=0)
    np.save(cache, counts)
    return counts


def main():
    official, checkpoints = official_scores()
    saved_source = read(P23/'source_manifest.json')['files']
    geometry_files = ['src/fgtrain/patch_preserve.py','src/fgtrain/patch_train.py','src/g1x/data.py',
                      'src/viz/c0_pca.py','src/g1eval/ade.py']
    verified_geometry = {path:sha(ROOT/path) for path in geometry_files}
    assert all(saved_source[path] == digest for path,digest in verified_geometry.items())
    with np.load(OUT/'val_confusion.npz') as archive:
        val = dict(archive)
    with np.load(OUT/'train_features.npz') as archive:
        train = dict(archive)
    classes_file = Path(ade.ROOT)/'objectInfo150.txt'
    labels = [line.split('\t')[-1].strip() for line in classes_file.read_text().splitlines()[1:] if line.strip()]
    assert len(labels) == 150
    class_short = [name.split(',')[0] for name in labels]
    val_ids = val['ids'].tolist()
    assert val_ids == [Path(ip).stem for ip,_ in ade.listing('val')]
    stored = sources(P23)
    reproduction = {}
    for k, name in enumerate(NAMES):
        _, directory, key = stored[name]
        with np.load(directory/'val_inter_union.npz') as archive:
            old_i, old_u = archive[f'I_{key}'], archive[f'U_{key}']
        new_i, new_u = val['intersection'][k], val['union'][k]
        base_iou = 100*old_i.sum(0)/old_u.sum(0)
        new_iou = 100*new_i.sum(0)/new_u.sum(0)
        reproduction[name] = dict(reference=official[name]['ade'], rerun=ade.miou(new_i,new_u),
                                  max_class_iou_delta=float(abs(new_iou-base_iou).max()),
                                  exact_per_image_counts=bool(np.array_equal(old_i,new_i) and np.array_equal(old_u,new_u)),
                                  images_differing=int(np.any((old_i!=new_i)|(old_u!=new_u),axis=1).sum()),
                                  delta_pp=ade.miou(new_i,new_u)-official[name]['ade'])
    confusion = val['confusion']
    iou = class_iou(confusion)
    gt_pixels = confusion[0,0].sum(1)
    assert np.all(confusion[:,0].sum(2) == gt_pixels[None,:])
    regions = {}
    for r,name in enumerate(('full','boundary','interior')):
        regions[name] = {model:dict(pixel_accuracy=100*float(confusion[k,r].trace()/confusion[k,r].sum()),
                                    miou=float(np.nanmean(iou[k,r])), pixels=int(confusion[k,r].sum()), classes=int(np.isfinite(iou[k,r]).sum()))
                         for k,model in enumerate(NAMES)}
    map_file = OUT/'ade20k_instance_catid_mapping.txt'
    mapping = [tuple(map(int,line.split())) for line in map_file.read_text().splitlines()[1:] if line.strip()]
    thing_ids = sorted({s-1 for _,s,_ in mapping})
    stuff_ids = sorted(set(range(150))-set(thing_ids))
    assert len(thing_ids)==100 and len(stuff_ids)==50
    type_stats = {name:{group:float(np.nanmean(iou[k,0,ids])) for group,ids in [('Thing',thing_ids),('Stuff',stuff_ids)]}
                  for k,name in enumerate(NAMES)}
    train_sem, distributions, pools, eligible = semantic_pairs(train)
    gt_image_count = val_presence_counts()
    assert gt_image_count.shape == (ade.NC,) and (gt_image_count > 0).all()
    rates = 100*confusion[:,0] / np.maximum(gt_pixels,1)[None,:,None]
    eligible_val = (gt_pixels >= 10000) & (gt_image_count >= 10)
    pairs = [(float((rates[1,a,b]+rates[1,b,a])/2), a,b) for a in range(150) for b in range(a+1,150) if eligible_val[a] and eligible_val[b]]
    pairs.sort(key=lambda q:(-q[0],q[1],q[2]))
    hard = []
    for score,a,b in pairs[:12]:
        hard.append(dict(class_ids=[a,b], labels=[class_short[a],class_short[b]], b0_symmetric_rate=score,
                         directions={name:[float(rates[k,a,b]),float(rates[k,b,a])] for k,name in enumerate(NAMES)},
                         train_auc=hard_pair_geometry(train['features'],train['metadata'],pools,a,b)))
    class_rows = []
    groups = set(thing_ids)
    for c in range(150):
        row = dict(class_id=c, name=class_short[c], aliases=labels[c], group='Thing' if c in groups else 'Stuff',
                   gt_pixels=int(gt_pixels[c]), gt_images=int(gt_image_count[c]), iou=dict(zip(NAMES,finite_list(iou[:,0,c]))),
                   delta_from_original=dict(zip(NAMES,finite_list(iou[:,0,c]-iou[0,0,c]))),
                   b0_to_p2=float(iou[3,0,c]-iou[1,0,c]), region_iou={region:dict(zip(NAMES,finite_list(iou[:,r,c]))) for r,region in [(1,'boundary'),(2,'interior')]})
        class_rows.append(row)
    relations = {}
    for region in REGIMES:
        before = np.array([r['auc'] for r in train_sem[region]['B0']['classes']])
        after = np.array([r['auc'] for r in train_sem[region]['P2']['classes']])
        improvement = iou[3,0,eligible]-iou[1,0,eligible]
        drift_gain = train['regional'][1,REGIMES.index(region),eligible,0]/train['regional'][1,REGIMES.index(region),eligible,1] - train['regional'][3,REGIMES.index(region),eligible,0]/train['regional'][3,REGIMES.index(region),eligible,1]
        relations[region] = dict(auc_gain_vs_iou_gain_spearman=float(spearmanr(after-before,improvement).statistic),
                                 cosine_gain_vs_iou_gain_spearman=float(spearmanr(drift_gain,improvement).statistic),
                                 positive_auc_and_iou=int(((after>before)&(improvement>0)).sum()), classes=len(eligible))
    means, pstats = {}, train['image_stats']
    for k,name in enumerate(NAMES):
        means[name] = {}
        for c,key in enumerate(('patch_cosine','normalized_l2','gram_mse')):
            vals = pstats[:,k,c]
            means[name][key] = dict(mean=float(vals.mean()), quantiles=np.quantile(vals,[.05,.25,.5,.75,.95]).tolist())
        for r,region in enumerate(REGIMES):
            means[name][region+'_cosine'] = float(1-train['regional'][k,r,:,0].sum()/train['regional'][k,r,:,1].sum())
    roi = read(P23/'patch_drift.json')
    results = dict(official=official, checkpoints=checkpoints, ade_train_ids=train['ids'].tolist(), ade_val_ids=val_ids,
                   val_reproduction=reproduction, regions=regions, class_rows=class_rows, hard_pairs=hard,
                   things_stuff=type_stats, things_stuff_mapping=dict(thing_ids_zero_based=thing_ids, mapping=MAP_SOURCE,
                      implementation=PANOPTIC_SOURCE, sha256=sha(map_file)), ade_benchmark=SCENE_SOURCE,
                   semantic=train_sem, semantic_distributions=distributions, eligible_train_classes=eligible,
                   separability_relations=relations, train_patch_counts=train['patch_counts'].tolist(),
                   drift=means, reused_coco_roi_drift=dict(source=str((P23/'patch_drift.json').relative_to(ROOT)),mean=roi['mean'],image_ids=roi['image_ids']),
                   seed=SEED, sampling=dict(train_images=len(train['ids']), unique_images_per_class_regime_cap=64,
                       minimum_distinct_images=8, paired_draws_per_class=256, sample_metadata=['region','class','train_sample_index','patch_index','purity']),
                   class_support_source='Presence of each non-ignored class in the 2000 actual Val label PNGs, counted once per image',
                   code=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      torch=torch.__version__, timm=timm.__version__,
                      matched_training_and_eval_source_sha256=verified_geometry,
                      files={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.glob('*.py'),ROOT/'src/g1eval/ade.py',ROOT/'src/fgtrain/common.py',ROOT/'src/fgtrain/patch_preserve.py']}))
    write_json(OUT/'diagnosis_metrics.json', results)
    print('Reproduction:', reproduction, flush=True)
    print('Drift:', means, flush=True)
    print('Separation:', {r:{m:round(x['auc'],5) for m,x in s.items()} for r,s in train_sem.items()}, relations, flush=True)
    print('Region:', regions, flush=True)


if __name__ == '__main__':
    main()
