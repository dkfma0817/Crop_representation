"""Offline retrieval and label-blind Teacher rank triplets on the saved ADE cache."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PRIOR = OUT.parent / 'p2_p3_dense_diagnosis'
MODELS = ('Original', 'B0', 'P2')
CACHE_MODELS = ('Original', 'B0', 'P1', 'P2')
REGIMES = ('Interior', 'Mixed')
KS = (1, 5, 10)
BANDS = ((6, 10), (11, 50), (51, 200), (201, None))
RELATIONS = ('same→different', 'same→same', 'different→different', 'different→same')
PAIR_SEED = 20261115
QUANTILES = np.linspace(0, 1, 6)


def sha(path):
    with path.open('rb') as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def average(values, classes):
    """Average a statistic by query count and by equally weighted represented classes."""
    if len(values) == 0:
        return {'micro': None, 'macro': None, 'n': 0, 'classes': 0}
    ids, inv = np.unique(classes, return_inverse=True)
    means = np.bincount(inv, weights=values) / np.bincount(inv)
    return {'micro': float(np.mean(values)), 'macro': float(means.mean()), 'n': len(values), 'classes': len(ids)}


def qrange(values):
    return np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist() if len(values) else None


def bins(values):
    edges = np.quantile(values, QUANTILES)
    return edges, np.searchsorted(edges[1:-1], values, side='right')


def rank_bank(features, image):
    x = np.ascontiguousarray(features, dtype=np.float32)
    score = x @ x.T
    np.clip(score, -1, 1, out=score)
    score[image[:, None] == image] = -np.inf
    order = np.argsort(-score, axis=1, kind='stable')
    return score, order


def retrieval_values(order, classes, images):
    matching = (classes[order] == classes[:, None]) & (images[order] != images[:, None])
    first = np.where(matching.any(1), matching.argmax(1) + 1, 0)
    return {**{f'p{k}': matching[:, :k].mean(1) for k in KS},
            **{f'hit{k}': matching[:, :k].any(1) for k in KS},
            'mrr': np.divide(1., first, out=np.zeros_like(first, dtype=float), where=first>0)}, first


def random_baselines(labels, images):
    classes = np.unique(labels)
    allowed = images[:, None] != images
    support = np.stack([allowed[:, labels == c].sum(1) for c in classes], axis=1)
    positives = support[np.arange(len(labels)), np.searchsorted(classes, labels)]
    eligible = allowed.sum(1)
    valid_classes = (support > 0).sum(1)
    frequency_p = positives / eligible
    balanced_p = (positives > 0) / valid_classes
    random = {f'p{k}': frequency_p for k in KS}
    balanced = {f'p{k}': balanced_p for k in KS}
    tail = np.ones(len(labels), dtype=float)
    # Exact marginal distribution for uniform permutations without replacement.
    random_mrr = np.zeros(len(labels), dtype=float)
    for k in range(1, int(eligible.max()) + 1):
        hazard = np.divide(positives, eligible - k + 1, out=np.zeros(len(labels)), where=eligible>=k).clip(0, 1)
        random_mrr += tail * hazard / k
        tail *= 1 - hazard
        if k in KS:
            random[f'hit{k}'] = 1 - tail.copy()
            balanced[f'hit{k}'] = 1 - (1 - balanced_p)**k
    random['mrr'] = random_mrr
    # Uniform class then uniform patch, independent draws. MRR uses N eligible draws.
    success = balanced_p.copy()
    balanced_mrr = np.zeros(len(labels), dtype=float)
    for k in range(1, int(eligible.max()) + 1):
        balanced_mrr += np.where(eligible >= k, success / k, 0)
        success *= 1 - balanced_p
    balanced['mrr'] = balanced_mrr
    return {'Frequency random': random, 'Class-balanced random': balanced}, eligible, positives, valid_classes


def overlap(reference, student, labels):
    values = {}
    for k in KS:
        original = reference[:, :k]
        retained = (original[:, :, None] == student[:, None, :k]).any(-1)
        relevant = labels[original] == labels[:, None]
        count = relevant.sum(1)
        kept = (retained & relevant).sum(1)
        values[f'retention{k}'] = retained.mean(1)
        values[f'teacher_same{k}'] = count
        values[f'retained_same{k}'] = kept
        values[f'student_same{k}'] = (labels[student[:, :k]] == labels[:, None]).sum(1)
    return values


def summarise_retention(values, classes):
    result = {}
    for k in KS:
        count = values[f'teacher_same{k}']
        kept = values[f'retained_same{k}']
        active = count > 0
        exact = average((kept[active] / count[active]), classes[active])
        exact['pooled_teacher_hits'] = float(kept.sum()/count.sum()) if count.sum() else None
        exact['teacher_hits'] = int(count.sum())
        exact['retained_teacher_hits'] = int(kept.sum())
        lost = int((count - kept).sum())
        new = int((values[f'student_same{k}'] - kept).sum())
        exact['lost_teacher_hits'] = lost
        exact['new_same_class_hits'] = new
        exact['new_per_lost'] = new / lost if lost else None
        result[str(k)] = {'overlap': average(values[f'retention{k}'], classes), 'same_class_id_retention': exact}
    return result


def query_stats(values, classes, mask=None):
    if mask is None:
        mask = slice(None)
    return {key: average(value[mask], classes[mask]) for key, value in values.items()}


def strata_summary(margins, relations, classes, selected):
    valid = selected & (margins[0] > 0)
    labels = classes[valid]
    count = int(valid.sum())
    relevant = relations[valid] == 0
    semantic = average(relevant, labels)
    stats = {name: average(margins[m, valid] <= 0, labels) for m, name in enumerate(MODELS)}
    return {'n': count, 'ties_excluded': int(selected.sum()) - count, 'teacher_margin_quantiles': qrange(margins[0, valid]),
            'same_different_share': semantic, 'violation': stats,
            'student_margin_quantiles': {name: qrange(margins[m, valid]) for m,name in enumerate(MODELS)}}


def confidence(teacher_top_scores, teacher_top, labels):
    score = {'top1_cosine': teacher_top_scores[:, 0],
             'gap_1_5': teacher_top_scores[:, 0] - teacher_top_scores[:, 4],
             'gap_1_10': teacher_top_scores[:, 0] - teacher_top_scores[:, 9]}
    correct = labels[teacher_top[:,0]] == labels
    result = {}
    for name, values in score.items():
        edges, group = bins(values)
        result[name] = {'edges': edges.tolist(), 'quantiles': qrange(values),
                        'groups': [dict(bin=i+1, lower=float(edges[i]), upper=float(edges[i+1]),
                                        **average(correct[group == i], labels[group == i])) for i in range(5)]}
    return result


def measure_regime(features, metadata, row, rng):
    sub = metadata[row]
    labels, images = sub[:,1].astype(int), sub[:,2].astype(int)
    queries = np.arange(len(row))
    random, eligible, positives, num_classes = random_baselines(labels, images)
    tops, top_scores, first, retrieval = [], [], [], dict(random)
    pair_query = np.broadcast_to(queries[:, None, None], (len(row), len(BANDS), 5))
    pair_upper, pair_lower, pair_k = None, None, None
    margins = []
    for m, name in enumerate(MODELS):
        scores, order = rank_bank(features[row, CACHE_MODELS.index(name)], images)
        values, ranks = retrieval_values(order, labels, images)
        retrieval[name] = values
        first.append(ranks)
        top = order[:, :10].copy()
        tops.append(top)
        top_scores.append(scores[queries[:, None], top])
        assert np.all(images[top] != images[:, None]) and np.isfinite(top_scores[-1]).all()
        if m == 0:
            pair_upper = np.broadcast_to(top[:, None, :5], pair_query.shape)
            pair_k = np.stack([rng.integers(lo, (eligible if hi is None else np.minimum(eligible, hi))[:, None] + 1,
                                             size=(len(row), 5)) for lo, hi in BANDS], axis=1)
            pair_lower = order[pair_query, pair_k - 1]
            assert np.all(images[pair_upper] != images[pair_query]) and np.all(images[pair_lower] != images[pair_query])
        upper, lower = scores[pair_query, pair_upper], scores[pair_query, pair_lower]
        margins.append(np.stack([upper, lower], axis=-1))
        del scores, order
    cosines = np.stack(margins)
    margins = (cosines[..., 0] - cosines[..., 1]).reshape(len(MODELS), -1)
    assert (margins[0] >= 0).all()
    relation = np.where(labels[pair_upper] == labels[pair_query], np.where(labels[pair_lower] == labels[pair_query], 1, 0),
                        np.where(labels[pair_lower] == labels[pair_query], 3, 2)).ravel()
    pair_labels = labels[pair_query].ravel()
    pair_bands = np.broadcast_to(np.arange(len(BANDS))[None,:,None], pair_query.shape).ravel()
    relevant = relation == 0
    real = margins[0] > 0
    edges, groups = bins(margins[0, real])
    quintiles = np.full(len(real), -1, dtype=np.int8)
    quintiles[real] = groups
    high = quintiles == 4
    retention = {name: overlap(tops[0], tops[m], labels) for m, name in enumerate(MODELS)}
    band_stats = []
    for i in range(len(BANDS)):
        band = pair_bands == i
        cutoff = float(np.quantile(margins[0, real & band], .8))
        band_stats.append(dict(band=i, all=strata_summary(margins,relation,pair_labels,band),
                               same_different=strata_summary(margins,relation,pair_labels,band&relevant),
                               band_p80=cutoff,
                               band_high_margin=strata_summary(margins,relation,pair_labels,band&(margins[0]>=cutoff)),
                               band_high_margin_same_different=strata_summary(margins,relation,pair_labels,band&relevant&(margins[0]>=cutoff))))
    top1_cutoff = float(np.quantile(top_scores[0][:,0], .8))
    confident_near = np.zeros(pair_query.shape, dtype=bool)
    confident_near[:,0,0] = ((top_scores[0][:,0] >= top1_cutoff) &
                             (margins[0].reshape(pair_query.shape)[:,0,0] >= band_stats[0]['band_p80']))
    confident_near = confident_near.ravel()
    teacher_right = retrieval['Original']['p1'] == 1
    top1_outcomes = {}
    for m, name in enumerate(MODELS):
        exact_top1 = tops[0][:,0] == tops[m][:,0]
        student_right = retrieval[name]['p1'] == 1
        outcomes = {'same_patch':exact_top1, 'new_same_class':~exact_top1 & student_right, 'different_class':~student_right}
        top1_outcomes[name] = {key:average(val[teacher_right],labels[teacher_right]) for key,val in outcomes.items()}
        top1_outcomes[name]['corrected_teacher_error'] = average(student_right[~teacher_right],labels[~teacher_right])
    stats = {'pool': {'patches':len(row), 'images': len(np.unique(images)), 'max_patches_per_image': int(np.bincount(images).max()),
                      'candidate_count_range': [int(eligible.min()), int(eligible.max())],
                      'positive_count_range': [int(positives.min()), int(positives.max())], 'candidate_classes_range':[int(num_classes.min()),int(num_classes.max())],
                      'has_same_class': average(positives>0, labels),
                      'per_class': {int(c):int((labels == c).sum()) for c in np.unique(labels)}},
             'retrieval': {name: query_stats(value, labels) for name,value in retrieval.items()},
             'retention': {name: summarise_retention(value, labels) for name,value in retention.items()},
             'top1_outcomes':top1_outcomes,
             'confidence': confidence(top_scores[0],tops[0],labels),
             'ranking': {'all':strata_summary(margins,relation,pair_labels,np.ones(len(real),bool)),
                         'same_different':strata_summary(margins,relation,pair_labels,relevant),
                         'high_margin':strata_summary(margins,relation,pair_labels,high),
                         'high_margin_same_different':strata_summary(margins,relation,pair_labels,high & relevant),
                         'high_margin_min':float(edges[-2]),
                         'margin_groups':[dict(bin=i+1, lower=float(edges[i]), upper=float(edges[i+1]),
                                               all=strata_summary(margins,relation,pair_labels,quintiles==i),
                                               same_different=strata_summary(margins,relation,pair_labels,(quintiles==i)&relevant)) for i in range(5)],
                         'by_band':band_stats,
                         'joint_confident_near': {'top1_cosine_p80':top1_cutoff,
                                                  'near_margin_p80':band_stats[0]['band_p80'],
                                                  'all':strata_summary(margins,relation,pair_labels,confident_near),
                                                  'same_different':strata_summary(margins,relation,pair_labels,confident_near&relevant),
                                                  'relation_counts':{name:int((confident_near&real&(relation==i)).sum()) for i,name in enumerate(RELATIONS)}},
                         'by_relation': {name:dict(share=average(relation[real]==i,pair_labels[real]),
                                                  **strata_summary(margins,relation,pair_labels,relation==i)) for i,name in enumerate(RELATIONS)},
                         'high_margin_band_shares': [float((pair_bands[high]==i).mean()) for i in range(len(BANDS))]},
             'classes': {int(c):dict(query_count=int((labels==c).sum()),
                                     p_at_1={name:float(retrieval[name]['p1'][labels==c].mean()) for name in MODELS},
                                     p_at_5={name:float(retrieval[name]['p5'][labels==c].mean()) for name in MODELS},
                                     p_at_10={name:float(retrieval[name]['p10'][labels==c].mean()) for name in MODELS},
                                     retention_at_10={name:float(retention[name]['retention10'][labels==c].mean()) for name in MODELS},
                                     semantic_violation={name: average(margins[i,real & relevant & (pair_labels==c)]<=0,pair_labels[real & relevant & (pair_labels==c)])
                                                         for i,name in enumerate(MODELS)},
                                     high_semantic_violation={name: average(margins[i,high & relevant & (pair_labels==c)]<=0,pair_labels[high & relevant & (pair_labels==c)])
                                                              for i,name in enumerate(MODELS)}) for c in np.unique(labels)}}
    arrays = dict(query_row=row, top_rows=row[np.stack(tops)], top_scores=np.stack(top_scores), first_true_rank=np.stack(first),
                  eligible_count=eligible, same_class_count=positives,
                  pair_query_row=row[pair_query], pair_upper_row=row[pair_upper], pair_lower_row=row[pair_lower], lower_teacher_rank=pair_k,
                  cosine_pairs=cosines, margin_quintile=quintiles.reshape(pair_query.shape), dominant_relation=relation.reshape(pair_query.shape))
    details = dict(retrieval=retrieval, retention=retention, labels=labels, margins=margins, relations=relation, pair_labels=pair_labels, quintiles=quintiles)
    return stats, arrays, details


def main():
    prior = read(PRIOR/'diagnosis_metrics.json')
    assert prior['sampling']['minimum_distinct_images'] == 8 and prior['sampling']['unique_images_per_class_regime_cap'] == 64
    names = [r['name'] for r in prior['class_rows']]
    eligible = np.asarray(prior['eligible_train_classes'])
    for name in MODELS:
        ckpt = prior['checkpoints'][name]
        assert sha(ROOT/ckpt['encoder']) == ckpt['sha256'], name
    sources = {path: sha(ROOT/path) for path in [f'{PRIOR.relative_to(ROOT)}/infer.py','src/g1eval/ade.py']}
    assert all(prior['code']['files'][path] == digest for path,digest in sources.items())
    cache = PRIOR/'train_features.npz'
    with np.load(cache) as archive:
        features, metadata, images = archive['features'], archive['metadata'], archive['ids'].tolist()
    assert len(images)==1024 and images == prior['ade_train_ids'] and features.shape == (len(metadata),4,768)
    assert np.isfinite(features).all() and features.dtype==np.float32
    norms = np.linalg.norm(features, axis=2)
    norm_error = float(np.max(np.abs(norms - 1)))
    assert norm_error < 1e-6
    assert np.all(np.where(metadata[:,0]==0, metadata[:,4]>=.90, (metadata[:,4]>=.50)&(metadata[:,4]<.90)))
    assert np.unique(metadata[:, :3],axis=0).shape[0] == len(metadata)
    class_counts = np.stack([np.bincount(metadata[metadata[:,0]==r,1].astype(int),minlength=len(names)) for r in range(2)])
    assert np.array_equal(np.flatnonzero(class_counts.min(0)>=8),eligible)
    assert class_counts.max()<=64
    omitted = [{'class_id':i,'name':names[i],'interior':int(class_counts[0,i]),'mixed':int(class_counts[1,i]),
                'reason':'fewer than 8 distinct sampled images in at least one regime'} for i in range(len(names)) if i not in eligible]
    rng = np.random.default_rng(PAIR_SEED)
    all_rows, raw, report, details = [], {}, {}, []
    for r, regime in enumerate(REGIMES):
        row = np.flatnonzero((metadata[:,0] == r) & np.isin(metadata[:,1],eligible))
        all_rows.extend(row.tolist())
        stats, arrays, detail = measure_regime(features,metadata,row,rng)
        report[regime] = stats
        raw.update({regime.lower()+'_'+key: value for key,value in arrays.items()})
        details.append(detail)
        print(regime, stats['pool']['patches'], 'queries; raw triplets', arrays['pair_query_row'].size, flush=True)
    labels = np.concatenate([q['labels'] for q in details])
    all_stats = {'patches':len(all_rows),'images_with_sampled_patches':int(len(np.unique(metadata[all_rows,2]))),
                 'max_patches_per_image':int(np.bincount(metadata[all_rows,2].astype(int)).max())}
    all_stats['retrieval'] = {name: query_stats({k: np.concatenate([x['retrieval'][name][k] for x in details]) for k in detail['retrieval'][name]},labels)
                              for name in detail['retrieval']}
    all_stats['retention'] = {name: summarise_retention({k:np.concatenate([x['retention'][name][k] for x in details]) for k in detail['retention'][name]},labels) for name in MODELS}
    margin = np.concatenate([x['margins'] for x in details],axis=1)
    relation = np.concatenate([x['relations'] for x in details])
    pair_labels = np.concatenate([x['pair_labels'] for x in details])
    high = np.concatenate([x['quintiles'] for x in details]) == 4
    all_stats['ranking'] = {'all':strata_summary(margin,relation,pair_labels,np.ones(len(relation),bool)),
                            'same_different':strata_summary(margin,relation,pair_labels,relation==0),
                            'high_margin_same_different':strata_summary(margin,relation,pair_labels,high & (relation==0)),
                            'high_margin':strata_summary(margin,relation,pair_labels,high)}
    csv_path = OUT/'patch_bank.csv'
    with csv_path.open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cache_row','image_id','regime','class_id_zero_based','class','patch_index_row_major','purity'])
        for i in all_rows:
            r,c,im,patch,purity = metadata[i]
            writer.writerow([i,images[int(im)],REGIMES[int(r)],int(c),names[int(c)],int(patch),format(purity,'.12g')])
    np.savez_compressed(OUT/'retrieval_and_triplets.npz', **raw)
    result = {'models':MODELS, 'image_ids':images, 'class_names':names, 'classes_included':eligible.tolist(), 'omitted_classes':omitted,
              'fixed_protocol': {'scope':'ADE20K Train only, unchanged previous image IDs and cached patch positions',
                  'preprocessing':'Whole RGB image, short side 384 with aspect ratio, bilinear RGB, nearest GT, pad right/bottom to Patch16; final LN excludes CLS + four registers; FP32 L2-normalized cache',
                  'regimes':'At least 128 non-ignore GT pixels; Interior purity >= .90; Mixed .50 <= purity < .90. Each regime queries its own bank only.',
                  'sampling':'Reuse one candidate per image/class/regime with max 64 images per class/regime; eligible classes have >=8 images in both regimes.',
                  'queries':'All eligible cached rows are both query and candidate; mask all same-image candidates before rank. No query-image holds out other query images.',
                  'tie_break':'FP32 cosine, descending; exact ties follow ascending immutable cache row.',
                  'retrieval':'Precision at k is fraction same class in top k. Hit at k is at least one same class in top k. MRR uses first true match in entire eligible rank, zero if absent.',
                  'random_frequency':'Exact expected scores under uniform random permutation of eligible patches without replacement.',
                  'random_balanced':'Exact expected scores under uniform eligible class then uniform patch, independent draws; MRR truncated to eligible candidate count.',
                  'aggregation':'Micro weights each query or triplet equally, Macro first averages within each anchor class then over classes with support. Pooled All combines the two disjoint searches.',
                  'retention':'Overlap |teacher top k ∩ student top k| / k. Same-class identity retention conditions on Teacher hits, with both per-query averages and pooled matched hits.',
                  'pairs':'Each Teacher rank j=1..5 paired with one independent uniform rank k in each fixed lower band [6,10], [11,50], [51,200], [201,N]. k and j may come from the same candidate image; neither can come from the query image. Same pairs for all models. GT unused for choosing ranks.',
                  'lower_bands':BANDS, 'pairs_per_query':20, 'pair_seed':PAIR_SEED,
                  'teacher_confidence':'Quintiles of top1 score, 1−5 and 1−10 gap within each regime. Pair confidence is Teacher margin quintile, top fifth is high-margin. The same quantile is also computed within each fixed lower-rank band to check whether the pooled top fifth is driven by far ranks. The joint near slice uses j=1, k=6–10, cosine in the top fifth of queries, and margin in the top fifth of the near band. Margin bins are diagnostic only.',
                  'violation':'Teacher margin > 0 and Student margin <= 0. Exact Teacher ties excluded from denominator.',
                  'semantic_relevance':'Interior same→different is the strict semantic-relevant subset. Same→same and different→different do not establish a semantic ranking; different→same can be an inverted Teacher relation. All Mixed rows remain ambiguous, their dominant-class analogues are descriptive.',
                  'relative_hyperparameters':'No learned thresholds or optimization; K=1,5,10 is reporting only. Diagnostic ADE labels and selected IDs stay in this results directory.',
                  'image_seed':prior['seed'],'cache_seed':prior['seed']+1, 'fp32_unit_norm_max_error':norm_error,
                  'array_layout':'top_rows/top_scores [model, query, top-rank]; pair arrays [query, lower-band, upper-rank]; cosine_pairs [model, query, lower-band, upper-rank, (upper,lower)]; rows index original cache.',
                  'class_relations':RELATIONS, 'metrics_precision':'raw values are fractions except baseline ADE mIoU and PACO AP; report renders most fractions as percent.'},
              'All':all_stats, **report,
              'previous_auc':{m:{r:prior['semantic'][k][m]['auc'] for r,k in zip(REGIMES,('interior','mixed'))} for m in MODELS},
              'previous_ade':{m:prior['official'][m]['ade'] for m in MODELS},
              'provenance': {'cache':str(cache.relative_to(ROOT)),'cache_sha256':sha(cache),
                             'previous_report':str((PRIOR/'report.html').relative_to(ROOT)),
                             'previous_metrics_sha256':sha(PRIOR/'diagnosis_metrics.json'),
                             'feature_sources_sha256':sources,
                             'checkpoint':{m:{k:prior['checkpoints'][m][k] for k in ('encoder','sha256','checksum')} for m in MODELS},
                             'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                             'numpy':np.__version__, 'analysis_source_sha256':{p.name:sha(p) for p in OUT.glob('*.py')}}}
    write(OUT/'metrics.json',result)
    print('Audit complete', flush=True)


if __name__ == '__main__':
    main()
