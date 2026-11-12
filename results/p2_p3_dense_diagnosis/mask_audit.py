"""Recompute the trained P3 mask on a fixed sample of the original stream."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from infer import OUT, ROOT, RUN, SEED
from src.config import COCO_TRAIN
from src.fgtrain.common import write_json
from src.fgtrain.patch_preserve import PatchData, weights
from src.viz.c0_pca import LB

P23 = ROOT/'output/2026-11-11/fgtrain_p23_s0'


def sha(path):
    with open(path, 'rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def weighted(values, weight):
    return float((values*weight).sum()/weight.sum()) if weight.sum() else None


def main():
    torch.set_num_threads(2)
    manifest = json.loads((P23/'data_manifest.json').read_text())
    stream = Path(manifest['stream'])
    roi = RUN/'teacher/rois.parquet'
    assert sha(stream) == manifest['stream_sha256'] and sha(roi) == manifest['roi_coordinates_sha256']
    cfg2 = json.loads((P23/'P2/config.json').read_text())
    cfg3 = json.loads((P23/'P3/config.json').read_text())
    assert cfg2['lambda_preserve'] == cfg3['lambda_preserve']
    for cfg in (cfg2, cfg3):
        assert cfg['steps'] == 12000
    stream_ids = np.load(stream)[:cfg3['steps']].ravel()
    stream_ids = stream_ids[~np.isin(stream_ids, manifest['base_manifest']['excluded_train_images'])]
    population, occurrences = np.unique(stream_ids, return_counts=True)
    sample = np.sort(np.random.default_rng(SEED).choice(len(population), size=4096, replace=False))
    ids = population[sample].astype(int)
    appearances = occurrences[sample]
    dataset = PatchData(RUN/'teacher')
    geometries = [dataset.geometry(int(iid)) for iid in ids]
    v = np.stack([item[0] for item in geometries]).astype(np.float64)
    m = np.stack([item[1] for item in geometries]).astype(np.float64)
    valid = v > 0
    raw = np.where(valid, 1-.75*m, 0.)
    w = weights(torch.from_numpy(v), torch.from_numpy(m), True).numpy().astype(np.float64)
    u = weights(torch.from_numpy(v), torch.from_numpy(m), False).numpy()
    assert np.array_equal(u, valid)
    avg = (v*w).sum(1)/v.sum(1)
    assert np.max(np.abs(avg-1)) < 1e-6
    fraction = (m*v).sum(1)/v.sum(1)
    ordered = np.argsort(fraction, kind='stable')
    example_idx = ordered[(np.array([.10, .40, .70, .95])*(len(ordered)-1)).astype(int)]
    examples = []
    # Exercise the actual training dataset path and its canvas coordinate assertions.
    sample_ids = list(dict.fromkeys([int(ids[i]) for i in example_idx]+list(map(int, stream_ids[:16]))))
    batch = dataset[sample_ids]
    for i in example_idx:
        iid = int(ids[i])
        j = sample_ids.index(iid)
        _, _, boxes = geometries[i]
        assert np.array_equal(batch['roi_union'][j].numpy(), m[i].astype(np.float32))
        assert np.array_equal(batch['box_c'][batch['roi_img'] == j].numpy(), boxes.astype(np.float32))
        source = Image.open(Path(COCO_TRAIN)/f'{iid:012d}.jpg').convert('RGB')
        geom = LB(*source.size, 384, 16)
        canvas = Image.new('RGB', (384, 384), '#e9eef4')
        canvas.paste(source.resize((geom.nw, geom.nh), Image.Resampling.BICUBIC), (geom.ox, geom.oy))
        draw = ImageDraw.Draw(canvas)
        kinds = dataset.by[iid].kind.tolist()
        for k, box in zip(kinds, boxes):
            draw.rectangle(tuple(box), outline={'gt':'#ffd564','random':'#55d6ff'}.get(k, '#f792cd'), width=2)
        canvas.save(OUT/f'roi_{iid}_boxes.jpg', quality=90)
        examples.append(dict(image_id=iid, coverage=float(fraction[i]), kinds=kinds, boxes=boxes.tolist(),
                             weights_range=[float(w[i,valid[i]].min()), float(w[i,valid[i]].max())]))
        np.savez_compressed(OUT/f'roi_{iid}_mask.npz', valid=v[i].reshape(24,24), union=m[i].reshape(24,24), weight=w[i].reshape(24,24))
    eps = 1e-6
    inside = valid & (m >= 1-eps)
    outside = valid & (m <= eps)
    boundary = valid & ~(inside | outside)
    exposure = appearances[:, None]
    area = v*exposure
    dev = np.abs(w-1)
    rows = dict(image_ids=ids.tolist(), stream_population=int(len(population)), stream_occurrences=int(len(stream_ids)),
                sample_seed=SEED, sample_occurrences=int(appearances.sum()), canvas_checks=len(sample_ids),
                threshold_abs_delta=.10, examples=examples, train_source_hashes=dict(stream=sha(stream), boxes=sha(roi)),
                union_mean=float(fraction.mean()), union_quantiles=np.quantile(fraction, [0,.05,.25,.5,.75,.95,1]).tolist(),
                union_above_90=float((fraction >= .90).mean()), union_full=float((fraction >= 1-eps).mean()),
                near_uniform_images=float((((v*dev).sum(1)/v.sum(1)) < .01).mean()),
                area_mean_abs_delta=weighted(dev, area), changed_area=weighted(dev >= .10, area),
                changed_patch_count=weighted(dev >= .10, valid*exposure),
                normalized_mean_max_error=float(np.max(np.abs(avg-1))),
                fraction_by_patch_count={k:weighted(b, valid*exposure) for k,b in [('inside',inside),('boundary',boundary),('outside',outside)]},
                fraction_by_valid_area={k:weighted(b, area) for k,b in [('inside',inside),('boundary',boundary),('outside',outside)]},
                weight_by_group={k:weighted(w, area*b) for k,b in [('inside',inside),('boundary',boundary),('outside',outside)]},
                fractional_inside_weight=weighted(w, area*m), fractional_outside_weight=weighted(w, area*(1-m)),
                union_check=json.loads((P23/'unit_verification.json').read_text()), lambda_preserve=cfg3['lambda_preserve'])
    stats, per_image_stats = {}, {}
    for name, values in [('P2 uniform', u), ('P3 raw', raw), ('P3 normalized', w)]:
        mean = weighted(values, area)
        image_mean = (v*values).sum(1)/v.sum(1)
        image_std = np.sqrt((v*(values-image_mean[:,None])**2).sum(1)/v.sum(1))
        image_min = np.where(valid, values, np.inf).min(1)
        image_max = np.where(valid, values, -np.inf).max(1)
        per_image_stats[name.replace(' ', '_')] = np.stack([image_min, image_mean, image_max, image_std], axis=1)
        stats[name] = dict(min=float(values[valid].min()), mean=mean, max=float(values[valid].max()),
                           std=weighted((values-mean)**2, area)**.5,
                           median_image_min=float(np.median(np.where(valid, values, np.inf).min(1))),
                           median_image_mean=float(np.median(image_mean)),
                           median_image_max=float(np.median(np.where(valid, values, -np.inf).max(1))),
                           median_image_std=float(np.median(image_std)))
    rows['weight_stats'] = stats
    rows['per_image_weight_columns'] = ['minimum', 'area_weighted_mean', 'maximum', 'area_weighted_std']
    gradients = []
    for name in ('P2','P3'):
        for line in (P23/name/'gradient_diagnostics.jsonl').read_text().splitlines():
            q = json.loads(line)
            gradients.append(dict(model=name, step=q['step'], crop=q['crop']['encoder'], raw=q['preserve']['encoder'],
                                  weighted=q['weighted_preservation_norm'], ratio=q['preservation_crop_ratio'],
                                  cosine=q['crop_preservation_cosine']))
    rows['gradients'] = gradients
    np.savez_compressed(OUT/'roi_stats_arrays.npz', union_fraction=fraction, image_ids=ids, appearances=appearances,
                        normalized=w.astype(np.float32), area=v.astype(np.float32), per_image_abs_delta=(v*dev).sum(1)/v.sum(1),
                        **per_image_stats)
    write_json(OUT/'roi_mask_audit.json', rows)
    print({k:rows[k] for k in ['union_mean','union_quantiles','union_above_90','changed_area','area_mean_abs_delta','weight_by_group']}, flush=True)


if __name__ == '__main__':
    main()
