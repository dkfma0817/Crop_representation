"""Teacher / student response to the local edit, on the 500-image held-out COCO dev set (never in the training stream). Used by the sanity check and after training.

For each image: one edited ROI drawn by a fixed seed from the dev ROI list; donor = next image. Per sample: ||dT||, ||dS||, cos(dS, dT), the same for the coarse-only
read A, and ||dS|| of a non-overlapping ROI of the same image (response leaking outside the edit).

  python -m src.siglip_cue.respond --models name=path[:cond] ... --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import COCO_TRAIN, FEAT
from src.msd.pretrain import eval_exclusions
from src.siglip_cue import transforms as T
from src.siglip_cue.data import LB, S, P, native_canvas, native_geo, load_rgb, overlaps
from src.siglip_cue.model import from_initial, load_m2
from src.siglip_cue.train import TAU
from src.siglip_pm.data import geometry
from src.siglip_pm.distill import cell_w, pool, read_q
from src.siglip_pm.model import load_hcf, load_vit, run_enc

BINS = [('tiny_lt4_tokens', 0, 32 ** 2), ('small_4to16_tokens', 32 ** 2, 64 ** 2), ('medium_16to64_tokens', 64 ** 2, 128 ** 2), ('large_ge64_tokens', 128 ** 2, 1e12)]    # = src.siglip_pm.size_diagnostic.BINS
DEV_SEED = 777


def dev_items():
    ex = set(eval_exclusions()['raev2_dev'])
    d = pd.read_parquet(os.path.join(FEAT, 'dev_rois.parquet'))
    d = d[~d.image_id.isin(ex)]
    return [(int(i), g[['l', 't', 'r', 'b']].values.astype(int)) for i, g in d.groupby('image_id', sort=True)]


def box_canvas(fl, W, H, b):
    l, t, r, bb = b
    return [fl.ox + l * fl.nw / W, fl.oy + t * fl.nh / H, fl.ox + r * fl.nw / W, fl.oy + bb * fl.nh / H]


def make_samples(items, region='full', op_control=None):
    out = []
    for k, (iid, boxes) in enumerate(items):
        arr, donor = load_rgb(iid), load_rgb(items[(k + 1) % len(items)][0])
        rng = T.item_rng(DEV_SEED, iid)
        ok = [j for j, b in enumerate(boxes) if min(b[2] - b[0], b[3] - b[1]) >= T.MIN_SIDE]
        if not ok:
            continue
        j = int(rng.choice(ok))
        bx = tuple(int(x) for x in boxes[j])
        if op_control == 'color':
            edited, m = T.color_control(arr, bx, rng), dict(op=9, sigma=0., strength=0., luma_abs=0., chroma_abs=0.)
        else:
            edited, m = T.apply(arr, bx, donor, rng, region)
        out.append(dict(iid=iid, arr=arr, edited=edited, box=bx, boxes=[tuple(int(x) for x in b) for b in boxes], j=j, meta=m))
    return out


def batch(chunk, dev, m2):
    full0, full1, valid, crop0, crop1, cov, box, roi_img, roi_box = [], [], [], [], [], [], [], [], []
    native0, native1, geo, cue, other = [], [], [], [], []
    offset = 0
    for k, s in enumerate(chunk):
        H, W = s['arr'].shape[:2]
        fl = LB(W, H, S, P)
        full0.append(fl.canvas(Image.fromarray(s['arr'])))
        full1.append(fl.canvas(Image.fromarray(s['edited'])))
        valid.append(torch.from_numpy(fl.coverage() > 0))
        crop, _, _ = geometry(W, H, s['box'], S, P)
        crop0.append(crop.canvas(Image.fromarray(s['arr']).crop(s['box'])))
        crop1.append(crop.canvas(Image.fromarray(s['edited']).crop(s['box'])))
        cov.append(torch.from_numpy(crop.coverage()))
        for q, b in enumerate(s['boxes']):
            roi_box.append(box_canvas(fl, W, H, b))
            roi_img.append(k)
        cue.append(offset + s['j'])
        far = [q for q, b in enumerate(s['boxes']) if not overlaps(b, s['box']) and q != s['j']]
        other.append(offset + far[0] if far else -1)
        offset += len(s['boxes'])
        if m2:
            native0.append(native_canvas(s['arr']))
            native1.append(native_canvas(s['edited']))
            geo.append(native_geo(W, H))
    b = dict(full=torch.stack(full0), full_p=torch.stack(full1), fvalid=torch.stack(valid), crops0=torch.stack(crop0), crops1=torch.stack(crop1), cue_cov=torch.stack(cov),
             roi_img=torch.tensor(roi_img), box_c=torch.tensor(roi_box, dtype=torch.float32), cue_index=torch.tensor(cue), other_index=torch.tensor(other))
    if m2:
        b.update(native0=torch.stack(native0), native1=torch.stack(native1), geo=torch.stack(geo))
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def run_model(model, enc0, samples, dev, m2, bs=8):
    rows = []
    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        b = batch(chunk, dev, m2)
        n = len(b['full'])
        x = torch.cat([b['full'], b['full_p']])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if m2:
                o = model.forward_all(x, native=torch.cat([b['native0'], b['native1']]), geo=torch.cat([b['geo'], b['geo']]))
            else:
                o = model.forward_all(x)
        o = dict(cls=o['cls'].float(), coarse=o['coarse'].float(), mem={k: v.float() for k, v in o['mem'].items()}, grids=o['grids'])
        w24, w48 = cell_w(b['box_c'], 24, 16), cell_w(b['box_c'], 48, 8)
        A0, Q0 = read_q(model, o, b['roi_img'], w24, w48)
        A1, Q1 = read_q(model, o, b['roi_img'] + n, w24, w48)
        tok0, _ = run_enc(enc0, b['crops0'])
        tok1, _ = run_enc(enc0, b['crops1'])
        cov = b['cue_cov'].float()
        T0, T1 = F.normalize(pool(tok0, cov), dim=-1), F.normalize(pool(tok1, cov), dim=-1)
        dT = T1 - T0
        c = b['cue_index']
        dS = F.normalize(Q1[c], dim=-1) - F.normalize(Q0[c], dim=-1)
        dA = F.normalize(A1[c], dim=-1) - F.normalize(A0[c], dim=-1)
        off = b['other_index']
        dO = F.normalize(Q1, dim=-1) - F.normalize(Q0, dim=-1)
        leak = torch.where(off >= 0, dO[off.clamp_min(0)].norm(dim=-1), torch.full((len(off),), float('nan'), device=dev))
        for k, s in enumerate(chunk):
            W, H = s['arr'].shape[1], s['arr'].shape[0]
            ratio = 384 / max(W, H)
            area = (s['box'][2] - s['box'][0]) * (s['box'][3] - s['box'][1]) * ratio * ratio
            rows.append(dict(image_id=s['iid'], op=int(s['meta']['op']), sigma=float(s['meta']['sigma']), native_w=s['box'][2] - s['box'][0], native_h=s['box'][3] - s['box'][1], canvas_area=float(area),
                             luma_abs=float(s['meta']['luma_abs']), chroma_abs=float(s['meta']['chroma_abs']), norm_dT=float(dT[k].norm()), norm_dS=float(dS[k].norm()), cos=float(F.cosine_similarity(dS[k], dT[k], dim=0)), norm_dA=float(dA[k].norm()),
                             cos_coarse=float(F.cosine_similarity(dA[k], dT[k], dim=0)), leak_dS=float(leak[k]), cos_T0_S0=float(F.cosine_similarity(F.normalize(Q0[c[k]], dim=-1), T0[k], dim=0))))
    df = pd.DataFrame(rows)
    df['size_bin'] = 'other'
    for name, lo, hi in BINS:
        df.loc[(df.canvas_area >= lo) & (df.canvas_area <= hi), 'size_bin'] = name
    df['valid'] = df.norm_dT >= TAU
    return df


def summarise(df):
    v = df[df.valid]
    groups = [('all', v)] + [(f'op_{T.OPS[o] if o < 3 else "color"}', v[v.op == o]) for o in sorted(v.op.unique()) if o >= 0] + [(f'size_{n}', v[v.size_bin == n]) for n, _, _ in BINS]
    out = []
    for name, g in groups:
        if len(g) == 0:
            continue
        out.append(dict(group=name, n=len(g), norm_dT=g.norm_dT.mean(), norm_dS=g.norm_dS.mean(), cos=g.cos.mean(), cos_coarse=g.cos_coarse.mean(), ratio_dS_dT=(g.norm_dS / g.norm_dT).mean(), leak_ratio=(g.leak_dS.dropna() / g.norm_dS[g.leak_dS.notna()]).mean()))
    return pd.DataFrame(out), dict(n_total=len(df), valid_fraction=float(df.valid.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='name=path[:M2]; "initial:<path>" also works')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    enc0 = load_vit(dev).eval().requires_grad_(False)
    samples = make_samples(dev_items())
    for spec in a.models:
        name, path = spec.split('=', 1)
        m2 = path.endswith(':M2')
        if m2:
            path = path[:-3]
            model = from_initial(path, dev)[0].eval().requires_grad_(False) if name == 'initial_m2' else load_m2(path, dev)[0]
        else:
            model = load_hcf(path, dev)[0]
        df = run_model(model, enc0, samples, dev, m2)
        df.to_parquet(out / f'{name}.parquet')
        table, info = summarise(df)
        table.to_csv(out / f'{name}.csv', index=False)
        (out / f'{name}.json').write_text(json.dumps(dict(info=info, table=table.to_dict('records')), indent=1))
        print(name, json.dumps(info), table.round(4).to_string(), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
