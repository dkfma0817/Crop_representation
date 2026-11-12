"""Pre-training checks 1-10 and the small-sample response statistics (spec section 6). Fails loudly on any violated invariant; writes sanity.json and example sheets.

  python -m src.siglip_cue.sanity --condition M1|M2 --run <run> --shared <shared>
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

import src.readout.common as RC
from src.config import STREAM
from src.ivc.ft import cosd
from src.msd.pretrain import eval_exclusions
from src.siglip_cue import transforms as T
from src.siglip_cue.data import CueDS, load_rgb, native_canvas, native_geo, overlaps
from src.siglip_cue.readout import BBNativeDS, encode_native
from src.siglip_cue.respond import dev_items, make_samples, run_model, summarise
from src.siglip_cue.train import LAM_CUE, TAU, batch_sha, build, grad_check, objective, step_update, teacher
from src.siglip_pm.distill import LAM_G, WARM, Slots, dev_diag, pool
from src.siglip_pm.cocodata import CocoDS
from src.siglip_pm.model import run_enc

N_STEPS = 12


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(n=int(len(x)), mean=float(np.nanmean(x)), std=float(np.nanstd(x)), p05=float(np.nanpercentile(x, 5)), p25=float(np.nanpercentile(x, 25)), p50=float(np.nanpercentile(x, 50)), p75=float(np.nanpercentile(x, 75)), p95=float(np.nanpercentile(x, 95)), min=float(np.nanmin(x)), max=float(np.nanmax(x)))


@torch.no_grad()
def regions_and_color(ds, stream, ex, enc0, dev):
    """Check 4: teacher response to the same delta field restricted to the feather band ('ring') vs the interior, and to a chroma-only control."""
    full_norm, ring_ratio, inter_cos, ring_cos, color_norm, color_cos = [], [], [], [], [], []
    for step in range(3):
        slots = next(iter(Slots(stream, 0, 1, ex, step)))[1]
        arrays = [load_rgb(iid) for _, iid in slots]
        for k, (col, iid) in enumerate(slots):
            donor = arrays[(k + 1) % len(arrays)]
            boxes, j, full, m = ds.edit_slot(step, col, iid, arrays[k], donor)
            if m['op'] < 0:
                continue
            _, _, ring, _ = ds.edit_slot(step, col, iid, arrays[k], donor, 'ring')
            _, _, inner, _ = ds.edit_slot(step, col, iid, arrays[k], donor, 'interior')
            color = T.color_control(arrays[k], boxes[j], np.random.default_rng([7, step, col]))
            ds_img = {n: np.asarray(v) for n, v in dict(orig=arrays[k], full=full, ring=ring, inner=inner, color=color).items()}
            from src.siglip_pm.data import geometry
            H, W = arrays[k].shape[:2]
            crop, _, _ = geometry(W, H, boxes[j], 384, 16)
            cov = torch.from_numpy(crop.coverage()).float().to(dev)[None]
            feats = {}
            for name, a in ds_img.items():
                x = crop.canvas(Image.fromarray(a).crop(boxes[j]))[None].to(dev)
                t, _ = run_enc(enc0, x)
                feats[name] = F.normalize(pool(t, cov), dim=-1)[0]
            d = {n: feats[n] - feats['orig'] for n in ('full', 'ring', 'inner', 'color')}
            nf = float(d['full'].norm())
            full_norm.append(nf)
            ring_ratio.append(float(d['ring'].norm()) / max(nf, 1e-8))
            inter_cos.append(float(F.cosine_similarity(d['inner'], d['full'], dim=0)))
            ring_cos.append(float(F.cosine_similarity(d['ring'], d['full'], dim=0)))
            color_norm.append(float(d['color'].norm()))
            color_cos.append(float(F.cosine_similarity(d['color'], d['full'], dim=0)))
    return dict(norm_dT_full=stats(full_norm), ring_over_full_norm=stats(ring_ratio), cos_interior_vs_full=stats(inter_cos), cos_ring_vs_full=stats(ring_cos), norm_dT_chroma_only_control=stats(color_norm), cos_chroma_control_vs_full=stats(color_cos))


def contact_sheet(ds, stream, ex, path, n=12, only=None):
    tiles = []
    step = 0
    while len(tiles) < n:
        slots = next(iter(Slots(stream, 0, 1, ex, step)))[1]
        arrays = [load_rgb(iid) for _, iid in slots]
        for k, (col, iid) in enumerate(slots):
            boxes, j, edited, m = ds.edit_slot(step, col, iid, arrays[k], arrays[(k + 1) % len(arrays)])
            if m['op'] < 0 or len(tiles) >= n or (only is not None and m['op'] != only):
                continue
            l, t, r, b = boxes[j]
            margin = max(int(.2 * max(r - l, b - t)), 6)
            H, W = arrays[k].shape[:2]
            L, Tt, R, B = max(0, l - margin), max(0, t - margin), min(W, r + margin), min(H, b + margin)
            orig = Image.fromarray(arrays[k][Tt:B, L:R]).resize((240, int(240 * (B - Tt) / (R - L))), Image.BICUBIC)
            edit = Image.fromarray(edited[Tt:B, L:R]).resize(orig.size, Image.BICUBIC)
            diff = np.abs(edited[Tt:B, L:R].astype(np.float32) - arrays[k][Tt:B, L:R].astype(np.float32)).mean(2)
            heat = Image.fromarray(np.uint8(np.clip(diff * 8, 0, 255))).resize(orig.size, Image.NEAREST).convert('RGB')
            draw = ImageDraw.Draw(orig)
            sx, sy = orig.size[0] / (R - L), orig.size[1] / (B - Tt)
            draw.rectangle([(l - L) * sx, (t - Tt) * sy, (r - L) * sx - 1, (b - Tt) * sy - 1], outline=(0, 255, 0))
            ImageDraw.Draw(heat).text((3, 3), f"{T.OPS[m['op']]} s={m['sigma']:.1f} a={m['strength']:.2f}", fill=(255, 255, 0))
            ImageDraw.Draw(edit).text((3, 3), f"{r - l}x{b - t}", fill=(255, 255, 0))
            tile = Image.new('RGB', (orig.size[0] * 3 + 8, orig.size[1]), 'white')
            tile.paste(orig, (0, 0))
            tile.paste(edit, (orig.size[0] + 4, 0))
            tile.paste(heat, (2 * orig.size[0] + 8, 0))
            tiles.append(tile)
        step += 1
    rows = [tiles[i:i + 2] for i in range(0, n, 2)]
    sheet_h = sum(max(t.size[1] for t in r) + 6 for r in rows)
    sheet = Image.new('RGB', (2 * tiles[0].size[0] + 12, sheet_h), 'white')
    y = 0
    for r in rows:
        x = 0
        for t in r:
            sheet.paste(t, (x, y))
            x += t.size[0] + 12
        y += max(t.size[1] for t in r) + 6
    sheet.save(path)


def run(a, dev):
    out = Path(a.run) / 'sanity'
    out.mkdir(parents=True, exist_ok=True)
    report = dict(condition=a.condition, lam_cue=LAM_CUE, lam_global=LAM_G, tau=TAU)
    args = argparse.Namespace(condition=a.condition, initial=str(Path(a.run) / 'initial.pt'))
    model, ema, opt, enc0, m2 = build(args, dev)
    ex = eval_exclusions()
    stream = np.load(STREAM)[:12000]
    ds = CueDS(a.shared, native=m2)
    sampler = Slots(stream, 0, 1, ex['raev2_stream'])
    items = [next(iter(Slots(stream, 0, 1, ex['raev2_stream'], s))) for s in range(N_STEPS)]
    batches = [RC.to_dev(ds[it], dev) for it in items]

    # 1, 2. Geometry and locality.
    outside_native, outside_canvas, inside_canvas, chroma, luma_abs, ops, areas = [], [], [], [], [], [], []
    for s, item in enumerate(items):
        step, slots, _ = item
        arrays = [load_rgb(iid) for _, iid in slots]
        b = batches[s]
        for k, (col, iid) in enumerate(slots):
            boxes, j, edited, m = ds.edit_slot(step, col, iid, arrays[k], arrays[(k + 1) % len(arrays)])
            assert b['cue_box'][k].tolist() == list(map(float, boxes[j])), 'cue box mismatch'
            ops.append(m['op'])
            if m['op'] < 0:
                continue
            l, t, r, bb = boxes[j]
            mask = np.zeros(arrays[k].shape[:2], bool)
            mask[t:bb, l:r] = True
            outside_native.append(int(np.abs(edited.astype(int) - arrays[k].astype(int))[~mask].sum()))
            chroma.append(m['chroma_abs'])
            luma_abs.append(m['luma_abs'])
            H, W = arrays[k].shape[:2]
            from src.siglip_pm.data import LB
            fl = LB(W, H, 384, 16)
            cl, ct, cr, cb = fl.ox + l * fl.nw / W - 2, fl.oy + t * fl.nh / H - 2, fl.ox + r * fl.nw / W + 2, fl.oy + bb * fl.nh / H + 2
            canvas_diff = (b['full'][k] - b['full_p'][k]).abs().amax(0).cpu().numpy()
            yy, xx = np.mgrid[:384, :384]
            inside = (xx + .5 >= cl) & (xx + .5 <= cr) & (yy + .5 >= ct) & (yy + .5 <= cb)
            outside_canvas.append(float(canvas_diff[~inside].max()))
            inside_canvas.append(float(canvas_diff[inside].max()))
            areas.append((r - l) * (bb - t))
    assert max(outside_native) == 0, 'native pixels changed outside the ROI'
    assert max(outside_canvas) < 1e-3, f'canvas changed outside the ROI by {max(outside_canvas)}'
    report['check1_2'] = dict(n_edits=len(outside_native), n_no_edit=int(np.sum(np.array(ops) < 0)), native_outside_sum_max=int(max(outside_native)), canvas_outside_max=float(max(outside_canvas)), canvas_inside_median_max=float(np.median(inside_canvas)), roi_native_area=stats(areas), op_counts={T.OPS[o]: int(np.sum(np.array(ops) == o)) for o in range(3)})

    # 3. Teacher response, including bf16 vs fp32 of the difference direction and the cached T0.
    rows = []
    bf_cos, cache_cos = [], []
    for s, b in enumerate(batches):
        t = teacher(enc0, b)
        with torch.no_grad():
            tok0 = torch.cat([enc0.forward_features(b['crops0'][i:i + 64]) for i in range(0, len(b['crops0']), 64)]).float()
            tok1 = torch.cat([enc0.forward_features(b['crops1'][i:i + 64]) for i in range(0, len(b['crops1']), 64)]).float()
            cov = b['cue_cov'].float()
            d32 = F.normalize(pool(tok1, cov), dim=-1) - F.normalize(pool(tok0, cov), dim=-1)
        d16 = F.normalize(t['T1'], dim=-1) - F.normalize(t['T0'], dim=-1)
        bf_cos.append(F.cosine_similarity(d16, d32, dim=-1).cpu().numpy())
        cache_cos.append(F.cosine_similarity(t['T0'], b['tpool'][b['cue_index']], dim=-1).cpu().numpy())
        rows.append(d16.norm(dim=-1).cpu().numpy())
    norms = np.concatenate(rows)
    report['check3'] = dict(norm_dT=stats(norms), fraction_below_tau=float((norms < TAU).mean()), cos_dT_bf16_vs_fp32=stats(np.concatenate(bf_cos)), cos_T0_online_vs_cache=stats(np.concatenate(cache_cos)))
    assert float(np.concatenate(cache_cos).min()) > .9995
    assert float(np.median(norms)) > 2 * TAU, 'teacher response too small'

    # 4. Artifact / colour controls.
    report['check4'] = regions_and_color(ds, stream, ex['raev2_stream'], enc0, dev)
    contact_sheet(ds, stream, ex['raev2_stream'], out / f'examples_{a.condition}.png')
    contact_sheet(ds, stream, ex['raev2_stream'], out / f'examples_swap_{a.condition}.png', n=16, only=0)

    # 5, 6. Gradients and the training step; 10 parity.
    model.train()
    history = []
    for step in range(3):
        b = batches[step]
        for group in opt.param_groups:
            group['lr'] = RC.C.lr_at(step, group['base'], WARM, 12000)
        t = objective(model, enc0, b, m2)
        check = grad_check(model, t)
        opt.zero_grad(set_to_none=True)
        t['total'].backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        grads = step_update(model, opt, ema, step, 12000)
        history.append(dict(step=step + 1, lr_new=opt.param_groups[-1]['lr'], total=float(t['total']), crop=float(t['crop']), crop0=float(t['crop0']), crop1=float(t['crop1']), glob=float(t['glob']), glob0=float(t['glob0']), glob1=float(t['glob1']), cue=float(t['cue']), dT_norm=float(t['nT'].mean()), dS_norm=float(t['nS'].mean()), cos=float(t['cos'][t['valid']].mean()), valid_frac=float(t['valid'].float().mean()), missing=missing, grad_clip=grads, **check))
    assert not any(h['missing'] for h in history), 'parameters without gradient'
    assert all(h['cue_grad_coarse'] > 0 and h['cue_grad_fine'] > 0 for h in history), 'L_cue gradient does not reach coarse/fine'
    report['training_steps'] = history
    if m2:
        assert history[0]['cue_grad_detail_proj'] > 0 and history[0]['cue_grad_detail_stem'] == 0, 'expected zero-init proj gradient only'
        assert history[-1]['cue_grad_detail_stem'] > 0, 'Detail stem has no L_cue gradient after the first update'
    report['batch_sha'] = {str(s): batch_sha(b) for s, b in enumerate(batches[:6])}

    # 9. Global / representation state at start.
    from src.config import COCO_TRAIN, FEAT
    import pandas as pd
    dev_rois = pd.read_parquet(f'{FEAT}/dev_rois.parquet')
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex['raev2_dev'])]
    dev_ids = sorted(int(i) for i in dev_rois.image_id.unique())[:64]
    dev_ds = CocoDS(dev_rois, COCO_TRAIN, 384, 16)
    chunks = [dev_ids[i:i + 8] for i in range(0, len(dev_ids), 8)]
    from src.siglip_cue.model import from_initial
    from src.siglip_pm.model import load_hcf
    initial = Path(a.run) / 'initial.pt'
    if m2:
        fresh, _ = from_initial(initial, dev)
        fresh.eval().requires_grad_(False)
        from src.siglip_cue.train import DevNative
        report['check9'] = dev_diag(fresh, enc0, DevNative(dev_ds, fresh), chunks, dev)
        p8, _ = load_hcf(initial, dev)
        x = batches[0]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            of = fresh.forward_all(x['full'], native=x['native0'], geo=x['geo'])
            op = p8.forward_all(x['full'])
        report['m2_equals_p8_at_start'] = dict(coarse_max_abs=float((of['coarse'].float() - op['coarse'].float()).abs().max()), fine_max_abs=float((of['mem']['fine'].float() - op['mem']['fine'].float()).abs().max()), cls_max_abs=float((of['cls'].float() - op['cls'].float()).abs().max()))
        assert max(report['m2_equals_p8_at_start'].values()) == 0, 'M2 start differs from P8'
        # 7. Detail really reads pre-resize pixels: a checkerboard at 1 native px is invisible to the 384 input but not to the Detail path.
        k = int(np.argmax([float((x['native0'][i] != 0).float().mean()) for i in range(len(x['native0']))]))
        probe = copy.deepcopy(fresh)
        probe.detail.proj.reset_parameters()
        native = x['native0'][k:k + 1]
        geo = x['geo'][k:k + 1]
        w, h = int(geo[0, 0]), int(geo[0, 1])
        pattern = torch.ones(1, 3, 640, 640, device=dev)
        pattern[:, :, ::2, :] *= -1
        pattern[:, :, :, ::2] *= -1
        pattern = pattern * (6 / 127.5)
        pattern[:, :, h:, :] = 0
        pattern[:, :, :, w:] = 0
        with torch.no_grad():
            r0 = probe.detail(native, geo).float()
            r1 = probe.detail((native + pattern).clamp(-1, 1), geo).float()
            pil0 = Image.fromarray(((native[0, :, :h, :w] + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
            pil1 = Image.fromarray((((native[0, :, :h, :w] + pattern[0, :, :h, :w]).clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
            from src.siglip_pm.data import LB
            fl = LB(w, h, 384, 16)
            canvas_diff = float((fl.canvas(pil1) - fl.canvas(pil0)).abs().max())
        stream_iid = items[0][1][k][1]
        arr = load_rgb(stream_iid)
        expected = native_canvas(arr).to(dev)
        report['check7'] = dict(native_matches_decoded_image=bool(torch.equal(expected, x['native0'][k])), canvas_max_change_from_native_checker=canvas_diff, detail_output_change=float((r1 - r0).abs().max()), image_wh=[w, h])
        assert report['check7']['native_matches_decoded_image'] and report['check7']['detail_output_change'] > 0
        # 8. Detail runs once per image however many ROIs are read.
        M = pd.read_parquet('output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet')
        import src.siglip_cue.readout as RD
        ids = sorted(M.image_id.unique())[:8]
        sub = M[M.image_id.isin(ids)].reset_index(drop=True)
        ds_val = BBNativeDS(sub, np.zeros((len(sub), 59), np.int8), False, 'bil', 384, 16)
        bv = RC.to_dev(ds_val[list(range(len(ds_val)))], dev)
        fresh.detail.images_seen = 0
        fresh.roi_mode = 'M2'
        with torch.no_grad():
            f = encode_native(fresh, bv, ['Q', 12])
        report['check8'] = dict(n_images=int(len(bv['x'])), n_rois=int(len(bv['rows'])), detail_images_seen=int(fresh.detail.images_seen), q_shape=list(f['Q'].shape))
        assert report['check8']['detail_images_seen'] == report['check8']['n_images'] and report['check8']['n_rois'] > report['check8']['n_images']
    else:
        p8, _ = load_hcf(initial, dev)
        report['check9'] = dev_diag(p8, enc0, dev_ds, chunks, dev)

    # Small-sample response statistics (initial state).
    samples = make_samples(dev_items()[:128])
    if m2:
        table, info = summarise(run_model(fresh, enc0, samples, dev, True))
    else:
        table, info = summarise(run_model(p8, enc0, samples, dev, False))
    report['response_initial'] = dict(info=info, table=table.to_dict('records'))
    report['status'] = 'passed'
    (out / f'sanity_{a.condition}.json').write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, default=str), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', required=True, choices=('M1', 'M2'))
    ap.add_argument('--run', required=True)
    ap.add_argument('--shared', required=True)
    a = ap.parse_args()
    RC.ddp_setup()
    run(a, torch.device('cuda'))


if __name__ == '__main__':
    main()
