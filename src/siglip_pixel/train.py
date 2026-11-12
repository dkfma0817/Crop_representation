"""S1/S2: SigLIP2 P8 with the DINOv3 Crop-PixelDiT auxiliary (crop_objectives.py is a byte-identical copy of its heads).

Port of hcf_material_causal/architecture_source/src/hcf/train_crop_objective.py (method pixeldit, crop weight 1), with the
centered term of hcf_followup_2h/source/src/hcf/train_crop_objective.py. Changes are limited to the SigLIP2 differences:
  dataset / geometry  src.siglip_pm TrainDS + data.geometry (bilinear resize, mean=std=0.5)
  clean pixel target  the canvas itself: rgb/127.5-1 equals DINOv3's (denorm)*2-1 range; padding 0 is masked by pixel_valid
  global targets      native MAP embedding (gembed) instead of CLS (gcls), the P8 rule
  optimizer groups    src.siglip_pm.distill.groups (the P8 ones); the watched coarse embedding is patch_embedding

  L = L_pixel + 1.0 L_crop + centered_weight L_centered + 0.5 L_global   (S1 centered_weight=0, S2 0.5)
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT, STREAM
from src.ivc.ft import cosd, ema_update
from src.msd.pretrain import eval_exclusions
from src.siglip_pm.cocodata import CocoDS
from src.siglip_pm.data import geometry
from src.siglip_pm.distill import LAM_G, M0_EMA, NEW_LR, WARM, WD, Slots, TrainDS, atomic_save, cell_w, dev_diag, fwd, groups, pool, read_q
from src.siglip_pm.model import MODEL_ID, REVISION, checksum, load_hcf, load_vit, n_params, run_enc
from src.siglip_pixel.crop_objectives import CropPixelDiT, pooled_grid, roi_tokens

HEADS_SHA = 'f7c4ff7517adc300d871f4cfa5bade2b350a379a4718fb0afdf749d0761a49b3'
STOP = False


def stop(*_):
    global STOP
    STOP = True


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class PixelDS(TrainDS):
    """P8 batch + 1 object and 1 nested (if present) RGB crop per image, deterministically rotated by step and stream column."""

    def __getitem__(self, item):
        b = super().__getitem__(item)
        step, slots, _ = item
        crops, cov, grids, valid, indices, ri = [], [], [], [], [], []
        offset = 0
        for k, (column, iid) in enumerate(slots):
            i = self.pos[iid]
            nested = [box for box in self.NB[step, column] if box[2] > box[0]]
            boxes = list(self.obj[i]) + nested
            selected = [(step + column) % 4]
            if nested:
                selected.append(4 + (step + column) % len(nested))
            image = Image.open(os.path.join(COCO_TRAIN, f'{iid:012d}.jpg')).convert('RGB')
            for j in selected:
                bx = tuple(int(x) for x in boxes[j])
                crop, _, grid = geometry(*image.size, bx, 384, 16)
                crops.append(crop.canvas(image.crop(bx)))
                cov.append(crop.coverage())
                grids.append(grid)
                mask = torch.zeros(1, 384, 384)
                mask[:, crop.oy:crop.oy + crop.nh, crop.ox:crop.ox + crop.nw] = 1.
                valid.append(mask)
                indices.append(offset + j)
                ri.append(k)
            offset += len(boxes)
        assert offset == len(b['tpool'])
        b.update(crops=torch.stack(crops), aux_cov=torch.from_numpy(np.stack(cov)), aux_grid=torch.from_numpy(np.stack(grids)), pixel_valid=torch.stack(valid),
                 aux_index=torch.tensor(indices), aux_ri=torch.tensor(ri))
        return b


def pixel_objective(head, memory, mcov, b, step, shuffle=False):
    if shuffle:
        memory = memory.roll(2, 0)
        mcov = mcov.roll(2, 0)
    r = len(memory)
    clean = b['crops']
    gen = torch.Generator(device=memory.device).manual_seed(200000 + step)
    t = .1 + .8 * torch.rand(r, device=memory.device, generator=gen)
    noise = torch.randn(clean.shape, device=memory.device, dtype=clean.dtype, generator=gen)
    noisy = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = head(noisy, t, memory, mcov)
    squared = (pred.float() - (noise - clean)).pow(2) * b['pixel_valid']
    loss = (squared.flatten(1).sum(1) / (3 * b['pixel_valid'].flatten(1).sum(1)).clamp_min(1)).mean()
    return loss, dict(time_mean=float(t.mean()))


def gnorm(grads):
    q = [g.float().square().sum() for g in grads if g is not None]
    return float(torch.stack(q).sum().sqrt()) if q else 0.


def losses(a, model, head, b, step, mu, sd):
    o = fwd(model, b['full'])
    A, Q = read_q(model, o, b['roi_img'], cell_w(b['box_c'], 24, 16), cell_w(b['box_c'], 48, 8))
    crop = cosd(Q, b['tpool']).mean()
    centered = cosd((Q - mu) / sd, (b['tpool'] - mu) / sd).mean()
    glob = .5 * (cosd(o['cls'], b['gembed']) + cosd(pool(o['coarse'], b['fvalid'].float()), b['gmean'])).mean()
    memory, mcov = pooled_grid(roi_tokens(model, o, b), b['aux_cov'], 8)
    pixel, diagnostics = pixel_objective(head, memory, mcov, b, step)
    return dict(o=o, A=A, Q=Q, crop=crop, centered=centered, glob=glob, pixel=pixel, memory=memory, mcov=mcov, diagnostics=diagnostics)


def total_loss(a, t):
    return t['pixel'] + a.crop_weight * t['crop'] + a.centered_weight * t['centered'] + LAM_G * t['glob']


def objective_check(a, model, head, b, step, t):
    """Per-loss gradients into the encoder / fine branch / head, and conditioning gap (same t and noise, shuffled ROI memory)."""
    vit, new, hp = list(model.vit.parameters()), [p for _, p in model.new_named()], list(head.parameters())
    named = dict(pixel=t['pixel'], crop=t['crop'], glob=t['glob'])
    if a.centered_weight:
        named['centered'] = t['centered']
    r = dict(step=step + 1, batch_sha=hashlib.sha1(b['box_c'].cpu().numpy().tobytes() + b['aux_index'].cpu().numpy().tobytes()).hexdigest())
    for name, value in named.items():
        grads = torch.autograd.grad(value, vit + new + hp, retain_graph=True, allow_unused=True)
        r[f'{name}_loss'] = float(value)
        r[f'{name}_grad_vit'] = gnorm(grads[:len(vit)])
        r[f'{name}_grad_new'] = gnorm(grads[len(vit):len(vit) + len(new)])
        r[f'{name}_grad_head'] = gnorm(grads[len(vit) + len(new):])
    watched = [model.vit.patch_embed.patch_embedding.weight, model.fine_embed.weight]
    grads = torch.autograd.grad(t['pixel'], watched, retain_graph=True, allow_unused=True)
    norms = [0. if g is None else float(g.float().norm()) for g in grads]
    assert sum(norms) > 0, 'Pixel objective has no gradient into the encoder embeddings'
    with torch.no_grad():
        shuffled, _ = pixel_objective(head, t['memory'].detach(), t['mcov'], b, step, shuffle=True)
    r.update(aux_grad_coarse_embed=norms[0], aux_grad_fine_embed=norms[1], shuffled_condition_loss=float(shuffled), matched_condition_loss=float(t['pixel']),
             conditioning_gap=float(shuffled - t['pixel']), **t['diagnostics'])
    return r


def build(a, dev):
    torch.manual_seed(0)
    np.random.seed(0)
    model, ck = load_hcf(a.initial, dev)
    assert ck['cond'] == 'P' and ck['step'] == 0 and ck['backbone'] == MODEL_ID
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    torch.manual_seed(700)
    head = CropPixelDiT().to(dev).train()
    pg = groups(model)
    for no_decay in (False, True):
        params = [p for p in head.parameters() if (p.ndim <= 1) == no_decay]
        pg.append(dict(params=params, base=NEW_LR, lr=NEW_LR, weight_decay=0. if no_decay else WD, mod='decoder'))
    opt = torch.optim.AdamW(pg, betas=(.9, .999))
    stats = np.load(a.stats)
    mu, sd = torch.tensor(stats['mu'], device=dev), torch.tensor(stats['sd'], device=dev)
    enc0 = load_vit(dev).eval()
    for p in enc0.parameters():
        p.requires_grad_(False)
    return model, ema, head, opt, mu, sd, enc0


def step_update(model, head, opt, ema, step, total):
    norms = [torch.nn.utils.clip_grad_norm_(list(model.vit.parameters()), 1.), torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.), torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)]
    if not all(bool(torch.isfinite(v)) for v in norms):
        raise FloatingPointError('gradient')
    opt.step()
    m = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
    ema_update(ema, model, m)
    return [float(v) for v in norms]


def config(a, model, head):
    return dict(args=vars(a), condition=a.condition, method='pixeldit', crop_weight=a.crop_weight, centered_weight=a.centered_weight, method_weight=1., global_weight=LAM_G, steps=a.steps, world=1, images_per_step=16,
                aux_crops_per_image='1 object + 1 nested if present', full_size=384, crop_size=384, head_params=n_params(head), encoder_params=n_params(model), initial_sha256=sha256(a.initial), teacher_stats_sha256=sha256(a.stats),
                teacher='RGB crop pixels (pixel objective); frozen SigLIP2 C0 crop means (crop/centered)', seed=0, head_seed=700, noise_seed='200000+step', source_sha256=sha256(__file__), heads_sha256=sha256(Path(__file__).with_name('crop_objectives.py')),
                backbone=MODEL_ID, revision=REVISION, torch=torch.__version__)


def smoke(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert sha256(Path(__file__).with_name('crop_objectives.py')) == HEADS_SHA, 'crop_objectives.py differs from the DINOv3 heads'
    model, ema, head, opt, mu, sd, enc0 = build(a, dev)
    teacher_sum = checksum(enc0.parameters())
    ex = eval_exclusions()['raev2_stream']
    stream = np.load(STREAM)[:a.steps]
    ds = PixelDS(a.shared)
    items = [next(iter(Slots(stream, 0, 1, ex, s))) for s in range(3)]
    report = dict(status='running', condition=a.condition, cfg=config(a, model, head))
    torch.cuda.reset_peak_memory_stats()
    for k, item in enumerate(items):
        b = RC.to_dev(ds[item], dev)
        started = time.perf_counter()
        for group in opt.param_groups:
            group['lr'] = RC.C.lr_at(k, group['base'], WARM, a.steps)
        t = losses(a, model, head, b, k, mu, sd)
        if k == 0:
            # Teacher: dimension, normalization and box <-> cached mean correspondence for the crops the head actually receives.
            assert b['crops'].shape[1:] == (3, 384, 384) and float(b['crops'].min()) >= -1 and float(b['crops'].max()) <= 1
            with torch.no_grad():
                T, _ = run_enc(enc0, b['crops'])
            online = pool(T, b['aux_cov'].float())
            cached = b['tpool'][b['aux_index']]
            cos = torch.nn.functional.cosine_similarity(online, cached, dim=-1)
            assert online.shape[1] == 768 == cached.shape[1] and float(cos.min()) > .9995, float(cos.min())
            # GT box <-> student grid: a crop cell more than half covered has its centre on crop content, so it maps inside the box on the full canvas.
            pix = (b['aux_grid'] + 1) * 192
            box = b['box_c'][b['aux_index']]
            inside = []
            for row in range(len(pix)):
                keep = b['aux_cov'][row] > .5
                if not bool(keep.any()):
                    continue
                x, y = pix[row, keep, 0], pix[row, keep, 1]
                inside.append(float(torch.stack([x.min() - box[row, 0], box[row, 2] - x.max(), y.min() - box[row, 1], box[row, 3] - y.max()]).min().clamp(max=0.)))
            grid_margin_px = min(inside)
            assert len(inside) > 0 and grid_margin_px > -.05, grid_margin_px
            assert (b['aux_ri'].bincount() >= 1).all() and b['aux_index'].max() < len(b['tpool'])
            # Conditioning really is the student memory: it needs gradient from the student features and reaches the encoder.
            assert t['memory'].requires_grad, 'memory is detached'
            coarse_g, fine_g = torch.autograd.grad(t['pixel'], [t['o']['coarse'], t['o']['mem']['fine']], retain_graph=True)
            memory_g = torch.autograd.grad(t['pixel'], t['memory'], retain_graph=True)[0]
            assert float(coarse_g.norm()) > 0 and float(fine_g.norm()) > 0 and float(memory_g.norm()) > 0
            shortcut = dict(crops_in_graph=bool(b['crops'].requires_grad), head_inputs='noisy crop, t, student ROI memory, memory coverage; clean crop only forms the target')
            assert not shortcut['crops_in_graph']
            # Crop / centered / global supervise the ROI-fused student features only; the head stays out of their graph.
            for name in ('crop', 'centered', 'glob'):
                g = torch.autograd.grad(t[name], list(head.parameters()), retain_graph=True, allow_unused=True)
                assert all(x is None for x in g), f'{name} reaches the head'
            check = objective_check(a, model, head, b, k, t)
            for key in ['pixel_grad_vit', 'pixel_grad_new', 'pixel_grad_head', 'crop_grad_vit', 'crop_grad_new', 'glob_grad_vit']:
                assert check[key] > 0, key
            if a.centered_weight:
                assert check['centered_grad_vit'] > 0
            with torch.no_grad():
                zero, _ = pixel_objective(head, t['memory'].detach() * 0, t['mcov'], b, k)
            report.update(teacher=dict(dim=int(online.shape[1]), cached_vs_online_cos_min=float(cos.min()), canvas_min=float(b['crops'].min()), canvas_max=float(b['crops'].max()), normalization='mean=std=0.5 (rgb/127.5-1)'),
                          grid_margin_px=grid_margin_px, n_aux=len(b['crops']), n_roi=len(b['tpool']), n_images=len(b['full']), grads=check, memory_grad_norm=float(memory_g.norm()), coarse_grad_norm=float(coarse_g.norm()), fine_grad_norm=float(fine_g.norm()),
                          shortcut=shortcut, zero_memory_loss=float(zero), crops_sha=hashlib.sha1(b['crops'].cpu().numpy().tobytes()).hexdigest())
        loss = total_loss(a, t)
        assert torch.isfinite(loss), 'non-finite smoke loss'
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if k == 0:
            bad = [n for n, p in list(model.named_parameters()) + list(head.named_parameters()) if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            assert not bad, bad
        norms = step_update(model, head, opt, ema, k, a.steps)
        report.setdefault('steps', []).append(dict(step=k + 1, loss=float(loss), pixel=float(t['pixel']), crop=float(t['crop']), centered=float(t['centered']), glob=float(t['glob']), grad_clip=norms, seconds=time.perf_counter() - started))
    assert checksum(enc0.parameters()) == teacher_sum
    ck = dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond='P', kind='ema', s0=.1, step=0, backbone=MODEL_ID, revision=REVISION)
    path = out / 'smoke_ckpt.pt'
    torch.save(ck, path)
    loaded, _ = load_hcf(path, dev)
    assert not any(k.startswith('head') for k in loaded.state_dict())
    path.unlink()
    report.update(status='passed', peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, teacher_unchanged=True, ema_loads_without_head=True)
    (out / 'smoke.json').write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, default=str), flush=True)


def train(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        raise FileExistsError(out)
    for folder in ('logs', 'checkpoints'):
        (out / folder).mkdir(exist_ok=True)
    assert sha256(Path(__file__).with_name('crop_objectives.py')) == HEADS_SHA
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    model, ema, head, opt, mu, sd, enc0 = build(a, dev)
    teacher_sum = checksum(enc0.parameters())
    ex = eval_exclusions()
    stream = np.load(STREAM)[:a.steps]
    total = len(stream)
    assert total == a.steps
    ds = PixelDS(a.shared)
    dl = DataLoader(ds, sampler=Slots(stream, 0, 1, ex['raev2_stream']), batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dev_rois = pd.read_parquet(os.path.join(FEAT, 'dev_rois.parquet'))
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex['raev2_dev'])]
    dev_ids = sorted(int(i) for i in dev_rois.image_id.unique())
    dev_ds = CocoDS(dev_rois, COCO_TRAIN, 384, 16)
    dev_chunks = [dev_ids[i:i + 8] for i in range(0, len(dev_ids), 8)]
    cfg = config(a, model, head)
    cfg.update(excluded_stream=ex['raev2_stream'], n_dev_images=len(dev_ids), teacher_checksum=teacher_sum)
    (out / 'config.json').write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps(cfg, default=str), flush=True)

    def diag(step, which):
        for name, m in which.items():
            record = dict(step=step, model=name, **dev_diag(m, enc0, dev_ds, dev_chunks, dev))
            with (out / 'logs/diag.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)

    def save(step):
        ck = dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond='P', kind='ema', s0=.1, step=step, backbone=MODEL_ID, revision=REVISION,
                  pretraining=dict(condition=a.condition, crop_weight=a.crop_weight, centered_weight=a.centered_weight, head_params=n_params(head)))
        atomic_save(ck, out / 'checkpoints/ema_last.pt')
        atomic_save(dict(model=model.state_dict(), ema=ema.state_dict(), head=head.state_dict(), opt=opt.state_dict(), step=step), out / 'checkpoints/recovery.pt')

    t0 = time.monotonic()
    last = t0
    values = []
    diag(0, dict(init=model))
    with (out / 'logs/train.jsonl').open('a') as log, (out / 'logs/objective_checks.jsonl').open('a') as checks:
        for step, b in enumerate(dl):
            b = RC.to_dev(b, dev)
            for group in opt.param_groups:
                group['lr'] = RC.C.lr_at(step, group['base'], WARM, total)
            t = losses(a, model, head, b, step, mu, sd)
            if step == 0 or (step + 1) % 3000 == 0:
                record = objective_check(a, model, head, b, step, t)
                checks.write(json.dumps(record) + '\n')
                checks.flush()
                print(json.dumps(dict(objective_check=record)), flush=True)
            loss = total_loss(a, t)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'loss at {step + 1}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            step_update(model, head, opt, ema, step, total)
            values.append([float(t['pixel']), float(t['crop']), float(t['centered']), float(t['glob']), float(cosd(t['A'], b['tpool']).mean())])
            if step == 0 or (step + 1) % a.log_every == 0:
                now = time.monotonic()
                v = np.mean(values, axis=0)
                record = dict(step=step + 1, elapsed_s=round(now - t0, 2), pixel_loss=float(v[0]), crop_cos_loss=float(v[1]), centered_loss=float(v[2]), global_loss=float(v[3]), coarse_cos_loss=float(v[4]), step_seconds=(now - last) / len(values), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
                log.write(json.dumps(record) + '\n')
                log.flush()
                print(json.dumps(record), flush=True)
                values = []
                last = now
            if (step + 1) % 3000 == 0:
                save(step + 1)
                if step + 1 < total:
                    diag(step + 1, dict(ema=ema))
            if STOP:
                save(step + 1)
                (out / 'interrupted.json').write_text(json.dumps(dict(step=step + 1, reason='signal')))
                return 143
    diag(total, dict(ema=ema, student=model))
    assert checksum(enc0.parameters()) == teacher_sum, 'teacher changed'
    result = dict(status='done', steps=total, time_s=round(time.monotonic() - t0), teacher_unchanged=True, head_removed_for_probe=True, ema_checksum=checksum(ema.parameters()))
    (out / 'done.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition', required=True, choices=('S1', 'S2'))
    ap.add_argument('--crop-weight', type=float, default=1.)
    ap.add_argument('--centered-weight', type=float, default=0.)
    ap.add_argument('--initial', required=True)
    ap.add_argument('--stats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shared', required=True)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    assert 'RANK' not in os.environ
    RC.ddp_setup()
    dev = torch.device('cuda')
    try:
        if a.smoke:
            smoke(a, dev)
        else:
            raise SystemExit(train(a, dev))
    except Exception:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / 'failed.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc()), indent=1))
        raise


if __name__ == '__main__':
    main()
