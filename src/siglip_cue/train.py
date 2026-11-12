"""M1 / M2: SigLIP2 P8 with Local Cue Response. Same COCO stream, optimizer, schedule and EMA as P8; both views of every image go through the student.

  L = L_crop + 0.5 L_global + 0.2 L_cue
  L_crop   = mean of the P8 crop cosine on I and on I' (I': the edited ROI against the online crop teacher, other ROIs only if they do not overlap it)
  L_global = mean of the P8 global term on I (cached) and on I' (frozen encoder on I')
  L_cue    = 1 - cos(dS, dT) over ROIs with ||dT|| >= TAU; dS = normalize(Q(I', r)) - normalize(Q(I, r)), dT from the frozen crop teacher (no gradient)

M1 is the P8 architecture unchanged; M2 adds the Detail path (src.siglip_cue.model). Identical loss, data order and edits for both.
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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT, STREAM
from src.ivc.ft import cosd, ema_update
from src.msd.pretrain import eval_exclusions
from src.siglip_cue.data import CueDS, native_canvas, native_geo, load_rgb
from src.siglip_cue.model import from_initial
from src.siglip_pm.cocodata import CocoDS
from src.siglip_pm.distill import LAM_G, M0_EMA, WARM, Slots, atomic_save, cell_w, dev_diag, groups, pool, read_q
from src.siglip_pm.model import MODEL_ID, REVISION, checksum, load_hcf, load_vit, n_params, run_enc

LAM_CUE = .2
TAU = .02
STOP = False


def stop(*_):
    global STOP
    STOP = True


def forward_two(model, b, m2):
    x = torch.cat([b['full'], b['full_p']])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if m2:
            o = model.forward_all(x, native=torch.cat([b['native0'], b['native1']]), geo=torch.cat([b['geo'], b['geo']]))
        else:
            o = model.forward_all(x)
    return dict(cls=o['cls'].float(), coarse=o['coarse'].float(), mem={k: v.float() for k, v in o['mem'].items()}, grids=o['grids'])


@torch.no_grad()
def teacher(enc0, b):
    tok0, _ = run_enc(enc0, b['crops0'])
    tok1, _ = run_enc(enc0, b['crops1'])
    cov = b['cue_cov'].float()
    gtok, gembed = run_enc(enc0, b['full_p'])
    return dict(T0=pool(tok0, cov), T1=pool(tok1, cov), gembed1=gembed.float(), gmean1=pool(gtok, b['fvalid'].float()))


def responses(Q0, Q1, t, cue_index):
    s0, s1 = F.normalize(Q0[cue_index], dim=-1), F.normalize(Q1[cue_index], dim=-1)
    t0, t1 = F.normalize(t['T0'], dim=-1), F.normalize(t['T1'], dim=-1)
    return s1 - s0, t1 - t0


def objective(model, enc0, b, m2, tau=TAU):
    n = len(b['full'])
    o = forward_two(model, b, m2)
    ri = b['roi_img']
    w24, w48 = cell_w(b['box_c'], 24, 16), cell_w(b['box_c'], 48, 8)
    A0, Q0 = read_q(model, o, ri, w24, w48)
    _, Q1 = read_q(model, o, ri + n, w24, w48)
    t = teacher(enc0, b)
    target1 = b['tpool'].clone()
    target1[b['cue_index']] = t['T1']
    crop0 = cosd(Q0, b['tpool']).mean()
    crop1 = cosd(Q1, target1)[b['crop1_mask']].mean()
    fv = b['fvalid'].float()
    glob0 = (.5 * (cosd(o['cls'][:n], b['gembed']) + cosd(pool(o['coarse'][:n], fv), b['gmean']))).mean()
    glob1 = (.5 * (cosd(o['cls'][n:], t['gembed1']) + cosd(pool(o['coarse'][n:], fv), t['gmean1']))).mean()
    dS, dT = responses(Q0, Q1, t, b['cue_index'])
    nT = dT.norm(dim=-1)
    valid = (nT >= tau) & (b['cue_meta'][:, 0] >= 0)
    cos = F.cosine_similarity(dS, dT, dim=-1)
    cue = (1 - cos)[valid].mean() if bool(valid.any()) else dS.sum() * 0
    crop = .5 * (crop0 + crop1)
    glob = .5 * (glob0 + glob1)
    return dict(o=o, Q0=Q0, Q1=Q1, A0=A0, crop=crop, crop0=crop0, crop1=crop1, glob=glob, glob0=glob0, glob1=glob1, cue=cue, dS=dS.detach(), dT=dT.detach(), nT=nT.detach(), nS=dS.detach().norm(dim=-1), cos=cos.detach(), valid=valid, t=t,
                total=crop + LAM_G * glob + LAM_CUE * cue)


def param_sets(model):
    r = dict(coarse=list(model.vit.parameters()), fine=[p for n, p in model.new_named() if not n.startswith('detail.')])
    if hasattr(model, 'detail'):
        r.update(detail_stem=list(model.detail.stem.parameters()), detail_proj=list(model.detail.proj.parameters()))
    return r


def gnorm(gs):
    q = [g.float().square().sum() for g in gs if g is not None]
    return float(torch.stack(q).sum().sqrt()) if q else 0.


def grad_check(model, t):
    sets = param_sets(model)
    flat, cuts, names = [], [], list(sets)
    for name in names:
        flat += sets[name]
        cuts.append(len(flat))
    r = {}
    for loss_name in ('crop', 'glob', 'cue'):
        g = torch.autograd.grad(t[loss_name], flat, retain_graph=True, allow_unused=True)
        start = 0
        for name, end in zip(names, cuts):
            r[f'{loss_name}_grad_{name}'] = gnorm(g[start:end])
            start = end
        r[f'{loss_name}_loss'] = float(t[loss_name])
    return r


def batch_sha(b):
    h = hashlib.sha1()
    for key in ('full', 'full_p', 'box_c', 'cue_index', 'cue_meta', 'crops1'):
        h.update(b[key].cpu().numpy().tobytes())
    return h.hexdigest()


def build(a, dev):
    torch.manual_seed(0)
    np.random.seed(0)
    m2 = a.condition == 'M2'
    if m2:
        model, ck = from_initial(a.initial, dev)
    else:
        model, ck = load_hcf(a.initial, dev)
        assert ck['cond'] == 'P' and ck['step'] == 0 and ck['backbone'] == MODEL_ID
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(groups(model), betas=(.9, .999))
    enc0 = load_vit(dev).eval()
    for p in enc0.parameters():
        p.requires_grad_(False)
    return model, ema, opt, enc0, m2


def step_update(model, opt, ema, step, total):
    norms = [torch.nn.utils.clip_grad_norm_(list(model.vit.parameters()), 1.), torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)]
    if not all(bool(torch.isfinite(v)) for v in norms):
        raise FloatingPointError('gradient')
    opt.step()
    m = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
    ema_update(ema, model, m)
    return [float(v) for v in norms]


class DevNative:
    """dev_diag helper for M2: hands the native image of every dev chunk to the model before it is encoded."""

    def __init__(self, ds, model):
        self.ds, self.model = ds, model

    def __getitem__(self, chunk):
        b = self.ds[chunk]
        arrays = [load_rgb(int(i)) for i in chunk]
        self.model.set_pending(torch.stack([native_canvas(x) for x in arrays]).to(next(self.model.parameters()).device),
                               torch.stack([native_geo(x.shape[1], x.shape[0]) for x in arrays]).to(next(self.model.parameters()).device))
        return b


def budget(a, stream, excluded):
    ids = stream[:a.steps].reshape(-1)
    used = ids[~np.isin(ids, list(excluded))]
    return dict(optimizer_steps=a.steps, images_per_step=16, unique_images=int(len(np.unique(used))), image_visits=int(len(used)), student_image_forwards=int(2 * len(used)), teacher_crop_forwards=int(2 * len(used)),
                teacher_full_forwards=int(len(used)), p8_student_image_forwards=int(len(used)), compute_matched_to_p8=False)


def config(a, model, excluded, stream):
    return dict(args=vars(a), condition=a.condition, method='local_cue', lambda_cue=LAM_CUE, lambda_global=LAM_G, tau=TAU, steps=a.steps, world=1, images_per_step=16, seed=0, n_params_total=n_params(model), n_params_vit=n_params(model.vit), n_params_new=n_params(model) - n_params(model.vit),
                backbone=MODEL_ID, revision=REVISION, torch=torch.__version__, data_budget=budget(a, stream, excluded), excluded_stream=sorted(excluded))


def train(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        raise FileExistsError(out)
    for folder in ('logs', 'checkpoints'):
        (out / folder).mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    model, ema, opt, enc0, m2 = build(a, dev)
    teacher_sum = checksum(enc0.parameters())
    ex = eval_exclusions()
    stream = np.load(STREAM)[:a.steps]
    total = len(stream)
    assert total == a.steps
    ds = CueDS(a.shared, native=m2)
    dl = DataLoader(ds, sampler=Slots(stream, 0, 1, ex['raev2_stream']), batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dev_rois = pd.read_parquet(os.path.join(FEAT, 'dev_rois.parquet'))
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex['raev2_dev'])]
    dev_ids = sorted(int(i) for i in dev_rois.image_id.unique())
    dev_ds = CocoDS(dev_rois, COCO_TRAIN, 384, 16)
    dev_chunks = [dev_ids[i:i + 8] for i in range(0, len(dev_ids), 8)]
    cfg = config(a, model, ex['raev2_stream'], stream)
    cfg.update(n_dev_images=len(dev_ids), teacher_checksum=teacher_sum)
    (out / 'config.json').write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps(cfg, default=str), flush=True)

    def diag(step, which):
        for name, m in which.items():
            record = dict(step=step, model=name, **dev_diag(m, enc0, DevNative(dev_ds, m) if m2 else dev_ds, dev_chunks, dev))
            with (out / 'logs/diag.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)

    def save(step):
        cond = 'M2' if m2 else 'P'
        ck = dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond=cond, kind='ema', s0=.1, step=step, backbone=MODEL_ID, revision=REVISION, pretraining=dict(condition=a.condition, lambda_cue=LAM_CUE, tau=TAU))
        atomic_save(ck, out / 'checkpoints/ema_last.pt')

    t0 = time.monotonic()
    last = t0
    values = []
    diag(0, dict(init=model))
    with (out / 'logs/train.jsonl').open('a') as log, (out / 'logs/objective_checks.jsonl').open('a') as checks:
        for step, b in enumerate(dl):
            b = RC.to_dev(b, dev)
            for group in opt.param_groups:
                group['lr'] = RC.C.lr_at(step, group['base'], WARM, total)
            t = objective(model, enc0, b, m2)
            if step in (0, 1, 2) or (step + 1) % 3000 == 0:
                record = dict(step=step + 1, batch_sha=batch_sha(b), **grad_check(model, t))
                checks.write(json.dumps(record) + '\n')
                checks.flush()
                print(json.dumps(dict(objective_check=record)), flush=True)
            loss = t['total']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'loss at {step + 1}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = step_update(model, opt, ema, step, total)
            v = t['valid']
            values.append([float(t['crop']), float(t['crop0']), float(t['crop1']), float(t['glob']), float(t['cue']), float(t['nT'].mean()), float(t['nS'].mean()), float(t['cos'][v].mean()) if bool(v.any()) else float('nan'), float(v.float().mean()), float(cosd(t['A0'], b['tpool']).mean()), grad[0], grad[1]])
            if step == 0 or (step + 1) % a.log_every == 0:
                now = time.monotonic()
                x = np.nanmean(values, axis=0)
                record = dict(step=step + 1, elapsed_s=round(now - t0, 2), crop_loss=x[0], crop0_loss=x[1], crop1_loss=x[2], global_loss=x[3], cue_loss=x[4], dT_norm=x[5], dS_norm=x[6], dS_dT_cos=x[7], valid_frac=x[8], coarse_cos_loss=x[9], grad_enc=x[10], grad_new=x[11], step_seconds=(now - last) / len(values), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
                log.write(json.dumps({k: float(v) for k, v in record.items()}) + '\n')
                log.flush()
                print(json.dumps({k: float(v) for k, v in record.items()}), flush=True)
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
    (out / 'done.json').write_text(json.dumps(dict(status='done', steps=total, time_s=round(time.monotonic() - t0), teacher_unchanged=True, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, ema_checksum=checksum(ema.parameters())), indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition', required=True, choices=('M1', 'M2'))
    ap.add_argument('--initial', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shared', required=True)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--log-every', type=int, default=100)
    a = ap.parse_args()
    assert 'RANK' not in os.environ
    RC.ddp_setup()
    dev = torch.device('cuda')
    try:
        raise SystemExit(train(a, dev))
    except Exception:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / 'failed.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc()), indent=1))
        raise


if __name__ == '__main__':
    main()
