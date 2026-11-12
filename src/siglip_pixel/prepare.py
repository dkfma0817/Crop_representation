"""S1/S2 common start: the same SigLIP2 P8 start state (official encoder + seed-0 fine branch + P8 calibrated scales).

P8 saved no step-0 checkpoint, so the state is rebuilt with the P8 builder and verified against what P8 recorded: encoder
checksum, residual scales and, on the same 8 images, the calibration diagnostics before and after applying the scales.
teacher_stats.npz is the DINOv3 centered-loss recipe (src.hcf.causal_distill.prepare) applied to the SigLIP2 teacher cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.config import COCO_TRAIN, STREAM
from src.msd.pretrain import eval_exclusions
from src.siglip_pm.data import LB
from src.siglip_pm.model import MODEL_ID, REVISION, build_student, checksum

TOL = 2e-3


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ratios(model, x):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = model.forward_all(x, collect=True)
        A = o['coarse']
        B = F.avg_pool2d(o['mem']['fine'].reshape(len(x), 48, 48, -1).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)
        q = model.fuse(A, dict(fine=B))
        ratio = float((q - A).norm(dim=-1).mean() / A.norm(dim=-1).mean())
    return ratio, dict(model.diag)


def fit_stats(shared, out):
    ex = set(eval_exclusions()['raev2_stream'])
    ids = np.load(f'{shared}/image_ids.npy')
    targets = np.load(f'{shared}/t_object.npy', mmap_mode='r')
    rows = np.flatnonzero(~np.isin(ids, list(ex)))
    rng = np.random.default_rng(0)
    flat = rng.choice(len(rows) * 4, min(100000, len(rows) * 4), replace=False)
    x = np.asarray(targets[rows[flat // 4], flat % 4], dtype=np.float64)
    assert np.isfinite(x).all()
    mu, sd = x.mean(0), x.std(0, ddof=1)
    floor = .1 * np.median(sd)
    sd = np.maximum(sd, floor)
    np.savez(out, mu=mu.astype(np.float32), sd=sd.astype(np.float32), n=len(x), sd_floor=floor)
    return dict(n=len(x), sd_floor=float(floor), sd_median=float(np.median(sd)), mu_norm=float(np.linalg.norm(mu)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--p8', required=True, help='completed P8 directory (read only)')
    ap.add_argument('--shared', required=True)
    a = ap.parse_args()
    run, p8 = Path(a.run), Path(a.p8)
    path = run / 'initial.pt'
    if path.exists():
        raise FileExistsError(path)
    dev = torch.device('cuda')
    cal = json.load(open(p8 / 'calib.json'))
    cfg = yaml.safe_load(open(p8 / 'config.yaml'))
    stream = np.load(STREAM)
    ex = set(eval_exclusions()['raev2_stream'])
    ids = [int(i) for i in stream[0] if int(i) not in ex][:8]
    x = []
    for iid in ids:
        img = Image.open(os.path.join(COCO_TRAIN, f'{iid:012d}.jpg')).convert('RGB')
        x.append(LB(*img.size, 384, 16).canvas(img))
    x = torch.stack(x).to(dev)
    model = build_student('P', dev, None, .1).eval()
    enc_sum = checksum(model.vit.parameters())
    assert enc_sum == cfg['init_enc_checksum'], (enc_sum, cfg['init_enc_checksum'])
    ratio0, diag0 = ratios(model, x)
    h0 = cal['history'][0]
    before = dict(q_ratio=(ratio0, h0['q_ratio']), **{k: (diag0[k], h0[k]) for k in diag0})
    assert all(abs(u - v) < TOL for u, v in before.values()), before
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in cal['scales']:
                p.fill_(cal['scales'][n])
    _, diag1 = ratios(model, x)
    after = {k: (diag1[k], cal['final_stage_ratios'][k]) for k in diag1}
    assert all(abs(u - v) < TOL for u, v in after.values()), after
    scales = {n: float(p) for n, p in model.named_parameters() if p.ndim == 0}
    assert scales == cal['scales'], (scales, cal['scales'])
    torch.save(dict(model=model.state_dict(), enc=model.vit.state_dict(), cond='P', kind='ema', s0=.1, step=0, backbone=MODEL_ID, revision=REVISION), path)
    stats = fit_stats(a.shared, run / 'teacher_stats.npz')
    meta = dict(initial_sha256=sha256(path), teacher_stats_sha256=sha256(run / 'teacher_stats.npz'), encoder_checksum=enc_sum,
                encoder_checksum_p8_config=cfg['init_enc_checksum'], p8_scales_equal=True, calib_before=before, calib_after=after, tolerance=TOL,
                stats=stats, centered_recipe='src.hcf.causal_distill.prepare (uniform 100000 object teacher means, seed 0, ex. raev2_stream, floor 0.1 median sd)',
                backbone=MODEL_ID, revision=REVISION, p8_dir=str(p8), shared=a.shared)
    (run / 'design.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
