"""Shared, fixed inputs of FA-A / FA-B (built once in a separate process; does not touch the training RNG):

  init.pt         FineAware initial state: torch.manual_seed(0) -> frozen DINOv3 (same call order as S3) -> FineAware(vit); sha256 recorded.
                  FA-A and FA-B both load this file, so their initial parameters are bit-identical.
  constants.json  FA-A normalisers from COCO train teacher features (512 stream images, rng 20261113, no flip, every content parent):
                  v_mean = mean over dims of the variance over parents of the parent mean (valid children only, centred);
                  v_res  = mean over valid children of ||child - parent mean||^2 / D.
  queue_prefill.pt FA-B queue prefill: 128 other stream images (rng 20261114), 32 teacher children each with the queue sampling rule
                  (parents first, diverse child offsets) -> 4,096 real train teacher features, L2-normalised.
PACO val/test duplicates excluded (eval_exclusions); overlap with training images allowed.

  python -m src.finedec.prep --out <shared>
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.config import STREAM
from src.finedec.losses import enqueue_samples
from src.finedec.model import FineAware, n_params
from src.msd.pretrain import eval_exclusions
from src.tsplit.features import frozen_vit, teacher, valid48
from src.tsplit.model import G, G2, rect_index
from src.tsplit.train import STEPS, load_pair


def sha_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def whole_rect(lb):
    y0, y1, x0, x1 = lb.oy // 16, (lb.oy + lb.nh - 1) // 16, lb.ox // 16, (lb.ox + lb.nw - 1) // 16
    return [y0, x0, y1 - y0 + 1, x1 - x0 + 1]


@torch.no_grad()
def teacher_rois(vit, ids, dev):
    """-> list of (Yt (1, N, 768), valid (1, N), idx (1, N)) over whole-content rectangles."""
    out = []
    for i in range(0, len(ids), 32):
        chunk = ids[i:i + 32]
        xs, cont, rects = [], [], []
        for iid in chunk:
            lb, _, t = load_pair(iid, False)
            xs.append(t)
            cont.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
            rects.append(whole_rect(lb))
        T = teacher(vit, torch.stack(xs).to(dev))
        v48 = valid48(torch.tensor(cont, dtype=torch.float32).to(dev))
        for k in range(len(chunk)):
            idx, cy, cx, ok = rect_index(torch.tensor([rects[k]], device=dev))
            out.append((T[k][idx], ok & v48[k][idx], idx, cy, cx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    vit = frozen_vit(dev)
    model = FineAware(vit)
    sd = model.state_dict()
    torch.save(sd, out / "init.pt")
    init_sha = sha_state(sd)
    ex = set(eval_exclusions()["raev2_stream"])
    pool = sorted(set(int(i) for i in np.load(STREAM)[:STEPS].ravel()) - ex)
    ids_c = sorted(int(i) for i in np.random.default_rng(20261113).choice(pool, 512, replace=False))
    rest = sorted(set(pool) - set(ids_c))
    ids_q = sorted(int(i) for i in np.random.default_rng(20261114).choice(rest, 128, replace=False))
    # FA-A constants
    mus, res_sum, res_n = [], 0.0, 0
    for Yt, v, idx, cy, cx in teacher_rois(vit, ids_c, dev):
        Y, vv = Yt[0].double(), v[0]
        par = (cy[0] // 2) * G + cx[0] // 2
        P = G * G
        w = vv.double()[:, None]
        s = torch.zeros(P, 768, device=dev, dtype=torch.float64).index_add_(0, par, Y * w)
        c = torch.zeros(P, device=dev, dtype=torch.float64).index_add_(0, par, vv.double())
        okp = c > 0
        mu = s[okp] / c[okp, None]
        mus.append(mu.cpu())
        mu_c = (s / c.clamp_min(1)[:, None])[par]
        res_sum += float(((Y - mu_c).pow(2).mean(-1) * vv.double()).sum())
        res_n += int(vv.sum())
    mus = torch.cat(mus)
    v_mean = float(mus.var(0, unbiased=False).mean())
    v_res = res_sum / res_n
    const = dict(v_mean=v_mean, v_res=v_res, ratio_mean_over_res=v_mean / v_res, n_images=len(ids_c), n_parents=len(mus), n_children=res_n, image_ids=ids_c,
                 definition="post-norm teacher final patch tokens (native 768 canvas, no flip); parent mean over VALID children; v_mean = mean_d Var_parents(mu_d); v_res = mean ||child - mu||^2 / D over valid children")
    (out / "constants.json").write_text(json.dumps(const, indent=1))
    # FA-B queue prefill
    feats = []
    for k, (Yt, v, idx, cy, cx) in enumerate(teacher_rois(vit, ids_q, dev)):
        feats.append(enqueue_samples(Yt, v, torch.zeros(1, dtype=torch.long, device=dev), idx, per_image=32, seed=20261114 * 1000 + k))
    q = torch.nn.functional.normalize(torch.cat(feats).float(), dim=-1).cpu()
    assert q.shape == (4096, 768), q.shape
    torch.save(dict(features=q, image_ids=ids_q), out / "queue_prefill.pt")
    man = dict(init_sha256=init_sha, trainable_params=n_params(model), all_params=sum(p.numel() for p in model.parameters()),
               queue_prefill_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest(), queue_images=len(ids_q), constants=dict(v_mean=v_mean, v_res=v_res))
    (out / "prep_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1), flush=True)


if __name__ == "__main__":
    main()
