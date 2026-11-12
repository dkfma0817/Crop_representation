"""Fixed inputs shared by RSP-Masked and RSP-Masked-noPix (built once, separate process, no training RNG touched).

  norm_stats.pt / .json : per-dim mean mu and std sigma of DENSE-TEACHER fine tokens (p_T = 4 px, post final norm) over 256 COCO
                          stream images (rng 20261112; PACO duplicates excluded; no flip), 8 random 1..6-parent ROIs each
                          (RSPStreamDS draws, rng (row, 0, 1)), every valid fine token. Loss normalisation norm(x) = (x - mu) / sigma (design doc:
                          'loss = mean | norm(S) - norm(T_crop) |', per-dimension standardisation; same mu / sigma for both runs).
  init_pix.pt / init_nopix.pt : torch.manual_seed(0) -> RSPPredictor(pix); sha256 recorded; every shared module bit-identical.

  python -m src.rsp.prep --out <shared>
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.config import STREAM
from src.msd.pretrain import eval_exclusions
from src.rsp.data import RSPStreamDS, frects_of
from src.rsp.model import RSPPredictor, n_params
from src.rsp.teacher import DenseTeacher
from src.tsplit.features import frozen_vit
from src.tsplit.train import STEPS


def sha_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def build_init(pix):
    torch.manual_seed(0)
    return RSPPredictor(pix=pix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_images", type=int, default=256)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    man = {}
    for pix in (True, False):
        m = build_init(pix)
        sd = m.state_dict()
        torch.save(sd, out / f"init_{'pix' if pix else 'nopix'}.pt")
        man[f"init_{'pix' if pix else 'nopix'}"] = dict(sha256=sha_state(sd), trainable_params=n_params(m))
    a_sd, b_sd = torch.load(out / "init_pix.pt"), torch.load(out / "init_nopix.pt")
    shared = sorted(set(a_sd) & set(b_sd))
    assert set(b_sd) <= set(a_sd) and all(torch.equal(a_sd[k], b_sd[k]) for k in shared)
    man["shared_init_identical"] = dict(n_tensors=len(shared), pix_only=sorted(set(a_sd) - set(b_sd)))
    torch.backends.cuda.matmul.allow_tf32 = True
    vit = frozen_vit(dev)
    teach = DenseTeacher(vit).to(dev)
    ex = set(eval_exclusions()["raev2_stream"])
    pool = sorted(set(int(i) for i in np.load(STREAM)[:STEPS].ravel()) - ex)
    ids = sorted(int(i) for i in np.random.default_rng(20261112).choice(pool, a.n_images, replace=False))
    s1 = torch.zeros(768, dtype=torch.float64, device=dev)
    s2 = torch.zeros(768, dtype=torch.float64, device=dev)
    n = 0
    ds = RSPStreamDS(np.array([[i] for i in ids]), ex, want_ev=False, flip=False)
    for k in range(0, len(ids), 16):
        items = [ds[j] for j in range(k, min(k + 16, len(ids)))]
        xs, content, rects, rimg = [], [], [], []
        for b_i, it in enumerate(items):
            xs.append(it["x"][0])
            content.append(it["content"][0])
            rects.append(it["rects"])
            rimg += [b_i] * len(it["rects"])
        x = torch.stack(xs).to(dev)
        content = torch.stack(content).to(dev)
        rects = torch.cat(rects).to(dev)
        rimg = torch.tensor(rimg, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res, _ = teach.targets(x, frects_of(rects), rimg, content)
        for T, v, _ in res:
            t = T[v].double()
            s1 += t.sum(0)
            s2 += t.pow(2).sum(0)
            n += len(t)
    mu = s1 / n
    sd = (s2 / n - mu.pow(2)).clamp_min(0).sqrt() + 1e-6
    torch.save(dict(mu=mu.float().cpu(), sd=sd.float().cpu(), n_tokens=n, image_ids=ids), out / "norm_stats.pt")
    st = dict(n_images=len(ids), n_tokens=n, mu_norm=float(mu.norm()), sd_min=float(sd.min()), sd_median=float(sd.median()), sd_max=float(sd.max()),
              top5_sd_dims=torch.topk(sd, 5).indices.tolist(), rule="256 stream images rng 20261112, no flip, 8 ROIs = RSPStreamDS rng (row, 0, 1), every valid fine token, post-norm, bf16 teacher")
    (out / "norm_stats.json").write_text(json.dumps(st, indent=1))
    man["norm_stats"] = st
    man["norm_stats_sha256"] = hashlib.sha256(torch.cat([mu.float().cpu(), sd.float().cpu()]).numpy().tobytes()).hexdigest()
    (out / "prep_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1), flush=True)


if __name__ == "__main__":
    main()
