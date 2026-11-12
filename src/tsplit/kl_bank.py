"""Fixed teacher patch bank for S3-Distill (built once, in its own process, with its own RNG -> no effect on the training RNG).

Images : unique COCO train images of the S3 stream rows 0..11,999 minus the PACO val/test duplicates (src.msd.pretrain.eval_exclusions);
         1,024 images drawn with numpy default_rng(BANK_SEED) without replacement; 8 teacher cells per image (no replacement) among the
         VALID cells of that image (the existing rule: 8-px child cell overlaps the letterbox content, cell_w(content, 48, 8) > 0).
Teacher: frozen timm vit_base_patch16_dinov3.lvd1689m, native 768 canvas (src.g1native.native.native_canvas, no flip), final-norm
         patch tokens (48 x 48 x 768), bf16 autocast -> fp32; then L2-normalised (fp32). Bank = 8,192 x 768.
Writes <out>/bank.pt {bank, raw_norm, image_id, cy, cx} and <out>/bank_manifest.json (incl. sha256 of the bank tensor bytes).

  python -m src.tsplit.kl_bank --out <run>/bank
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import STREAM
from src.msd.pretrain import eval_exclusions
from src.tsplit.features import frozen_vit, teacher, valid48
from src.tsplit.train import STEPS, load_pair

BANK_SEED, N_IMG, PER_IMG = 20261111, 1024, 8


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ex = set(eval_exclusions()["raev2_stream"])
    pool = sorted(set(int(i) for i in np.load(STREAM)[:STEPS].ravel()) - ex)
    rng = np.random.default_rng(BANK_SEED)
    ids = sorted(int(i) for i in rng.choice(pool, N_IMG, replace=False))
    vit = frozen_vit(dev)
    feats, norms, iid_l, cy_l, cx_l = [], [], [], [], []
    for i in range(0, len(ids), 32):
        chunk = ids[i:i + 32]
        xs, cont = [], []
        for iid in chunk:
            lb, _, t = load_pair(iid, False)
            xs.append(t)
            cont.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
        T = teacher(vit, torch.stack(xs).to(dev))
        v = valid48(torch.tensor(cont, dtype=torch.float32)).numpy()
        for k, iid in enumerate(chunk):
            cells = np.flatnonzero(v[k])
            pick = np.sort(rng.choice(cells, PER_IMG, replace=False))
            f = T[k, torch.as_tensor(pick, device=dev)]
            feats.append(F.normalize(f, dim=-1).cpu())
            norms.append(f.norm(dim=-1).cpu())
            iid_l += [iid] * PER_IMG
            cy_l += (pick // 48).tolist()
            cx_l += (pick % 48).tolist()
    bank = torch.cat(feats).float().contiguous()
    raw = torch.cat(norms)
    assert bank.shape == (N_IMG * PER_IMG, 768) and torch.isfinite(bank).all()
    sha = hashlib.sha256(bank.numpy().tobytes()).hexdigest()
    torch.save(dict(bank=bank, raw_norm=raw, image_id=torch.tensor(iid_l), cy=torch.tensor(cy_l), cx=torch.tensor(cx_l)), out / "bank.pt")
    sim = bank @ bank.T
    off = sim[~torch.eye(len(bank), dtype=torch.bool)]
    manifest = dict(
        size=len(bank), dim=768, sha256_bank_fp32=sha, bank_seed=BANK_SEED, n_images=N_IMG, per_image=PER_IMG,
        teacher="timm vit_base_patch16_dinov3.lvd1689m (pretrained, frozen, eval), forward_features final-norm patch tokens, 5 prefix tokens dropped, bf16 autocast -> fp32",
        preprocessing="original COCO image resized (PIL bicubic) to 2 x (nw, nh) of the 384 letterbox and pasted at 2 x (ox, oy) on a 768 canvas, ImageNet mean/std, no flip (src.g1native.native.native_canvas)",
        normalization="L2 per row (fp32) after the teacher final LayerNorm", valid_rule="48 x 48 cell (8 px on the 384 canvas) overlaps the letterbox content (cell_w > 0), = the S3 training valid-child rule",
        image_pool="unique image ids of the COCO stream rows 0..11,999 minus eval_exclusions()['raev2_stream'] (PACO val/test duplicates by id/md5/pixel md5/dHash); overlap with S3 training images allowed (user spec)",
        excluded=sorted(ex), image_ids=ids, cells=[[i, y, x] for i, y, x in zip(iid_l, cy_l, cx_l)],
        stats=dict(raw_norm_mean=float(raw.mean()), raw_norm_std=float(raw.std()), offdiag_cos_mean=float(off.mean()), offdiag_cos_p99=float(off.quantile(.99)) if off.numel() < 16_000_000 else float(off[torch.randperm(off.numel(), generator=torch.Generator().manual_seed(0))[:1_000_000]].quantile(.99)),
                   max_offdiag_cos=float(off.max()), n_pairs_cos_gt_0p99=int((off > .99).sum()) // 2))
    (out / "bank_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("image_ids", "cells", "excluded")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
