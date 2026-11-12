"""Conditioning dependency on held-out COCO dev GT ROIs (frozen encoder + head from a recovery checkpoint; no training).

For the same noisy crop x_t and timestep: A = correct M_r, B = zero M_r, C = M_r of a different image with the closest aspect ratio (same tensor shape, the donor's coverage).
The student encoder and head of recovery.pt are used, because the head was trained against the student. Timestep bins cover the training range [0.1, 0.9].
MSE = valid-pixel velocity MSE of the training objective. Differences are descriptive: B and C are outside the training distribution.

  CUDA_VISIBLE_DEVICES=0 python -m src.pdit.diag --condition P1 --recovery <run>/P1/checkpoints/recovery.pt --out <run>/diag/P1.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import COCO_TRAIN, FEAT, STREAM
from src.hcf.distill import fwd
from src.hcf.model import load_hcf
from src.msd.pretrain import eval_exclusions
from src.pdit.crop_objectives import roi_tokens
from src.pdit.train import build_head, conditioning
from src.viz.c0_pca import MEAN_T, STD_T, geometry

BINS = [(.1, .3), (.3, .5), (.5, .7), (.7, .9)]
INITIAL = "output/2026-11-08/hcf_material_causal/initial.pt"


class DevDS(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(os.path.join(COCO_TRAIN, f"{int(r.image_id):012d}.jpg")).convert("RGB")
        bx = (int(r.l), int(r.t), int(r.r), int(r.b))
        crop, full, grid = geometry(*img.size, bx, 384, 16)
        mask = torch.zeros(1, 384, 384)
        mask[:, crop.oy:crop.oy + crop.nh, crop.ox:crop.ox + crop.nw] = 1.
        return dict(full=full.canvas(img), crops=crop.canvas(img.crop(bx)), aux_grid=torch.from_numpy(grid), aux_cov=torch.from_numpy(crop.coverage()), pixel_valid=mask)


def dev_rows():
    ex = eval_exclusions()
    d = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    d = d[~d.image_id.isin(ex["raev2_dev"])]
    d = d[d.kind == "gt"].reset_index(drop=True)
    d = d[(d.r > d.l) & (d.b > d.t)].reset_index(drop=True)
    overlap = len(set(d.image_id.astype(int)) & set(np.load(STREAM)[:12000].reshape(-1).astype(int).tolist()))
    assert overlap == 0, f"{overlap} dev images are in the training stream"
    d["aspect"] = (d.r - d.l) / (d.b - d.t)
    return d.sort_values(["aspect", "image_id", "l", "t"]).reset_index(drop=True)


def donors(ids):
    """Index of the next row in aspect order from a different image (circular)."""
    n, out = len(ids), np.zeros(len(ids), int)
    for i in range(n):
        j = (i + 1) % n
        while ids[j] == ids[i]:
            j = (j + 1) % n
        out[i] = j
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=("baseline", "P1", "P2"))
    ap.add_argument("--recovery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.recovery, map_location="cpu", weights_only=False)
    model, _ = load_hcf(INITIAL, dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    head = build_head(a.condition).to(dev)
    head.load_state_dict(ck["head"], strict=True)
    head.eval()
    rows = dev_rows()
    dl = DataLoader(DevDS(rows), batch_size=a.batch, num_workers=a.workers, shuffle=False, pin_memory=True)
    feats, covs, data = [], [], []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        b["aux_ri"] = torch.arange(len(b["full"]), device=dev)
        o = fwd(model, b["full"])
        feats.append(roi_tokens(model, o, b).cpu())
        covs.append(b["aux_cov"].cpu())
        data.append(dict(crops=b["crops"].cpu(), pixel_valid=b["pixel_valid"].cpu()))
    F_, C_ = torch.cat(feats), torch.cat(covs)
    X_ = torch.cat([d["crops"] for d in data])
    V_ = torch.cat([d["pixel_valid"] for d in data])
    donor = donors(rows.image_id.values.astype(int))
    names = ("A_correct", "B_zero", "C_other_image")
    mse = {n: [] for n in names}
    for k, (lo, hi) in enumerate(BINS):
        gen = torch.Generator(device=dev).manual_seed(300000 + k)
        per = {n: [] for n in names}
        for s in range(0, len(F_), a.batch):
            idx = np.arange(s, min(s + a.batch, len(F_)))
            clean = (X_[idx].to(dev) * STD_T.to(dev) + MEAN_T.to(dev)) * 2 - 1
            valid = V_[idx].to(dev)
            t = lo + (hi - lo) * torch.rand(len(idx), device=dev, generator=gen)
            noise = torch.randn(clean.shape, device=dev, dtype=clean.dtype, generator=gen)
            noisy = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
            target = noise - clean
            cases = {"A_correct": (F_[idx], C_[idx]), "B_zero": (torch.zeros_like(F_[idx]), C_[idx]), "C_other_image": (F_[donor[idx]], C_[donor[idx]])}
            for name, (f, c) in cases.items():
                memory, mcov = conditioning(a.condition, f.to(dev), c.to(dev))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = head(noisy, t, memory, mcov)
                sq = (pred.float() - target).pow(2) * valid
                per[name].append((sq.flatten(1).sum(1) / (3 * valid.flatten(1).sum(1)).clamp_min(1)).cpu())
        for n in names:
            mse[n].append(torch.cat(per[n]).numpy())
    arr = {n: np.stack(v, 1) for n, v in mse.items()}                 # (N, 4 bins)
    images = rows.image_id.values.astype(int)
    uniq, inv = np.unique(images, return_inverse=True)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(uniq), size=(1000, len(uniq)))
    weight = np.stack([np.bincount(b, minlength=len(uniq))[inv] for b in boot]).astype(np.float64)

    def ci(v):
        est = (weight * v).sum(1) / weight.sum(1)
        return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]

    report = dict(condition=a.condition, encoder="student of recovery.pt", n_rois=len(rows), n_images=len(uniq), step=int(ck["step"]), bins=BINS, t_note="t ~ U(bin) per ROI; the same x_t and t for A/B/C", per_bin={}, overall={})
    for k, label in enumerate(f"{lo:.1f}-{hi:.1f}" for lo, hi in BINS):
        rec = {n: float(arr[n][:, k].mean()) for n in names}
        rec.update(B_minus_A=float((arr["B_zero"][:, k] - arr["A_correct"][:, k]).mean()), C_minus_A=float((arr["C_other_image"][:, k] - arr["A_correct"][:, k]).mean()))
        rec.update(B_minus_A_ci=ci(arr["B_zero"][:, k] - arr["A_correct"][:, k]), C_minus_A_ci=ci(arr["C_other_image"][:, k] - arr["A_correct"][:, k]))
        report["per_bin"][label] = rec
    all_ = {n: arr[n].mean(1) for n in names}
    rec = {n: float(all_[n].mean()) for n in names}
    rec.update(B_minus_A=float((all_["B_zero"] - all_["A_correct"]).mean()), C_minus_A=float((all_["C_other_image"] - all_["A_correct"]).mean()), B_minus_A_ci=ci(all_["B_zero"] - all_["A_correct"]), C_minus_A_ci=ci(all_["C_other_image"] - all_["A_correct"]))
    report["overall"] = rec
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
