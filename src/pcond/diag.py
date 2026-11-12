"""Conditioning dependency of a trained PixelDiT head (frozen; no training). PACO Val images, GT ROIs, one ROI per image (object/part alternated), four timestep bins.

Same x_t and timestep for A correct, B zero, C different-image conditioning (memory and coverage of the next image's ROI in the batch, matching shape and normalisation). Uses the student encoder with its head (the trained pair) from recovery.pt.
Removing or swapping the conditioning is an out-of-distribution input: a loss gap measures how much the head reads the conditioning, not attribute discrimination.
  python -m src.pcond.diag --condition B|P1|P2 --ckpt <recovery.pt> --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.hcf.distill import fwd
from src.hcf.model import load_vit
from src.hcf.model import CoarseFine
from src.pcond.heads import build_head, memory_for
from src.siglip_pixel.crop_objectives import roi_tokens
from src.viz.c0_pca import MEAN_T, STD_T, LB, geometry

BINS = ((.1, .3), (.3, .5), (.5, .7), (.7, .9))
MANIFEST = "output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet"
MODES = ("correct", "zero", "different")


class RoiDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path).convert("RGB")
        box = (int(r.l), int(r.t), int(r.r), int(r.b))
        crop, full, grid = geometry(*img.size, box, 384, 16)
        mask = torch.zeros(1, 384, 384)
        mask[:, crop.oy:crop.oy + crop.nh, crop.ox:crop.ox + crop.nw] = 1.
        return dict(full=full.canvas(img), crops=crop.canvas(img.crop(box)), aux_cov=torch.from_numpy(crop.coverage()), aux_grid=torch.from_numpy(grid), pixel_valid=mask, level=int(r.level == "part"), row=i)


def rois(n):
    """One ROI per image, the first half of images taking an object ROI and the rest a part ROI, fixed by seed 0."""
    M = pd.read_parquet(MANIFEST)
    rng = np.random.default_rng(0)
    ids = M.image_id.unique().copy()
    rng.shuffle(ids)
    ids = ids[:n]
    out = []
    for k, iid in enumerate(ids):
        g = M[M.image_id == iid]
        level = "object" if k % 2 == 0 else "part"
        pick = g[g.level == level]
        if len(pick) == 0:
            pick = g
        out.append(pick.iloc[int(rng.integers(len(pick)))])
    return pd.DataFrame(out).reset_index(drop=True)


def load(condition, ckpt, dev):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = CoarseFine(load_vit(dev), s0=.1).to(dev)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    head = build_head(condition).to(dev).eval()
    head.load_state_dict(state["head"])
    head.requires_grad_(False)
    return model, head, int(state["step"])


@torch.no_grad()
def run(condition, model, head, dl, dev):
    rows = []
    for n, b in enumerate(dl):
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        b["aux_ri"] = torch.arange(len(b["full"]), device=dev)
        o = fwd(model, b["full"])
        features = roi_tokens(model, o, b)
        memory, mcov = memory_for(condition, features, b["aux_cov"])
        clean = (b["crops"] * STD_T.to(dev) + MEAN_T.to(dev)) * 2 - 1
        for k, (lo, hi) in enumerate(BINS):
            gen = torch.Generator(device=dev).manual_seed(300000 + n * 10 + k)
            t = lo + (hi - lo) * torch.rand(len(clean), device=dev, generator=gen)
            noise = torch.randn(clean.shape, device=dev, dtype=clean.dtype, generator=gen)
            noisy = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
            row = dict(bin=np.full(len(clean), k), level=b["level"].cpu().numpy(), roi=b["row"].cpu().numpy(), t=t.cpu().numpy())
            for mode, (mem, cov) in dict(correct=(memory, mcov), zero=(torch.zeros_like(memory), mcov), different=(memory.roll(1, 0), mcov.roll(1, 0))).items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = head(noisy, t, mem, cov)
                squared = (pred.float() - (noise - clean)).pow(2) * b["pixel_valid"]
                row[mode] = ((squared.flatten(1).sum(1) / (3 * b["pixel_valid"].flatten(1).sum(1)).clamp_min(1)).cpu().numpy())
            rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def summarize(df):
    def block(g):
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(g), (1000, len(g)))
        out = dict(n=len(g))
        for m in MODES:
            out[m] = float(g[m].mean())
        for m in ("zero", "different"):
            d = (g[m] - g["correct"]).to_numpy()
            s = d[idx].mean(1)
            out[f"{m}_minus_correct"] = dict(point=float(d.mean()), lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)))
        return out

    out = dict(all=block(df))
    for k, (lo, hi) in enumerate(BINS):
        out[f"t[{lo:.1f},{hi:.1f})"] = block(df[df.bin == k])
    for name, value in (("object", 0), ("part", 1)):
        out[name] = block(df[df.level == value])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=("B", "P1", "P2"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model, head, step = load(a.condition, a.ckpt, dev)
    df = rois(a.n)
    dl = DataLoader(RoiDS(df), batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True, drop_last=True)
    result = run(a.condition, model, head, dl, dev)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out.with_suffix(".parquet"))
    report = dict(condition=a.condition, ckpt=str(a.ckpt), step=step, n_rois=int(result.roi.nunique()), protocol="PACO Val GT ROIs (one per image); student encoder + head; same x_t and t per mode", note="Out-of-distribution interventions: loss gaps measure use of the conditioning, not attribute discrimination.", **summarize(result))
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
