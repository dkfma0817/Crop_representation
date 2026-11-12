"""Build the crop-teacher logits once (shared by all jobs): frozen original DINOv3 on each GT crop + the existing C0 R2 reader, for both image flips. Training never runs the crop encoder.
The reader is the C0 probe (ivc_v3/probe_c0) at its dev-selected epoch (max PACO-dev M0), so the teacher uses no Val information. It was trained on the same PACO train labels: train-fit M2 78.9 vs Val 47.9, so
its logits on train are partly memorized labels. That is why a supervised-only control exists.
  CUDA_VISIBLE_DEVICES=0 python -m src.bgjobs.teacher --tag crop_aware_pretrain_bg
"""
import argparse
import fcntl
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.bgjobs import data as DT
from src.crop_aware_pretrain import ckpt as CK
from src.crop_aware_pretrain import config as CF
from src.ivc.compare_bb import pick
from src.ivc.ft import run_enc
from src.readout.models import Reader
from src.viz.c0_pca import BACKBONES, load_encoder

BASE = os.path.join(CF.ROOT, "configs", "bgjobs", "base.yaml")


@torch.no_grad()
def build(cfg, dev, workers=16):
    root, paths, done = DT.cache_paths(cfg)
    os.makedirs(root, exist_ok=True)
    lock = open(os.path.join(root, "build.lock"), "w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)         # a second job waits here, then finds the cache done
    if os.path.exists(done):
        return
    df, _ = DT.load_rows()
    S, P = BACKBONES[cfg["backbone"]]["S"], BACKBONES[cfg["backbone"]]["P"]
    probe = os.path.join(cfg["data"]["teacher_probe"], "r2")
    ep, dm0, _ = pick(probe)
    reader = Reader("r2", layers=(12,)).to(dev)
    reader.load_state_dict(torch.load(os.path.join(probe, f"ep{ep}.pt"), map_location=dev, weights_only=False)["reader"])
    reader.eval().requires_grad_(False)
    vit = load_encoder(cfg["backbone"], dev)
    t0 = time.time()
    for flip, path in enumerate(paths):
        if os.path.exists(path):
            continue
        ds = DT.TeacherDS(df, bool(flip), S, P)
        L = np.full((len(df), 59), np.nan, np.float32)
        chunks = [list(range(i, min(len(ds), i + 16))) for i in range(0, len(ds), 16)]
        for k, b in enumerate(DataLoader(ds, sampler=chunks, batch_size=1, num_workers=workers, collate_fn=RC.collate_one, pin_memory=True)):
            b = RC.to_dev(b, dev)
            tok, _ = run_enc(vit, b["x"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L[b["rows"].cpu().numpy()] = reader({12: tok}, b)["logits"].float().cpu().numpy()
            if k % 200 == 0:
                print(f"flip {flip} chunk {k}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
        assert np.isfinite(L).all()
        tmp = path + f".tmp.{os.getpid()}.npy"
        np.save(tmp, L)
        os.replace(tmp, path)
    CK.atomic_json(dict(probe=probe, epoch=int(ep), dev_M0=float(dm0), n_rows=len(df), time=time.strftime("%F %T")), done)
    print(json.dumps(dict(epoch=int(ep), dev_M0=float(dm0), n_rows=len(df), seconds=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = CF.load(BASE)
    build(cfg, torch.device("cuda"))
