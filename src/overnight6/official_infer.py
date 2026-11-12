"""PACO official-manifest inference for one six-pilot condition = src.g1up.official_infer with the condition name free. R2 reader only; GT box + GT category choose the ROI.

  --split val   all 10 reader epochs, with the reproduction check against the probe's saved Val scores (Pearson r > 0.999)
  --split test  only the epoch frozen in selection/<name>.json
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.g1up.official_infer import DATA, load_reader
from src.hcf.model import load_hcf
from src.hcf.probe import encode as hcf_encode
from src.ivc.common import EXCL
from src.pdit.bbdata import BBImageDS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt", default="", help="encoder checkpoint (default <run>/<cond>/checkpoints/ema_last.pt)")
    ap.add_argument("--probe", default="", help="R2 reader dir with ep*.pt (default <run>/<cond>/probe/r2fused)")
    ap.add_argument("--epochs", default="", help="val only: comma list of reader epochs (default 1..10)")
    ap.add_argument("--soft-check", action="store_true", help="record the reproduction Pearson r without failing (consistency audits of older probes)")
    a = ap.parse_args()
    dev, t0, run = torch.device("cuda"), time.time(), Path(a.run)
    root = run / "eval"
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    name = f"{a.cond}_r2"
    if a.split == "test":
        assert set(M.split.unique()) == {"test"}
        epochs = [json.loads((root / "selection" / f"{name}.json").read_text())["epoch"]]
    else:
        epochs = [int(e) for e in a.epochs.split(",")] if a.epochs else list(range(1, 11))
    model, ck = load_hcf(Path(a.ckpt) if a.ckpt else run / a.cond / "checkpoints/ema_last.pt", dev)
    assert ck["kind"] == "ema" and ck["step"] == 12000, ck["step"]
    probe = Path(a.probe) if a.probe else run / a.cond / "probe/r2fused"
    readers = {e: load_reader(probe / f"ep{e}.pt", dev) for e in epochs}
    ds = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    L = {e: torch.full((len(M), 59), float("nan")) for e in epochs}
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True):
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        f = {12: hcf_encode(model, b, ["Q"])["Q"]}
        for e, r in readers.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L[e][rows] = r(f, b)["logits"].float().cpu()
    for e, v in L.items():
        assert not torch.isnan(v).any(), e
        out = root / "preds" / name / f"ep{e}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"probs_{a.split}.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    report = dict(cond=a.cond, split=a.split, epochs=epochs, ckpt=str(a.ckpt or "default"), probe=str(probe), n_rows=len(M), n_images=int(M.image_id.nunique()), seconds=round(time.time() - t0))
    if a.split == "val":
        df = RC.load_data(EXCL)["val"][0]
        idx = pd.Series(np.arange(len(M)), index=M.ann_id.values).loc[df.ann_id.values].values
        check = {}
        for e in [x for x in (1, 10) if x in epochs]:
            saved = np.load(probe / f"scores_val_ep{e}.npy").astype(np.float64).ravel()
            new = L[e][idx]
            check[f"ep{e}"] = max(float(np.corrcoef(saved, new.numpy().astype(np.float64).ravel())[0, 1]), float(np.corrcoef(saved, torch.sigmoid(new).numpy().astype(np.float64).ravel())[0, 1]))
            assert a.soft_check or check[f"ep{e}"] > .999, check
        report["reproduction_pearson"] = check
    (root / "verify").mkdir(exist_ok=True)
    (root / "verify" / f"{a.cond}_{a.split}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
