"""PACO official-manifest inference for one token-splitting condition (= src.g1native.official_infer with the RoiEncoder read).
R2 reader only; GT box + GT category choose the ROI.

  --split val   all 10 reader epochs + reproduction check against the probe's saved Val scores (Pearson r > 0.999)
  --split test  only the epoch fixed in <root>/selection_frozen.json
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import src.g1native.bbdata as BD
import src.readout.common as RC
from src.ivc.common import EXCL
from src.readout.models import Reader
from src.tsplit.features import ContentDS, RoiEncoder

DATA = "output/2026-11-07/paco_oracle_attr/data"


def load_reader(path, dev):
    r = Reader("r2", layers=(12,)).to(dev)
    r.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["reader"])
    return r.eval().requires_grad_(False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="eval root (preds/, verify/, selection_frozen.json)")
    ap.add_argument("--probe", required=True, help="probe dir of this condition (job.yaml + r2/ep*.pt)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev, t0, root, probe = torch.device("cuda"), time.time(), Path(a.root), Path(a.probe)
    job = yaml.safe_load((probe / "job.yaml").read_text())
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    name = f"{a.name}_r2"
    if a.split == "test":
        assert set(M.split.unique()) == {"test"}
        epochs = [json.loads((root / "selection_frozen.json").read_text())[name]["epoch"]]
    else:
        epochs = list(range(1, 11))
    enc = RoiEncoder(job["mode"], dev, ckpt=job.get("split_ckpt"), layer=job.get("layer"))
    readers = {e: load_reader(probe / "r2" / f"ep{e}.pt", dev) for e in epochs}
    BD.NATIVE = job["mode"] in ("T768", "T768M")
    ds = ContentDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    L = {e: torch.full((len(M), 59), float("nan")) for e in epochs}
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True):
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        f = {12: enc(b)}
        for e, r in readers.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L[e][rows] = r(f, b)["logits"].float().cpu()
    for e, v in L.items():
        assert not torch.isnan(v).any(), e
        out = root / "preds" / name / f"ep{e}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"probs_{a.split}.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values,
                 logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    report = dict(cond=a.name, mode=job["mode"], split=a.split, epochs=epochs, n_rows=len(M), n_images=int(M.image_id.nunique()), seconds=round(time.time() - t0))
    if a.split == "val":
        df = RC.load_data(EXCL)["val"][0]
        idx = pd.Series(np.arange(len(M)), index=M.ann_id.values).loc[df.ann_id.values].values
        check = {}
        for e in (1, 10):
            saved = np.load(probe / "r2" / f"scores_val_ep{e}.npy").astype(np.float64).ravel()
            new = L[e][idx]
            check[f"ep{e}"] = max(float(np.corrcoef(saved, new.numpy().astype(np.float64).ravel())[0, 1]),
                                  float(np.corrcoef(saved, torch.sigmoid(new).numpy().astype(np.float64).ravel())[0, 1]))
            assert check[f"ep{e}"] > .999, check
        report["reproduction_pearson"] = check
    (root / "verify").mkdir(parents=True, exist_ok=True)
    (root / "verify" / f"{a.name}_{a.split}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
