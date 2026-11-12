"""PACO official-manifest inference for the interaction ablation. R2 reader; GT box + GT category choose the ROI.

  names: A1_r2, A2_r2 (Coarse + Fine fused, Q) and A0_coarse, A1_coarse, A2_coarse (Coarse only). A0_r2 is the existing r384_cosine result.
  --split val   all 10 reader epochs + reproduction check against the probe's saved Val scores (Pearson r > 0.999)
  --split test  only the epoch frozen in selection_frozen.json
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
from src.arch_ablate.bbdata import BBImageDS
from src.arch_ablate.model import load_ablation
from src.hcf.probe import encode as hcf_encode
from src.ivc.common import EXCL
from src.readout.models import Reader

DATA = "output/2026-11-07/paco_oracle_attr/data"
A0_CKPT = "output/2026-11-08/hcf_material_causal/r384_cosine/P/checkpoints/ema_last.pt"
NAMES = ("A1_r2", "A2_r2", "A0_coarse", "A1_coarse", "A2_coarse")


def probe_dir(run, name):
    arch, kind = name.split("_")
    return Path(run) / arch / "probe" / "r2fused" if kind == "r2" else Path(run) / arch / "probe_coarse" / "r2coarse"


def ckpt(run, name):
    arch = name.split("_")[0]
    return A0_CKPT if arch == "A0" else Path(run) / arch / "checkpoints/ema_last.pt"


def load_reader(path, dev):
    r = Reader("r2", layers=(12,)).to(dev)
    r.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["reader"])
    return r.eval().requires_grad_(False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True, choices=NAMES)
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev, t0, root = torch.device("cuda"), time.time(), Path(a.root)
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    if a.split == "test":
        assert set(M.split.unique()) == {"test"}
        epochs = [json.loads((root / "selection_frozen.json").read_text())[a.name]["epoch"]]
    else:
        epochs = list(range(1, 11))
    model, ck = load_ablation(ckpt(a.run, a.name), dev)
    assert ck["kind"] == "ema" and ck["step"] == 12000, ck["step"]
    assert ck.get("interaction", "A0") == a.name.split("_")[0]
    key = "Q" if a.name.endswith("_r2") else 12
    probe = probe_dir(a.run, a.name)
    readers = {e: load_reader(probe / f"ep{e}.pt", dev) for e in epochs}
    ds = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    L = {e: torch.full((len(M), 59), float("nan")) for e in epochs}
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True):
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        f = {12: hcf_encode(model, b, [key])[key]}
        for e, r in readers.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L[e][rows] = r(f, b)["logits"].float().cpu()
    for e, v in L.items():
        assert not torch.isnan(v).any(), e
        out = root / "preds" / a.name / f"ep{e}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"probs_{a.split}.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    report = dict(name=a.name, split=a.split, epochs=epochs, n_rows=len(M), n_images=int(M.image_id.nunique()), seconds=round(time.time() - t0))
    if a.split == "val":
        df = RC.load_data(EXCL)["val"][0]
        idx = pd.Series(np.arange(len(M)), index=M.ann_id.values).loc[df.ann_id.values].values
        check = {}
        for e in (1, 10):
            saved = np.load(probe / f"scores_val_ep{e}.npy").astype(np.float64).ravel()
            new = L[e][idx]
            check[f"ep{e}"] = max(float(np.corrcoef(saved, new.numpy().astype(np.float64).ravel())[0, 1]), float(np.corrcoef(saved, torch.sigmoid(new).numpy().astype(np.float64).ravel())[0, 1]))
            assert check[f"ep{e}"] > .999, check
        report["reproduction_pearson"] = check
    (root / "verify").mkdir(exist_ok=True)
    (root / "verify" / f"{a.name}_{a.split}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
