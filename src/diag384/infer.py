"""PACO official-manifest inference for the two new diagnosis readers (R2, GT box + GT category choose the ROI).

  C1_r2     Frozen original DINOv3 on the Crop-from-384 canvas
  A1_fine   A1 encoder, Fine contribution only

  --split val   all 10 reader epochs + reproduction check against the probe's saved Val scores (Pearson r > 0.999)
  --split test  only the epoch frozen in <root>/selection_frozen.json
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
from src.diag384.bbdata import BBImageDS
from src.diag384.model import load_probe_encoder
from src.hcf.model import load_vit
from src.hcf.probe import encode as hcf_encode
from src.ivc.common import EXCL
from src.readout.models import Reader

DATA = "output/2026-11-07/paco_oracle_attr/data"
A1 = "output/2026-11-09/coarse_fine_ablation_s0/A1/checkpoints/ema_last.pt"


def load_reader(path, dev):
    r = Reader("r2", layers=(12,)).to(dev)
    r.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["reader"])
    return r.eval().requires_grad_(False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--probe", required=True, help="probe dir holding <reader>/ep<k>.pt")
    ap.add_argument("--name", required=True, choices=("C1_r2", "A1_fine"))
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
    probe = Path(a.probe)
    readers = {e: load_reader(probe / f"ep{e}.pt", dev) for e in epochs}
    if a.name == "C1_r2":
        vit = load_vit(dev).eval().requires_grad_(False)
        ds, chunk = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "c1", 384, 16), 6

        def enc(b):
            toks = []
            for i in range(0, len(b["x"]), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    toks.append(vit.forward_features(b["x"][i:i + 64])[:, vit.num_prefix_tokens:].float())
            return torch.cat(toks)
    else:
        model, ck = load_probe_encoder(A1, dev, fine_only=True)
        assert ck["kind"] == "ema" and ck["step"] == 12000, ck["step"]
        ds, chunk = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16), 12
        enc = lambda b: hcf_encode(model, b, ["Q"])["Q"]  # noqa: E731
    L = {e: torch.full((len(M), 59), float("nan")) for e in epochs}
    chunks = [list(range(i, min(len(ds), i + chunk))) for i in range(0, len(ds), chunk)]
    for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True):
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        f = {12: enc(b)}
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
