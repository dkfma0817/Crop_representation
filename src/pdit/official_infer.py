"""PACO official-Val-manifest inference for P1 / P2 and the two baselines without official Mean predictions. Val only (no way to read Test). All 10 reader epochs; GT box + category choose the ROI.

  --cond P1|P2         both readers (r2fused -> <cond>_r2, r0 -> <cond>_linear)
  --cond Initial|Hybrid Mean probe only (their R2 Val comes from output/2026-11-09/hcf_official_val)

Reproduction check: predictions on the 11,109 project-M2 rows must match the saved scores_val_ep<k>.npy of the same reader (Pearson r > 0.999).
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
from src.hcf.model import load_hcf
from src.hcf.probe import encode
from src.ivc.common import EXCL, R0
from src.pdit.bbdata import BBImageDS
from src.readout.models import Reader

DATA = "output/2026-11-07/paco_oracle_attr/data"
CAUSAL = Path("output/2026-11-08/hcf_material_causal")


def load_reader(kind, path, dev):
    r = R0().to(dev) if kind == "r0" else Reader("r2", layers=(12,)).to(dev)
    r.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["reader"])
    return r.eval().requires_grad_(False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", required=True, choices=("P1", "P2", "Initial", "Hybrid"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev, t0, root = torch.device("cuda"), time.time(), Path(a.root)
    probe = Path(a.run) / a.cond / "probe" if a.cond in ("P1", "P2") else CAUSAL / {"Initial": "r384_initial", "Hybrid": "r384_pixeldit_hybrid"}[a.cond] / "probe"
    ckpt = json.loads((probe / "config.json").read_text())["job"]["encoder_ckpt"]
    model, _ = load_hcf(ckpt, dev)
    specs = [("r2", "r2", probe / "r2fused"), ("linear", "r0", probe / "r0")] if a.cond in ("P1", "P2") else [("linear", "r0", probe / "r0")]
    M = pd.read_parquet(os.path.join(DATA, "manifest_val.parquet"))
    ds = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    readers = {(key, e): load_reader(kind, path / f"ep{e}.pt", dev) for key, kind, path in specs for e in range(1, 11)}
    L = {k: torch.full((len(M), 59), float("nan")) for k in readers}
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
    for b in dl:
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        f = {12: encode(model, b, ["Q"])["Q"]}
        for k, r in readers.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L[k][rows] = r(f, b)["logits"].float().cpu()
    for (key, e), v in L.items():
        assert not torch.isnan(v).any(), (key, e)
        out = root / "preds" / f"{a.cond}_{key}" / f"ep{e}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "probs_val.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    df = RC.load_data(EXCL)["val"][0]
    pos = pd.Series(np.arange(len(M)), index=M.ann_id.values)
    idx = pos.loc[df.ann_id.values].values
    check = {}
    for key, _, path in specs:
        for e in (1, 10):
            saved = np.load(path / f"scores_val_ep{e}.npy").astype(np.float64).ravel()
            new = L[(key, e)][idx]
            r = max(float(np.corrcoef(saved, new.numpy().astype(np.float64).ravel())[0, 1]), float(np.corrcoef(saved, torch.sigmoid(new).numpy().astype(np.float64).ravel())[0, 1]))
            check[f"{key}_ep{e}"] = r
            assert r > .999, (key, e, r)
    report = dict(cond=a.cond, n_rows=len(M), n_images=int(M.image_id.nunique()), encoder=ckpt, seconds=round(time.time() - t0), reproduction_pearson_vs_saved_m2_scores=check)
    (root / "verify").mkdir(exist_ok=True)
    (root / "verify" / f"{a.cond}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
