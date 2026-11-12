"""PACO official-manifest inference for G0 (Full384, Up768) and G1 (G1U, G1R). R2 reader only; GT box + GT category choose the ROI.

  --split val   all 10 reader epochs, with a reproduction check against the probe's saved Val scores (Pearson r > 0.999)
  --split test  only the epoch fixed in selection.json (needs that file; G1 only)
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.hcf.model import load_hcf, load_vit
from src.hcf.probe import encode as hcf_encode
from src.ivc.common import EXCL
from src.pdit.bbdata import BBImageDS
from src.readout.models import Reader

DATA = "output/2026-11-07/paco_oracle_attr/data"
FROZEN = ("Full384", "Up768")


def probe_dir(run, cond):
    return Path(run) / "g0" / f"probe_{cond}" / "r2" if cond in FROZEN else Path(run) / cond / "probe" / "r2fused"


def load_reader(path, dev):
    r = Reader("r2", layers=(12,)).to(dev)
    r.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["reader"])
    return r.eval().requires_grad_(False)


@torch.no_grad()
def frozen_encode(vit, b, up):
    x, toks = b["x"], []
    if up:
        x = F.interpolate(x, size=(768, 768), mode="bilinear", align_corners=False)
    for i in range(0, len(x), 24 if up else 64):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = vit.forward_features(x[i:i + (24 if up else 64)])
        toks.append(t[:, vit.num_prefix_tokens:].float())
    tok, g = torch.cat(toks), 48 if up else 24
    maps = tok.transpose(1, 2).reshape(tok.shape[0], -1, g, g)[b["img_index"]]
    out = F.grid_sample(maps, b["bil_grid"].view(-1, 24, 24, 2), mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)
    b["img_index"] = torch.arange(len(out), device=out.device)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", required=True, choices=FROZEN + ("G1U", "G1R"))
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev, t0, root = torch.device("cuda"), time.time(), Path(a.root)
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    name = f"{a.cond}_r2"
    if a.split == "test":
        assert a.cond not in FROZEN, "G0 is Val only"
        assert set(M.split.unique()) == {"test"}
        epochs = [json.loads((root / "selection.json").read_text())[name]["epoch"]]
    else:
        epochs = list(range(1, 11))
    if a.cond in FROZEN:
        vit = load_vit(dev).eval().requires_grad_(False)
        enc = lambda b: frozen_encode(vit, b, a.cond == "Up768")  # noqa: E731
    else:
        model, ck = load_hcf(Path(a.run) / a.cond / "checkpoints/ema_last.pt", dev)
        assert ck["kind"] == "ema" and ck["step"] == 12000, ck["step"]
        enc = lambda b: hcf_encode(model, b, ["Q"])["Q"]  # noqa: E731
    probe = probe_dir(a.run, a.cond)
    readers = {e: load_reader(probe / f"ep{e}.pt", dev) for e in epochs}
    ds = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
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
        np.savez(out / f"probs_{a.split}.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    report = dict(cond=a.cond, split=a.split, epochs=epochs, n_rows=len(M), n_images=int(M.image_id.nunique()), seconds=round(time.time() - t0))
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
    (root / "verify" / f"{a.cond}_{a.split}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
