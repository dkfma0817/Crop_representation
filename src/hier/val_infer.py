"""PACO official-Val-manifest inference for the H0/H1 pilot (no way to read Test): all 10 reader epochs, GT box + category choose the ROI.

  --cond H0|H1 : both readers (r2fused -> <name>_r2, r0 -> <name>_linear) on the frozen H encoder
  --cond P384  : the P384 Mean probe (r384_cosine/probe/r0); its R2 Val comes from output/2026-11-09/hcf_official_val
  --cond C0    : the DINOv3 C0 Mean probe (ivc_v3/probe_c0/r0); its R2 Val comes from output/2026-11-08/hcf_pilot
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

import src.hier.bbdata as HB
import src.ivc.bbdata as IB
import src.readout.common as RC
from src.hcf.model import load_hcf, load_vit
from src.hcf.probe import encode as hcf_encode
from src.hier.model import load_hier
from src.hier.probe import encode as hier_encode
from src.ivc.common import R0
from src.readout.models import Reader

DATA = "output/2026-11-07/paco_oracle_attr/data"
P384 = Path("output/2026-11-08/hcf_material_causal/r384_cosine")
C0 = Path("output/2026-11-06/ivc_v3/probe_c0")


def load_reader(kind, path, dev):
    r = R0().to(dev) if kind == "r0" else Reader("r2", layers=(12,)).to(dev)
    r.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["reader"])
    return r.eval().requires_grad_(False)


def save(root, name, epoch, M, logits):
    out = Path(root) / "preds" / name / f"ep{epoch}"
    out.mkdir(parents=True, exist_ok=True)
    assert not torch.isnan(logits).any(), (name, epoch)
    np.savez(out / "probs_val.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=logits.numpy().astype(np.float32), probs=torch.sigmoid(logits).numpy().astype(np.float32))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", required=True, choices=("H0", "H1", "P384", "C0"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev, t0 = torch.device("cuda"), time.time()
    M = pd.read_parquet(os.path.join(DATA, "manifest_val.parquet"))
    Y0 = np.zeros((len(M), 59), np.int8)
    if a.cond in ("H0", "H1"):
        HB.DETAIL_COND = a.cond
        model, _ = load_hier(Path(a.run) / a.cond / "checkpoints/ema_last.pt", dev)
        probe = Path(a.run) / f"probe_{a.cond}"
        specs = [("r2", "r2", probe / "r2fused"), ("linear", "r0", probe / "r0")]
        ds, bs = HB.BBImageDS(M, Y0, False, "bil", 384, 16), 12

        def encode(b):
            return {12: hier_encode(model, b, ["Q"])["Q"]}
    elif a.cond == "P384":
        model, _ = load_hcf(P384 / "P/checkpoints/ema_last.pt", dev)
        specs = [("linear", "r0", P384 / "probe/r0")]
        ds, bs = IB.BBImageDS(M, Y0, False, "bil", 384, 16), 12

        def encode(b):
            return {12: hcf_encode(model, b, ["Q"])["Q"]}
    else:
        model = load_vit(dev).eval().requires_grad_(False)
        specs = [("linear", "r0", C0 / "r0")]
        ds, bs = IB.BBImageDS(M, Y0, False, "c0", 384, 16), 6

        def encode(b):
            x, toks = b["x"], []
            for i in range(0, len(x), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    toks.append(model.forward_features(x[i:i + 64])[:, 5:].float())
            return {12: torch.cat(toks)}
    readers = {(key, e): load_reader(kind, path / f"ep{e}.pt", dev) for key, kind, path in specs for e in range(1, 11)}
    L = {k: torch.full((len(M), 59), float("nan")) for k in readers}
    chunks = [list(range(i, min(len(ds), i + bs))) for i in range(0, len(ds), bs)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
    for b in dl:
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        f = encode(b)
        for k, r in readers.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L[k][rows] = r(f, b)["logits"].float().cpu()
    for (key, e), v in L.items():
        save(a.root, f"{a.cond}_{key}", e, M, v)
    print(json.dumps(dict(cond=a.cond, n_rows=len(M), n_images=int(M.image_id.nunique()), seconds=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
