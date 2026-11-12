"""PACO official Test inference at the epochs fixed on official Val, for H0/H1 and the DINOv3 baselines (R2 and Mean probe). No epoch is chosen here."""
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
from src.hier.val_infer import C0, DATA, P384, load_reader


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--cond", required=True, choices=("H0", "H1", "P384", "C0"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev, t0 = torch.device("cuda"), time.time()
    selection = json.loads(Path(a.selection).read_text())
    M = pd.read_parquet(os.path.join(DATA, "manifest_test.parquet"))
    Y0 = np.zeros((len(M), 59), np.int8)
    if a.cond in ("H0", "H1"):
        HB.DETAIL_COND = a.cond
        model, _ = load_hier(Path(a.run) / a.cond / "checkpoints/ema_last.pt", dev)
        probe = Path(a.run) / f"probe_{a.cond}"
        dirs = dict(r2=probe / "r2fused", linear=probe / "r0")
        ds, bs = HB.BBImageDS(M, Y0, False, "bil", 384, 16), 12

        def encode(b):
            return {12: hier_encode(model, b, ["Q"])["Q"]}
    elif a.cond == "P384":
        model, _ = load_hcf(P384 / "P/checkpoints/ema_last.pt", dev)
        dirs = dict(r2=P384 / "probe/r2fused", linear=P384 / "probe/r0")
        ds, bs = IB.BBImageDS(M, Y0, False, "bil", 384, 16), 12

        def encode(b):
            return {12: hcf_encode(model, b, ["Q"])["Q"]}
    else:
        model = load_vit(dev).eval().requires_grad_(False)
        dirs = dict(r2=C0 / "r2", linear=C0 / "r0")
        ds, bs = IB.BBImageDS(M, Y0, False, "c0", 384, 16), 6

        def encode(b):
            x, toks = b["x"], []
            for i in range(0, len(x), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    toks.append(model.forward_features(x[i:i + 64])[:, 5:].float())
            return {12: torch.cat(toks)}
    epochs = {key: selection[f"{a.cond}_{key}"]["epoch"] for key in dirs}
    readers = {key: load_reader("r2" if key == "r2" else "r0", dirs[key] / f"ep{epochs[key]}.pt", dev) for key in dirs}
    L = {key: torch.full((len(M), 59), float("nan")) for key in readers}
    chunks = [list(range(i, min(len(ds), i + bs))) for i in range(0, len(ds), bs)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
    for b in dl:
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        f = encode(b)
        for key, r in readers.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L[key][rows] = r(f, b)["logits"].float().cpu()
    for key, v in L.items():
        assert not torch.isnan(v).any(), key
        out = Path(a.root) / "preds" / f"{a.cond}_{key}" / f"ep{epochs[key]}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "probs_test.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    print(json.dumps(dict(cond=a.cond, epochs=epochs, n_rows=len(M), n_images=int(M.image_id.nunique()), seconds=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
