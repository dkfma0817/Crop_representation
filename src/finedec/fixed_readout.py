"""Fixed-readout substitution: the F768 (T768) R2 reader of output/2026-11-11/token_split_x2_s0, frozen at its official-Val epoch (6),
fed with another representation's 24 x 24 ROI read instead of the native-768 teacher read. Same manifest rows, same ROI geometry
(384 crop grid), no reader training. Reported separately from the independently trained R2 results.

  python -m src.finedec.fixed_readout --mode FA --ckpt <model.pt> --name FA-A --split val --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.g1native.bbdata as BD
import src.readout.common as RC
from src.readout.models import Reader
from src.tsplit.features import ContentDS, RoiEncoder

DATA = "output/2026-11-07/paco_oracle_attr/data"
T768 = Path("output/2026-11-11/token_split_x2_s0")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ep = json.loads((T768 / "eval/selection_frozen.json").read_text())["T768_r2"]["epoch"]
    r = Reader("r2", layers=(12,)).to(dev)
    r.load_state_dict(torch.load(T768 / f"probes/T768/r2/ep{ep}.pt", map_location=dev, weights_only=False)["reader"])
    r.eval().requires_grad_(False)
    enc = RoiEncoder(a.mode, dev, ckpt=a.ckpt)
    BD.NATIVE = a.mode == "T768"
    M = pd.read_parquet(f"{DATA}/manifest_{a.split}.parquet")
    ds = ContentDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    L = torch.full((len(M), 59), float("nan"))
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=8, collate_fn=RC.collate_one, pin_memory=True):
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        f = {12: enc(b)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            L[rows] = r(f, b)["logits"].float().cpu()
    assert not torch.isnan(L).any()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"probs_{a.split}.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=L.numpy(), probs=torch.sigmoid(L).numpy())
    (out / f"info_{a.split}.json").write_text(json.dumps(dict(name=a.name, mode=a.mode, ckpt=a.ckpt, reader=str(T768 / f"probes/T768/r2/ep{ep}.pt"), reader_epoch=ep, split=a.split, n_rows=len(M))))
    print("done", a.name, a.split, flush=True)


if __name__ == "__main__":
    main()
