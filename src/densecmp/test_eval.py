"""ONE-TIME PACO test scoring of a finished probe (no training, no selection on test).

Epoch = max PACO-dev M0 Overall over epochs 1..10 (curves_dev.jsonl; IVC/MSD rule). Same encoder / dataset / reader code as
training, no flip, encoder fp16 autocast, reader bf16 autocast. First re-scores VAL and compares with the saved
scores_val_ep<k>.npy (max |diff| recorded, must be < 0.05 logits), then scores ALL test ROIs once -> scores_test.npy.

  CUDA_VISIBLE_DEVICES=2 python -m src.densecmp.test_eval --probe <probe dir>
"""
import argparse
import json
import os

os.environ["READOUT_ALLOW_TEST"] = "1"          # this script only

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.densecmp.data import EvaPacoDS
from src.densecmp.encoders import PATCH, Encoder
from src.densecmp.train import EvaR2
from src.ivc.common import EXCL


def select_epoch(probe):
    dv = [json.loads(l) for l in open(os.path.join(probe, "r2", "curves_dev.jsonl"))]
    dv = [x for x in dv if x["epoch"] >= 1]
    best = max(dv, key=lambda x: x["dev"]["M0"]["overall"])
    return best["epoch"], best["dev"]["M0"]["overall"]


@torch.no_grad()
def score(df, Y, enc, reader, mode, side, dev, workers):
    ds = EvaPacoDS(df, Y, flip=False, mode=mode, side=side, patch=PATCH)
    bs = 8 if mode == "full" else 6
    chunks = [list(range(i, min(len(ds), i + bs))) for i in range(0, len(ds), bs)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=workers, collate_fn=RC.collate_one, pin_memory=True)
    S = np.zeros((len(df), 59), np.float32)
    seen = np.zeros(len(df), bool)
    chunk = 8 if (mode == "full" and side >= 768) else 64
    for b in dl:
        b = RC.to_dev(b, dev)
        x = b["x"]
        f = torch.cat([enc(x[i:i + chunk]) for i in range(0, len(x), chunk)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = reader({12: f}, b)
        rows = b["rows"].cpu().numpy()
        S[rows] = o["logits"].float().cpu().numpy()
        seen[rows] = True
    assert seen.all()
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.probe, "config.json")))
    assert os.path.isfile(os.path.join(a.probe, "done.json")), "probe not finished"
    enc_name, mode, side = cfg["args"]["enc"], cfg["args"]["mode"], cfg["args"]["side"]
    ep, dev_m0 = select_epoch(a.probe)
    dev = torch.device("cuda")
    enc = Encoder(enc_name).to(dev)
    reader = EvaR2().to(dev)
    ck = torch.load(os.path.join(a.probe, "r2", f"ep{ep}.pt"), map_location="cpu")
    assert ck["epoch"] == ep
    reader.load_state_dict(ck["reader"])
    reader.eval()
    data = RC.load_data(EXCL, splits=("train", "val", "test"))
    assert not (set(data["train"][0].image_id) & set(data["test"][0].image_id)), "train / test image overlap"
    assert not (set(data["val"][0].image_id) & set(data["test"][0].image_id)), "val / test image overlap"
    dva, Yva = data["val"]
    Sv = score(dva, Yva, enc, reader, mode, side, dev, a.workers)
    ref = np.load(os.path.join(a.probe, "r2", f"scores_val_ep{ep}.npy"))
    chk = dict(epoch=ep, dev_M0=dev_m0, max_abs_diff=float(np.abs(Sv - ref).max()), mean_abs_diff=float(np.abs(Sv - ref).mean()))
    print(json.dumps(chk), flush=True)
    json.dump(chk, open(os.path.join(a.probe, "val_recheck.json"), "w"), indent=1)
    assert chk["max_abs_diff"] < 0.05, "val re-score does not reproduce the saved scores"
    dte, Yte = data["test"]
    St = score(dte, Yte, enc, reader, mode, side, dev, a.workers)
    np.save(os.path.join(a.probe, "scores_test.npy"), St)
    json.dump(dict(encoder=enc_name, mode=mode, side=side, epoch=ep, dev_M0=dev_m0, n_test_rows=len(dte),
                   n_test_images=int(dte.image_id.nunique())), open(os.path.join(a.probe, "test_done.json"), "w"), indent=1)
    print("TEST SCORED", enc_name, mode, side, "epoch", ep, flush=True)


if __name__ == "__main__":
    main()
