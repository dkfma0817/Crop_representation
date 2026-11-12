"""SigLIP2 frozen feature + independent R2 inference in official PACO manifest order. All ten epochs on VAL; selected epoch only on TEST."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.siglip_pm.model import load_vit
from src.siglip_pm.probe import encode
from src.siglip_extra.probe import load_frozen
from src.siglip_pm.bbdata import BBImageDS
from src.readout.models import Reader

DATA = "output/2026-11-07/paco_oracle_attr/data"
S, P = 384, 16


def load_reader(path, layers, dev):
    torch.manual_seed(0)
    r = Reader("r2", layers=tuple(layers)).to(dev)
    r.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["reader"])
    r.eval()
    for p in r.parameters():
        p.requires_grad_(False)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--job", default="")
    ap.add_argument("--c0", action="store_true")
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--select", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda")
    t0 = time.time()
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    if a.limit:
        M = M[M.image_id.isin(set(M.image_id.drop_duplicates().iloc[: a.limit]))].reset_index(drop=True)
    sel = json.load(open(a.select)) if a.select else None
    if a.c0:
        tag, specs, keys, enc = "C0", [dict(name="r2", layers=[12])], [12], load_vit(dev).eval()
        pdir = f"{a.run}/probe_C0"
        ds = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "c0", S, P)
        bs = 6
    else:
        job = yaml.safe_load(open(a.job))
        tag, specs = job["cond"], job["readers"]
        keys = sorted({l for s in specs for l in s["layers"]}, key=str)
        enc = load_frozen(job, dev)
        pdir = f"{a.run}/{job['name']}"
        ds = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "c0" if tag == "C0" else "bil", S, P)
        bs = 12
    for p in enc.parameters():
        p.requires_grad_(False)
    R = []                                                  # (name, epoch, reader, key-drop flag)
    for s in specs:
        name = f"{tag}_{s['name']}"
        if a.split == "val":
            eps = list(range(1, 11))
        else:
            eps = [sel[name]["epoch"]]
        for ep in eps:
            R.append((name, ep, load_reader(f"{pdir}/{s['name']}/ep{ep}.pt", s["layers"], dev), False))
        if False:                             # Fine memory removed from the SAME trained reader (spec section 8, auxiliary)
            for ep in eps:
                R.append(("P_r2fused_nofine", ep, load_reader(f"{pdir}/{s['name']}/ep{ep}.pt", s["layers"], dev), True))
    chunks = [list(range(i, min(len(ds), i + bs))) for i in range(0, len(ds), bs)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
    L = {(n, e): torch.full((len(M), 59), float("nan")) for n, e, _, _ in R}
    need_drop = any(d for *_, d in R)
    with torch.no_grad():
        for b in dl:
            b = RC.to_dev(b, dev)
            rows = b["rows"].cpu()
            if a.c0:
                x, toks = b["x"], []
                for i in range(0, len(x), 64):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        toks.append(enc.forward_features(x[i:i + 64]).float())
                f = {12: torch.cat(toks)}
                fd = None
            else:
                ii = b["img_index"].clone()
                f = encode(enc, b, keys)
                fd = {**f, "Q": f[12]} if need_drop else None
            for n, e, r, d in R:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    L[(n, e)][rows] = r(fd if d else f, b)["logits"].float().cpu()
    for (n, e), v in L.items():
        assert not torch.isnan(v).any(), (n, e)
        od = f"{a.run}/preds/{n}/ep{e}"
        os.makedirs(od, exist_ok=True)
        np.savez(f"{od}/probs_{a.split}.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values,
                 logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    print(json.dumps(dict(split=a.split, tag=tag, n_rows=len(M), n_images=int(M.image_id.nunique()), preds=len(L), s=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    from src.siglip_extra.resources import root_from_argv, gpu_lease
    with gpu_lease(root_from_argv('--run')):
        main()
