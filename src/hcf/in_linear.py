"""ImageNet-1K linear probe on cached frozen CLS features (768 -> 1000), one independent head per encoder, plus the
Original-head transfer diagnostic. Protocol fixed BEFORE any result and applied identically to every encoder:

  features   final-LayerNorm CLS (fp16 cache -> fp32), no normalisation / projection / adapter
  head       nn.Linear(768, 1000), init torch.manual_seed(0) (same initial head for every encoder)
  loss       cross-entropy (no label smoothing)
  optimiser  AdamW lr 1e-3, betas (0.9, 0.999), weight decay 1e-4 on the weight (not the bias)
  schedule   20 epochs, batch 1024, linear warmup 1 epoch -> cosine to 0, data order torch.Generator seed 0 (same
             permutation for every encoder)
  selection  final epoch (no checkpoint selection on val; val is only evaluated once at the end + per-epoch logging
             of train loss only)
Transfer diagnostic: the Original encoder's trained head is applied unchanged to the G1-pool / G1-cross features.

  CUDA_VISIBLE_DEVICES=2 python -m src.g1in.linear --cache <dir> --out <dir>
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.hcf.in_common import MODELS

CFG = dict(lr=1e-3, betas=(0.9, 0.999), wd=1e-4, epochs=20, batch=1024, warmup_epochs=1, seed=0, label_smoothing=0.0)


def load(cache, split, m, dev):
    x = torch.from_numpy(np.load(os.path.join(cache, f"{split}_cls_{m}.npy"))).to(dev).float()
    return x


@torch.no_grad()
def evaluate(head, X, y, bs=8192):
    loss, c1, c5, top5 = 0.0, 0, 0, []
    for i in range(0, len(X), bs):
        lo = head(X[i:i + bs])
        loss += float(F.cross_entropy(lo, y[i:i + bs], reduction="sum"))
        t5 = lo.topk(5, dim=1).indices
        top5.append(t5.short().cpu())
        c1 += int((t5[:, 0] == y[i:i + bs]).sum())
        c5 += int((t5 == y[i:i + bs, None]).any(1).sum())
    n = len(X)
    return dict(loss=loss / n, top1=100 * c1 / n, top5=100 * c5 / n), torch.cat(top5).numpy()


def train_head(X, y, dev, log):
    torch.manual_seed(CFG["seed"])
    head = torch.nn.Linear(X.shape[1], 1000).to(dev)
    opt = torch.optim.AdamW([{"params": [head.weight], "weight_decay": CFG["wd"]}, {"params": [head.bias], "weight_decay": 0.0}],
                            lr=CFG["lr"], betas=CFG["betas"])
    g = torch.Generator(device="cpu").manual_seed(CFG["seed"])
    n, bs = len(X), CFG["batch"]
    spe = math.ceil(n / bs)
    total, warm = spe * CFG["epochs"], spe * CFG["warmup_epochs"]
    step = 0
    for ep in range(1, CFG["epochs"] + 1):
        perm = torch.randperm(n, generator=g).to(dev)
        run, t0 = 0.0, time.time()
        for i in range(0, n, bs):
            lr = CFG["lr"] * (step + 1) / warm if step < warm else CFG["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            for gr in opt.param_groups:
                gr["lr"] = lr
            b = perm[i:i + bs]
            loss = F.cross_entropy(head(X[b]), y[b], label_smoothing=CFG["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss) * len(b)
            step += 1
        log(dict(epoch=ep, train_running_loss=run / n, lr_end=lr, s=round(time.time() - t0, 1)))
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    json.dump(CFG, open(os.path.join(a.out, "linear_config.json"), "w"), indent=1)
    ytr = torch.from_numpy(np.load(os.path.join(a.cache, "train_labels.npy")).astype(np.int64)).to(dev)
    yva = torch.from_numpy(np.load(os.path.join(a.cache, "val_labels.npy")).astype(np.int64)).to(dev)
    assert int(ytr.min()) >= 0 and int(ytr.max()) <= 999 and int(yva.min()) >= 0 and int(yva.max()) <= 999
    res, heads = {}, {}
    for m in MODELS:                                                    # independent heads, identical protocol
        Xtr = load(a.cache, "train", m, dev)
        fh = open(os.path.join(a.out, f"curve_{m}.jsonl"), "w")

        def log(r, m=m, fh=fh):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(m, json.dumps(r), flush=True)
        head = train_head(Xtr, ytr, dev, log)
        heads[m] = head
        torch.save(head.state_dict(), os.path.join(a.out, f"head_{m}.pt"))
        tr, _ = evaluate(head, Xtr, ytr)
        del Xtr
        Xva = load(a.cache, "val", m, dev)
        va, t5 = evaluate(head, Xva, yva)
        np.save(os.path.join(a.out, f"val_top5_{m}.npy"), t5)
        res[m] = dict(train=tr, val=va)
        print(m, json.dumps(res[m]), flush=True)
        del Xva
    # transfer diagnostic: the ORIGINAL head on every encoder's features
    tf = {}
    for m in MODELS:
        Xtr = load(a.cache, "train", m, dev)
        tr, _ = evaluate(heads["original"], Xtr, ytr)
        del Xtr
        Xva = load(a.cache, "val", m, dev)
        va, t5 = evaluate(heads["original"], Xva, yva)
        np.save(os.path.join(a.out, f"val_top5_transfer_{m}.npy"), t5)
        tf[m] = dict(train=tr, val=va)
        del Xva
        print("transfer", m, json.dumps(tf[m]), flush=True)
    json.dump(dict(config=CFG, independent=res, transfer_original_head=tf), open(os.path.join(a.out, "linear_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
