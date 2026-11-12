"""ADE20K (SceneParsing 150) frozen-feature LINEAR segmentation for Original / G1-pool / G1-cross encoders (2026-11-07).

Model: image -> frozen DINOv3-B/16 (eval, no grad, bf16) -> forward_features[:, 5:] (final-LayerNorm PATCH tokens only;
CLS + 4 registers dropped) -> Linear(768 -> 150) per token -> bilinear upsample of the logits -> pixel cross-entropy.
No BN, no multi-layer fusion, no decoder. The three heads are trained on the SAME augmented batches (one input tensor
per step, fed to all three encoders), same init (seed 0), same optimiser -> identical protocol.

Labels: official ADEChallengeData2016 annotations (uint8, 0 = 'other' -> ignore 255, 1..150 -> 0..149).
Protocol fixed before training (DINOv2/v3 linear-seg mmseg recipe scaled from 512 to 384, photometric distortion omitted):
  train : resize keeping aspect so the shorter side = 384 * U(0.5, 2.0) (image bilinear, mask NEAREST), random crop 384x384
          (smaller -> pad: image 0 after normalisation, mask 255 = ignore -> padding excluded from the loss), hflip 0.5,
          ImageNet mean / std
  optim : AdamW lr 1e-3, betas (0.9, 0.999), wd 1e-4 on weights, 40,000 iterations, batch 16 (2 GPUs x 8), linear warmup
          1,500 iters (from 1e-6 x lr) then poly(1.0) decay to 0; final iteration is evaluated (no checkpoint selection)
  eval  : whole image, shorter side resized to 384 (bilinear), padded to a multiple of 16 (pad region cropped away from
          the logits), logits bilinearly resized to the ORIGINAL H x W, argmax; IoU accumulated per class over the val set
          (mIoU = mean over classes with union > 0), per-image intersection / union kept for the image bootstrap.

  torchrun --nproc_per_node 2 -m src.g1eval.ade --out <dir> [--iters N --val_limit N]   (smoke: small N)
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import src.readout.common as RC
from src.g1in.common import G1, MEAN, MODELS, PACO_PROBE, checksum, load_encoder

ROOT = "/home/irteam/datasets/ade20k/ADEChallengeData2016"
NC, IGN, S, P, NPT = 150, 255, 384, 16, 5
CFG = dict(lr=1e-3, betas=(0.9, 0.999), wd=1e-4, iters=40000, batch=16, warmup=1500, warmup_ratio=1e-6, poly_power=1.0,
           ratio_range=(0.5, 2.0), crop=S, flip=0.5, eval_short=S, seed=0)
MEAN_T = torch.tensor(MEAN).view(3, 1, 1)
STD_T = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def listing(split):
    d = {"train": "training", "val": "validation"}[split]
    ims = sorted(os.listdir(os.path.join(ROOT, "images", d)))
    return [(os.path.join(ROOT, "images", d, f), os.path.join(ROOT, "annotations", d, f[:-4] + ".png")) for f in ims]


def load_pair(ip, ap):
    img = Image.open(ip).convert("RGB")
    lab = np.array(Image.open(ap), dtype=np.int64)
    lab = np.where(lab == 0, IGN, lab - 1)                    # reduce_zero_label
    return img, lab


def norm(img):
    return (torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1) - MEAN_T) / STD_T


class TrainDS(Dataset):
    def __init__(self):
        self.items = listing("train")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, lab = load_pair(*self.items[i])
        W, H = img.size
        s = S * random.uniform(*CFG["ratio_range"]) / min(W, H)
        nw, nh = max(1, round(W * s)), max(1, round(H * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        lab = np.array(Image.fromarray(lab.astype(np.uint8)).resize((nw, nh), Image.NEAREST), np.int64)
        x = norm(img)
        c = CFG["crop"]
        ox, oy = random.randint(0, max(0, nw - c)), random.randint(0, max(0, nh - c))
        x, lab = x[:, oy:oy + c, ox:ox + c], lab[oy:oy + c, ox:ox + c]
        X = torch.zeros(3, c, c)
        Y = torch.full((c, c), IGN, dtype=torch.long)
        X[:, :x.shape[1], :x.shape[2]] = x
        Y[:lab.shape[0], :lab.shape[1]] = torch.from_numpy(lab)
        if random.random() < CFG["flip"]:
            X, Y = X.flip(-1), Y.flip(-1)
        return X, Y


class Sampler:
    """Global permutation per epoch (seed + epoch); rank r takes items r::world of each global batch of 16."""

    def __init__(self, n, rank, world, iters, bs):
        self.n, self.rank, self.world, self.iters, self.bs = n, rank, world, iters, bs

    def __iter__(self):
        k, ep = 0, 0
        while True:
            perm = np.random.default_rng(CFG["seed"] + ep).permutation(self.n)
            for i in range(0, self.n - self.bs + 1, self.bs):
                if k >= self.iters:
                    return
                yield from perm[i:i + self.bs][self.rank::self.world].tolist()
                k += 1
            ep += 1

    def __len__(self):
        return self.iters * self.bs // self.world


def lr_at(it):
    if it < CFG["warmup"]:
        k = CFG["warmup_ratio"] + (1 - CFG["warmup_ratio"]) * it / CFG["warmup"]
        return CFG["lr"] * k
    return CFG["lr"] * (1 - (it - CFG["warmup"]) / (CFG["iters"] - CFG["warmup"])) ** CFG["poly_power"]


def feats(e, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = e.forward_features(x)
    h, w = x.shape[-2] // P, x.shape[-1] // P
    t = t[:, NPT:].float()
    assert t.shape[1] == h * w
    return t.transpose(1, 2).reshape(len(x), -1, h, w)


def logits(head, f):
    return F.conv2d(f, head.weight[:, :, None, None], head.bias)


@torch.no_grad()
def evaluate(encs, heads, rank, world, dev, limit=0):
    items = listing("val")[: (limit or None)]
    mine = list(range(len(items)))[rank::world]
    I = {m: np.zeros((len(items), NC), np.int64) for m in MODELS}
    U = {m: np.zeros((len(items), NC), np.int64) for m in MODELS}
    for i in mine:
        img, lab = load_pair(*items[i])
        W, H = img.size
        s = CFG["eval_short"] / min(W, H)
        nw, nh = max(1, round(W * s)), max(1, round(H * s))
        x = norm(img.resize((nw, nh), Image.BILINEAR))
        pw, ph = math.ceil(nw / P) * P, math.ceil(nh / P) * P
        X = torch.zeros(1, 3, ph, pw)
        X[0, :, :nh, :nw] = x
        X = X.to(dev)
        y = torch.from_numpy(lab).to(dev)
        valid = y != IGN
        for m in MODELS:                                          # same input tensor X for every encoder
            lo = logits(heads[m], feats(encs[m], X))
            lo = F.interpolate(lo, size=(ph, pw), mode="bilinear", align_corners=False)[:, :, :nh, :nw]
            pred = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False).argmax(1)[0]
            p, t = pred[valid], y[valid]
            inter = torch.bincount(t[p == t], minlength=NC)[:NC]
            ap = torch.bincount(p, minlength=NC)[:NC]
            at = torch.bincount(t, minlength=NC)[:NC]
            I[m][i], U[m][i] = inter.cpu().numpy(), (ap + at - inter).cpu().numpy()
    for m in MODELS:
        I[m] = RC.all_sum(torch.from_numpy(I[m]).to(dev)).cpu().numpy()
        U[m] = RC.all_sum(torch.from_numpy(U[m]).to(dev)).cpu().numpy()
    return I, U


def miou(I, U):
    i, u = I.sum(0), U.sum(0)
    ok = u > 0
    return 100 * float((i[ok] / u[ok]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.iters:
        CFG["iters"] = a.iters
        CFG["warmup"] = min(CFG["warmup"], max(1, a.iters // 4))
    rank, world = RC.ddp_setup()
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    random.seed(CFG["seed"] * 100 + rank)
    torch.manual_seed(CFG["seed"] * 100 + rank)
    encs = {m: load_encoder(m, dev) for m in MODELS}
    ref = {m: json.load(open(os.path.join(G1, PACO_PROBE[m], "frozen_check.json")))["checksum_before"] for m in MODELS}
    ck0 = {m: checksum(e) for m, e in encs.items()}
    assert all(abs(ck0[m] - ref[m]) < 1e-6 for m in MODELS), (ck0, ref)
    assert all(not p.requires_grad for e in encs.values() for p in e.parameters()) and not any(e.training for e in encs.values())
    heads = {}
    for m in MODELS:
        torch.manual_seed(CFG["seed"])                           # identical initial head for every encoder
        heads[m] = torch.nn.Linear(768, NC).to(dev)
    opts = {m: torch.optim.AdamW([{"params": [h.weight], "weight_decay": CFG["wd"]}, {"params": [h.bias], "weight_decay": 0.0}],
                                 lr=CFG["lr"], betas=CFG["betas"]) for m, h in heads.items()}
    ds = TrainDS()
    bs_rank = CFG["batch"] // world
    dl = DataLoader(ds, batch_size=bs_rank, sampler=Sampler(len(ds), rank, world, CFG["iters"], CFG["batch"]), num_workers=a.workers,
                    pin_memory=True, drop_last=False, persistent_workers=True, prefetch_factor=4,
                    worker_init_fn=lambda w: (random.seed(1000 * rank + w), np.random.seed(1000 * rank + w)))
    info = dict(cfg=CFG, world=world, n_train=len(ds), n_val=len(listing("val")), checksum=ck0, checksum_paco=ref,
                feature="forward_features(x)[:, 5:] = final-LN patch tokens (CLS + 4 registers dropped)")
    if rank == 0:
        json.dump(info, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
        print(json.dumps(info, default=str), flush=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w") if rank == 0 else None
    t0, run, it = time.time(), {m: 0.0 for m in MODELS}, 0
    for x, y in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        lr = lr_at(it)
        nvalid = RC.all_sum((y != IGN).sum().float())
        for m in MODELS:                                          # same augmented input tensor for every encoder
            with torch.no_grad():
                f = feats(encs[m], x)
            lo = F.interpolate(logits(heads[m], f), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(lo, y, ignore_index=IGN, reduction="sum") / nvalid
            for g in opts[m].param_groups:
                g["lr"] = lr
            opts[m].zero_grad(set_to_none=True)
            loss.backward()
            RC.allreduce_grads(list(heads[m].parameters()))
            opts[m].step()
            run[m] += float(loss) * world
        it += 1
        if it % 100 == 0 and rank == 0:
            rec = dict(it=it, s=round(time.time() - t0), lr=lr, **{f"loss_{m}": round(run[m] / 100, 4) for m in MODELS})
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run = {m: 0.0 for m in MODELS}
    if rank == 0:
        for m in MODELS:
            torch.save(heads[m].state_dict(), os.path.join(a.out, f"head_{m}.pt"))
    for h in heads.values():
        h.eval()
    t1 = time.time()
    I, U = evaluate(encs, heads, rank, world, dev, a.val_limit)
    if rank == 0:
        res = {m: dict(mIoU=miou(I[m], U[m]), aAcc=None) for m in MODELS}
        np.savez_compressed(os.path.join(a.out, "val_inter_union.npz"), **{f"I_{m}": I[m] for m in MODELS}, **{f"U_{m}": U[m] for m in MODELS})
        ck1 = {m: checksum(e) for m, e in encs.items()}
        json.dump(dict(results=res, train_time_s=round(t1 - t0), eval_time_s=round(time.time() - t1), iters=it,
                       checksum_after=ck1, unchanged={m: ck1[m] == ck0[m] for m in MODELS}),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
        print("DONE", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
