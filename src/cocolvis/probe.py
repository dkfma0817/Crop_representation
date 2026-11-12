"""COCO / LVIS GT-box region-classification headroom probe (2026-11-07). One condition per process, ONE GPU.

Condition = (dataset, mode); encoder is ALWAYS the frozen original DINOv3-B/16 lvd1689m (no G1, no checkpoint loading):
  full  : full image 384 letterbox, encoded once; R2 reads the 576 full-grid tokens with the GT-box geometry (cov > 0)
  tight : C0 tight crop of the GT box -> 384 letterbox -> re-encoded; R2 reads the object (= letterbox content) region
Inputs / geometry / hflip / step packing = the VAW probe (src.vaw.data.VawDS via src.cocolvis.data.RegionDS).
Reader = src.readout.models.Reader('r2', layers=[12]) with ONE Linear(768 -> K) head (K = 80 COCO, 1203 LVIS),
init torch.manual_seed(7). Loss = cross-entropy over the K classes (plain, natural sampling, no class re-weighting).
Optim (identical for every condition of a dataset): AdamW reader lr 3e-4 / head lr 1e-3, wd 0.05 (none on 1-D/query),
betas (0.9, 0.999), 1 epoch warmup + cosine, clip 1.0, ~128 ROIs per step (C.full_epoch_batches, data seed 0).
Uses ONLY the GT box (no category) at inference. Val = checkpoint (epoch) selection by val top-1 + the reported numbers.
Final: the selected epoch's reader is reloaded, must reproduce its val top-1, and the full fp32 softmax over K classes
for every val GT box is saved (val_prob.npy, rows = data/instances_{ds}_val.parquet order) for the AP / bootstrap report.

  CUDA_VISIBLE_DEVICES=2 python -m src.cocolvis.probe --ds coco --mode full --out <run>/coco_full
"""
import argparse
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.readout.common import C
from src.readout.models import Reader, n_params, param_groups
from src.cocolvis.common import NCLS, RUN
from src.cocolvis.data import RegionDS
from src.cocolvis.metrics import per_instance, ranks_topk, summary
from src.viz.c0_pca import load_encoder

LR, HEAD_LR, WD = 3e-4, 1e-3, 0.05
EPOCHS = {"coco": 5, "lvis": 5}


class HeadK(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.lin = nn.Linear(768, k)

    def forward(self, f, lv):
        return self.lin(f).float()


def build_reader(k, dev):
    torch.manual_seed(7)
    r = Reader("r2", layers=(12,))
    r.heads = HeadK(k)
    return r.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["coco", "lvis"])
    ap.add_argument("--mode", required=True, choices=["full", "tight"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(RUN, "data"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N images per split")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke: steps per epoch")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    a.epochs = a.epochs or EPOCHS[a.ds]
    os.makedirs(a.out, exist_ok=True)
    try:
        run(a)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


def run(a):
    dev = torch.device("cpu" if a.cpu else "cuda")
    t0 = time.time()
    K = NCLS[a.ds]
    D = {sp: pd.read_parquet(os.path.join(a.data, f"instances_{a.ds}_{sp}.parquet")) for sp in ("train", "val")}
    if a.limit:
        for sp in D:
            keep = set(D[sp].image_id.drop_duplicates().iloc[: a.limit])
            D[sp] = D[sp][D[sp].image_id.isin(keep)].reset_index(drop=True)
    Y = {sp: torch.as_tensor(D[sp].label.values, dtype=torch.long) for sp in D}
    freq = None
    if a.ds == "lvis":
        cc = pd.read_csv(os.path.join(a.data, "class_counts_lvis.csv"))
        freq = cc.frequency.values[D["val"].label.values]
    vit = load_encoder("dinov3", dev)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    sum0 = float(sum(p.double().sum() for p in vit.parameters()))
    reader = build_reader(K, dev)
    opt = torch.optim.AdamW(param_groups(reader, LR, HEAD_LR, WD), lr=LR, betas=(0.9, 0.999))
    tr_ds = RegionDS(D["train"], Y["train"], True, a.mode)
    sampler = RC.RankStepSampler(tr_ds.n_roi, 0, 1, seed=0)
    torch.manual_seed(0)
    dl = DataLoader(tr_ds, sampler=sampler, batch_size=1, num_workers=a.workers, pin_memory=not a.cpu,
                    collate_fn=RC.collate_one, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    steps_ep = [len(C.full_epoch_batches(tr_ds.n_roi, e, 0)) for e in range(1, a.epochs + 1)]
    if a.max_steps:
        steps_ep = [min(s, a.max_steps) for s in steps_ep]
    total, warm = sum(steps_ep), steps_ep[0]
    cfg = dict(args=vars(a), encoder="frozen original DINOv3-B/16 lvd1689m (timm vit_base_patch16_dinov3.lvd1689m)",
               encoder_checksum=sum0, n_classes=K, n_train_rois=len(D["train"]), n_train_groups=len(tr_ds),
               n_train_images=int(D["train"].image_id.nunique()), n_val_rois=len(D["val"]), steps_per_epoch=steps_ep,
               total_steps=total, loss="cross-entropy over K classes, natural sampling, no re-weighting",
               optim=dict(lr=LR, head_lr=HEAD_LR, wd=WD, betas=[0.9, 0.999], warmup_steps=warm, schedule="cosine", clip=1.0),
               reader_params=n_params(reader), head=f"Linear(768, {K})", reader_init_seed=7, data_seed=0, hflip=0.5,
               selection="max val top-1 (ties -> earlier epoch)", torch=torch.__version__, start=time.strftime("%F %T"))
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
    print(json.dumps(cfg, default=str), flush=True)
    ac = lambda: torch.autocast(dev.type, dtype=torch.bfloat16)  # noqa: E731

    @torch.no_grad()
    def encode(x):
        toks = []
        for i in range(0, len(x), 64):
            with ac():
                t = vit.forward_features(x[i:i + 64])
            toks.append(t[:, vit.num_prefix_tokens:].float())
        return {12: torch.cat(toks)}

    @torch.no_grad()
    def score(full_prob=False):
        ds = RegionDS(D["val"], Y["val"], False, a.mode)
        bs = 16 if a.mode == "full" else 8
        chunks = [list(range(i, min(len(ds), i + bs))) for i in range(0, len(ds), bs)]
        dle = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=not a.cpu)
        n = len(D["val"])
        top5 = torch.full((n, 5), -1, dtype=torch.long)
        prob = torch.full((n, K), float("nan")) if full_prob else None
        reader.eval()
        for b in dle:
            b = RC.to_dev(b, dev)
            f = encode(b["x"])
            with ac():
                o = reader(f, b)["logits"].float()
            r = b["rows"].cpu()
            top5[r] = o.topk(5, dim=1).indices.cpu()
            if full_prob:
                prob[r] = F.softmax(o, dim=1).cpu()
        reader.train()
        assert (top5[:, 0] >= 0).all()
        if full_prob:
            assert not torch.isnan(prob).any()
        return top5.numpy(), (prob.numpy() if full_prob else None)

    def metrics(top5):
        y = D["val"].label.values
        c1, c5 = per_instance(top5, y)
        return summary(c1, c5, y, D["val"]["size"].values, freq, K)

    hist, step = [], 0
    for ep in range(1, a.epochs + 1):
        sampler.set_epoch(ep)
        rl, n, te = torch.zeros((), device=dev), 0, time.time()
        for k, b in enumerate(dl):
            if a.max_steps and k >= a.max_steps:
                break
            b = RC.to_dev(b, dev)
            f = encode(b["x"])
            for g in opt.param_groups:
                g["lr"] = C.lr_at(step, g["base"], warm, total)
            with ac():
                o = reader(f, b)
            loss = F.cross_entropy(o["logits"].float(), b["Y"].long())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            rl += loss.detach()
            n += 1
            step += 1
            if step % 500 == 0:
                if not torch.isfinite(rl):
                    raise FloatingPointError(f"non-finite loss before step {step}")
                print(json.dumps(dict(step=step, ep=ep, s=round(time.time() - t0), s_per_step=round((time.time() - te) / n, 3),
                                      loss=round(float(rl / n), 4))), flush=True)
        tv = time.time()
        top5, _ = score()
        m = metrics(top5)
        rec = dict(epoch=ep, step=step, val=m, train_loss=float(rl / max(n, 1)), epoch_train_s=round(tv - te, 1),
                   val_s=round(time.time() - tv, 1), elapsed_s=round(time.time() - t0, 1))
        hist.append(rec)
        with open(os.path.join(a.out, "curves.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        np.save(os.path.join(a.out, f"val_top5_ep{ep}.npy"), top5.astype(np.int16))
        torch.save(dict(reader=reader.state_dict(), epoch=ep, val=m), os.path.join(a.out, f"reader_ep{ep}.pt"))
        print(json.dumps(rec), flush=True)
    best = max(hist, key=lambda r: (r["val"]["top1"], -r["epoch"]))
    sel = dict(selected_epoch=best["epoch"], val_top1=best["val"]["top1"], rule="max val top-1 (ties -> earlier)",
               all={r["epoch"]: r["val"]["top1"] for r in hist}, last_epoch=hist[-1]["epoch"])
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    sum1 = float(sum(p.double().sum() for p in vit.parameters()))
    json.dump(dict(checksum_before=sum0, checksum_after=sum1, unchanged=sum1 == sum0,
                   requires_grad_any=any(p.requires_grad for p in vit.parameters())),
              open(os.path.join(a.out, "frozen_check.json"), "w"), indent=1)
    assert sum1 == sum0, "encoder changed"
    guard = os.path.join(a.out, "final_scored.json")
    assert not os.path.exists(guard), "final scoring already done for this condition"
    reader.load_state_dict(torch.load(os.path.join(a.out, f"reader_ep{best['epoch']}.pt"), map_location=dev)["reader"])
    top5, prob = score(full_prob=True)
    m = metrics(top5)
    assert abs(m["top1"] - best["val"]["top1"]) < 2e-3, "reloaded reader does not reproduce val top-1"
    np.save(os.path.join(a.out, "val_prob.npy"), prob.astype(np.float32))
    np.save(os.path.join(a.out, "val_top5.npy"), top5.astype(np.int16))
    json.dump(dict(epoch=best["epoch"], time=time.strftime("%F %T"), n=len(prob), val=m), open(guard, "w"), indent=1)
    json.dump(dict(status="done", time_s=round(time.time() - t0), time=time.strftime("%F %T")),
              open(os.path.join(a.out, "done.json"), "w"), indent=1)
    print("DONE", json.dumps(sel), flush=True)


if __name__ == "__main__":
    main()
