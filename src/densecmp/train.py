"""PACO R2 probe on a frozen EVA-family encoder (EVA-CLIP / CLIPSelf / DeCLIP / EVA crop oracle).

= src.ivc.train (11-05 IVC / MSD probe protocol, the protocol of the DINOv2 test-primary numbers) restricted to R2, with
the encoder swapped. Everything else is the same code path / constants:
  data      RC.load_data(EXCL) (PACO train minus data_check exclusions), PACO dev = 1,000 train images held out (rng 0),
            image packs of ~128 ROIs (C.full_epoch_batches, data seed 0) split over 2 ranks (RC.RankStepSampler), hflip p=0.5
  reader    src.readout.models.Reader('r2', layers=(12,)) built right after torch.manual_seed(7) (identical init order),
            PLUS an input adapter in_proj = LayerNorm(512) -> Linear(512, 768) (the encoders output 512-d; R2 is 768-d).
            Adapter params: reader lr, wd 0.05 on the Linear weight (RR.reader_param_groups rule).
  loss/opt  masked BCE / global #known labels, AdamW(0.9, 0.999) lr 3e-4 (head 1e-3), wd 0.05, warmup 1 epoch + cosine,
            grad clip 1.0, 10 epochs, reader under bf16 autocast
  eval      every epoch: PACO dev + PACO VAL scores (M0 / M2, RC.m2_eval); every epoch's reader saved (ep<k>.pt).
            Selected epoch = max PACO-dev M0 Overall (MSD/IVC rule, applied in report.py). Test is never read here.
Encoder: src.densecmp.encoders.Encoder(name) frozen (requires_grad False, checksum before/after -> frozen_check.json).

  torchrun --nproc_per_node 2 -m src.densecmp.train --enc clipself --mode full --side 1024 --out <dir>
"""
import argparse
import json
import os
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.densecmp.data import EvaPacoDS
from src.densecmp.encoders import DIM, PATCH, Encoder
from src.ivc.common import EXCL
from src.readout.common import C
from src.readout.models import Reader, n_params, param_groups


class EvaR2(Reader):
    """R2 with a 512 -> 768 token adapter. Reader parameters are created first (same RNG order as the DINOv2 R2)."""

    def __init__(self, in_dim=DIM):
        super().__init__("r2", layers=(12,))
        self.in_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 768))

    def forward(self, feats, b, ctx_mode="normal"):
        return super().forward({12: self.in_proj(feats[12])}, b, ctx_mode)


def build_reader(seed, dev):
    torch.manual_seed(1000 * seed + 7)
    return EvaR2().to(dev)


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "c0"])
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images per split")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: steps per epoch")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--enc_chunk", type=int, default=0, help="encoder micro-batch (0 = auto)")
    a = ap.parse_args()
    rank, world = RC.ddp_setup()
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    try:
        run(a, rank, world, dev)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if rank == 0:
            json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


def run(a, rank, world, dev):
    lr, head_lr, wd, epochs = 3e-4, 1e-3, 0.05, a.epochs
    data = RC.load_data(EXCL, a.limit)
    Ytr_by = RC.ytr_by_level(data)
    dtr, Ytr = data["train"]
    dva, Yva = data["val"]
    ids = np.array(sorted(dtr.image_id.unique()))
    dev_ids = set(np.random.default_rng(0).choice(ids, min(1000, len(ids) // 2), replace=False).tolist())
    m = dtr.image_id.isin(dev_ids).values
    ddev, Ydev = dtr[m].reset_index(drop=True), Ytr[m]
    dtr, Ytr = dtr[~m].reset_index(drop=True), Ytr[~m]
    assert not (set(dtr.image_id) & set(dva.image_id)) and not (set(dtr.image_id) & set(ddev.image_id))
    reader = build_reader(a.seed, dev)
    opt = torch.optim.AdamW(param_groups(reader, lr, head_lr, wd), lr=lr, betas=(0.9, 0.999))
    enc = Encoder(a.enc).to(dev)
    frozen = list(enc.parameters())
    sum0 = float(sum(p.double().sum() for p in frozen))
    chunk = a.enc_chunk or (8 if (a.mode == "full" and a.side >= 768) else 64)

    def DS(df, Y, flip):
        return EvaPacoDS(df, Y, flip=flip, mode=a.mode, side=a.side, patch=PATCH)

    tr_ds = DS(dtr, Ytr, True)
    sampler = RC.RankStepSampler(tr_ds.n_roi, rank, world, seed=0)
    torch.manual_seed(a.seed * 100 + rank)
    dl = DataLoader(tr_ds, sampler=sampler, batch_size=1, num_workers=a.workers, pin_memory=True,
                    collate_fn=RC.collate_one, persistent_workers=True, prefetch_factor=4)
    steps_per_ep = [len(C.full_epoch_batches(tr_ds.n_roi, e, 0)) for e in range(1, epochs + 1)]
    if a.max_steps:
        steps_per_ep = [min(s, a.max_steps) for s in steps_per_ep]
    total, warm = sum(steps_per_ep), steps_per_ep[0]
    G = a.side // PATCH
    cfg = dict(args=vars(a), world=world, encoder=a.enc, encoder_load=enc.load_report, token_grid=[G, G], feat_dim=DIM,
               n_train_rows=len(dtr), n_train_images=int(dtr.image_id.nunique()), n_dev_rows=len(ddev), n_dev_images=int(ddev.image_id.nunique()),
               n_val_rows=len(dva), n_val_images=int(dva.image_id.nunique()), steps_per_epoch=steps_per_ep, total_steps=total,
               n_params_reader=n_params(reader), n_params_in_proj=n_params(reader.in_proj), lr=lr, head_lr=head_lr, wd=wd,
               encoder_checksum_before=sum0, torch=torch.__version__)
    if rank == 0:
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
        os.makedirs(os.path.join(a.out, "r2"), exist_ok=True)
    log(rank, json.dumps({k: v for k, v in cfg.items() if k != "encoder_load"}, default=str))

    def encode(b):
        x = b["x"]
        f = torch.cat([enc(x[i:i + chunk]) for i in range(0, len(x), chunk)])
        assert f.shape[1] == G * G and f.shape[2] == DIM and b["cov"].shape[1] == G * G, (f.shape, b["cov"].shape)
        return {12: f}

    @torch.no_grad()
    def score(df, Y, rows_subset=None):
        ds = DS(df, Y, False)
        idx = np.arange(len(ds)) if rows_subset is None else rows_subset
        es = RC.EvalSampler(len(idx), 8 if a.mode == "full" else 6, rank, world)
        chunks = [[int(idx[i]) for i in ch] for ch in es]
        dle = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
        Sc = torch.zeros(len(df), 59, device=dev)
        reader.eval()
        for b in dle:
            b = RC.to_dev(b, dev)
            f = encode(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = reader(f, b)
            Sc[b["rows"]] = o["logits"].float()
        reader.train()
        return RC.all_sum(Sc).cpu().numpy()

    t0, step = time.time(), 0

    def evaluate(ep, stats):
        Sv = score(dva, Yva)
        Sd = score(ddev, Ydev)
        if rank != 0:
            return
        rd = RC.m2_eval(Sd, ddev, Ydev, Ytr_by)
        with open(os.path.join(a.out, "r2", "curves_dev.jsonl"), "a") as fh:
            fh.write(json.dumps(dict(epoch=ep, dev=dict(loss=rd["loss"], M0=rd["M0"], M2=rd["M2"]))) + "\n")
        np.save(os.path.join(a.out, "r2", f"scores_dev_ep{ep}.npy"), Sd)
        r = RC.m2_eval(Sv, dva, Yva, Ytr_by)
        if not np.isfinite(r["loss"]):
            raise FloatingPointError(f"non-finite val loss at epoch {ep}")
        rec = dict(epoch=ep, step=step, elapsed_s=round(time.time() - t0, 1), **stats,
                   val=dict(loss=r["loss"], object_loss=r["object_loss"], part_loss=r["part_loss"], M0=r["M0"], M2=r["M2"]))
        with open(os.path.join(a.out, "r2", "curves.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        np.save(os.path.join(a.out, "r2", f"scores_val_ep{ep}.npy"), Sv)
        print(json.dumps(dict(epoch=ep, s=round(time.time() - t0), dev_M0=round(rd["M0"]["overall"], 2), val_M0=round(r["M0"]["overall"], 2),
                              val_M2=round(r["M2"]["overall"], 2), val_loss=round(r["loss"], 4))), flush=True)

    evaluate(0, {})
    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        run_l, run_n, bad = torch.zeros((), device=dev), 0.0, torch.zeros((), device=dev)
        for k, b in enumerate(dl):
            if a.max_steps and k >= a.max_steps:
                break
            b = RC.to_dev(b, dev)
            f = encode(b)
            Y = b["Y"]
            nt = RC.all_sum(torch.tensor(float((Y >= 0).sum()), device=dev))
            for g in opt.param_groups:
                g["lr"] = C.lr_at(step, g["base"], warm, total)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = reader(f, b)
            ls, _ = C.masked_bce_sum(o["logits"].float(), Y)
            loss = ls / nt
            bad = bad + (~torch.isfinite(loss.detach())).float()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            RC.allreduce_grads(list(reader.parameters()))
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            run_l += ls.detach()
            run_n += float(nt) / world
            step += 1
            if step % 100 == 0:
                if float(bad) > 0:
                    raise FloatingPointError(f"non-finite train loss before step {step}")
                log(rank, json.dumps(dict(step=step, epoch=ep, s=round(time.time() - t0), loss_run=round(float(run_l) / max(run_n, 1), 4))))
        if float(bad) > 0:
            raise FloatingPointError(f"non-finite train loss in epoch {ep}")
        tot_l = float(RC.all_sum(run_l.clone()))
        tot_n = float(RC.all_sum(torch.tensor(run_n, device=dev)))
        evaluate(ep, dict(train_running_loss=tot_l / max(tot_n, 1)))
        if rank == 0:
            torch.save(dict(reader=reader.state_dict(), epoch=ep), os.path.join(a.out, "r2", f"ep{ep}.pt"))
    # train-subset fit (2,000 fixed train images, no flip) = src.ivc.train final extra
    rng = np.random.default_rng(0)
    groups = C.image_groups(dtr)
    sub = np.sort(rng.choice(len(groups), min(2000, len(groups)), replace=False))
    St = score(dtr, Ytr, rows_subset=sub)
    sum1 = float(sum(p.double().sum() for p in frozen))
    if rank == 0:
        rows = np.concatenate([groups[i] for i in sub])
        r = RC.m2_eval(St[rows], dtr.iloc[rows].reset_index(drop=True), Ytr[rows], Ytr_by)
        json.dump(dict(train_subset_loss=r["loss"], train_subset_M0=r["M0"], train_subset_M2=r["M2"], n_images=len(sub), n_rows=len(rows)),
                  open(os.path.join(a.out, "train_fit.json"), "w"), indent=1)
        json.dump(dict(requires_grad_any=any(p.requires_grad for p in frozen), grads_present=any(p.grad is not None for p in frozen),
                       checksum_before=sum0, checksum_after=sum1, unchanged=sum1 == sum0), open(os.path.join(a.out, "frozen_check.json"), "w"), indent=1)
        assert sum1 == sum0, "frozen encoder changed during probe training"
        json.dump(dict(status="done", epochs=epochs, steps=step, time_s=round(time.time() - t0), time=time.strftime("%F %T")),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
    log(rank, "DONE")


if __name__ == "__main__":
    main()
