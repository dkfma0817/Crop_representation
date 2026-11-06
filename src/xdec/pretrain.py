"""COCO decoder training for the decoder-architecture pilot (E1-E7, R00). GPU 2+3 DDP, seed 0. No PACO data.

Loss (fixed, = RAEv2 K=1 Feature-only 'A' / R00 = src.losses.feature, verbatim): raw final-LN tokens (no normalisation),
per ROI with w = C0 teacher-cell content coverage (padding cells 0):
    L_mse = sum_p w_p mean_c (Z_p - T_p)^2 / sum_p w_p ;  L_cos = sum_p w_p (1 - cos(Z_p, T_p)) / sum_p w_p
    L = mean over the step's ROIs of (L_mse + 1.0 * L_cos), fp32, teacher under no_grad (stop-gradient).
Recipe (= src.engine.train / src.msd.pretrain): AdamW lr 1e-4 (0.9, 0.95) wd 0.05 on all decoder parameters, warmup 600 +
cosine to 0 at 12,000 steps, clip 1.0, BF16 autocast for the decoder (bicubic B and the loss in fp32), 64 ROIs / step =
one row of the existing 12k stream (16 COCO train images x 4 ROIs, seed 2027; 2 images shared with PACO val/test
removed), no flip / no augmentation. Each rank takes 8 images of the row; loss divided by the GLOBAL ROI count and the
gradients summed (= one 64-ROI step). Encoder and teacher: the same frozen DINOv2-B/14 (timm lvd142m), 336.
Dev = COCO dev ROIs minus 32 images shared with PACO val/test (468 images, 1,872 ROIs), evaluated every 1,000 steps
(+ step 0). Checkpoints 3k / 6k / 9k / 12k (with optimizer / scheduler -> resumable); selected = min dev feature loss
among these four (rule fixed before training).

  torchrun --nproc_per_node 2 -m src.xdec.pretrain --cond E3 --out <run>/checkpoints/E3
"""
import argparse
import glob
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.msd.common as MS
import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT, STREAM
from src.losses.feature import dense_cos, dense_mse
from src.msd.pretrain import eval_exclusions
from src.xdec import geometry as XG
from src.xdec import models as XM

STATS = "output/2026-11-05/msd/teacher_stats.npz"       # raw teacher final-LN channel mean / std (first 47 stream steps)
SAVE_AT = (3000, 6000, 9000, 12000)


def teacher12(vit, crops, chunk=48):
    with torch.no_grad():
        return torch.cat([RC.encode_layers(vit, crops[i:i + chunk], (12,))[12] for i in range(0, len(crops), chunk)])


def roi_terms(Z, T, w):
    """per-ROI (mse, cos) exactly src.losses.feature.dense_mse / dense_cos (fp32)."""
    return dense_mse(Z, T, w), dense_cos(Z, T, w)


def gnorm(ps):
    g = [p.grad.detach().float().norm() for p in ps if p.grad is not None]
    return float(torch.stack(g).norm()) if g else 0.0


def flat(ps):
    return torch.cat([p.detach().float().reshape(-1) for p in ps]) if ps else torch.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(XM.COND_SPEC) + ["R00"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_at", default=",".join(map(str, SAVE_AT)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dev_max", type=int, default=0, help="smoke: first N dev images")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rank, world = RC.ddp_setup()
    dev = torch.device("cuda")
    try:
        train(a, rank, world, dev)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if rank == 0:
            json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


def train(a, rank, world, dev):
    XG.install()
    ex = eval_exclusions()
    rois = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    dev_rois = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex["raev2_dev"])]
    dev_ids = sorted(dev_rois.image_id.unique())[: (a.dev_max or None)]
    stream = np.load(STREAM)[: a.steps]
    z = np.load(STATS)
    mu, sd = torch.as_tensor(z["mean_12"], device=dev), torch.as_tensor(z["std_12"], device=dev)
    dec = XM.build(a.cond).to(dev)
    dec.set_stats(mu, sd)
    student = MS.Student(adapters=False).to(dev).eval()    # frozen F336 student; its vit is also the frozen teacher encoder
    vit = student.vit
    for p in vit.parameters():
        assert not p.requires_grad
    params = list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / a.warmup if s < a.warmup else 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup))))
    save_at = sorted(int(x) for x in a.save_at.split(",") if x)
    # ---------------------------------------------------------------- resume from the newest complete checkpoint
    step = 0
    cks = sorted(glob.glob(os.path.join(a.out, "ckpt_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    if cks:
        ck = torch.load(cks[-1], map_location="cpu")
        dec.load_state_dict(ck["decoder"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        step = ck["step"]
    gr = dec.groups()
    theta0 = {k: flat(v).to(dev) for k, v in XM.build(a.cond).groups().items()}       # deterministic init (seed-0 canonical)
    ds = MS.CocoStreamDS(rois, COCO_TRAIN)
    sampler = MS.StreamSampler(stream, rank, world, exclude=ex["raev2_stream"], start=step)
    dl = DataLoader(ds, sampler=sampler, batch_size=1, num_workers=a.workers, collate_fn=MS.collate_one, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    dev_ds = MS.CocoStreamDS(dev_rois, COCO_TRAIN)
    dev_chunks = [dev_ids[i:i + 8] for i in range(0, len(dev_ids), 8)][rank::world]
    info = dict(cond=a.cond, arch=dec.arch, spec=XM.COND_SPEC.get(a.cond, "R00 CoordMLP"), steps=a.steps, world=world,
                loss="mean_ROI [ MSE_w + 1.0 (1 - cos)_w ] on raw final-LN tokens (src.losses.feature)",
                stats=STATS, excluded=dict(stream=ex["raev2_stream"], dev=ex["raev2_dev"]), n_dev_images=len(dev_ids),
                n_dev_rois=int(dev_rois.image_id.isin(dev_ids).sum()), n_params=XM.n_params(dec),
                n_params_groups={k: int(sum(p.numel() for p in v)) for k, v in gr.items()},
                recipe=dict(lr=a.lr, betas=(0.9, 0.95), wd=a.wd, warmup=a.warmup, clip=1.0, bf16=True, rois_per_step=64),
                save_at=save_at, selection="min dev feature loss among save_at", resumed_from_step=step,
                cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), local_device=torch.cuda.current_device(),
                device_name=torch.cuda.get_device_name(), device_uuid=str(getattr(torch.cuda.get_device_properties(dev), "uuid", "")))
    if rank == 0:
        json.dump(info, open(os.path.join(a.out, "config.json" if step == 0 else f"config_resume_{step}.json"), "w"), indent=1, default=str)
        print(json.dumps(info, default=str), flush=True)

    def forward(b):
        with torch.no_grad():
            feats, _ = student(b["full"])
            T = teacher12(vit, b["crops"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Z, parts = dec(feats, b["geom"], return_parts=True)
        Z = Z.float()
        w = b["geom"]["valid_w"].float()
        mse, cos = roi_terms(Z, T, w)
        return dict(Z=Z, T=T, w=w, mse=mse, cos=cos, feats=feats, parts=parts)

    @torch.no_grad()
    def dev_eval():
        dec.eval()
        acc = torch.zeros(4, device=dev, dtype=torch.float64)          # sum loss, sum mse, sum cos, n rois
        for ch in dev_chunks:
            b = MS.to_dev(dev_ds[ch], dev)
            o = forward(b)
            acc += torch.stack([(o["mse"] + 1 - o["cos"]).sum(), o["mse"].sum(), o["cos"].sum(), torch.tensor(float(len(o["mse"])), device=dev)]).double()
        acc = RC.all_sum(acc)
        dec.train()
        return dict(loss=float(acc[0] / acc[3]), mse=float(acc[1] / acc[3]), cos=float(acc[2] / acc[3]), n_rois=int(acc[3]))

    log_f = open(os.path.join(a.out, "train_log.jsonl"), "a") if rank == 0 else None
    t0 = time.time()
    if step == 0:
        ev = dev_eval()
        if rank == 0:
            log_f.write(json.dumps(dict(step=0, dev=ev)) + "\n"); log_f.flush()
            print(json.dumps(dict(step=0, dev=ev)), flush=True)
    torch.cuda.reset_peak_memory_stats()
    gc, run = {}, torch.zeros(4, device=dev)
    for b in dl:
        b = MS.to_dev(b, dev)
        o = forward(b)
        R_glob = RC.all_sum(torch.tensor(float(len(o["mse"])), device=dev))
        loss = (o["mse"] + 1.0 * (1 - o["cos"])).sum() / R_glob
        if step == 0 and rank == 0:
            p = o["parts"]
            gc["step0"] = dict(output_shape=list(o["Z"].shape), teacher_shape=list(o["T"].shape),
                               feats_requires_grad=bool(o["feats"][12].requires_grad), teacher_requires_grad=bool(o["T"].requires_grad),
                               finite=bool(torch.isfinite(o["Z"]).all()))
            if p.get("B") is not None and XM.COND_SPEC.get(a.cond, {}).get("out") == "skip":
                gc["step0"].update(max_abs_head=float(p["head"].abs().max()), max_abs_Z_minus_B=float((o["Z"] - p["B"]).abs().max()),
                                   Z_equals_B=bool(torch.equal(o["Z"], p["B"])))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        RC.allreduce_grads(params)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if step < 3 and rank == 0:
            gc[f"step{step + 1}_grad_norm"] = {k: gnorm(v) for k, v in gr.items()}
            before = {k: flat(v).clone() for k, v in gr.items()}
        opt.step()
        sched.step()
        if f"step{step + 1}_grad_norm" in gc:
            gc[f"step{step + 1}_update_norm"] = {k: float((flat(v).to(dev) - before[k].to(dev)).norm()) for k, v in gr.items()}
        run += torch.stack([loss.detach() * world, (o["mse"].sum() / R_glob).detach() * world, (o["cos"].sum() / R_glob).detach() * world, gn.detach()])
        step += 1
        if step == 3 and rank == 0:
            json.dump(gc, open(os.path.join(a.out, "grad_check.json"), "w"), indent=1)
        if step % 50 == 0:
            r = (RC.all_sum(run.clone()) / (50 * world)).tolist()
            if not all(np.isfinite(r)):
                raise FloatingPointError(f"non-finite loss at step {step}: {r}")
            rec = dict(step=step, lr=sched.get_last_lr()[0], loss=r[0], mse=r[1], cos=r[2], grad_norm=r[3], s=round(time.time() - t0))
            run.zero_()
            if rank == 0:
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                if step % 500 == 0:
                    print(json.dumps(rec), flush=True)
        if step % a.eval_every == 0 or step == a.steps or step in save_at:
            ev = dev_eval()
            if rank == 0:
                log_f.write(json.dumps(dict(step=step, dev=ev)) + "\n"); log_f.flush()
                print(json.dumps(dict(step=step, dev=ev)), flush=True)
                if step in save_at:
                    tmp = os.path.join(a.out, f".ckpt_{step}.tmp")
                    torch.save(dict(arch=dec.arch, cond=a.cond, decoder=dec.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                    step=step, dev=ev, adapters=None), tmp)
                    os.replace(tmp, os.path.join(a.out, f"ckpt_{step}.pt"))
        if step >= a.steps:
            break
    if rank == 0:
        devs = {}
        for ln in open(os.path.join(a.out, "train_log.jsonl")):
            r = json.loads(ln)
            if "dev" in r and r["step"] in save_at:
                devs[r["step"]] = r["dev"]
        sel = min(devs, key=lambda s: devs[s]["loss"])
        ck = torch.load(os.path.join(a.out, f"ckpt_{sel}.pt"), map_location="cpu")
        torch.save(dict(arch=ck["arch"], cond=a.cond, decoder=ck["decoder"], step=sel, dev=ck["dev"], adapters=None), os.path.join(a.out, "selected.pt"))
        delta = {k: float((flat(v).to(dev) - theta0[k]).norm()) for k, v in gr.items()}
        json.dump(dict(status="done", steps=step, time_s=round(time.time() - t0), peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                       weight_change_norm=delta, dev_at_ckpts=devs, selected=dict(step=sel, dev=devs[sel]), time=time.strftime("%F %T")),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
