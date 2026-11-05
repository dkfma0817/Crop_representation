"""COCO decoder pre-training for H0-H3 (no PACO data, no attribute labels). GPU 2+3 DDP, seed 0.

  H0: encoder frozen,              decoder trained, loss = L12
  H1: encoder frozen,              decoder trained, loss = L12 + 0.5 L8 + 0.5 L4
  H2: encoder adapters (4/8/12) + decoder trained, loss = L12                 + 1.0 L_glob
  H3: encoder adapters (4/8/12) + decoder trained, loss = L12 + 0.5 L8 + 0.5 L4 + 1.0 L_glob
L_d = coverage-weighted mean over teacher cells of [mean_c (p - t)^2 + (1 - cos(p, t))] on per-depth teacher-normalised
features ((x - mu_d) / sd_d, stats = 3,000 stream ROIs); p_12 = decoder output Z, p_4 / p_8 = aux heads on Z.
Recipe = existing COCO decoder recipe (src.engine.train / multilevel_relation_decoder): AdamW lr 1e-4 (0.9, 0.95) wd 0.05,
warmup 600 + cosine to 0 at 12,000 steps, clip 1.0, BF16 autocast, 64 ROIs / step = one row of the existing 12k stream
(16 COCO train images x 4 ROIs, seed 2027), no flip. Dev = existing COCO dev ROIs minus images shared with PACO val/test;
checkpoint = best dev L12 (evaluated every 1,000 steps). Grad / update checks at steps 1-2 -> grad_check.json.

  torchrun --nproc_per_node 2 -m src.msd.pretrain --cond H1 --out output/2026-11-05/msd/H1
  python -m src.msd.pretrain stats --out output/2026-11-05/msd       (teacher statistics, once)
"""
import argparse
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

CONDS = {"H0": dict(adapters=False, multi=False), "H1": dict(adapters=False, multi=True),
         "H2": dict(adapters=True, multi=False), "H3": dict(adapters=True, multi=True),
         # S-series (2026-11-05 pilot 2): frozen encoder + H1 loss, decoder architecture varies
         "A0": dict(adapters=False, multi=True, arch="flat"), "A1": dict(adapters=False, multi=True, arch="hier"),
         "B0": dict(adapters=False, multi=True, arch="oneway"), "B1": dict(adapters=False, multi=True, arch="twoway")}
LAM_AUX, LAM_GLOB = 0.5, 1.0
KEYS = "output/2026-11-04/readout_queue/data_check/image_keys.parquet"


def eval_exclusions():
    """COCO images whose id / byte md5 / pixel md5 / dHash equals a PACO val or test image (data_check keys)."""
    t = pd.read_parquet(KEYS)
    ev = t[t.source.isin(["paco_val", "paco_test"])]
    out = {}
    for src in ("raev2_stream", "raev2_dev"):
        s = t[t.source == src]
        hit = np.zeros(len(s), bool)
        for k in ("image_id", "md5", "pix_md5", "dhash"):
            hit |= s[k].isin(set(ev[k])).values
        out[src] = sorted(int(i) for i in s.image_id.values[hit])
    return out


def stats(out, dev, n_steps=47):
    """Per-depth per-channel teacher mean / std over valid cells of the first 47 stream steps (= 3,008 ROIs)."""
    rois = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    ds = MS.CocoStreamDS(rois, COCO_TRAIN)
    vit = RC.load_vit().to(dev)
    stream = np.load(STREAM)
    acc = {d: [] for d in MS.DEPTHS}
    ex = set(eval_exclusions()["raev2_stream"])
    for row in stream[:n_steps]:
        b = MS.to_dev(ds[[int(i) for i in row if int(i) not in ex]], dev)
        T = MS.teacher_feats(vit, b["crops"])
        m = b["geom"]["valid_w"] > 0
        for d in MS.DEPTHS:
            acc[d].append(T[d][m].double().cpu())
    z = {}
    for d in MS.DEPTHS:
        x = torch.cat(acc[d])
        z[f"mean_{d}"], z[f"std_{d}"] = x.mean(0).float().numpy(), (x.std(0) + 1e-6).float().numpy()
        print(d, len(x), float(x.std(0).mean()), flush=True)
    np.savez(os.path.join(out, "teacher_stats.npz"), **z)


def grad_norm(params):
    g = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(g).norm()) if g else None


def groups(student, dec):
    gr = {"encoder_backbone (frozen, = teacher weights)": list(student.vit.parameters()),
          "decoder_input_adapters A4/A8": list(dec.in_adapt.parameters())}
    if hasattr(dec, "body"):
        gr["decoder_body (CoordMLP)"] = list(dec.body.parameters())
    else:
        skip = {id(p) for m in (dec.in_adapt, dec.out, dec.aux) for p in m.parameters()}
        gr["decoder_trunk (attention / FFN / embeddings)"] = [p for p in dec.parameters() if id(p) not in skip]
        gr["decoder_output_head (zero-init)"] = list(dec.out.parameters())
    gr["aux_heads (removed after training)"] = list(dec.aux.parameters())
    if student.adapters is not None:
        for d in MS.DEPTHS:
            a = student.adapters[str(d)]
            gr[f"encoder_adapter_{d}.fc1"] = list(a.fc1.parameters()) + list(a.ln.parameters())
            gr[f"encoder_adapter_{d}.fc2 (zero-init)"] = list(a.fc2.parameters())
    return gr


def flat(params):
    return torch.cat([p.detach().float().reshape(-1) for p in params]) if params else torch.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="train", choices=["train", "stats"])
    ap.add_argument("--cond", default="H0", choices=list(CONDS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_at", default="4000,8000,12000")
    ap.add_argument("--stats", default="output/2026-11-05/msd/teacher_stats.npz")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dev_max", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rank, world = RC.ddp_setup()
    dev = torch.device("cuda")
    if a.cmd == "stats":
        stats(a.out, dev)
        return
    try:
        train(a, rank, world, dev)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if rank == 0:
            json.dump(dict(status="failed", error=tb), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


def train(a, rank, world, dev):
    cfg = CONDS[a.cond]
    ex = eval_exclusions()
    rois = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    dev_rois = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex["raev2_dev"])]
    dev_ids = sorted(dev_rois.image_id.unique())[: (a.dev_max or None)]
    stream = np.load(STREAM)[: a.steps]
    norm = MS.Norm(a.stats, dev)
    torch.manual_seed(0)
    dec = MS.build_decoder(cfg.get("arch", "coordmlp")).to(dev)
    dec.set_stats(norm.mu[12], norm.sd[12])
    torch.manual_seed(1)
    student = MS.Student(adapters=cfg["adapters"]).to(dev)
    for p in student.vit.parameters():
        assert not p.requires_grad
    trainable = list(dec.parameters()) + (list(student.adapters.parameters()) if student.adapters is not None else [])
    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / a.warmup if s < a.warmup else 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup))))
    gr = groups(student, dec)
    theta0 = {k: flat(v).clone() for k, v in gr.items()}
    ds = MS.CocoStreamDS(rois, COCO_TRAIN)
    sampler = MS.StreamSampler(stream, rank, world, exclude=ex["raev2_stream"])
    dl = DataLoader(ds, sampler=sampler, batch_size=1, num_workers=a.workers, collate_fn=MS.collate_one, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    dev_ds = MS.CocoStreamDS(dev_rois, COCO_TRAIN)
    dev_chunks = [dev_ids[i:i + 8] for i in range(0, len(dev_ids), 8)][rank::world]
    info = dict(cond=a.cond, cfg=cfg, arch=cfg.get("arch", "coordmlp"), lam_aux=LAM_AUX if cfg["multi"] else 0.0, lam_glob=LAM_GLOB if cfg["adapters"] else 0.0,
                steps=a.steps, world=world, excluded=dict(stream=ex["raev2_stream"], dev=ex["raev2_dev"]), n_dev_images=len(dev_ids),
                n_dev_rois=int(dev_rois.image_id.isin(dev_ids).sum()), trainable_params={k: int(sum(p.numel() for p in v if p.requires_grad)) for k, v in gr.items()},
                trainable_total=int(sum(p.numel() for p in trainable)), recipe=dict(lr=a.lr, betas=(0.9, 0.95), wd=a.wd, warmup=a.warmup, clip=1.0))
    if rank == 0:
        json.dump(info, open(os.path.join(a.out, "config.json"), "w"), indent=1)
        print(json.dumps(info), flush=True)

    def forward(b, train=True):
        x, g = b["full"], b["geom"]
        if cfg["adapters"] and train:
            feats, cls = student(x)
        else:
            with torch.no_grad():
                feats, cls = student(x)
        T = MS.teacher_feats(student.vit, b["crops"])                  # frozen original weights, no adapters, no grad
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Z = dec(feats, g)
        Z = Z.float()
        Zn = norm.n(Z, 12)
        aux = {d: v.float() for d, v in dec.aux_pred(Zn).items()}
        w = g["valid_w"].float()
        num, den = {}, None
        for d, p_ in ((12, Zn), (4, aux[4]), (8, aux[8])):
            num[d], den = MS.feat_loss(p_, norm.n(T[d], d), w)
        out = dict(num=num, den=den, feats=feats, cls=cls, x=x, Z=Z, T=T, w=w)
        return out

    @torch.no_grad()
    def dev_eval():
        dec.eval()
        acc = torch.zeros(7, device=dev)            # num4, num8, num12, den, cos12 sum, n tokens, n rois
        for ch in dev_chunks:
            b = MS.to_dev(dev_ds[ch], dev)
            o = forward(b, train=False)
            acc[0] += o["num"][4]; acc[1] += o["num"][8]; acc[2] += o["num"][12]; acc[3] += o["den"]
            m = o["w"] > 0
            acc[4] += torch.nn.functional.cosine_similarity(o["Z"], o["T"][12], dim=-1)[m].sum()
            acc[5] += m.sum(); acc[6] += len(o["w"])
        acc = RC.all_sum(acc)
        dec.train()
        return dict(L4=float(acc[0] / acc[3]), L8=float(acc[1] / acc[3]), L12=float(acc[2] / acc[3]), cos12_raw=float(acc[4] / acc[5]), n_rois=int(acc[6]))

    save_at = {int(x) for x in a.save_at.split(",") if x}
    best = float("inf")
    log_f = open(os.path.join(a.out, "train_log.jsonl"), "a") if rank == 0 else None
    t0, step = time.time(), 0
    gc = {}
    run = torch.zeros(5, device=dev)
    for b in dl:
        b = MS.to_dev(b, dev)
        o = forward(b)
        den_g = RC.all_sum(o["den"].detach().clone())
        lam = LAM_AUX if cfg["multi"] else 0.0
        loss = (o["num"][12] + lam * (o["num"][4] + o["num"][8])) / den_g
        lg = torch.zeros((), device=dev)
        if cfg["adapters"]:
            with torch.no_grad():
                f_o, cls_o = student(o["x"], use_adapters=False)
            lg = MS.global_loss(o["cls"], o["feats"][12].mean(1), cls_o, f_o[12].mean(1)) / b["n_img_global"]
            loss = loss + LAM_GLOB * lg
        if step == 0:   # does the multi-depth (aux) loss reach the spatial decoder body, not only the aux heads?
            body = [p for n_, p in dec.named_parameters() if not n_.startswith("aux.")]
            ga = torch.autograd.grad((o["num"][4] + o["num"][8]) / den_g, body, retain_graph=True, allow_unused=True)
            gs = [x.float().norm() for x in ga if x is not None]
            gc["aux_loss_only_grad_norm_into_decoder_body"] = float(torch.stack(gs).norm()) if gs else 0.0
            gc["cache_checksum_before"] = {d: float(v.double().sum()) for d, v in o["feats"].items()}
            gc["decoder_output_shape"] = list(o["Z"].shape)
            gc["teacher_shapes"] = {d: list(v.shape) for d, v in o["T"].items()}
            gc["n_rois_rank0_step0"] = int(o["Z"].shape[0])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        RC.allreduce_grads(trainable)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if step < 5:
            gc[f"step{step + 1}_grad_norm"] = {k: grad_norm(v) for k, v in gr.items()}
            before = {k: flat(v).clone() for k, v in gr.items()}
        opt.step()
        sched.step()
        if step == 0:
            gc["cache_checksum_after_backward"] = {d: float(v.double().sum()) for d, v in o["feats"].items()}
            gc["cache_unchanged"] = gc["cache_checksum_after_backward"] == gc["cache_checksum_before"]
            gc["cache_requires_grad"] = {d: bool(v.requires_grad) for d, v in o["feats"].items()}
        if step < 5:
            gc[f"step{step + 1}_weight_update_norm"] = {k: float((flat(v) - before[k]).norm()) for k, v in gr.items()}
        run += torch.stack([o["num"][12].detach() / den_g * world, o["num"][4].detach() / den_g * world, o["num"][8].detach() / den_g * world,
                            lg.detach() * world, gnorm.detach()])
        step += 1
        if step == 5 and rank == 0:
            json.dump(gc, open(os.path.join(a.out, "grad_check.json"), "w"), indent=1)
        if step % 50 == 0:
            r = (RC.all_sum(run.clone()) / (50 * world)).tolist()
            if not all(np.isfinite(r)):
                raise FloatingPointError(f"non-finite loss at step {step}: {r}")
            rec = dict(step=step, lr=sched.get_last_lr()[0], L12=r[0], L4=r[1], L8=r[2], L_glob=r[3], grad_norm=r[4], s=round(time.time() - t0))
            run.zero_()
            if rank == 0:
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                if step % 200 == 0:
                    print(json.dumps(rec), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = dev_eval()
            if rank == 0:
                log_f.write(json.dumps(dict(step=step, dev=ev)) + "\n"); log_f.flush()
                print(json.dumps(dict(step=step, dev=ev)), flush=True)
                state = dict(arch=cfg.get("arch", "coordmlp"), decoder=dec.state_dict(), adapters=student.adapters.state_dict() if student.adapters is not None else None,
                             cond=a.cond, step=step, dev=ev)
                if ev["L12"] < best:
                    best = ev["L12"]
                    torch.save(state, os.path.join(a.out, "best_dev.pt"))
                if step in save_at:
                    torch.save(state, os.path.join(a.out, f"ckpt_{step}.pt"))
        if step >= a.steps:
            break
    if rank == 0:
        delta = {k: float((flat(v) - theta0[k]).norm()) for k, v in gr.items()}
        sel = torch.load(os.path.join(a.out, "best_dev.pt"), map_location="cpu")
        json.dump(dict(status="done", steps=step, time_s=round(time.time() - t0), peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                       weight_change_norm_total=delta, selected=dict(step=sel["step"], dev=sel["dev"])),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
