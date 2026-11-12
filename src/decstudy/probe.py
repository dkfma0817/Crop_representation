"""Decoder-study R2 probe (2026-11-07) = src.g1clip.probe (copied; not modified) with
  * feature paths from src.decstudy.paths.load_path(job): kind enc (G1-cross encoder, full-canvas geometry), dec_existing
    (G1-cross encoder + its jointly trained decoder), bilinear (B0), residual (D1 / L1 / L2 EMA decoder), c0 (original
    CLIP on the GT crop) -- every crop-grid path uses the C0 crop-grid R2 geometry (PacoDS 'dec')
  * --stop_epoch K: the 10-epoch schedule (warmup / cosine over 10 epochs) is unchanged; training stops after epoch K and
    epoch K is THE checkpoint (fixed in advance; no epoch selection). selected_epoch.json records the fixed rule.
Everything else (data, dev split, reader init, optimiser, batches, masked BCE, DDP rule) is the g1clip / maskemb probe.

Original docstring follows.
G1-CLIP probe (2026-11-07) = src.maskemb.probe (copied; that file is not modified) with the encoder taken from a G1
checkpoint (job.ckpt, key 'enc' = final EMA, loaded into the original CLIP-L architecture) and a third input mode:
  mode 'full' : G1 encoder full-image tokens, R2 full-canvas geometry (= G1-pool / G1-cross Encoder Only)
  mode 'dec'  : the SAME G1-cross checkpoint's encoder + its ROI decoder (key 'dec', src.g1clip.model.ROIDecoderD,
                bf16 autocast as in the original G1 probe) -> 576 crop-grid tokens per ROI, R2 reads them with the C0
                crop-grid geometry (src.g1x.data.PacoDS 'dec', = src.g1x.probe mode 'dec')
Everything else below is the maskemb probe unchanged.

Original docstring follows.
R2 + attribute-head probe on a FROZEN CLIP ViT-L/14@336 or MaskEmbed encoder (2026-11-07) = src.g1x.probe protocol
(src.ivc.train_bb lineage) with only these changes:
  * encoder = src.maskemb.common.build_encoder(job.encoder) (frozen, eval), tokens = final-norm patch tokens (1024-d),
    encoder precision = job.precision ('tf32' | 'fp32' | 'fp16'), fixed BEFORE training by the smoke check (same for all jobs)
  * canvases: S = 336, P = 14 (24 x 24 grid, = the existing grid), OpenAI CLIP mean / std (common.set_clip_norm)
  * reader: R2 only (src.maskemb.reader.R2, width D = 1024; = the existing R2 at D = 768 bit for bit); R0 not trained
  * PACO val is NOT scored during training (no per-epoch val curve, no best_val.pt): the official val is read only by
    src.maskemb.infer AFTER the epoch is fixed on the train-internal dev (max dev M0 Overall over epochs 1..10)
Unchanged: data (PACO train minus data_check exclusions, dev = 1,000 train images rng 0), input modes ('full' = PacoDS
full-canvas R2 geometry, cov = box ∩ cell, no interpolation; 'c0' = BBImageDS GT tight crop -> letterbox, cov = crop
content coverage), image packs of ~128 ROIs (C.full_epoch_batches, data seed 0), 10 epochs, hflip p = 0.5, masked BCE
over KNOWN labels only (Y = -1 unknown type -> masked, never a negative) / #known in the global step, AdamW (reader lr
3e-4, head lr 1e-3, wd 0.05, no wd on 1-D / query), 1 epoch warmup + cosine, grad clip 1.0, reader init seed 7, DDP 2
GPUs with one fused all-reduce, bf16 autocast for the reader, per-epoch checkpoints, frozen-encoder checksum.

  torchrun --nproc_per_node 2 -m src.maskemb.probe --job <yaml> --out <dir>
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.readout.common import C
from src.readout.models import param_groups
from src.maskemb import common as MC
from src.maskemb.reader import build as build_r2, n_params
from src.g1clip.model import ROIDecoderD
from src.ivc.bbdata import BBImageDS
from src.g1x.data import PacoDS
from src.decstudy.paths import load_path

EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
EVAL_BS = {"c0": 6, "full": 12, "dec": 12}


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def make_ds(df, Y, flip, mode):
    return PacoDS(df, Y, flip, mode, MC.S, MC.P) if mode in ("full", "dec") else BBImageDS(df, Y, flip, "c0", MC.S, MC.P)


def load_g1(ckpt, mode, dev):
    """-> (vit, dec or None, info). Encoder Only: the decoder is NOT built / loaded."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["kind"] == "ema" and ck["arch"] == MC.CLIP_ARCH
    vit, _ = MC.build_encoder("clip", dev)
    vit.load_state_dict(ck["enc"])
    vit.eval()
    dec = None
    if mode == "dec":
        assert ck["dec"] is not None, "dec mode needs a G1-cross checkpoint"
        dec = ROIDecoderD(MC.S, MC.P, MC.DIM).to(dev)
        dec.load_state_dict(ck["dec"])
        dec.eval()
    for p in list(vit.parameters()) + (list(dec.parameters()) if dec is not None else []):
        p.requires_grad_(False)
    return vit, dec, dict(ckpt=ckpt, ckpt_sha256=MC.sha256(ckpt), cond=ck["cond"], step=ck["step"], kind=ck["kind"], decoder_loaded=dec is not None)


def set_precision(prec):
    tf = prec == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf
    torch.backends.cudnn.allow_tf32 = tf


def encoder_tokens(vit, x, prec):
    if prec == "fp16":
        return MC.tokens(vit, x, amp=True, dtype=torch.float16)
    return MC.tokens(vit, x, amp=False)          # tf32 / fp32: fp32 activations (matmul mode set by set_precision)


def split_dev(dtr, Ytr, n_dev):
    ids = np.array(sorted(dtr.image_id.unique()))
    dev_ids = set(np.random.default_rng(0).choice(ids, int(n_dev), replace=False).tolist())
    m = dtr.image_id.isin(dev_ids).values
    return dtr[~m].reset_index(drop=True), Ytr[~m], dtr[m].reset_index(drop=True), Ytr[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images per split")
    ap.add_argument("--epochs", type=int, default=0, help="override (smoke test only)")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: steps per epoch")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stop_epoch", type=int, default=6)
    a = ap.parse_args()
    job = yaml.safe_load(open(a.job))
    rank, world = RC.ddp_setup()
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    try:
        run(job, a, rank, world, dev)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if rank == 0:
            json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


def run(job, a, rank, world, dev):
    assert not os.path.exists(os.path.join(a.out, "done.json")), "run exists (never overwritten)"
    CP = MC.set_clip_norm()
    assert torch.allclose(CP.MEAN_T.flatten(), torch.tensor(MC.CLIP_MEAN)) and torch.allclose(CP.STD_T.flatten(), torch.tensor(MC.CLIP_STD))
    epochs = a.epochs or job.get("epochs", 10)
    mode, prec = job["mode"], job["precision"]
    if prec == "auto":                                # fixed by the smoke-step precision check (src.maskemb.prec_check)
        prec = json.load(open(MC.run_path("sanity", "precision.json")))["choice"]
    assert mode in ("full", "dec", "c0") and prec in ("tf32", "fp32", "fp16")
    set_precision(prec)
    lr, head_lr, wd = job.get("lr", 3e-4), job.get("head_lr", 1e-3), job.get("wd", 0.05)
    data = RC.load_data(EXCLUDE, a.limit, splits=("train",))
    Ytr_by = RC.ytr_by_level(data)                    # unit inclusion from the FULL train labels (= existing probes)
    dtr, Ytr = data["train"]
    assert set(np.unique(Ytr)) <= {-1, 0, 1}
    n_dev = job.get("paco_dev_images", 1000) if not a.limit else max(2, a.limit // 10)
    dtr, Ytr, ddev, Ydev = split_dev(dtr, Ytr, n_dev)
    reader = build_r2(MC.DIM, job.get("seed", 0)).to(dev)
    opt = torch.optim.AdamW(param_groups(reader, lr, head_lr, wd), lr=lr, betas=(0.9, 0.999))
    vit, dec, einfo = load_path(job, dev)
    assert einfo["mode"] == mode
    frozen = list(vit.parameters()) + (list(dec.parameters()) if dec is not None else [])
    sum0 = float(sum(p.double().sum() for p in frozen))
    tr_ds = make_ds(dtr, Ytr, True, mode)
    sampler = RC.RankStepSampler(tr_ds.n_roi, rank, world, seed=job.get("data_seed", 0))
    prep = lambda b: RC.to_dev(b, dev)  # noqa: E731
    torch.manual_seed(job.get("seed", 0) * 100 + rank)
    dl = DataLoader(tr_ds, batch_sampler=None, sampler=sampler, batch_size=1, num_workers=a.workers, pin_memory=True,
                    collate_fn=RC.collate_one, persistent_workers=True, prefetch_factor=4)
    steps_per_ep = [len(C.full_epoch_batches(tr_ds.n_roi, e, job.get("data_seed", 0))) for e in range(1, epochs + 1)]
    if a.max_steps:
        steps_per_ep = [min(s, a.max_steps) for s in steps_per_ep]
    total, warm = sum(steps_per_ep), steps_per_ep[0]
    cfg = dict(job=job, args=vars(a), world=world, n_train_rows=len(dtr), n_train_images=int(dtr.image_id.nunique()),
               n_dev_rows=len(ddev), n_dev_images=int(ddev.image_id.nunique()), steps_per_epoch=steps_per_ep, total_steps=total,
               n_params=dict(total=n_params(reader), reader_only=n_params(reader) - n_params(reader.heads), heads=n_params(reader.heads)),
               encoder=einfo, encoder_params=sum(p.numel() for p in frozen), S=MC.S, P=MC.P, D=MC.DIM, precision=prec,
               tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32),
               norm=dict(mean=MC.CLIP_MEAN, std=MC.CLIP_STD), torch=torch.__version__, val_scored_during_training=False)
    if rank == 0:
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
        os.makedirs(os.path.join(a.out, "r2"), exist_ok=True)
    log(rank, json.dumps(cfg, default=str))

    @torch.no_grad()
    def encode(b):
        tok = encoder_tokens(vit, b["x"], prec)
        if dec is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = dec(tok, b["img_index"], b["box_c"], b["qrel"], b["fvalid"]).float()
            b["img_index"] = torch.arange(len(tok), device=tok.device)
        return {12: tok}

    @torch.no_grad()
    def score(df, Y):
        ds = make_ds(df, Y, False, mode)
        es = RC.EvalSampler(len(ds), EVAL_BS[mode], rank, world)
        dle = DataLoader(ds, sampler=[list(ch) for ch in es], batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
        Sc = torch.zeros(len(df), 59, device=dev)
        reader.eval()
        for b in dle:
            b = prep(b)
            f = encode(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Sc[b["rows"]] = reader(f, b)["logits"].float()
        reader.train()
        return RC.all_sum(Sc).cpu().numpy()

    procs = []

    def evaluate(ep, stats):
        """GPU scoring of the dev rows here (unchanged call order); the CPU metrics (RC.m2_eval) run in a separate
        process (src.maskemb.dev_eval) so the GPUs do not idle while rank 0 computes M0 / M2."""
        Sd = score(ddev, Ydev)
        if rank != 0:
            return
        np.save(os.path.join(a.out, "r2", f"scores_dev_ep{ep}.npy"), Sd)
        json.dump(dict(epoch=ep, step=step, elapsed_s=round(time.time() - t0, 1), **stats), open(os.path.join(a.out, "r2", f"stats_ep{ep}.json"), "w"))
        env = dict(os.environ, OMP_NUM_THREADS="8", CUDA_VISIBLE_DEVICES="")
        cmd = [sys.executable, "-m", "src.maskemb.dev_eval", "--out", a.out, "--ep", str(ep), "--limit", str(a.limit), "--n_dev", str(n_dev)]
        procs.append((ep, subprocess.Popen(cmd, env=env, stdout=open(os.path.join(a.out, "r2", f"dev_ep{ep}.log"), "w"), stderr=subprocess.STDOUT)))

    step, t0 = 0, time.time()
    evaluate(0, {})
    params = list(reader.parameters())
    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        run_l, bad, run_n = torch.zeros((), device=dev), torch.zeros((), device=dev), 0.0
        for k, b in enumerate(dl):
            if a.max_steps and k >= a.max_steps:
                break
            b = prep(b)
            f = encode(b)
            Y = b["Y"]
            nt = b["nt_global"].float() if "nt_global" in b else RC.all_sum(torch.tensor(float((Y >= 0).sum()), device=dev))
            for g in opt.param_groups:
                g["lr"] = C.lr_at(step, g["base"], warm, total)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = reader(f, b)
            ls, _ = C.masked_bce_sum(o["logits"].float(), Y)
            loss = ls / nt
            bad = bad + (~torch.isfinite(loss.detach())).float()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            RC.allreduce_grads(params)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run_l += ls.detach()
            run_n += (b["nt_cpu"] if "nt_cpu" in b else float(nt)) / world
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
            torch.save(dict(reader=reader.state_dict(), epoch=ep, D=MC.DIM, encoder=job.get("ckpt", "original_clip"), kind=job["kind"],
                            dec_ckpt=job.get("dec_ckpt"), mode=mode, precision=prec), os.path.join(a.out, "r2", f"ep{ep}.pt"))
        if ep == a.stop_epoch:
            break
    if rank == 0:
        sum1 = float(sum(p.double().sum() for p in frozen))
        json.dump(dict(requires_grad_any=any(p.requires_grad for p in frozen), grads_present=any(p.grad is not None for p in frozen),
                       checksum_before=sum0, checksum_after=sum1, unchanged=sum1 == sum0, encoder=job.get("ckpt"), info=einfo),
                  open(os.path.join(a.out, "frozen_check.json"), "w"), indent=1)
        assert sum1 == sum0, "frozen encoder changed during probe training"
        for ep_, p in procs:
            assert p.wait() == 0, f"dev_eval epoch {ep_} failed (r2/dev_ep{ep_}.log)"
        dv = []
        with open(os.path.join(a.out, "r2", "curves_dev.jsonl"), "w") as fh:
            for ep_, _ in procs:
                rec = dict(**json.load(open(os.path.join(a.out, "r2", f"stats_ep{ep_}.json"))), **{k: v for k, v in json.load(open(os.path.join(a.out, "r2", f"dev_ep{ep_}.json"))).items() if k != "epoch"})
                fh.write(json.dumps(rec) + "\n")
                dv.append(rec)
                print(json.dumps(dict(epoch=ep_, dev_M0=round(rec["dev"]["M0"]["overall"], 2), dev_M2=round(rec["dev"]["M2"]["overall"], 2))), flush=True)
        best = [r for r in dv if r["epoch"] == a.stop_epoch][0]
        json.dump(dict(rule=f"FIXED in advance: epoch {a.stop_epoch} of the {epochs}-epoch schedule (no selection)", epoch=best["epoch"],
                       dev_M0=best["dev"]["M0"]["overall"], dev_M2=best["dev"]["M2"]["overall"]),
                  open(os.path.join(a.out, "r2", "selected_epoch.json"), "w"), indent=1)
        json.dump(dict(status="done", epochs=epochs, stopped_after_epoch=a.stop_epoch, steps=step, time_s=round(time.time() - t0), time=time.strftime("%F %T")),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
    log(rank, "DONE")


if __name__ == "__main__":
    main()
