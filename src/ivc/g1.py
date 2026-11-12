"""G1: teach the FULL-IMAGE encoder the crop representation (no PACO labels; PACO train images + boxes only).

  teacher : x_crop -> E_0 (original DINOv3, frozen) -> T (576 crop-grid tokens), and E_0(x_full) for L_glob
  student : x_full -> E_s (trainable, whole encoder) -> full tokens -> ROI read on the C0 crop grid (bilinear, = the
            Bilinear probe input) -> [pre] -> Z
  G1-pool (pre: none)  : L_roi = 1 - cos(pool(Z), pool(T))                       (CLIPSelf-style: one vector per ROI)
  G1-pre  (pre: tiny)  : L_roi = token term (coverage-weighted mean 1 - cos) + pooled term; Z = bilinear + pre(...)
  L_glob = 1/2 [(1 - cos CLS) + (1 - cos mean patch token)] of E_s(x_full) vs E_0(x_full)      (weight `glob`)
pool = coverage-weighted mean over valid C0 cells. Teacher outputs are detached (E_0 never trains).

pre (tiny transformer): h = Linear(768->384)(Z0) + pos_emb(576) ; 2 x [self-attn over the crop grid (keys = valid cells),
cross-attn to the student's full-image tokens of that image (+ full-canvas geometry MLP), MLP x4] ; Z = Z0 + Linear_out(LN h),
Linear_out zero-init -> at step 0, Z = Bilinear. LP-FT: with a predictor, the encoder lr is 0 during epoch 1 (the
predictor alone fits the teacher), then the encoder warms up for one epoch and follows the cosine schedule.
Encoder: AdamW, layer-wise lr decay 0.75 from enc_lr at the top block, wd 0.05 (none on 1-D / tokens); pre: lr 3e-4.
EMA copies of E_s and pre (m: 0.999 -> 1 cosine) are kept; the probe uses the FINAL EMA weights (fixed rule, no labels).
Every epoch on the PACO dev images (held out from probe training): token / pooled cosine to the teacher for the
student and the EMA, and the plain Bilinear (no pre) pooled cosine -> curves.jsonl.

  torchrun --nproc_per_node 2 -m src.ivc.g1 --job configs/ivc_g1/G1pool.yaml --out output/2026-11-06/ivc_g1/G1pool --exclude <json>
"""
import argparse
import copy
import json
import math
import os
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.readout.common import C
from src.ivc.bbdata import BBImageDS, bil_sample
from src.ivc.ft import enc_param_groups, ema_update, run_enc
from src.viz.c0_pca import BACKBONES, load_encoder

D, G = 768, 24


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.sa = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2, self.ca = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln3, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h, mem, kpm_self, kpm_mem):
        x = self.ln1(h)
        h = h + self.sa(x, x, x, key_padding_mask=kpm_self, need_weights=False)[0]
        h = h + self.ca(self.ln2(h), mem, mem, key_padding_mask=kpm_mem, need_weights=False)[0]
        return h + self.mlp(self.ln3(h))


class Pre(nn.Module):
    """Tiny transformer predictor on the crop grid; output = bilinear input + zero-init correction."""

    def __init__(self, d=384, heads=6, depth=2):
        super().__init__()
        self.inp, self.mem_in = nn.Linear(D, d), nn.Linear(D, d)
        self.geo = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, G * G, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, D)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, Z0, full_tok, b):
        h = self.inp(Z0) + self.pos
        mem = self.mem_in(full_tok[b["roi_img"]]) + self.geo(b["fgeo"])
        kpm_s, kpm_m = ~(b["cov"] > 0), ~b["fvalid"][b["roi_img"]]
        for blk in self.blocks:
            h = blk(h, mem, kpm_s, kpm_m)
        return Z0 + self.out(self.ln(h))


def pool(x, w):
    return (x * w[..., None]).sum(1) / w.sum(1, keepdim=True)


def cosd(a, b):
    return 1 - F.cosine_similarity(a, b, dim=-1)


def student_roi(enc, pre, b):
    """-> Z (R, 576, D) on the crop grid, Z0 (bilinear), full tokens, full CLS."""
    ftok, fcls = run_enc(enc, b["full"])
    Z0 = bil_sample(ftok, dict(img_index=b["roi_img"], bil_grid=b["bil_grid"]))
    if pre is None:
        return Z0, Z0, ftok, fcls
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Z = pre(Z0, ftok, b).float()
    return Z, Z0, ftok, fcls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    job = yaml.safe_load(open(a.job))
    rank, world = RC.ddp_setup()
    os.makedirs(a.out, exist_ok=True)
    try:
        run(job, a, rank, world, torch.device("cuda"))
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if rank == 0:
            json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


def run(job, a, rank, world, dev):
    epochs = a.epochs or job.get("epochs", 10)
    use_pre, glob_w = job.get("pre", "none") == "tiny", float(job.get("glob", 0.5))
    backbone = job.get("backbone", "dinov3")
    S, P = BACKBONES[backbone]["S"], BACKBONES[backbone]["P"]
    data = RC.load_data(a.exclude or None, a.limit, splits=("train",))
    dtr, Ytr = data["train"]
    ids = np.array(sorted(dtr.image_id.unique()))
    dev_ids = set(np.random.default_rng(0).choice(ids, int(job["paco_dev_images"]), replace=False).tolist())
    m = dtr.image_id.isin(dev_ids).values
    ddev, Ydev = dtr[m].reset_index(drop=True), Ytr[m]
    dtr, Ytr = dtr[~m].reset_index(drop=True), Ytr[~m]          # labels are carried by the dataset but never used

    torch.manual_seed(1000 * job.get("seed", 0) + 7)
    pre = Pre().to(dev) if use_pre else None
    enc = load_encoder(backbone, dev)
    enc0 = load_encoder(backbone, dev)                          # frozen teacher (original weights)
    for p in enc.parameters():
        p.requires_grad_(True)
    enc.train()
    enc_t = copy.deepcopy(enc).eval()
    pre_t = copy.deepcopy(pre).eval() if use_pre else None
    for p in list(enc_t.parameters()) + (list(pre_t.parameters()) if use_pre else []):
        p.requires_grad_(False)
    wd = job.get("wd", 0.05)
    eg = enc_param_groups(enc, job.get("enc_lr", 5e-5), job.get("llrd", 0.75), wd)
    for g in eg:
        g["is_enc"] = True
    groups = eg
    if use_pre:
        pg = {}
        for n, p in pre.named_parameters():
            k = p.ndim <= 1 or n == "pos"
            pg.setdefault(k, dict(params=[], weight_decay=0.0 if k else wd, base=job.get("lr", 3e-4), lr=job.get("lr", 3e-4), is_enc=False))
            pg[k]["params"].append(p)
        groups = eg + list(pg.values())
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    m0 = float(job.get("ema", 0.999))

    tr_ds = BBImageDS(dtr, Ytr, True, "c0ctx", S, P)
    sampler = RC.RankStepSampler(tr_ds.n_roi, rank, world, seed=job.get("data_seed", 0))
    torch.manual_seed(job.get("seed", 0) * 100 + rank)
    dl = DataLoader(tr_ds, sampler=sampler, batch_size=1, num_workers=a.workers, pin_memory=True,
                    collate_fn=RC.collate_one, persistent_workers=True, prefetch_factor=4)
    steps_per_ep = [len(C.full_epoch_batches(tr_ds.n_roi, e, job.get("data_seed", 0))) for e in range(1, epochs + 1)]
    if a.max_steps:
        steps_per_ep = [min(s, a.max_steps) for s in steps_per_ep]
    total, ep1 = sum(steps_per_ep), steps_per_ep[0]
    lp = ep1 if use_pre else 0                                  # LP-FT: encoder lr 0 for the first epoch with a predictor

    def lr_of(g, step):
        if g["is_enc"]:
            return 0.0 if step < lp else C.lr_at(step - lp, g["base"], ep1, total - lp)
        return C.lr_at(step, g["base"], ep1, total)

    cfg = dict(job=job, args=vars(a), world=world, n_train_rows=len(dtr), n_train_images=int(dtr.image_id.nunique()),
               n_dev_rows=len(ddev), steps_per_epoch=steps_per_ep, total_steps=total, lp_steps=lp,
               n_params_pre=sum(p.numel() for p in pre.parameters()) if use_pre else 0, encoder=BACKBONES[backbone]["name"])
    if rank == 0:
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
        print(json.dumps(cfg, default=str), flush=True)

    def losses(Z, T, w, fcls, ftok, fv, tcls, tftok):
        l_pool = cosd(pool(Z, w), pool(T, w)).mean()
        l_tok = (cosd(Z, T) * w).sum(1).div(w.sum(1)).mean()
        l_glob = 0.5 * (cosd(fcls, tcls).mean() + cosd(pool(ftok, fv), pool(tftok, fv)).mean())
        return l_pool, l_tok, l_glob

    @torch.no_grad()
    def dev_eval(ep):
        ds = BBImageDS(ddev, Ydev, False, "c0ctx", S, P)
        es = RC.EvalSampler(len(ds), 6, rank, world)
        dle = DataLoader(ds, sampler=list(es), batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one)
        acc = torch.zeros(7, device=dev, dtype=torch.float64)
        enc.eval()
        for b in dle:
            b = RC.to_dev(b, dev)
            T, _ = run_enc(enc0, b["x"])
            w = b["cov"].float()
            pT = pool(T, w)
            row = []
            for e, pr in ((enc, pre), (enc_t, pre_t)):
                Z, Z0, _, _ = student_roi(e, pr, b)
                row += [F.cosine_similarity(pool(Z, w), pT, dim=-1).sum(), (F.cosine_similarity(Z, T, dim=-1) * w).sum(1).div(w.sum(1)).sum(),
                        F.cosine_similarity(pool(Z0, w), pT, dim=-1).sum()]
            acc += torch.stack(row + [torch.tensor(float(len(T)), device=dev)]).double()
        enc.train()
        RC.all_sum(acc)
        if rank == 0:
            n = float(acc[6])
            r = dict(epoch=ep, s=round(time.time() - t0), dev_pool_cos=round(float(acc[0]) / n, 4), dev_tok_cos=round(float(acc[1]) / n, 4),
                     dev_bil_pool_cos=round(float(acc[2]) / n, 4), ema_pool_cos=round(float(acc[3]) / n, 4),
                     ema_tok_cos=round(float(acc[4]) / n, 4), ema_bil_pool_cos=round(float(acc[5]) / n, 4))
            with open(os.path.join(a.out, "curves.jsonl"), "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)

    t0, step = time.time(), 0
    dev_eval(0)
    trainable = list(enc.parameters()) + (list(pre.parameters()) if use_pre else [])
    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        run_l = np.zeros(4)
        for k, b in enumerate(dl):
            if a.max_steps and k >= a.max_steps:
                break
            b = RC.to_dev(b, dev)
            for g in opt.param_groups:
                g["lr"] = lr_of(g, step)
            with torch.no_grad():
                T, _ = run_enc(enc0, b["x"])
                tftok, tcls = run_enc(enc0, b["full"])
            Z, Z0, ftok, fcls = student_roi(enc, pre, b)
            w, fv = b["cov"].float(), b["fvalid"].float()
            l_pool, l_tok, l_glob = losses(Z, T, w, fcls, ftok, fv, tcls, tftok)
            l_roi = l_pool + (l_tok if use_pre else 0.0)
            loss = (l_roi + glob_w * l_glob) / world               # per-rank mean -> summed by the all-reduce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            RC.allreduce_grads(trainable)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            if use_pre:
                torch.nn.utils.clip_grad_norm_(pre.parameters(), 1.0)
            opt.step()
            mt = 1 - (1 - m0) * (math.cos(math.pi * step / total) + 1) / 2
            ema_update(enc_t, enc, mt)
            if use_pre:
                ema_update(pre_t, pre, mt)
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"non-finite loss at step {step}")
            run_l += np.array([float(l_pool), float(l_tok), float(l_glob), 1.0])
            step += 1
            if step % 100 == 0 and rank == 0:
                print(json.dumps(dict(step=step, epoch=ep, s=round(time.time() - t0), l_pool=round(run_l[0] / run_l[3], 4),
                                      l_tok=round(run_l[1] / run_l[3], 4), l_glob=round(run_l[2] / run_l[3], 5),
                                      enc_lr_top=round(max(g["lr"] for g in opt.param_groups if g["is_enc"]), 8),
                                      mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))), flush=True)
        dev_eval(ep)
        if rank == 0:
            torch.save(dict(enc=enc_t.state_dict(), pre=pre_t.state_dict() if use_pre else None, epoch=ep, kind="ema"),
                       os.path.join(a.out, "ema_last.pt"))
            torch.save(dict(enc=enc.state_dict(), pre=pre.state_dict() if use_pre else None, epoch=ep, kind="student"),
                       os.path.join(a.out, "student_last.pt"))
    if rank == 0:
        json.dump(dict(status="done", epochs=epochs, steps=step, time_s=round(time.time() - t0), time=time.strftime("%F %T")),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
