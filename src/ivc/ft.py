"""Oracle C0 crop + ENCODER fine-tuning with an EMA teacher (PACO train, dev-selected epoch, VAL only; test never read).

  F0  keep 0,  ctx false : x_crop -> E_s (trainable) -> R2 -> attributes                       (plain crop FT)
  F1  keep 1,  ctx false : F0 + preservation  L_keep = E_s(.) ~ E_t(.) on the crop AND the full image
  F2  keep 1,  ctx true  : F1 + the R2 classifier also reads the full-image tokens E_s(x_full) as context
E_t = EMA(E_s) (no grad), momentum m(t) = 1 - (1 - m0)(cos(pi t / T) + 1) / 2 (m0 -> 1, DINO rule); the classifier has an
EMA copy too. Every epoch both the STUDENT (E_s + R2) and the EMA model (E_t + R2_ema) are scored on PACO dev and val
(<out>/r2 and <out>/r2_ema, same file layout as the IVC probes -> src.ivc.compare_bb). Epoch = max dev M0 Overall.

L_keep (cosine distance, final-norm outputs, teacher detached): crop = coverage-weighted mean over valid crop tokens
of (1 - cos) + (1 - cos) of CLS; full = mean over content tokens + CLS; L_keep = (crop + full) / 2 / 2.
Loss = masked BCE / known labels + keep * L_keep. F0 keeps the EMA copy (scored, no loss) so F1 - F0 isolates the loss.

Data / sampler / flips / dev split / reader / reader lr = src.ivc.train (IVC protocol). Encoder: AdamW wd 0.05 (no wd
on 1-D / tokens), layer-wise lr decay 0.75 from enc_lr 5e-5 at the top block (= the 10-28 paco_crop_finetune recipe),
same warmup (1 epoch) + cosine schedule, grad clip 1.0 for encoder and reader separately, bf16 autocast.

  CUDA_VISIBLE_DEVICES=0 python -m src.ivc.ft --job configs/ivc_ft/F0.yaml --out output/2026-11-06/ivc_ft/F0 --exclude <json>
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
from src.readout.models import Reader, n_params, param_groups
from src.ivc.bbdata import BBImageDS
from src.viz.c0_pca import BACKBONES, load_encoder


class CtxReader(Reader):
    """R2 over [crop tokens ; full-image context tokens] (+ token-type embedding), one query, same blocks / heads."""

    def __init__(self):
        super().__init__("r2", layers=(12,))
        self.type_emb = nn.Parameter(torch.zeros(2, 1, 1, 768))
        nn.init.trunc_normal_(self.type_emb, std=0.02)

    def forward(self, feats, b, ctx_mode="normal"):
        lv, q = b["lv"], self._q0(b)
        kv_c = self._kv(feats[12][b["img_index"]], b["geo"]) + self.type_emb[0]
        kv_f = self._kv(feats["full"][b["roi_img"]], b["fgeo"]) + self.type_emb[1]
        mask = torch.cat([b["cov"] > 0, b["fvalid"][b["roi_img"]]], 1)
        q = self._run(self.blocks, q, torch.cat([kv_c, kv_f], 1), mask)
        return dict(logits=self.heads(self.ln_out(q[:, 0]), lv), extra={})


def enc_param_groups(vit, base, decay, wd):
    nb = len(vit.blocks)
    g = {}
    for n, p in vit.named_parameters():
        if n.startswith("blocks."):
            lid = int(n.split(".")[1]) + 1
        elif n.startswith(("norm", "fc_norm", "head")):
            lid = nb + 1
        else:
            lid = 0                                                     # patch_embed, cls / reg tokens, pos
        lr = base * decay ** (nb + 1 - lid)
        no_wd = p.ndim <= 1 or "token" in n or "pos_embed" in n
        k = (lid, no_wd)
        g.setdefault(k, dict(params=[], weight_decay=0.0 if no_wd else wd, base=lr, lr=lr))
        g[k]["params"].append(p)
    return list(g.values())


def run_enc(vit, x, chunk=96):
    out = []
    for i in range(0, len(x), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(vit.forward_features(x[i:i + chunk]))
    t = torch.cat(out).float()
    return t[:, vit.num_prefix_tokens:], t[:, 0]


def cosd(a, b):
    return 1 - F.cosine_similarity(a, b, dim=-1)


@torch.no_grad()
def ema_update(dst, src, m):
    for pd, ps in zip(dst.parameters(), src.parameters()):
        pd.mul_(m).add_(ps.detach(), alpha=1 - m)
    for bd, bs in zip(dst.buffers(), src.buffers()):
        bd.copy_(bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
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
    epochs = a.epochs or job.get("epochs", 10)
    keep, ctx = float(job.get("keep", 0.0)), bool(job.get("ctx", False))
    use_full = keep > 0 or ctx
    mode = "c0ctx" if use_full else "c0"
    backbone = job.get("backbone", "dinov3")
    S, P = BACKBONES[backbone]["S"], BACKBONES[backbone]["P"]
    data = RC.load_data(a.exclude or None, a.limit)
    Ytr_by = RC.ytr_by_level(data)
    dtr, Ytr = data["train"]
    dva, Yva = data["val"]
    ids = np.array(sorted(dtr.image_id.unique()))
    dev_ids = set(np.random.default_rng(0).choice(ids, int(job["paco_dev_images"]), replace=False).tolist())
    m = dtr.image_id.isin(dev_ids).values
    ddev, Ydev = dtr[m].reset_index(drop=True), Ytr[m]
    dtr, Ytr = dtr[~m].reset_index(drop=True), Ytr[~m]

    torch.manual_seed(1000 * job.get("seed", 0) + 7)
    reader = (CtxReader() if ctx else Reader("r2", layers=(12,))).to(dev)
    enc = load_encoder(backbone, dev)
    for p in enc.parameters():
        p.requires_grad_(True)
    enc.train()
    if job.get("grad_ckpt", False):
        enc.set_grad_checkpointing(True)
    enc_t, reader_t = copy.deepcopy(enc).eval(), copy.deepcopy(reader).eval()
    for p in list(enc_t.parameters()) + list(reader_t.parameters()):
        p.requires_grad_(False)
    lr, head_lr, wd = job.get("lr", 3e-4), job.get("head_lr", 1e-3), job.get("wd", 0.05)
    groups = enc_param_groups(enc, job.get("enc_lr", 5e-5), job.get("llrd", 0.75), wd) + param_groups(reader, lr, head_lr, wd)
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))
    m0 = float(job.get("ema", 0.999))

    tr_ds = BBImageDS(dtr, Ytr, True, mode, S, P)
    sampler = RC.RankStepSampler(tr_ds.n_roi, rank, world, seed=job.get("data_seed", 0))
    torch.manual_seed(job.get("seed", 0) * 100 + rank)
    dl = DataLoader(tr_ds, sampler=sampler, batch_size=1, num_workers=a.workers, pin_memory=True,
                    collate_fn=RC.collate_one, persistent_workers=True, prefetch_factor=4)
    steps_per_ep = [len(C.full_epoch_batches(tr_ds.n_roi, e, job.get("data_seed", 0))) for e in range(1, epochs + 1)]
    if a.max_steps:
        steps_per_ep = [min(s, a.max_steps) for s in steps_per_ep]
    total, warm = sum(steps_per_ep), steps_per_ep[0]
    names = ("r2", "r2_ema")
    cfg = dict(job=job, args=vars(a), world=world, mode=mode, n_train_rows=len(dtr), n_train_images=int(dtr.image_id.nunique()),
               n_val_rows=len(dva), n_dev_rows=len(ddev), steps_per_epoch=steps_per_ep, total_steps=total,
               n_params_reader=n_params(reader), n_params_encoder=n_params(enc), encoder=BACKBONES[backbone]["name"], S=S, P=P)
    if rank == 0:
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
        for n in names:
            os.makedirs(os.path.join(a.out, n), exist_ok=True)
    if rank == 0:
        print(json.dumps(cfg, default=str), flush=True)

    def feats_of(e, b):
        tok, cls = run_enc(e, b["x"])
        f = {12: tok, "cls": cls}
        if use_full:
            ftok, fcls = run_enc(e, b["full"])
            f.update(full=ftok, full_cls=fcls)
        return f

    @torch.no_grad()
    def score(df, Y):
        ds = BBImageDS(df, Y, False, mode, S, P)
        es = RC.EvalSampler(len(ds), 6, rank, world)
        dle = DataLoader(ds, sampler=list(es), batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
        Sc = {n: torch.zeros(len(df), 59, device=dev) for n in names}
        enc.eval()
        reader.eval()
        for b in dle:
            b = RC.to_dev(b, dev)
            for n, (e, r) in (("r2", (enc, reader)), ("r2_ema", (enc_t, reader_t))):
                f = feats_of(e, b)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Sc[n][b["rows"]] = r(f, b)["logits"].float()
        enc.train()
        reader.train()
        return {n: RC.all_sum(v).cpu().numpy() for n, v in Sc.items()}

    t0, step = time.time(), 0

    def evaluate(ep, stats):
        Sv, Sd = score(dva, Yva), score(ddev, Ydev)
        if rank != 0:
            return
        line = {}
        for n in names:
            rd_, rv = RC.m2_eval(Sd[n], ddev, Ydev, Ytr_by), RC.m2_eval(Sv[n], dva, Yva, Ytr_by)
            with open(os.path.join(a.out, n, "curves_dev.jsonl"), "a") as fh:
                fh.write(json.dumps(dict(epoch=ep, dev=dict(loss=rd_["loss"], M0=rd_["M0"], M2=rd_["M2"]))) + "\n")
            with open(os.path.join(a.out, n, "curves.jsonl"), "a") as fh:
                fh.write(json.dumps(dict(epoch=ep, step=step, elapsed_s=round(time.time() - t0, 1), **stats,
                                         val=dict(loss=rv["loss"], M0=rv["M0"], M2=rv["M2"]))) + "\n")
            np.save(os.path.join(a.out, n, f"scores_val_ep{ep}.npy"), Sv[n])
            np.save(os.path.join(a.out, n, f"scores_dev_ep{ep}.npy"), Sd[n])
            line[n] = dict(dev_M0=round(rd_["M0"]["overall"], 2), val_M0=round(rv["M0"]["overall"], 2))
        print(json.dumps(dict(epoch=ep, s=round(time.time() - t0), **line)), flush=True)

    evaluate(0, {})
    trainable = [p for p in enc.parameters()] + [p for p in reader.parameters()]
    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        acc = dict(task=0.0, keep=0.0, n=0)
        for k, b in enumerate(dl):
            if a.max_steps and k >= a.max_steps:
                break
            b = RC.to_dev(b, dev)
            Y = b["Y"]
            nt = RC.all_sum(torch.tensor(float((Y >= 0).sum()), device=dev))
            for g in opt.param_groups:
                g["lr"] = C.lr_at(step, g["base"], warm, total)
            f = feats_of(enc, b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = reader(f, b)["logits"].float()
            ls, _ = C.masked_bce_sum(logits, Y)
            loss = ls / nt
            lk = torch.zeros((), device=dev)
            if keep > 0:
                with torch.no_grad():
                    ft = feats_of(enc_t, b)
                w = b["cov"].float()
                crop = ((cosd(f[12], ft[12]) * w).sum(1) / w.sum(1)).mean() + cosd(f["cls"], ft["cls"]).mean()
                fv = b["fvalid"].float()
                full = ((cosd(f["full"], ft["full"]) * fv).sum(1) / fv.sum(1)).mean() + cosd(f["full_cls"], ft["full_cls"]).mean()
                lk = (crop + full) / 4
                loss = loss + keep * lk / world                 # per-rank mean -> summed over ranks by all-reduce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            RC.allreduce_grads(trainable)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            mt = 1 - (1 - m0) * (math.cos(math.pi * step / total) + 1) / 2
            ema_update(enc_t, enc, mt)
            ema_update(reader_t, reader, mt)
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"non-finite loss at step {step}")
            acc["task"] += float(ls) / max(float(nt), 1)
            acc["keep"] += float(lk)
            acc["n"] += 1
            step += 1
            if step % 100 == 0 and rank == 0:
                print(json.dumps(dict(step=step, epoch=ep, s=round(time.time() - t0), task=round(acc["task"] / acc["n"], 4),
                                      keep=round(acc["keep"] / acc["n"], 5), ema_m=round(mt, 6),
                                      mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))), flush=True)
        evaluate(ep, dict(train_task=acc["task"] / max(acc["n"], 1), train_keep=acc["keep"] / max(acc["n"], 1)))
        if rank == 0:
            torch.save(dict(enc=enc.state_dict(), reader=reader.state_dict(), enc_ema=enc_t.state_dict(),
                            reader_ema=reader_t.state_dict(), epoch=ep), os.path.join(a.out, "last.pt"))
            for n, (e, r) in (("r2", (enc, reader)), ("r2_ema", (enc_t, reader_t))):
                torch.save(dict(reader=r.state_dict(), epoch=ep), os.path.join(a.out, n, f"ep{ep}.pt"))
                torch.save(e.state_dict(), os.path.join(a.out, n, f"enc_ep{ep}.pt"))
    if rank == 0:
        json.dump(dict(status="done", epochs=epochs, steps=step, time_s=round(time.time() - t0), time=time.strftime("%F %T")),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
