#!/usr/bin/env python
"""Attribute-aware background training of the DINOv3 full-image encoder on one GPU. Same contract as src.crop_aware_pretrain.train (run lock, heartbeat, last_train_exit.json, SIGTERM = safe stop).

  student   DINOv3-B/16 @384, last 4 blocks + final norm train; the full image is encoded once; a learning-only R2 head reads ROIs (R2 full-canvas geometry, same as the eval probe)
  A         L = lambda_gt L_gt + lambda_kd L_kd + lambda_preserve L_preserve   (dinov3_attribute_distill)
  A control lambda_kd = 0, same data and labels                                (dinov3_attribute_distill_supervised)
  B         A + lambda_rank L_rank                                             (dinov3_discriminative_region)

  L_gt       masked BCE over known labels / #known labels of the step (unknown = ignored)
  L_kd       BCE with logits vs sigmoid(teacher crop logits), mean over all ROI-attribute entries (= E12 form)
  L_rank     softplus(s_j,a - s_i,a) over ROI pairs of the same level and category within the step with Y_i,a = 1 and Y_j,a = 0 explicit (unknown never used); mean over triplets
  L_preserve as in crop_aware_pretrain: 1 - cos vs the frozen encoder for CLS, mean patch and the patch tokens outside every ROI of the image (equal weights)

Eval: exported EMA encoders go through src.crop_aware_pretrain.evals (R2 retrained from scratch, official AP). The learning-only head is never used for evaluation.
"""
import argparse
import fcntl
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("CUDA_CACHE_PATH", os.path.join(_ROOT, "cache", "cuda_jit"))
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")
os.environ.setdefault("CUDA_CACHE_DISABLE", "0")

import json       # noqa: E402
import math       # noqa: E402
import random     # noqa: E402
import signal     # noqa: E402
import time       # noqa: E402
import traceback  # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import src.readout.common as RC                   # noqa: E402
from src.bgjobs import data as DT                 # noqa: E402
from src.bgjobs import teacher as TE              # noqa: E402
from src.crop_aware_pretrain import ckpt as CK    # noqa: E402
from src.crop_aware_pretrain import config as CF  # noqa: E402
from src.crop_aware_pretrain import gpu as GP     # noqa: E402
from src.crop_aware_pretrain import model as MD   # noqa: E402
from src.crop_aware_pretrain.data import worker_init  # noqa: E402
from src.crop_aware_pretrain.train import ema_m, lr_factor  # noqa: E402
from src.ivc.ft import run_enc                    # noqa: E402
from src.readout.common import C                  # noqa: E402
from src.readout.models import Reader, param_groups as reader_groups  # noqa: E402
from src.viz.c0_pca import BACKBONES              # noqa: E402

EX_LOCKED, EX_BUSY = 75, 76


class Stop:
    flag, signum = False, None

    @classmethod
    def handler(cls, sig, _f):
        cls.flag, cls.signum = True, sig


def write_exit(ctl, reason, step, extra=None):
    CK.atomic_json(dict(reason=reason, step=step, time=time.strftime("%F %T"), launch_id=os.environ.get("BGJOB_LAUNCH_ID", ""), **(extra or {})), os.path.join(ctl, "last_train_exit.json"))


def cosd(a, b):
    return 1 - F.cosine_similarity(a, b, dim=-1)


def pool(x, w):
    return (x * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)


def losses(b, ftok, fcls, otok, ocls, logits, tl, lc, lam_rank):
    fv = b["fvalid"].float()
    ri = b["img_index"]
    cover = torch.zeros(fv.shape, device=fv.device)
    cover.index_add_(0, ri, (b["cov"] > 0).float())
    outside = ((cover == 0) & (fv > 0)).float()
    n_img = float(b["x"].shape[0])
    l_cls = cosd(fcls, ocls).sum() / n_img
    l_mean = cosd(pool(ftok, fv), pool(otok, fv)).sum() / n_img
    l_tok = ((cosd(ftok, otok) * outside).sum(1) / outside.sum(1).clamp_min(1.0)).sum() / n_img
    l_pres = (l_cls + l_mean + l_tok) / 3
    gsum, _ = C.masked_bce_sum(logits, b["Y"])
    l_gt = gsum / float(b["nt"])
    l_kd = F.binary_cross_entropy_with_logits(logits, torch.sigmoid(tl), reduction="sum") / (float(b["n_rows"]) * 59)
    Y, cat, lv = b["Y"], b["cat"], b["lv"]
    same = (cat[:, None] == cat[None, :]) & (lv[:, None] == lv[None, :])
    same.fill_diagonal_(False)
    m = (same[:, :, None] & (Y == 1)[:, None, :] & (Y == 0)[None, :, :]).float()
    d = logits[None, :, :] - logits[:, None, :]
    pairs = m.sum()
    l_rank = (F.softplus(d) * m).sum() / pairs.clamp(min=1.0)
    loss = lc["lambda_gt"] * l_gt + lc["lambda_kd"] * l_kd + lc["lambda_preserve"] * l_pres + lam_rank * l_rank
    return dict(loss=loss, l_gt=l_gt, l_kd=l_kd, l_preserve=l_pres, l_cls=l_cls, l_mean=l_mean, l_tok=l_tok, l_rank=l_rank), pairs.detach()


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--stop-after-steps", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--managed", action="store_true")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--bg-class", default="", help="process tag; other queues exempt `crop_aware_pretrain` processes from their GPU gates and stop them through `ctl reserve`, which this job honours")
    return ap.parse_args()


def main():
    a = parse()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    cfg = CF.load(a.config, a.set)
    if a.save_every:
        cfg["ckpt"]["save_every"] = a.save_every
    D = CF.dirs(cfg)
    for k in D.values():
        os.makedirs(k, exist_ok=True)
    signal.signal(signal.SIGTERM, Stop.handler)
    signal.signal(signal.SIGINT, Stop.handler)
    if not a.managed and not a.allow_busy_gpu:
        st = GP.query()
        for i in [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]:
            bad, why = GP.busy_external(i, st)
            if bad:
                print(f"GPU {i} is in use: {why}. Not starting.", flush=True)
                sys.exit(EX_BUSY)
    lock = open(os.path.join(D["control"], "train.lock"), "w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("run lock is held: another copy of this job is running", flush=True)
        sys.exit(EX_LOCKED)
    try:
        sys.exit(train(a, cfg, D))
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        write_exit(D["control"], "failed", -1, dict(error=tb))
        raise


def train(a, cfg, D):
    tc, lc = cfg["train"], dict(cfg["loss"])
    seed, total = int(cfg["seed"]), int(tc["max_steps"])
    dev = torch.device("cuda")
    S, P = BACKBONES[cfg["backbone"]]["S"], BACKBONES[cfg["backbone"]]["P"]
    df, Y = DT.load_rows()
    TE.build(cfg, dev)
    _, tpaths, _ = DT.cache_paths(cfg)
    T = [np.load(p, mmap_mode="r") for p in tpaths]
    assert all(t.shape == (len(df), 59) for t in T)
    plan = DT.Plan(df, seed)
    torch.manual_seed(1000 * seed + 7)
    reader = Reader("r2", layers=(12,)).to(dev)
    enc, enc0 = MD.build(cfg, dev)
    ema = MD.make_ema(enc)
    pg = MD.param_groups(enc, cfg)
    for g in pg:
        g["mod"] = "enc"
    rg = reader_groups(reader, float(tc["reader_lr"]), float(tc["head_lr"]), float(tc["weight_decay"]))
    for g in rg:
        g["mod"] = "reader"
    opt = torch.optim.AdamW(pg + rg, betas=(0.9, 0.999))
    warm = int(tc["warmup_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warm, total))
    enc_p = [p for _, p in MD.trainable_named(enc)]
    rd_p = list(reader.parameters())
    CK.clean_stale(D["ckpt"])
    step0, src, st = 0, None, None
    if a.resume != "never":
        src, st = CK.load_latest(D["ckpt"], print)
    plan_fp = dict(seed=seed, n_rows=len(df), n_images=len(plan.groups))
    if st is not None:
        enc.load_state_dict(st["student"])
        ema.load_state_dict(st["ema"])
        reader.load_state_dict(st["reader"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step0 = int(st["global_step"])
        assert st["data_state"]["plan"] == plan_fp, ("data plan changed", st["data_state"]["plan"], plan_fp)
        rs = CK.load_rng(src, 0)
        if rs is not None:
            CK.set_rng_state(rs)
        exp = lr_factor(step0, warm, total) * opt.param_groups[0]["initial_lr"]
        assert abs(opt.param_groups[0]["lr"] - exp) < 1e-12, (opt.param_groups[0]["lr"], exp)
    else:
        np.random.seed(seed * 1000)
        random.seed(seed * 1000)
    lam_rank = 0.0 if lc.get("lambda_rank", 0) in (0, 0.0) else None
    lam_file = os.path.join(D["run"], "lambda.json")

    def forward(b):
        b = RC.to_dev(b, dev)
        rows, flips = b["rows"].cpu().numpy(), b["flips"].cpu().numpy()
        tl = torch.from_numpy(np.where(flips[:, None], T[1][rows], T[0][rows]).astype(np.float32)).to(dev)
        with torch.no_grad():
            otok, ocls = run_enc(enc0, b["x"])
        ftok, fcls = run_enc(enc, b["x"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = reader({12: ftok}, b)["logits"].float()
        return b, ftok, fcls, otok, ocls, logits, tl

    W = int(cfg["data"]["workers"])

    def loader(start, stop):
        return DataLoader(DT.AttrDS(df, Y, S, P), sampler=DT.StepSampler(plan, Y, start, stop, accum), batch_size=None, num_workers=W, pin_memory=True, persistent_workers=W > 0, prefetch_factor=4 if W else None, worker_init_fn=worker_init if W else None)

    accum = int(tc["accum"])
    if lam_rank is None:
        if str(lc["lambda_rank"]) == "auto":
            if os.path.exists(lam_file):
                lam_rank = float(json.load(open(lam_file))["lambda_rank"])
            else:
                ratios = []
                for k, b in enumerate(loader(0, 8)):
                    b, ftok, fcls, otok, ocls, logits, tl = forward(b)
                    p, n = losses(b, ftok, fcls, otok, ocls, logits, tl, lc, 0.0)
                    gg = torch.autograd.grad(p["l_gt"], enc_p, retain_graph=True, allow_unused=True)
                    gr = torch.autograd.grad(p["l_rank"], enc_p, allow_unused=True)
                    n1, n2 = [float(torch.sqrt(sum((x.double() ** 2).sum() for x in g if x is not None))) for g in (gg, gr)]
                    ratios.append(n1 / max(n2, 1e-12))
                lam_rank = float(f"{float(np.median(ratios)):.2g}")
                CK.atomic_json(dict(lambda_rank=lam_rank, rule="median ||g_enc L_gt|| / ||g_enc L_rank|| over the first 8 steps at the initial weights", ratios=ratios, time=time.strftime("%F %T")), lam_file)
                print(json.dumps(dict(lambda_rank=lam_rank, ratios=ratios)), flush=True)
        else:
            lam_rank = float(lc["lambda_rank"])
    eval_steps = sorted(int(s) for s in cfg["eval"]["steps"])

    def export(step, pending):
        d = os.path.join(D["evals"], f"step_{step:08d}")
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "enc_ema.pt")):
            CK.atomic_save(dict(enc=ema.state_dict(), step=step, kind="ema"), os.path.join(d, "enc_ema.pt"))
        if pending and not any(os.path.exists(os.path.join(d, x)) for x in ("PENDING", "RUNNING", "DONE", "FAILED")):
            open(os.path.join(d, "PENDING"), "w").write(time.strftime("%F %T"))

    stop_at = min(total, step0 + a.stop_after_steps) if a.stop_after_steps else total
    logf = open(os.path.join(D["logs"], "train.csv"), "a")
    header = st is None and os.path.getsize(logf.name) == 0
    if header:
        logf.write("step,epoch,lr_enc,lr_reader,loss,l_gt,l_kd,l_preserve,l_cls,l_mean,l_tok,l_rank,pairs,enc_grad,reader_grad,s_per_step,mem_gb,launch_id\n")
        logf.flush()
    rec = dict(time=time.strftime("%F %T"), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), start_step=step0, resumed_from=src, lam_rank=lam_rank, model=MD.describe(enc), n_rows=len(df), n_images=len(plan.groups), steps_per_epoch=len(plan._epoch(1)), loss={k: lc[k] for k in ("lambda_gt", "lambda_kd", "lambda_preserve")}, accum=accum, argv=sys.argv[1:], launch_id=os.environ.get("BGJOB_LAUNCH_ID", ""))
    with open(os.path.join(D["logs"], "launch_history.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps(rec, default=str), flush=True)
    keys = ("loss", "l_gt", "l_kd", "l_preserve", "l_cls", "l_mean", "l_tok", "l_rank")
    every, gc_every, hb_every, save_every, an_every = (int(x) for x in (cfg["log"]["every"], cfg["log"]["grad_check_every"], cfg["log"]["heartbeat_every"], cfg["ckpt"]["save_every"], cfg["ckpt"]["analysis_every"]))

    def heartbeat(done):
        CK.atomic_json(dict(step=done, time=time.time()), os.path.join(D["logs"], "heartbeat.json"))

    def ckpt(done):
        CK.save(D["ckpt"], done, 0, lambda: dict(student=enc.state_dict(), ema=ema.state_dict(), reader=reader.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=None, global_step=done, epoch=plan.locate(done)[0], data_state=dict(plan=plan_fp, next_step=done), config=CF.snapshot(cfg), time=time.strftime("%F %T")),
                dict(step=done, time=time.strftime("%F %T"), lr_enc=opt.param_groups[0]["lr"]), int(cfg["ckpt"]["keep_last"]), lambda: None)

    it = iter(loader(step0, stop_at))
    zero = lambda: torch.zeros((), device=dev, dtype=torch.float64)  # noqa: E731  running sums stay on the GPU; reading them forces a host sync, so that happens only at log time
    win, win_n, win_t, gsum, grsum, pairs_sum = {k: zero() for k in keys}, 0, time.time(), zero(), zero(), zero()
    done, reason = step0, None
    heartbeat(step0)
    for g in range(step0, stop_at):
        opt.zero_grad(set_to_none=True)
        for m in range(accum):
            raw = next(it)
            assert (raw["step"], raw["m"]) == (g, m), (raw["step"], raw["m"], g, m)
            b, ftok, fcls, otok, ocls, logits, tl = forward(raw)
            p, n = losses(b, ftok, fcls, otok, ocls, logits, tl, lc, lam_rank)
            if m == 0 and (g % gc_every == 0 or g < 3):
                grads = {}
                for k in ("l_gt", "l_kd", "l_preserve", "l_rank"):
                    gg = torch.autograd.grad(p[k], enc_p, retain_graph=True, allow_unused=True)
                    grads[k] = float(torch.sqrt(sum((x.double() ** 2).sum() for x in gg if x is not None))) if p[k].requires_grad else 0.0
                frozen = sum(1 for _, q in enc.named_parameters() if not q.requires_grad and q.grad is not None)
                with open(os.path.join(D["logs"], "grad_check.jsonl"), "a") as fh:
                    fh.write(json.dumps(dict(step=g, enc_grad_norm=grads, loss={k: float(p[k]) for k in keys}, pairs=float(n), lam_rank=lam_rank, frozen_params_with_grad=frozen, time=time.strftime("%F %T"))) + "\n")
            p["loss"].backward()
            for k in keys:
                win[k] += p[k].detach().double()
            pairs_sum += n.double()
        ge = torch.nn.utils.clip_grad_norm_(enc_p, float(tc["grad_clip"])).detach().double()
        gr = torch.nn.utils.clip_grad_norm_(rd_p, float(tc["grad_clip"])).detach().double()
        opt.step()
        sched.step()
        MD.ema_step(ema, enc, ema_m(g, float(tc["ema_m0"]), total))
        done = g + 1
        gsum += ge
        grsum += gr
        win_n += 1
        if done % hb_every == 0:
            heartbeat(done)
        stop_now = Stop.flag
        if done % every == 0 or done == stop_at or stop_now:
            if not (math.isfinite(float(gsum)) and math.isfinite(float(grsum)) and math.isfinite(float(win["loss"]))):    # checked before any checkpoint of this step
                raise FloatingPointError(f"non-finite loss or gradient in the window ending at step {done}")
            lrs = {x["mod"]: x["lr"] for x in reversed(opt.param_groups)}
            row = [done, plan.locate(done)[0], lrs["enc"], lrs["reader"]] + [f"{float(win[k]) / win_n:.6f}" for k in keys] + [f"{float(pairs_sum) / win_n:.1f}", f"{float(gsum) / win_n:.4f}", f"{float(grsum) / win_n:.4f}", f"{(time.time() - win_t) / win_n:.4f}", f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}", os.environ.get("BGJOB_LAUNCH_ID", "manual")]
            logf.write(",".join(str(x) for x in row) + "\n")
            logf.flush()
            if done % (every * 20) == 0 or done <= step0 + every:
                print(",".join(str(x) for x in row), flush=True)
            win, win_n, win_t, gsum, grsum, pairs_sum = {k: zero() for k in keys}, 0, time.time(), zero(), zero(), zero()
        if done % an_every == 0 or done in eval_steps:
            export(done, done in eval_steps)
        if done % save_every == 0 or stop_now or done == stop_at:
            ckpt(done)
        if stop_now:
            reason = f"signal_{Stop.signum}"
            break
    if reason is None:
        reason = "completed" if done >= total else "stop_after_steps"
    del it
    write_exit(D["control"], reason, done)
    if reason == "completed":
        CK.atomic_json(dict(status="done", steps=done, time=time.strftime("%F %T")), os.path.join(D["run"], "done.json"))
    print(json.dumps(dict(exit=reason, step=done)), flush=True)
    return 0


if __name__ == "__main__":
    main()
