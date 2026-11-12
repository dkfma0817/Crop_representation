#!/usr/bin/env python
"""dinov3_crop_aware_pretrain: crop-aware fine-tuning of the DINOv3-B/16 full-image encoder from COCO object ROIs (frozen crop teacher).

  cd <repo>; python -m src.crop_aware_pretrain.train --gpu 3 --resume auto --stop-after-steps 5000
  (also: cd src/crop_aware_pretrain; python train.py --gpu 3 --resume auto --stop-after-steps 5000)

Teacher: frozen original DINOv3, GT crop from original pixels -> C0 letterbox -> crop tokens (training only); frozen original on the full image
 = preservation reference. Student: DINOv3 full image encoded ONCE; ROI coordinates only select which tokens are read (area-weighted pooling on the
full-image token grid, = R2 full-canvas geometry). Never re-encodes per ROI. Gradient reaches the last blocks (+ final norm) -> see model.py.

Stopping: SIGTERM/SIGINT sets a flag; all ranks agree at the next optimizer-step boundary and write a checkpoint before exiting (exit code 0,
reason in control/last_train_exit.json). Exit 75 = another copy holds the run lock; 76 = GPU busy and not managed by the supervisor.
Under DDP (one process per GPU, started by supervisor.py with RANK/WORLD_SIZE/LOCAL_RANK) there is one student; gradients are summed with an
all-reduce (= RC.allreduce_grads, the project's DDP helper) and the world size only changes how the identical step batch is split (see data.py).
"""
import argparse
import fcntl
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("NCCL_IB_DISABLE", "1")     # single node: GPUs talk over NVLink (= src.engine.train)
os.environ.setdefault("CUDA_CACHE_PATH", os.path.join(_ROOT, "cache", "cuda_jit"))   # NCCL/CUDA JIT cache (= project queues; init minutes -> seconds)
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")
os.environ.setdefault("CUDA_CACHE_DISABLE", "0")

import json       # noqa: E402
import math       # noqa: E402
import signal     # noqa: E402
import time       # noqa: E402
import traceback  # noqa: E402

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import torch         # noqa: E402
import yaml          # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import src.readout.common as RC                  # noqa: E402
from src.config import COCO_TRAIN, FEAT          # noqa: E402
from src.crop_aware_pretrain import ckpt as CK   # noqa: E402
from src.crop_aware_pretrain import config as CF  # noqa: E402
from src.crop_aware_pretrain import data as DT   # noqa: E402
from src.crop_aware_pretrain import gpu as GP    # noqa: E402
from src.crop_aware_pretrain import losses as LS  # noqa: E402
from src.crop_aware_pretrain import model as MD  # noqa: E402
from src.g1x.data import CocoDS                  # noqa: E402
from src.g1x.distill import dev_diag             # noqa: E402
from src.ivc.ft import run_enc                   # noqa: E402
from src.msd.pretrain import eval_exclusions     # noqa: E402
from src.viz.c0_pca import BACKBONES             # noqa: E402

EX_LOCKED, EX_BUSY = 75, 76


class Stop:
    flag, signum = False, None

    @classmethod
    def handler(cls, sig, _frame):
        cls.flag, cls.signum = True, sig


def lr_factor(step, warm, total):
    """Warmup + cosine to 0 = C.lr_at (COCO stream recipe), as a LambdaLR factor."""
    return (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def ema_m(step, m0, total):
    return 1 - (1 - m0) * (math.cos(math.pi * step / total) + 1) / 2


def gnorm_by_block(named):
    groups = {}
    for n, p in named:
        k = n.split(".")[0] + ("." + n.split(".")[1] if n.startswith("blocks.") else "")
        if p.grad is not None:
            groups.setdefault(k, []).append(p.grad.detach().double().pow(2).sum())
    return {k: float(torch.sqrt(torch.stack(v).sum())) for k, v in groups.items()}


def write_exit(ctl, rank, reason, step, world, extra=None):
    if rank == 0:
        CK.atomic_json(dict(reason=reason, step=step, world=world, time=time.strftime("%F %T"), launch_id=os.environ.get("CROP_AWARE_LAUNCH_ID", ""), **(extra or {})),
                       os.path.join(ctl, "last_train_exit.json"))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="a.b=c override (recorded in config_effective.yaml)")
    ap.add_argument("--resume", default="auto", help="auto = newest checkpoint, or start from step 0 if there is none; never = refuse if a checkpoint exists")
    ap.add_argument("--stop-after-steps", type=int, default=0, help="run this many ADDITIONAL optimizer steps, checkpoint, exit")
    ap.add_argument("--stop-at-step", type=int, default=0, help="stop at this global step at the latest (supervisor: gate step until the PACO stop decision exists)")
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint period in optimizer steps (overrides ckpt.save_every)")
    ap.add_argument("--gpu", default=None, help="single physical GPU index for a manual run (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--managed", action="store_true", help="started by the supervisor (skips the GPU-busy guard, which the supervisor applies itself)")
    ap.add_argument("--allow-busy-gpu", action="store_true", help="manual runs only: skip the GPU-busy guard")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="cpu = logic tests only")
    return ap.parse_args()


def main():
    a = parse()
    if a.gpu is not None:
        assert "," not in a.gpu, "--gpu is for a single GPU; multi-GPU runs are launched by the supervisor"
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    cfg = CF.load(a.config, a.set)
    if a.save_every:
        cfg["ckpt"]["save_every"] = a.save_every
    D = CF.dirs(cfg)
    for k in D.values():
        os.makedirs(k, exist_ok=True)
    rank, world = RC.ddp_setup() if a.device == "cuda" else (0, 1)
    dev = torch.device(a.device)
    if world > 1:
        RC.all_sum(torch.zeros(1, device="cuda"))        # build the communicator before data workers are forked
        torch.cuda.synchronize()
    signal.signal(signal.SIGTERM, Stop.handler)
    signal.signal(signal.SIGINT, Stop.handler)
    if not a.managed and a.device == "cuda" and not a.allow_busy_gpu:
        st = GP.query()
        vis = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        for i in vis or [x["index"] for x in st["gpus"]]:
            bad, why = GP.busy_external(i, st)
            if bad:
                print(f"GPU {i} is in use: {why}. Not starting (use the supervisor, or --allow-busy-gpu to override).", flush=True)
                sys.exit(EX_BUSY)
    lock_file = open(os.path.join(D["control"], "train.lock"), "w")
    locked = 0.0
    if rank == 0:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            locked = 1.0
    if a.device == "cuda" and world > 1:
        t = torch.tensor([locked], device=dev)
        RC.all_sum(t)
        locked = float(t.item())
    if locked:
        print("run lock is held: another copy of this experiment is running", flush=True)
        sys.exit(EX_LOCKED)
    try:
        code = train(a, cfg, D, rank, world, dev)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        write_exit(D["control"], rank, "failed", -1, world, dict(error=tb))
        raise
    sys.exit(code)


def train(a, cfg, D, rank, world, dev):
    tc, lc = cfg["train"], cfg["loss"]
    total, seed = int(tc["max_steps"]), int(cfg["seed"])
    accum, micro = CF.world_setup(cfg, world)
    B_cfg = BACKBONES[cfg["backbone"]]
    S, P = B_cfg["S"], B_cfg["P"]
    log = open(os.path.join(D["logs"], "train.csv"), "a") if rank == 0 else None
    table = os.path.join(D["data"], "roi_table.parquet")
    assert os.path.exists(table), f"missing {table}: run python -m src.crop_aware_pretrain.checks tables first"
    DT.verify_exclusion(D["data"])
    plan = DT.StepPlan(table, cfg)
    assert total <= plan.spe * 10_000, "implausible max_steps"
    enc, enc0 = MD.build(cfg, dev)
    ema = MD.make_ema(enc)
    pg = MD.param_groups(enc, cfg)
    opt = torch.optim.AdamW(pg, betas=(0.9, 0.999))
    warm = int(tc["warmup_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warm, total))
    named_tr = MD.trainable_named(enc)
    params_tr = [p for _, p in named_tr]
    def sync():
        if world > 1:
            t = torch.zeros(1, device=dev)
            RC.all_sum(t)
            torch.cuda.synchronize()

    if rank == 0:
        CK.clean_stale(D["ckpt"])
    sync()
    step0, src, st = 0, None, None
    if a.resume != "never":
        src, st = CK.load_latest(D["ckpt"], print)
    elif CK.list_ckpts(D["ckpt"]):
        raise SystemExit("--resume never but checkpoints exist")
    plan_fp = dict(seed=seed, n_images=plan.n, B=plan.B, spe=plan.spe, R=plan.R, probs=list(plan.probs), K=plan.K)
    if st is not None:
        enc.load_state_dict(st["student"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step0 = int(st["global_step"])
        assert st["data_state"]["plan"] == plan_fp, ("data plan changed since the checkpoint", st["data_state"]["plan"], plan_fp)
        rs = CK.load_rng(src, rank)
        if rs is not None:
            CK.set_rng_state(rs)
    else:
        torch.manual_seed(seed * 1000 + rank)
        np.random.seed(seed * 1000 + rank)
        import random as _r
        _r.seed(seed * 1000 + rank)
    exp_lr = lr_factor(step0, warm, total) * max(g["base"] for g in pg)
    got_lr = max(g["lr"] for g in opt.param_groups)
    assert abs(got_lr - exp_lr) < 1e-12, (got_lr, exp_lr)
    resume_info = dict(start_step=step0, resumed_from=src, lr_top_at_start=got_lr, lr_expected=exp_lr, resumed_rng=st is not None)

    if rank == 0:
        effective = dict(CF.snapshot(cfg))
        fn = os.path.join(D["run"], "config_effective.yaml")
        if os.path.exists(fn):
            old = yaml.safe_load(open(fn))
            cmp_a = {k: effective[k] for k in effective if k != "supervisor"}
            cmp_b = {k: old[k] for k in old if k != "supervisor"}
            if cmp_a != cmp_b:
                print(f"WARNING: effective config differs from the first run's config (kept: {fn})", flush=True)
        else:
            yaml.safe_dump(effective, open(fn, "w"), sort_keys=False)
        rec = dict(time=time.strftime("%F %T"), world=world, accum=accum, micro=micro, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                   launch_id=os.environ.get("CROP_AWARE_LAUNCH_ID", ""), argv=sys.argv[1:], model=MD.describe(enc), n_images=plan.n, steps_per_epoch=plan.spe, **resume_info)
        with open(os.path.join(D["logs"], "launch_history.jsonl"), "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(json.dumps(rec, default=str), flush=True)
        if st is None:
            log.write("step,epoch,lr_top,lr_bottom,loss,l_crop,l_pool,l_spatial,l_preserve,l_cls,l_mean,l_tok,roi_pool_cos,spatial_cos,cls_cos,mean_cos,out_tok_cos,student_norm,teacher_norm,grad_norm,s_per_step,mem_gb,world,accum,launch_id\n")
            log.flush()
    eval_steps = sorted(int(s) for s in cfg["eval"]["steps"])

    def export_eval(step):
        if rank != 0:
            return
        d = os.path.join(D["evals"], f"step_{step:08d}")
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "enc_ema.pt")):
            CK.atomic_save(dict(enc=ema.state_dict(), step=step, kind="ema"), os.path.join(d, "enc_ema.pt"))
        if not any(os.path.exists(os.path.join(d, x)) for x in ("PENDING", "RUNNING", "DONE", "FAILED")):
            open(os.path.join(d, "PENDING"), "w").write(time.strftime("%F %T"))

    if step0 == 0:
        export_eval(0)
    sync()

    dev_rois = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    ex = eval_exclusions()
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex["raev2_dev"])]
    dev_ids = sorted(int(i) for i in dev_rois.image_id.unique())
    dev_ds = CocoDS(dev_rois, COCO_TRAIN, S, P)
    dev_chunks = [dev_ids[i:i + 8] for i in range(0, len(dev_ids), 8)][rank::world]
    diag_every = int(cfg["log"]["diag_every"])

    def diag(step, which):
        if a.device != "cuda":
            return
        for nm, m in which.items():
            r = dev_diag(m, enc0, None, dev_ds, dev_chunks, dev)
            if rank == 0:
                with open(os.path.join(D["logs"], "diag.jsonl"), "a") as fh:
                    fh.write(json.dumps(dict(step=step, model=nm, **r, time=time.strftime("%F %T"))) + "\n")

    if step0 == 0:
        diag(0, {"init": ema})

    stop_at = min(total, step0 + a.stop_after_steps) if a.stop_after_steps else total
    if a.stop_at_step:
        stop_at = min(stop_at, a.stop_at_step)
    W = int(cfg["data"]["workers"])
    sampler = DT.ShardSampler(plan, step0, stop_at, world, accum, rank)
    ds = DT.CocoRoiDS(S, P, plan.K)
    dl = DataLoader(ds, sampler=sampler, batch_size=None, num_workers=W, pin_memory=(a.device == "cuda"), persistent_workers=W > 0,
                    prefetch_factor=4 if W else None, worker_init_fn=DT.worker_init if W else None)
    it = iter(dl)
    save_every = int(cfg["ckpt"]["save_every"])
    every, gc_every, hb_every = int(cfg["log"]["every"]), int(cfg["log"]["grad_check_every"]), int(cfg["log"]["heartbeat_every"])

    def checkpoint(done):
        def build():
            return dict(student=enc.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=None,
                        global_step=done, epoch=plan.epoch_pos(done)[0], data_state=dict(plan=plan_fp, next_step=done, epoch_pos=plan.epoch_pos(done)),
                        config=CF.snapshot(cfg), world_at_save=world, accum_at_save=accum, time=time.strftime("%F %T"))
        CK.save(D["ckpt"], done, rank, build, dict(step=done, world=world, accum=accum, time=time.strftime("%F %T"), lr_top=max(g["lr"] for g in opt.param_groups)),
                int(cfg["ckpt"]["keep_last"]), sync)

    def heartbeat(done):
        if rank == 0:
            CK.atomic_json(dict(step=done, time=time.time(), world=world, accum=accum), os.path.join(D["logs"], "heartbeat.json"))

    keys = ("loss", "l_crop", "l_pool", "l_spatial", "l_preserve", "l_cls", "l_mean", "l_tok")
    win = torch.zeros(len(keys) + 9, device=dev, dtype=torch.float64)
    win_t, win_steps, grad_win = time.time(), 0, 0.0
    done = step0
    reason = None
    heartbeat(step0)
    for g in range(step0, stop_at):
        opt.zero_grad(set_to_none=True)
        for m in range(accum):
            b = next(it)
            assert (b["step"], b["m"]) == (g, m), ((b["step"], b["m"]), (g, m))
            b = RC.to_dev(b, dev)
            with torch.no_grad():
                T, _ = run_enc(enc0, b["crops"])
                otok, ocls = run_enc(enc0, b["full"])
            ftok, fcls = run_enc(enc, b["full"])
            parts, stats = LS.compute(b, ftok, fcls, T, otok, ocls, lc)
            if m == 0 and rank == 0 and (g % gc_every == 0 or g < 3):
                gr = {}
                for k in ("l_pool", "l_spatial", "l_preserve"):
                    gs = torch.autograd.grad(parts[k], params_tr, retain_graph=True, allow_unused=True)
                    gr[k] = float(torch.sqrt(sum((x.double() ** 2).sum() for x in gs if x is not None)))
                frozen_grad = sum(1 for _, p in enc.named_parameters() if not p.requires_grad and p.grad is not None)
                with open(os.path.join(D["logs"], "grad_check.jsonl"), "a") as fh:
                    fh.write(json.dumps(dict(step=g, loss_values={k: float(parts[k]) for k in keys}, per_loss_grad_local=gr, frozen_params_with_grad=frozen_grad,
                                             student_token_norm=stats["student_norm"], teacher_token_norm=stats["teacher_norm"], time=time.strftime("%F %T"))) + "\n")
            parts["loss"].backward()
            with torch.no_grad():
                row = torch.tensor([float(parts[k]) for k in keys] + [stats[k] for k in ("roi_pool_cos", "spatial_cos", "cls_cos", "mean_cos", "out_tok_cos")] + [float(len(T)), float(b["sv"].sum()), float(b["full"].shape[0]), 1.0],
                                   device=dev, dtype=torch.float64)
                win += row
        RC.allreduce_grads(params_tr)
        gn = float(torch.nn.utils.clip_grad_norm_(params_tr, float(tc["grad_clip"])))
        opt.step()
        sched.step()
        MD.ema_step(ema, enc, ema_m(g, float(tc["ema_m0"]), total))
        done = g + 1
        win_steps += 1
        grad_win += gn
        flags = torch.tensor([float(Stop.flag), 0.0 if math.isfinite(gn) and math.isfinite(float(parts["loss"])) else 1.0], device=dev)
        RC.all_sum(flags)
        if flags[1] > 0:
            raise FloatingPointError(f"non-finite loss or gradient at step {done}")
        stop_now = flags[0] > 0
        if done % hb_every == 0:
            heartbeat(done)
        if (done % every == 0 or done == stop_at or stop_now or (g == step0 and step0 == 0)):
            RC.all_sum(win)
            w = win.cpu().numpy()
            win[:] = 0
            n_roi, n_sp, n_img, n_micro = w[len(keys) + 5:]
            if rank == 0:
                vals = dict(zip(keys, w[:len(keys)] / max(win_steps, 1)))
                roi_cos, sp_cos = w[len(keys)] / max(n_roi, 1), w[len(keys) + 1] / max(n_sp, 1)
                cls_cos, mean_cos, out_cos = (w[len(keys) + k] / max(n_img, 1) for k in (2, 3, 4))
                lrs = [g2["lr"] for g2 in opt.param_groups]
                row = [done, plan.epoch_pos(done)[0], max(lrs), min(lrs)] + [f"{vals[k]:.6f}" for k in keys] + [f"{roi_cos:.5f}", f"{sp_cos:.5f}", f"{cls_cos:.5f}", f"{mean_cos:.5f}", f"{out_cos:.5f}",
                       f"{stats['student_norm']:.3f}", f"{stats['teacher_norm']:.3f}", f"{grad_win / max(win_steps, 1):.5f}", f"{(time.time() - win_t) / max(win_steps, 1):.4f}",
                       f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}" if a.device == "cuda" else "0", world, accum, os.environ.get("CROP_AWARE_LAUNCH_ID", "manual")]
                log.write(",".join(str(x) for x in row) + "\n")
                log.flush()
                if done % (every * 20) == 0 or done <= step0 + every:
                    print(",".join(str(x) for x in row), flush=True)
            win_t, win_steps, grad_win = time.time(), 0, 0.0
        if done in eval_steps:
            export_eval(done)
        if diag_every and done % diag_every == 0:      # 0 = never inside training; evals.py runs the diagnostic in its own process instead
            diag(done, {"ema": ema})
        if done % save_every == 0 or stop_now or done == stop_at:
            checkpoint(done)
            sync()
        if stop_now:
            reason = f"signal_{Stop.signum}"
            break
    if reason is None:
        reason = "completed" if done >= total else "stop_after_steps"
    del it
    if done in eval_steps:
        export_eval(done)
    write_exit(D["control"], rank, reason, done, world)
    if rank == 0 and reason == "completed":
        CK.atomic_json(dict(status="done", steps=done, time=time.strftime("%F %T")), os.path.join(D["run"], "done.json"))
    if rank == 0:
        print(json.dumps(dict(exit=reason, step=done, world=world, accum=accum)), flush=True)
    return 0


if __name__ == "__main__":
    main()
