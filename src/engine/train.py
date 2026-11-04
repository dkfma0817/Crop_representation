"""A / B / C on the RAEv2 DINOv2-B K1 latent, CoordMLP (R00 structure: MLRDecoder('final') = CoordMLP body + zero-init adapter),
fresh weights from the SAME init file for all three runs; only the loss differs:

  A Feature-only : L = L_feat                                  L_feat = MSE_w + 1.0 (1 - cos)_w   (raw latent, losses_mlr)
  B Pixel-only   : L = lam * L_pix                             L_pix  = L1 + beta * LPIPS          (pixel.pix_terms)
  C Feature+Pixel: L = L_feat + lam * L_pix
  (MSE / cos coefficients identical in A and C; L1 / LPIPS coefficients lam, lam*beta identical in B and C; fixed for the run.)
Pixel path: Z_pred (raw) -> frozen official RAE decoder (BF16 autocast, params requires_grad False, eval) -> unclamped RGB 256
-> loss vs the GT crop canvas on content pixels. No no_grad / detach / clamp on the student path.
Optimisation (= multilevel_relation_decoder / visual_recon_loss recipe): AdamW lr 1e-4 (0.9, 0.95) wd 0.05, warmup 600 + cosine to 0
at --steps, clip 1.0, BF16 autocast CoordMLP, 64 ROIs / step = one stream row (16 images x 4 ROIs, existing 12k stream seed 2027),
torch seed 0. The 64 ROIs are processed in fixed micro-batches of --micro ROIs with gradient accumulation (loss of each micro-batch =
sum of its per-ROI losses / 64, so the update equals the full-batch one); identical partition in A, B, C.
Frozen features come from the cache (extract_cache.py). Logs: train_log.jsonl (every 50 steps), grad norms every --gn_every steps
(||dL/dZ|| and ||dL/dtheta|| for L_feat and lam*L_pix on the first micro-batch, incl. A/B where the term is not optimised),
COCO dev (2,000 ROIs) every --eval_every steps, ckpt_<step>.pt at --save_at.

  CUDA_VISIBLE_DEVICES=0 python -m src.engine.train --run A --w_feat 1 --lam 0  --beta B --micro M --out cache/runs/A
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

import src.rae.common as RC  # noqa: E402
import src.losses.pixel as PX  # noqa: E402
import src.data.dataset as DR  # noqa: E402
from src.models.coordmlp import MLRDecoder  # noqa: E402

from src.config import INIT, OUTPUT  # noqa: E402
from src.utils.yaml_config import with_config  # noqa: E402
ROI_KEYS = ("img_index", "tok", "coords", "pad", "inside", "valid_w", "full_cov", "bil_grid", "n_in", "n_ctx")


def make_init():
    """One fresh MLRDecoder('final') (torch seed 0) + teacher-stat buffers, shared by A/B/C (and nothing else)."""
    torch.manual_seed(0)
    m = MLRDecoder("final")
    mu, sd = DR.teacher_stats()
    m.body.set_stats(mu, sd)
    os.makedirs(os.path.dirname(INIT), exist_ok=True)
    torch.save(dict(state_dict=m.state_dict(), seed=0, n_params=sum(p.numel() for p in m.parameters()),
                    note="fresh init; mean_T/std_T = per-channel teacher stats of the first 3,000 stream-cache ROIs (content cells)"), INIT)
    print("saved", INIT, sum(p.numel() for p in m.parameters()))


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MLRDecoder("final")
    m.load_state_dict(ck["state_dict"])
    return m.to(dev)


def sub(g, ix):
    return {k: (v[ix] if k in ROI_KEYS else v) for k, v in g.items()}


def micro_terms(m, rae, lp, b, ix, P, need_pix, beta):
    """-> Z (n,256,768) fp32, per-ROI feat loss, per-ROI pix loss (or None), parts."""
    g = sub(b["geom"], ix)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Z = m(RC.gather_roi(b["full"], g), None, g, P).float()
    w = g["valid_w"]
    mse, cos = PX.feat_mse_cos(Z, b["teacher"][ix], w)
    feat = mse + 1.0 * (1 - cos)
    parts = dict(mse=mse, cos=cos)
    pix = None
    if need_pix:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = rae.decode(Z).float()
        l1, lpv = PX.pix_terms(y, b["rgb"][ix], b["mask"][ix], lp)
        pix = l1 + beta * lpv
        parts.update(l1=l1, lpips=lpv)
    return Z, feat, pix, parts


def grad_record(m, rae, lp, b, ix, P, beta, lam, nroi):
    ps = [p for p in m.parameters() if p.requires_grad]
    Z, feat, pix, _ = micro_terms(m, rae, lp, b, ix, P, True, beta)
    lf, lv = feat.sum() / nroi, lam * pix.sum() / nroi if lam > 0 else pix.sum() / nroi
    gz_f = torch.autograd.grad(lf, Z, retain_graph=True)[0]
    gz_v = torch.autograd.grad(lv, Z, retain_graph=True)[0]
    gp_f = torch.autograd.grad(lf, ps, retain_graph=True, allow_unused=True)
    gp_v = torch.autograd.grad(lv, ps, allow_unused=True)
    nrm = lambda gs: float(torch.sqrt(sum((x.float() ** 2).sum() for x in gs if x is not None)))
    return dict(gz_feat=float(gz_f.norm()), gz_pix=float(gz_v.norm()), gp_feat=nrm(gp_f), gp_pix=nrm(gp_v),
                cos_gz=float(torch.nn.functional.cosine_similarity(gz_f.flatten(), gz_v.flatten(), dim=0)),
                pix_scaled_by_lam=lam > 0, n_tensors_pix_grad=sum(x is not None and float(x.abs().sum()) > 0 for x in gp_v))


@torch.no_grad()
def evaluate(m, rae, lp, dl, dev, P, beta, max_rois=None):
    m.eval()
    agg = {}
    n = 0
    for batch in dl:
        b = DR.to_dev(batch, dev)
        g = b["geom"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Z = m(RC.gather_roi(b["full"], g), None, g, P).float()
            y = rae.decode(Z, chunk=64).float()
        w = g["valid_w"]
        mse, cos = PX.feat_mse_cos(Z, b["teacher"], w)
        l1, lpv = PX.pix_terms(y, b["rgb"], b["mask"], lp)
        met = PX.rgb_metrics(y.clamp(0, 1), b["rgb"], b["mask"], lp)
        vals = dict(mse=mse, cos=cos, feat_loss=mse + 1 - cos, pix_l1=l1, pix_lpips=lpv, pix_loss=l1 + beta * lpv,
                    svar=PX.svar_ratio(Z, b["teacher"], w), **{f"rgb_{k}": v for k, v in met.items()})
        for k, v in vals.items():
            agg.setdefault(k, []).append(v.float().cpu())
        n += len(w)
        if max_rois and n >= max_rois:
            break
    m.train()
    return {k: float(torch.cat(v).mean()) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--w_feat", type=float, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--micro", type=int, required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=INIT)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_at", default="4000,8000,12000")
    ap.add_argument("--gn_every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dev_max", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: output/<YYYY-MM-DD>/<run> (run.py passes it explicitly)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)   # --resume / --no-resume
    ap.add_argument("--slots", default="", help="overfit only: json list of fixed step slot lists (repeated)")
    a = with_config(ap)        # --config configs/raev2_crop_recon/train/<run>.yaml; CLI flags override the YAML
    # Multi-GPU (torchrun --nproc_per_node N): every rank loads the same stream row, takes micro-batches rank, rank+N, ... and the
    # gradients are summed with all_reduce -> the same update as single-GPU accumulation (up to float summation order).
    # Rank 0 alone does the dev evaluation, logging and checkpoints.
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        import torch.distributed as dist
        os.environ.setdefault("NCCL_IB_DISABLE", "1")     # single node: GPUs talk over NVLink; probing the 8 IB NICs stalls init
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        dist.all_reduce(torch.zeros(1, device="cuda"))   # build the communicator now, before data workers are forked
        torch.cuda.synchronize()
    main0 = rank == 0
    if not a.out:
        import datetime
        a.out = os.path.join(OUTPUT, datetime.date.today().isoformat(), a.run)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    RC.set_fp32()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rae = RC.FrozenRAE(dev)                       # encoder unused in training (cache); decoder frozen, eval
    del rae.enc
    lp = PX.load_lpips(dev)
    assert not any(p.requires_grad for p in list(rae.dec.parameters()) + list(lp.parameters()))
    assert not rae.dec.training and not lp.training
    fp0 = dict(decoder=RC.param_hash(rae.dec), lpips=RC.param_hash(lp))
    m = load_model(a.init, dev)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    sched_fn = lambda s: (s + 1) / a.warmup if s < a.warmup else 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)
    P = torch.as_tensor(RC.query_coords(), device=dev)
    if a.slots:
        fixed = json.load(open(a.slots))
        steps_slots = [fixed[s % len(fixed)] for s in range(a.steps)]
        ds_tr = DR.CachedSet(json.load(open(a.slots + ".set")) if os.path.exists(a.slots + ".set") else "dev", rgb=True)
    else:
        steps_slots = DR.stream_slots(a.steps)
        ds_tr = DR.CachedSet("stream", rgb=True)
    ds_dv = DR.CachedSet("dev", rgb=True)
    save_at = {int(x) for x in a.save_at.split(",") if x}
    need_pix = a.lam > 0
    cfg = dict(vars(a), n_params=sum(p.numel() for p in m.parameters()), torch=torch.__version__,
               loss=(f"{a.w_feat} * (MSE_w + 1.0 (1 - cos)_w)" if a.w_feat else "") + (f" + {a.lam} * (L1 + {a.beta} LPIPS)" if a.lam else ""),
               decoder=os.path.join(RC.MODELS, "decoder.pt"), fingerprints_start=fp0, gpu=torch.cuda.get_device_name(0),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), world_size=world)
    step, t0 = 0, time.time()
    rpath = os.path.join(a.out, "resume.pt")
    log = open(os.path.join(a.out, "train_log.jsonl"), "a") if main0 else open(os.devnull, "w")
    if a.resume and os.path.exists(rpath):
        rs = torch.load(rpath, map_location="cpu", weights_only=False)
        m.load_state_dict(rs["state_dict"]), opt.load_state_dict(rs["opt"]), sched.load_state_dict(rs["sched"])
        step, t0 = rs["step"], time.time() - rs["elapsed_s"]
        torch.set_rng_state(rs["rng"])
        print(json.dumps(dict(resumed_at=step, world_size=world)), file=log, flush=True)
    elif main0:
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=float)
    if main0:
        print(json.dumps(cfg, default=float), flush=True)
    t_start, step_start = time.time(), step
    ps_all = [p for p in m.parameters() if p.requires_grad]
    dv_slots = fixed if a.slots else DR.seq_slots(ds_dv, 16)      # overfit: evaluate on the same fixed 256 ROIs
    dv_loader = lambda: DR.loader(ds_dv, dv_slots, 6)
    if step == 0 and a.eval_every and main0:
        ev = evaluate(m, rae, lp, dv_loader(), dev, P, a.beta, a.dev_max or None)
        for f in (log, sys.stdout):
            print(json.dumps(dict(step=0, dev=ev)), file=f, flush=True)
    torch.cuda.reset_peak_memory_stats()
    keys = ("loss", "feat", "pix", "mse", "cos", "l1", "lpips")
    run, nrun, npix = np.zeros(len(keys)), 0, 0
    for batch in DR.loader(ds_tr, steps_slots[step:a.steps], a.workers):
        b = DR.to_dev(batch, dev)
        R = b["teacher"].shape[0]
        chunks = [torch.arange(i, min(R, i + a.micro), device=dev) for i in range(0, R, a.micro)]
        gn_rec = grad_record(m, rae, lp, b, chunks[0], P, a.beta, a.lam, R) if main0 and ((step + 1) % a.gn_every == 0 or step == 0) else None
        opt.zero_grad(set_to_none=True)
        tot = np.zeros(len(keys))
        for ix in chunks[rank::world]:
            Z, feat, pix, parts = micro_terms(m, rae, lp, b, ix, P, need_pix, a.beta)
            loss = a.w_feat * feat.sum() / R
            if need_pix:
                loss = loss + a.lam * pix.sum() / R
            loss.backward()
            tot += [float(loss) * 1.0, float(feat.sum()) / R, float(pix.sum()) / R if need_pix else 0.0,
                    float(parts["mse"].sum()) / R, float(parts["cos"].sum()) / R,
                    float(parts["l1"].sum()) / R if need_pix else 0.0, float(parts["lpips"].sum()) / R if need_pix else 0.0]
        if world > 1:
            for p in ps_all:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            flat = torch.cat([p.grad.flatten() for p in ps_all])
            dist.all_reduce(flat)
            o = 0
            for p in ps_all:
                p.grad.copy_(flat[o:o + p.numel()].view_as(p))
                o += p.numel()
            tt = torch.as_tensor(tot, device=dev, dtype=torch.float64)
            dist.all_reduce(tt)
            tot = tt.cpu().numpy()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip))
        if not all(math.isfinite(x) for x in tot) or not math.isfinite(gn):
            raise FloatingPointError(f"non-finite at step {step}: {tot} gn {gn}")
        opt.step()
        sched.step()
        run += tot
        nrun += 1
        step += 1
        if gn_rec is not None:
            print(json.dumps(dict(step=step, grad=gn_rec, clip_total=gn)), file=log, flush=True)
        if step % 50 == 0 and main0:
            r = run / nrun
            rec = dict(step=step, lr=sched.get_last_lr()[0], train={k: v for k, v in zip(keys, r)}, grad_norm=gn,
                       s_per_step=(time.time() - t_start) / max(1, step - step_start), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            print(json.dumps(rec), file=log, flush=True)
            if step % 500 == 0:
                print(json.dumps(rec), flush=True)
        if step % 50 == 0:
            run[:] = 0
            nrun = 0
        do_eval = a.eval_every and (step % a.eval_every == 0 or step == a.steps)
        if (do_eval or step in save_at) and main0:
            ev = evaluate(m, rae, lp, dv_loader(), dev, P, a.beta, a.dev_max or None) if do_eval else None
            if step in save_at or step == a.steps:
                torch.save(dict(arch="mlr", source="final", state_dict=m.state_dict(), step=step, dev=ev, cfg=cfg), os.path.join(a.out, f"ckpt_{step}.pt"))
            torch.save(dict(state_dict=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                            elapsed_s=time.time() - t0, rng=torch.get_rng_state()), rpath)
            rec = dict(step=step, dev=ev, elapsed_s=time.time() - t0, peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            for f in (log, sys.stdout):
                print(json.dumps(rec), file=f, flush=True)
    if world > 1:
        dist.barrier()
    if main0:
        fp1 = dict(decoder=RC.param_hash(rae.dec), lpips=RC.param_hash(lp))
        json.dump(dict(steps=step, time_s=time.time() - t0, peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                       frozen_unchanged={k: fp0[k] == fp1[k] for k in fp0}, fingerprints_end=fp1, world_size=world),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
        if not a.save_at:
            torch.save(dict(arch="mlr", source="final", state_dict=m.state_dict(), step=step, cfg=cfg), os.path.join(a.out, f"ckpt_{step}.pt"))
        print("DONE", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1] == "make_init":
        make_init()
    else:
        main()
