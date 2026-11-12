"""FA-A / FA-B training (one condition per process, one GPU each). Identical for both: FineAware architecture and initial state
(<shared>/init.pt), data and ROIs (src.tsplit.train.StreamDS: COCO stream rows 0..11,999, 16 images / step minus the 2 PACO duplicates,
hflip p 0.5, 8 ROIs of 1..6 parents per image), AdamW lr 3e-4 betas (0.9, 0.999) wd 0.05 on weights with ndim >= 2, warmup 600 + cosine,
12,000 optimizer steps, no accumulation, clip 1.0 (trainable params), bf16 autocast for the decoder forward, TF32 allowed outside the
losses (= S3), loss statistics in true FP32. Frozen: DINOv3 (384 student / 768 teacher), copied overlapping patch embedding, copied norm.
  --loss A : L = L_mean / v_mean + L_res / v_res                         (src.finedec.losses.mean_residual)
  --loss B : L = patch CE (queue 4,096, tau 0.07) + 0.05 SIGReg(Z)        (src.finedec.losses.patch_ce / sigreg); FIFO update after the step
Separate RNG streams (never the data RNG): SIGReg sampling seed 1e6 + step, directions 2e6 + step, queue enqueue sampling 3e6 + step.

  python -m src.finedec.train --loss A --shared <shared> --out <run>/train
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import STREAM
from src.finedec.feats import student_pre
from src.finedec.losses import TeacherQueue, enqueue_samples, mean_residual, parent_stats, patch_ce, sample_latents, sigreg
from src.finedec.model import FineAware
from src.msd.pretrain import eval_exclusions
from src.tsplit.features import frozen_vit, teacher, valid48
from src.tsplit.model import G, rect_index
from src.tsplit.train import STEPS, StreamDS

LR, WD, WARM, CLIP = 3e-4, 0.05, 600, 1.0
LAM_SIG, TAU = 0.05, 0.07
STOP = False


def _stop(*_):
    global STOP
    STOP = True


def lr_at(step):
    if step < WARM:
        return LR * (step + 1) / WARM
    return LR * 0.5 * (1 + math.cos(math.pi * (step - WARM) / (STEPS - WARM)))


def make_opt(m):
    tr = m.trainable()
    return torch.optim.AdamW([dict(params=[p for p in tr if p.ndim >= 2], weight_decay=WD), dict(params=[p for p in tr if p.ndim < 2], weight_decay=0.)], lr=LR, betas=(.9, .999))


def build(dev, shared):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    vit = frozen_vit(dev)
    m = FineAware(vit).to(dev)
    m.load_state_dict(torch.load(Path(shared) / "init.pt", map_location=dev, weights_only=False))
    return vit, m


def state_sha(m):
    h = hashlib.sha256()
    sd = m.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def batch_tensors(vit, m, b, dev):
    x, x768 = b["x"].to(dev, non_blocking=True), b["x768"].to(dev, non_blocking=True)
    rects, rimg, content = b["rects"].to(dev), b["rimg"].to(dev), b["content"].to(dev)
    final, xpre = student_pre(vit, x)
    T = teacher(vit, x768)
    idx, cy, cx, ok = rect_index(rects)
    key_ok = ok & valid48(content)[rimg[:, None], idx]
    lp = (cy // 2 - rects[:, 0:1]) * rects[:, 3:4] + (cx // 2 - rects[:, 1:2])
    lp = torch.where(ok, lp, torch.zeros_like(lp))
    return dict(x=x, final=final, xpre=xpre, Tt=T[rimg[:, None], idx], rects=rects, rimg=rimg, idx=idx, cy=cy, cx=cx, ok=ok, key_ok=key_ok, lp=lp, n_img=len(x))


def forward(m, d):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fine, kvs = m.image_side(d["x"], d["final"])
        Y, Z, dm, dr = m.decode(fine, kvs, d["xpre"], d["rimg"], d["idx"], d["cy"], d["cx"], d["ok"], d["key_ok"], d["rects"][:, 0], d["rects"][:, 1], d["rects"][:, 3])
    return Y, Z, dm, dr


@torch.no_grad()
def feature_stats(Y, Tt, d, dm, dr, Z):
    v = d["key_ok"]
    vf = v.float()
    Yf = Y.float()
    mus_c, _, _ = parent_stats(Yf, d["lp"], v)
    mut_c, _, _ = parent_stats(Tt.float(), d["lp"], v)
    rS, rT = Yf - mus_c, Tt.float() - mut_c
    full = vf.sum()
    cos_r = F.cosine_similarity(rS, rT, dim=-1)
    nz = (rT.norm(dim=-1) > 1e-6) & v
    return dict(child_mse=float(((Yf - Tt).pow(2).mean(-1) * vf).sum() / full), cos=float((F.cosine_similarity(Yf, Tt.float(), dim=-1) * vf).sum() / full),
                norm_ratio=float((Yf.norm(dim=-1) * vf).sum() / (Tt.float().norm(dim=-1) * vf).sum()),
                parent_mean_mse=float(((mus_c - mut_c).pow(2).mean(-1) * vf).sum() / full), residual_mse=float(((rS - rT).pow(2).mean(-1) * vf).sum() / full),
                residual_r2=1 - float(((rS - rT).pow(2).sum(-1) * vf).sum() / (rT.pow(2).sum(-1) * vf).sum().clamp_min(1e-12)),
                residual_norm_ratio=float((rS.norm(dim=-1) * vf).sum() / (rT.norm(dim=-1) * vf).sum().clamp_min(1e-12)), residual_cos=float(cos_r[nz].mean()) if nz.any() else float("nan"),
                delta_mean_norm=float((dm.norm(dim=-1) * vf).sum() / full), delta_res_norm=float((dr.norm(dim=-1) * vf).sum() / full),
                xpre_norm=float((d["xpre"].float().norm(dim=-1)).mean()), Z_std=float(Z.float()[v].std(0).mean()), Z_mean_abs=float(Z.float()[v].mean(0).abs().mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=("A", "B"), required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        run(a, out)
    except Exception:
        (out / "failed.json").write_text(json.dumps(dict(status="failed", error=traceback.format_exc()), indent=1))
        raise


def run(a, out):
    if (out / "done.json").exists():
        return
    dev = torch.device("cuda")
    vit, m = build(dev, a.shared)
    init_sha = state_sha(m)
    man = json.loads((Path(a.shared) / "prep_manifest.json").read_text())
    assert init_sha == man["init_sha256"], "init state differs from the shared init"
    m.train()
    opt = make_opt(m)
    const = json.loads((Path(a.shared) / "constants.json").read_text())
    queue = TeacherQueue(torch.load(Path(a.shared) / "queue_prefill.pt", weights_only=False)["features"].to(dev)) if a.loss == "B" else None
    vit_sum0 = float(sum(p.double().sum() for p in vit.parameters()))
    frozen_sum0 = float(sum(p.double().sum() for p in m.parameters() if not p.requires_grad))
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:STEPS]
    start = 0
    rec = out / "recovery.pt"
    if rec.exists():
        ck = torch.load(rec, map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        if queue is not None:
            queue.features.copy_(ck["queue"].to(dev))
            queue.pointer = ck["queue_pointer"]
    n_steps = 30 if a.smoke else STEPS
    cfg = dict(condition=f"FA-{a.loss}", loss=("L_mean / v_mean + 1.0 L_res / v_res" if a.loss == "A" else "patch CE (queue 4096, tau 0.07) + 0.05 SIGReg(Z)"),
               steps=STEPS, images_per_step=16, rois_per_image=8, roi_parent_side=[1, 6], grad_accumulation=1, lr=LR, betas=[.9, .999], wd=WD, warmup=WARM, schedule="cosine to 0",
               clip=CLIP, amp="bf16 autocast for the decoder forward, no GradScaler; TF32 allowed outside the losses; loss statistics FP32 with TF32 off",
               init_sha256=init_sha, constants=dict(v_mean=const["v_mean"], v_res=const["v_res"]), seed=0, smoke=a.smoke, torch=torch.__version__,
               rng_streams=dict(data="StreamDS numpy rng per (step, slot)", sigreg_sampling="1e6 + step", sigreg_directions="2e6 + step", queue_enqueue="3e6 + step"))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    dl = DataLoader(StreamDS(stream, ex), sampler=range(start, n_steps), batch_size=None, num_workers=a.workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    logs = {n: (out / f"{n}.jsonl").open("a") for n in ("train", "grads")}
    params = m.trainable()

    def save(step, final=False):
        tmp = out / "recovery.tmp"
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), step=step, queue=None if queue is None else queue.features.cpu(), queue_pointer=None if queue is None else queue.pointer), tmp)
        tmp.replace(rec)
        if final:
            torch.save(dict(kind="FA", loss=a.loss, model=m.state_dict(), step=step, config=cfg), out / "model.pt")

    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.time()
    acc = {}
    for b in dl:
        step = b["step"]
        d = batch_tensors(vit, m, b, dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        Y, Z, dm, dr = forward(m, d)
        if a.loss == "A":
            loss, t1, t2, st = mean_residual(Y, d["Tt"], d["lp"], d["key_ok"], const["v_mean"], const["v_res"])
            terms = dict(mean=t1 / (const["v_mean"] + 1e-8), res=t2 / (const["v_res"] + 1e-8))
        else:
            Q = queue.snapshot()
            ce, st = patch_ce(Y, d["Tt"], d["key_ok"], d["rimg"], d["n_img"], Q, TAU)
            zs = sample_latents(Z.float(), d["key_ok"], d["rimg"], d["idx"], budget=1024, seed=1_000_000 + step)
            sg = sigreg(zs, seed=2_000_000 + step)
            loss = ce + LAM_SIG * sg
            terms = dict(ce=ce, sigreg=LAM_SIG * sg)
            st.update(sigreg=float(sg), sigreg_samples=len(zs))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss at step {step + 1}")
        opt.zero_grad(set_to_none=True)
        diag = step < 3 or (step + 1) % 100 == 0
        grec = dict(step=step + 1)
        if diag:
            names = {id(p): n for n, ps in m.groups().items() for p in ps}
            for tn, tv in terms.items():
                gs = torch.autograd.grad(tv, params, retain_graph=True, allow_unused=True)
                grec[f"grad_{tn}"] = float(torch.cat([g.float().flatten() for g in gs if g is not None] or [torch.zeros(1, device=dev)]).norm())
                for gn in m.groups():
                    v = [g.float().flatten() for p, g in zip(params, gs) if g is not None and names.get(id(p)) == gn]
                    grec[f"grad_{tn}/{gn}"] = float(torch.cat(v).norm()) if v else 0.0
            assert all(p.grad is None for p in params)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"grad at step {step + 1}")
        if diag:
            grec["total_before_clip"] = float(gn)
            for gname, ps in m.groups().items():
                gg = [p.grad.float().norm() for p in ps if p.grad is not None]
                grec[f"total/{gname}"] = float(torch.stack(gg).norm()) if gg else 0.0
            assert all(p.grad is None for p in vit.parameters()) and all(p.grad is None for p in m.parameters() if not p.requires_grad)
            logs["grads"].write(json.dumps(grec) + "\n")
            logs["grads"].flush()
        opt.step()
        if queue is not None:
            queue.enqueue(enqueue_samples(d["Tt"].float(), d["key_ok"], d["rimg"], d["idx"], per_image=32, seed=3_000_000 + step))
        vals = dict(loss=float(loss), **st, **(feature_stats(Y, d["Tt"], d, dm, dr, Z) if (step < 3 or (step + 1) % 10 == 0) else {}))
        for k, v in vals.items():
            acc.setdefault(k, []).append(v)
        if step == start or (step + 1) % 50 == 0:
            now = time.time()
            r = dict(step=step + 1, lr=lr_at(step), elapsed_s=round(now - t0, 1), sec_per_step=round((now - last) / max(len(acc["loss"]), 1), 3),
                     peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), **{k: float(np.nanmean(v)) for k, v in acc.items()})
            logs["train"].write(json.dumps(r) + "\n")
            logs["train"].flush()
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k in ("step", "sec_per_step", "loss", "L_mean", "L_res", "patch_ce", "patch_kl", "sigreg", "cos", "residual_r2", "parent_mean_mse", "norm_ratio", "peak_gib")}), flush=True)
            acc, last = {}, now
        if (step + 1) % 1000 == 0:
            save(step + 1)
        if STOP:
            save(step + 1)
            (out / "interrupted.json").write_text(json.dumps(dict(step=step + 1)))
            return
    save(n_steps, final=True)
    assert float(sum(p.double().sum() for p in vit.parameters())) == vit_sum0
    assert float(sum(p.double().sum() for p in m.parameters() if not p.requires_grad)) == frozen_sum0
    (out / "done.json").write_text(json.dumps(dict(status="done", steps=n_steps, time_s=round(time.time() - t0), peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                                                   frozen_unchanged=True, smoke=a.smoke), indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
