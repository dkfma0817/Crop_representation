"""RSP-Residual training = a copy of src.rsp.train (RSP-Masked, pix) with ONLY the output structure changed (src.rspres.model):
pred_n = (F384_bil - mu) / sigma + Delta_n, Delta_n = zero-initialised output Linear. Init = the RSP shared init_pix with out.weight /
out.bias set to 0 (every other tensor asserted equal). Norm stats, teacher, data, ROI / query draws, optimiser, schedule, steps, seed:
the RSP files and code unchanged (read from the RSP shared dir, never written). Diff vs src/rsp/train.py: <run>/train_py.diff.

Original RSP docstring:
RSP-Masked / RSP-Masked-noPix training (one condition per process, GPU 1).

Identical for both runs: frozen DINOv3-B/16 (student 384 context AND dense teacher, same original weights), dense teacher p_T = 4 px
(efficient multi-ROI masked teacher, src.rsp.teacher, == single-ROI reference), data / ROI / query draws (src.rsp.data.RSPStreamDS),
normalisation stats and every shared initial parameter (<shared>/init_*.pt, prep), optimiser and schedule = the S3 / FA decoder recipe:
AdamW lr 3e-4, betas (0.9, 0.999), wd 0.05 on weights with ndim >= 2, warmup 600 + cosine to 0, 12,000 optimizer updates,
16 images x 8 ROIs per update, K = min(256, #valid) queries per ROI, no accumulation, clip 1.0, bf16 autocast, seed 0.
Loss (only): L = mean_ROI [ mean_{valid sampled queries} mean_d | pred_n - (T - mu) / sigma | ]  (L1, FP32, teacher stop-grad).

  python -m src.rsp.train --pix 1 --shared <shared> --out <run>/train [--smoke]
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
from src.msd.pretrain import eval_exclusions
from src.rsp.common import GF, K_TRAIN, PT
from src.rsp.data import RSPStreamDS, frects_of
from src.rspres.model import RSPResidual
from src.rsp.teacher import DenseTeacher
from src.tsplit.features import frozen_vit

STEPS, WARM = 12000, 600
LR, WD, CLIP = 3e-4, 0.05, 1.0
STOP = False


def _stop(*_):
    global STOP
    STOP = True


def lr_at(step):
    if step < WARM:
        return LR * (step + 1) / WARM
    return LR * 0.5 * (1 + math.cos(math.pi * (step - WARM) / (STEPS - WARM)))


def make_opt(m):
    tr = [p for p in m.parameters() if p.requires_grad]
    return torch.optim.AdamW([dict(params=[p for p in tr if p.ndim >= 2], weight_decay=WD), dict(params=[p for p in tr if p.ndim < 2], weight_decay=0.)], lr=LR, betas=(.9, .999))


def state_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def query_centres(frects, q):
    """frects (R, 4) [fy, fx, h, w]; q (R, K) local row-major index -> centres (R, K, 2) canvas px, flat fine idx (R, K)."""
    w = frects[:, 3:4]
    uy = frects[:, 0:1] + torch.div(q, w, rounding_mode="floor")
    ux = frects[:, 1:2] + q % w
    return torch.stack([(uy.float() + .5) * PT, (ux.float() + .5) * PT], -1), uy * GF + ux


@torch.no_grad()
def teacher_targets(teach, x, frects, rimg, content, q, q_ok, budget=50000):
    """Coarse stream once per image + chunked multi-ROI fine streams -> final (B, 581, 768) (student context),
    targets at the sampled queries (R, K, 768) fp32, plus gather checks."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        final, kv = teach.coarse(x)
        femb = teach.fine_embed(x)
    R, K = q.shape
    tgt = torch.zeros(R, K, 768, device=x.device)
    n = (frects[:, 2] * frects[:, 3]).tolist()
    order = sorted(range(R), key=lambda r: n[r])
    bad = 0
    i = 0
    while i < R:
        j = i + 1
        while j < R and n[order[j]] * (j + 1 - i) <= budget:
            j += 1
        sel = torch.tensor(order[i:j], device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            T, valid, idx = teach.fine(femb, kv, frects[sel], rimg[sel], content[rimg[sel]])
        qs = q[sel]
        tgt[sel] = torch.gather(T.float(), 1, qs[..., None].expand(-1, -1, 768))
        okq = q_ok[sel]
        bad += int((okq & ~torch.gather(valid, 1, qs)).sum())                # a sampled query on an invalid teacher token
        i = j
    return final.float(), tgt, bad


def roi_l1(pred_n, tgt_n, q_ok):
    e = (pred_n.float() - tgt_n).abs().mean(-1)
    w = q_ok.float()
    return ((e * w).sum(1) / w.sum(1).clamp_min(1)).mean()


@torch.no_grad()
def feature_stats(pred_n, tgt, tgt_n, final, rimg, cen, q_ok, mu, sd, delta_n=None):
    """Teacher-student metrics on this batch's sampled queries (raw = de-normalised space) + the F384 bilinear reference."""
    w = q_ok.float()
    tot = w.sum()
    pr = pred_n.float() * sd + mu
    fm = final[:, 5:].transpose(1, 2).reshape(final.shape[0], 768, 24, 24)
    g = (cen.flip(-1) / 384 * 2 - 1)[:, :, None]                               # (R, K, 1, 2) (x, y)
    fb = F.grid_sample(fm[rimg], g, mode="bilinear", padding_mode="border", align_corners=False)[..., 0].transpose(1, 2)
    out = {}
    for name, y in (("", pr), ("ref_f384bil_", fb)):
        out[name + "cos"] = float((F.cosine_similarity(y, tgt, dim=-1) * w).sum() / tot)
        out[name + "mse_raw"] = float(((y - tgt).pow(2).mean(-1) * w).sum() / tot)
        out[name + "l1_norm"] = float((((y - mu) / sd - tgt_n).abs().mean(-1) * w).sum() / tot)
        out[name + "norm_ratio"] = float((y.norm(dim=-1) * w).sum() / (tgt.norm(dim=-1) * w).sum())
    if delta_n is not None:                                                      # RSP-Residual: |Delta| / |F384_bil| (raw space)
        out["delta_over_bil_norm"] = float(((delta_n.float() * sd).norm(dim=-1) * w).sum() / (fb.norm(dim=-1) * w).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
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
        (out / "failed.json").write_text(json.dumps(dict(status="failed", error=traceback.format_exc(), time=time.strftime("%F %T")), indent=1))
        raise


def run(a, out):
    if (out / "done.json").exists():
        print("already done", flush=True)
        return
    (out / "failed.json").unlink(missing_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    pix = True
    shared = Path(a.shared)
    vit = frozen_vit(dev)
    teach = DenseTeacher(vit).to(dev)
    st = torch.load(shared / "norm_stats.pt", map_location=dev)
    mu, sd = st["mu"].to(dev), st["sd"].to(dev)
    m = RSPResidual(mu, sd).to(dev)
    init_name = "init_pix"
    init_pix = torch.load(shared / f"{init_name}.pt", map_location=dev)
    m.load_state_dict(init_pix)
    man = json.loads((shared / "prep_manifest.json").read_text())
    assert state_sha(m.state_dict()) == man[init_name]["sha256"]
    m.zero_delta()                                                               # the ONLY init change: Delta = 0 at step 0
    for k, v in m.state_dict().items():
        if k not in ("out.weight", "out.bias"):
            assert torch.equal(v, init_pix[k]), k
    init_res_sha = state_sha(m.state_dict())
    m.train()
    opt = make_opt(m)
    vit_sum0 = float(sum(p.double().sum() for p in vit.parameters()))
    w4_sum0 = float(teach.w4.double().sum())
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:STEPS]
    start = 0
    rec = out / "recovery.pt"
    if rec.exists():
        ck = torch.load(rec, map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed from step {start}", flush=True)
    n_steps = 40 if a.smoke else STEPS
    cfg = dict(condition="RSP-Residual", output="pred_n = (F384_bil - mu)/sigma + Delta_n; Delta_n = out(out_norm(h)) zero-init (weight + bias); Y_hat = F384_bil + sigma*Delta_n",
               init_residual_sha256=init_res_sha, pix=pix, steps=STEPS, images_per_step=16, rois_per_image=8, roi_parent_side=[1, 6],
               K_train=K_TRAIN, p_T=PT, grad_accumulation=1, lr=LR, betas=[.9, .999], wd=WD, wd_rule="ndim >= 2 only", warmup=WARM, schedule="linear warmup + cosine to 0",
               clip=CLIP, amp="bf16 autocast (teacher, predictor); loss and normalisation FP32", loss="mean_ROI mean_valid_query mean_d |pred_n - (T - mu)/sigma| (L1 only)",
               init=init_name, init_sha256=man[init_name]["sha256"], norm_stats_sha256=man["norm_stats_sha256"], seed=0, smoke=a.smoke, torch=torch.__version__,
               predictor=dict(d=m.d, depth=len(m.blocks), heads=m.blocks[0].h, mlp=4, rope=m.rope_cfg, trainable_params=sum(p.numel() for p in m.parameters())),
               teacher=dict(kind="dense masked multi-ROI (efficient path)", p_T=PT, patch_embed="PI-resize 16->4 of the frozen DINOv3 kernel, exact upsampling form w4 = U^T w (U = bilinear 4->16), bias copied",
                            fine_keys="prefix + coarse outside ROI parents + own valid fine", coarse_rows="prefix + coarse only (= plain student forward)"),
               data="src.rsp.data.RSPStreamDS (S3/FA stream rows 0..11,999; rng (step, slot) flip, (step, slot, 1) ROIs, (step, slot, 2, roi) queries)")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    dl = DataLoader(RSPStreamDS(stream, ex, want_ev=pix), sampler=range(start, n_steps), batch_size=None, num_workers=a.workers, pin_memory=True,
                    prefetch_factor=4, persistent_workers=True)
    logs = {k: (out / f"{k}.jsonl").open("a") for k in ("train", "grads")}
    params = [p for p in m.parameters() if p.requires_grad]

    def save(step, final=False):
        tmp = out / "recovery.tmp"
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), step=step), tmp)
        tmp.replace(rec)
        if final:
            torch.save(dict(kind="RSPRES", pix=pix, model=m.state_dict(), step=step, config=cfg, mu=mu.cpu(), sd=sd.cpu()), out / "model.pt")

    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.time()
    acc = {}
    step = start
    for b in dl:
        step = b["step"]
        x = b["x"].to(dev, non_blocking=True)
        rects, rimg, content = b["rects"].to(dev), b["rimg"].to(dev), b["content"].to(dev)
        q, q_ok, box = b["q"].to(dev), b["q_ok"].to(dev), b["box"].to(dev)
        ev = b["ev"].to(dev, non_blocking=True) if pix else None
        assert pix == ("ev" in b)
        fr = frects_of(rects)
        final, tgt, bad = teacher_targets(teach, x, fr, rimg, content, q, q_ok)
        if bad:
            raise RuntimeError(f"{bad} sampled queries fall on invalid teacher tokens at step {step}")
        cen, _ = query_centres(fr, q)
        tgt_n = (tgt - mu) / sd
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_n, _, delta_n = m.parts(final, rimg, box, ev, cen)
        loss = roi_l1(pred_n, tgt_n, q_ok)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss at step {step + 1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"grad at step {step + 1}")
        if step < 3 or (step + 1) % 100 == 0:
            grec = dict(step=step + 1, total_before_clip=float(gn))
            for gname, ps in m.groups().items():
                gg = [p_.grad.float().norm() for p_ in ps if p_.grad is not None]
                grec[gname] = float(torch.stack(gg).norm()) if gg else 0.0
            grec["frozen_grads_none"] = all(p_.grad is None for p_ in vit.parameters())
            assert grec["frozen_grads_none"]
            logs["grads"].write(json.dumps(grec) + "\n")
            logs["grads"].flush()
        opt.step()
        vals = dict(loss=float(loss), n_query=float(q_ok.sum()))
        if step < 3 or (step + 1) % 10 == 0:
            vals.update(feature_stats(pred_n, tgt, tgt_n, final, rimg, cen, q_ok, mu, sd, delta_n))
        for k, v in vals.items():
            acc.setdefault(k, []).append(v)
        if step == start or (step + 1) % 50 == 0:
            now = time.time()
            r = dict(step=step + 1, lr=lr_at(step), elapsed_s=round(now - t0, 1), sec_per_step=round((now - last) / max(len(acc["loss"]), 1), 3),
                     peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), **{k: float(np.nanmean(v)) for k, v in acc.items()})
            logs["train"].write(json.dumps(r) + "\n")
            logs["train"].flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("step", "sec_per_step", "loss", "cos", "delta_over_bil_norm", "l1_norm", "ref_f384bil_cos", "ref_f384bil_l1_norm", "norm_ratio", "peak_gib")}), flush=True)
            acc, last = {}, now
        if (step + 1) % 500 == 0:
            save(step + 1)
        if STOP:
            save(step + 1)
            (out / "interrupted.json").write_text(json.dumps(dict(step=step + 1, time=time.strftime("%F %T"))))
            print(f"interrupted at {step + 1}", flush=True)
            return
    save(n_steps, final=True)
    assert float(sum(p.double().sum() for p in vit.parameters())) == vit_sum0, "frozen DINOv3 changed"
    assert float(teach.w4.double().sum()) == w4_sum0, "PI-resized teacher kernel changed"
    (out / "done.json").write_text(json.dumps(dict(status="done", steps=n_steps, time_s=round(time.time() - t0), peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                                                   frozen_unchanged=True, smoke=a.smoke, final_sha256=state_sha(m.state_dict()), time=time.strftime("%F %T")), indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
