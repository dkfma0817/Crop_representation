"""S3-Distill: the S3 decoder of src.tsplit.train with ONLY the loss replaced (L2 -> teacher-bank KL + ROI local relation KL).

Identical to the S3-L2 run (src.tsplit.train): data stream / ROI sampler (StreamDS: same images, flips and rectangles per step), frozen
DINOv3 384 / native-768 features, TokenSplit('S3') architecture and initialisation (torch.manual_seed(0) -> frozen_vit -> TokenSplit S1, S2,
S3 constructed in the same order, S1 / S2 discarded, so S3 gets the same initial parameters), AdamW lr 3e-4 / wd 0.05 (ndim >= 2),
warmup 600 + cosine, 12,000 steps, 16 images x 8 ROIs per step, no accumulation, clip 1.0, bf16 autocast for the decoder forward
(no GradScaler, so the logged gradient norm needs no unscale), tf32 on, seed 0.
Loss (src.tsplit.kl_loss, FP32, autocast off): L = L_bank + 1.0 * L_local, tau_bank = tau_local = 0.07; fixed 8,192 x 768 bank (src.tsplit.kl_bank).
Analysis only (no gradient): raw feature MSE vs teacher, mean L2 norms, bank / local entropies, diagonal mass; per-loss gradient norms on
the same decoder parameters with torch.autograd.grad (never written to .grad), at steps 1-3 and every 100 steps.

  python -m src.tsplit.train_kl --out <run>/train --bank <run>/bank/bank.pt
"""
import argparse
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import STREAM
from src.msd.pretrain import eval_exclusions
from src.tsplit.features import frozen_vit, student, teacher, up2, valid48
from src.tsplit.kl_loss import bank_kl, local_kl
from src.tsplit.model import TokenSplit, n_params, param_report, rect_index
from src.tsplit.train import CLIP, LR, MAXS, ROIS_PER_IMG, STEPS, WARM, WD, Held, StreamDS, dev_set, lr_at, make_opt, roi_mse

TAU_B, TAU_L, LAM_L = 0.07, 0.07, 1.0
STOP = False


def _stop(*_):
    global STOP
    STOP = True


def build(dev):
    """Exactly the RNG sequence of src.tsplit.train.run up to the decoder construction."""
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    vit = frozen_vit(dev)
    models = {k: TokenSplit(k).to(dev).train() for k in ("S1", "S2", "S3")}
    return vit, models["S3"]


def flatg(gs):
    return torch.cat([g.float().flatten() for g in gs if g is not None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-every", type=int, default=2000)
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
        print("already done", flush=True)
        return
    dev = torch.device("cuda")
    vit, m = build(dev)
    sum0 = float(sum(p.double().sum() for p in vit.parameters()))
    init_sum = float(sum(p.double().sum() for p in m.parameters()))
    opt = make_opt(m)
    bk = torch.load(a.bank, map_location="cpu", weights_only=False)
    Q = bk["bank"].to(dev).float().requires_grad_(False)
    assert Q.shape == (8192, 768) and torch.allclose(Q.norm(dim=-1), torch.ones(len(Q), device=dev), atol=1e-4)
    q_sum0 = float(Q.double().sum())
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:STEPS]
    start = 0
    rec = out / "recovery.pt"
    if rec.exists():
        ck = torch.load(rec, map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(json.dumps(dict(resumed_from=start)), flush=True)
    n_steps = 30 if a.smoke else STEPS
    cfg = dict(condition="S3-Distill", steps=STEPS, images_per_step=16, rois_per_image=ROIS_PER_IMG, roi_parent_side=[1, MAXS], lr=LR, wd=WD, warmup=WARM, clip=CLIP,
               grad_accumulation=1, seed=0, loss="L_bank + 1.0 * L_local (src.tsplit.kl_loss), FP32", tau_bank=TAU_B, tau_local=TAU_L, lambda_local=LAM_L,
               bank=str(a.bank), bank_size=len(Q), init_param_sum=init_sum, params=param_report()["S3"], n_params=n_params(m), excluded_stream_images=ex,
               same_as_S3_L2="architecture, init RNG sequence, StreamDS images / flips / ROI rectangles, optimizer, schedule, steps, batch; only the loss differs",
               smoke=a.smoke, torch=torch.__version__)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    held = None if a.smoke else Held(vit, dev, dev_set())
    dl = DataLoader(StreamDS(stream, ex), sampler=range(start, n_steps), batch_size=None, num_workers=a.workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    logs = {n: (out / f"{n}.jsonl").open("a") for n in ("train", "grads", "heldout")}
    params = list(m.parameters())

    def heldout(step):
        if held is None:
            return
        m.eval()
        t = time.time()
        r = held.evaluate({"S3": m})
        m.train()
        logs["heldout"].write(json.dumps(dict(step=step, seconds=round(time.time() - t, 1), metrics=r)) + "\n")
        logs["heldout"].flush()
        print(json.dumps(dict(heldout=step, S3=dict(mse=round(r["S3"]["all"]["mse"], 5), r2=round(r["S3"]["all"]["within_parent_r2"], 4), pm=round(r["S3"]["all"]["parent_mean_mse"], 5)))), flush=True)

    def save(step, final=False):
        tmp = out / "recovery.tmp"
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), step=step), tmp)
        tmp.replace(rec)
        if final:
            torch.save(dict(kind="S3", model=m.state_dict(), step=step, config=cfg), out / "S3.pt")

    if start == 0:
        heldout(0)
    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.time()
    acc = {}
    for b in dl:
        step = b["step"]
        x, x768 = b["x"].to(dev, non_blocking=True), b["x768"].to(dev, non_blocking=True)
        rects, rimg, content = b["rects"].to(dev), b["rimg"].to(dev), b["content"].to(dev)
        fin, ear = student(vit, x)
        T = teacher(vit, x768)
        v48 = valid48(content)
        idx, cy, cx, ok = rect_index(rects)
        key_ok = ok & v48[rimg[:, None], idx]
        Tt = T[rimg[:, None], idx]
        one = (rects[:, 2] == 1) & (rects[:, 3] == 1)
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cmap = m.child_map(fin, ear)
            y = m.refine(cmap, rimg, idx, cy, cx, key_ok)
        L_b, sb = bank_kl(y, Tt, key_ok, Q, TAU_B)
        L_l, sl = local_kl(y, Tt, key_ok, TAU_L, one_by_one=one)
        loss = L_b + LAM_L * L_l
        if not (torch.isfinite(L_b) and torch.isfinite(L_l)):
            raise FloatingPointError(f"non-finite loss at step {step + 1}: bank {float(L_b)} local {float(L_l)}")
        diag = step < 3 or (step + 1) % 100 == 0
        grec = dict(step=step + 1)
        opt.zero_grad(set_to_none=True)
        if diag:
            gb = torch.autograd.grad(L_b, params, retain_graph=True, allow_unused=True)
            gl = torch.autograd.grad(L_l, params, retain_graph=True, allow_unused=True)
            fb, fl = flatg(gb), flatg(gl)
            grec.update(grad_bank=float(fb.norm()), grad_local=float(fl.norm()), grad_cos_bank_local=float(torch.nn.functional.cosine_similarity(fb, fl, dim=0)),
                        grad_bank_to_local_ratio=float(fb.norm() / fl.norm().clamp_min(1e-12)))
            names = {id(p): n for n, ps in m.groups().items() for p in ps}
            for nm in m.groups():
                grec[f"grad_bank/{nm}"] = float(torch.cat([g.float().flatten() for p, g in zip(params, gb) if g is not None and names.get(id(p)) == nm] or [torch.zeros(1, device=dev)]).norm())
                grec[f"grad_local/{nm}"] = float(torch.cat([g.float().flatten() for p, g in zip(params, gl) if g is not None and names.get(id(p)) == nm] or [torch.zeros(1, device=dev)]).norm())
            assert all(p.grad is None for p in params), "analysis gradients leaked into .grad"
        loss.backward()
        if diag:
            for gname, ps in m.groups().items():
                grec[f"total/{gname}"] = float(torch.stack([p.grad.float().norm() for p in ps if p.grad is not None]).norm())
            gw = m.split.lin.weight.grad
            grec["total/split_final_cols"], grec["total/split_early_cols"] = float(gw[:, :768].float().norm()), float(gw[:, 768:].float().norm())
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"non-finite gradient at step {step + 1}")
        if diag:
            grec["total_before_clip"] = float(gn)
            logs["grads"].write(json.dumps(grec) + "\n")
            logs["grads"].flush()
        opt.step()
        with torch.no_grad():
            yf = y.float()
            vals = dict(L_bank=float(L_b), L_local=float(L_l), L_total=float(loss), mse=float(roi_mse(yf, Tt, key_ok).mean()),
                        mse_B0=float(roi_mse(up2(fin)[rimg[:, None], idx], Tt, key_ok).mean()),
                        norm_student=float(yf.norm(dim=-1)[key_ok].mean()), norm_teacher=float(Tt.norm(dim=-1)[key_ok].mean()), **sb, **sl)
        for k, v in vals.items():
            acc.setdefault(k, []).append(v)
        if step == start or (step + 1) % 50 == 0:
            now = time.time()
            r = dict(step=step + 1, lr=lr, elapsed_s=round(now - t0, 1), sec_per_step=round((now - last) / max(len(acc["L_total"]), 1), 3), n_roi=len(rects), n_1x1=int(one.sum()),
                     peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), **{k: float(np.nanmean(v)) for k, v in acc.items()})
            logs["train"].write(json.dumps(r) + "\n")
            logs["train"].flush()
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k in ("step", "sec_per_step", "L_bank", "L_local", "mse", "local_diag_student", "local_diag_teacher", "bank_H_student", "bank_H_teacher", "peak_gib")}), flush=True)
            acc, last = {}, now
        if (step + 1) % 1000 == 0:
            save(step + 1)
        if (step + 1) % a.eval_every == 0 and step + 1 < n_steps:
            heldout(step + 1)
        if STOP:
            save(step + 1)
            (out / "interrupted.json").write_text(json.dumps(dict(step=step + 1)))
            return
    save(n_steps, final=True)
    heldout(n_steps)
    assert float(sum(p.double().sum() for p in vit.parameters())) == sum0, "frozen encoder changed"
    assert float(Q.double().sum()) == q_sum0, "bank changed"
    (out / "done.json").write_text(json.dumps(dict(status="done", steps=n_steps, time_s=round(time.time() - t0), peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                                                   encoder_unchanged=True, bank_unchanged=True, smoke=a.smoke), indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
