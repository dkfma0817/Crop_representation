"""P1 / P2 training (one model per process, one GPU each) and the one-off SIGReg weight calibration.

Common (fixed before training; P1 and P2 identical except the model and its LR groups):
  data      : src.tsplit.train.StreamDS = the S3 / FA stream: COCO rows 0..11,999, 16 images per update minus the 2 PACO val/test
              duplicates, hflip p 0.5 (rng per (step, slot)), 8 ROIs per image of 1..6 x 1..6 parents (clipped to the content)
  updates   : 12,000 optimizer updates, microbatch = the whole update (16 images, 128 ROIs), no gradient accumulation
  optimizer : AdamW betas (0.9, 0.95), wd 0.05 on ndim >= 2 params, warmup 600 (5 %) + cosine to 0, clip 1.0 (all trainable params)
              P1: lr 2e-4 for every trainable parameter; P2: copied blocks 9-12 lr 1e-5, child split projection lr 2e-4
  precision : bf16 autocast for the frozen encoders and the decoder forward; loss statistics in true FP32 (TF32 off inside losses)
  loss      : L = patch CE + lambda * SIGReg(Z)
              patch CE: Tn, Sn, Qn L2-normalised; p_T = softmax(Tn Qn^T / 0.07) over the 4,096 queue rows, CE = -sum p_T log p_S per valid
              child; mean over the valid children of a ROI, then mean over ROIs. Invalid children (letterbox: 8-px child cell does not
              overlap the content = the S3 rule; padding parents) are excluded from CE, SIGReg sampling and enqueue.
              SIGReg: reference kernel (src.finedec.losses.sigreg), raw Z, 1,024 samples (image-balanced, parent-first, children of
              overlapping ROIs deduplicated by fine id; fewer if fewer valid children, never duplicated), 256 directions, new each step.
  queue     : 4,096 rows, prefilled with the FA prefill (output/2026-11-11/fine_aware_shared_s0/queue_prefill.pt: 128 train images x 32,
              rng 20261114); after each successful optimizer step 32 teacher children per image are enqueued (rng 3e6 + step).
              P1 and P2 therefore see the same teacher reference sequence (the enqueue depends on the teacher only).
  RNG       : data = StreamDS (numpy per step / slot); SIGReg sampling 1e6 + step, directions 2e6 + step; enqueue 3e6 + step;
              calibration uses 9e6 + i / 9.1e6 + i and never touches these.
Calibration (--calibrate): 20 fixed batches (stream steps 0..19), init state, prefill snapshot, no optimizer update; per batch the
  unclipped gradient norms of CE and of unweighted SIGReg on the parameters on the path to Z (reached by both);
  lambda = 0.1 * sum ||g_CE|| / (sum ||g_SIG|| + 1e-12). Training reloads init.pt and starts at step 0 (state equal to pre-calibration).

  python -m src.fte.train --model P1 --run <run> --calibrate
  python -m src.fte.train --model P1 --run <run>
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
from src.finedec.losses import TeacherQueue, enqueue_samples, fp32, sample_latents, sigreg
from src.fte.models import P1, P2
from src.msd.pretrain import eval_exclusions
from src.tsplit.features import PREFIX, frozen_vit, teacher, valid48
from src.fte.common import child_ids
from src.tsplit.train import STEPS, StreamDS

PREFILL = "output/2026-11-11/fine_aware_shared_s0/queue_prefill.pt"
WARM, CLIP, TAU, WD, BETAS = 600, 1.0, 0.07, 0.05, (0.9, 0.95)
LRS = dict(P1=dict(all=2e-4), P2=dict(blocks=1e-5, split=2e-4))
STOP = False


def _stop(*_):
    global STOP
    STOP = True


@torch.no_grad()
def student_h8(vit, x, chunk=64):
    """One original frozen forward: final patch tokens (fp32) and blocks[7] output (all 581 tokens, raw, as produced)."""
    fin, h8 = [], []
    for i in range(0, len(x), chunk):
        store = []
        hk = vit.blocks[7].register_forward_hook(lambda m, a, o: store.append(o))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = vit.forward_features(x[i:i + chunk])
        finally:
            hk.remove()
        fin.append(t[:, PREFIX:].float())
        h8.append(store[0])
    return torch.cat(fin), torch.cat(h8)


def build(kind, dev):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    vit = frozen_vit(dev)
    m = (P1 if kind == "P1" else P2)(vit).to(dev)
    return vit, m


def sha_state(m):
    h = hashlib.sha256()
    sd = m.state_dict()
    for k in sorted(sd):
        if k.startswith("rope."):
            continue
        h.update(k.encode())
        h.update(sd[k].detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def make_opt(kind, m):
    groups = []
    if kind == "P1":
        named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        lr = LRS["P1"]["all"]
        groups += [dict(params=[p for n, p in named if p.ndim >= 2], weight_decay=WD, base=lr), dict(params=[p for n, p in named if p.ndim < 2], weight_decay=0.0, base=lr)]
    else:
        for part, lr in (("blocks", LRS["P2"]["blocks"]), ("split", LRS["P2"]["split"])):
            ps = list(getattr(m, part).parameters())
            groups += [dict(params=[p for p in ps if p.ndim >= 2], weight_decay=WD, base=lr), dict(params=[p for p in ps if p.ndim < 2], weight_decay=0.0, base=lr)]
    for g in groups:
        g["lr"] = g["base"]
    return torch.optim.AdamW(groups, betas=BETAS)


def lr_scale(step):
    if step < WARM:
        return (step + 1) / WARM
    return 0.5 * (1 + math.cos(math.pi * (step - WARM) / (STEPS - WARM)))


def prepare(vit, b, dev):
    x, x768 = b["x"].to(dev, non_blocking=True), b["x768"].to(dev, non_blocking=True)
    rects, rimg, content = b["rects"].to(dev), b["rimg"].to(dev), b["content"].to(dev)
    final, h8 = student_h8(vit, x)
    T = teacher(vit, x768)
    return dict(final=final, h8=h8, T=T, rects=rects, rimg=rimg, v48=valid48(content), n_img=len(x))


def run_model(kind, m, d):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Y, Z, pid, pok = m(d["final"] if kind == "P1" else d["h8"], d["rimg"], d["rects"])
    cid, _, _ = child_ids(pid)
    valid = pok[..., None] & d["v48"][d["rimg"][:, None, None], cid]
    Tt = d["T"][d["rimg"][:, None, None], cid]
    return Y, Z, cid, valid, Tt


def patch_ce_roi(Y, Tt, valid, Q, tau=TAU):
    """Per ROI: mean over valid children of -sum_a p_T,a log p_S,a; then mean over ROIs (ROIs with no valid child skipped)."""
    with fp32():
        R = Y.shape[0]
        v = valid.reshape(R, -1)
        roi = torch.arange(R, device=Y.device)[:, None].expand_as(v)[v]
        s = F.normalize(Y.reshape(R, -1, Y.shape[-1]).float()[v], dim=-1)
        t = F.normalize(Tt.reshape(R, -1, Tt.shape[-1]).float()[v], dim=-1)
        Qn = F.normalize(Q.float(), dim=-1)
        ls = F.log_softmax(s @ Qn.T / tau, -1)
        with torch.no_grad():
            lt = F.log_softmax(t @ Qn.T / tau, -1)
            pt = lt.exp()
            ent = -(pt * lt).sum(-1)
        ce = -(pt * ls).sum(-1)
        cnt = torch.zeros(R, device=Y.device).index_add_(0, roi, torch.ones_like(ce))
        ok = cnt > 0
        ce_roi = torch.zeros(R, device=Y.device).index_add_(0, roi, ce)[ok] / cnt[ok]
        ent_roi = torch.zeros(R, device=Y.device).index_add_(0, roi, ent)[ok] / cnt[ok]
        loss = ce_roi.mean()
    return loss, dict(patch_ce=float(loss), teacher_entropy=float(ent_roi.mean()), patch_kl=float(loss) - float(ent_roi.mean()), n_valid_children=int(v.sum()))


def sig_term(Z, cid, valid, rimg, seed_s, seed_d):
    R = Z.shape[0]
    zs = sample_latents(Z.reshape(R, -1, Z.shape[-1]).float(), valid.reshape(R, -1), rimg, cid.reshape(R, -1), budget=1024, seed=seed_s)
    return sigreg(zs, seed=seed_d), len(zs)


def flat_norm(gs):
    v = [g.float().flatten() for g in gs if g is not None]
    return torch.cat(v) if v else torch.zeros(1)


def calibrate(a, run, dev):
    vit, m = build(a.model, dev)
    m.load_state_dict(torch.load(run / "init.pt", map_location=dev, weights_only=False))
    assert sha_state(m) == json.loads((run / "init.json").read_text())["sha256"]
    m.train()
    Q = TeacherQueue(torch.load(PREFILL, weights_only=False)["features"].to(dev)).snapshot()
    ds = StreamDS(np.load(STREAM)[:STEPS], eval_exclusions()["raev2_stream"])
    P = m.latent_params()
    recs, sc, ss = [], 0.0, 0.0
    for i in range(20):
        d = prepare(vit, ds[i], dev)
        Y, Z, cid, valid, Tt = run_model(a.model, m, d)
        ce, st = patch_ce_roi(Y, Tt, valid, Q)
        sg, n = sig_term(Z, cid, valid, d["rimg"], 9_000_000 + i, 9_100_000 + i)
        gc = flat_norm(torch.autograd.grad(ce, P, retain_graph=True, allow_unused=True))
        gs = flat_norm(torch.autograd.grad(sg, P, allow_unused=True))
        assert all(p.grad is None for p in m.parameters())
        recs.append(dict(batch=i, ce=float(ce), sigreg=float(sg), samples=n, grad_ce=float(gc.norm()), grad_sigreg=float(gs.norm()), cos=float(F.cosine_similarity(gc, gs, dim=0))))
        sc += float(gc.norm())
        ss += float(gs.norm())
        print(json.dumps(recs[-1]), flush=True)
    lam = 0.1 * sc / (ss + 1e-12)
    out = dict(model=a.model, lambda_sigreg=lam, rule="lambda = 0.1 * sum ||grad CE|| / (sum ||grad SIGReg|| + 1e-12); 20 fixed batches (stream steps 0..19), init state, prefill queue snapshot, no optimizer update, unclipped gradients on the parameters on the path to Z",
               param_set=[n for n, p in m.named_parameters() if any(p is q for q in P)][:3] + ["..."], n_params_compared=sum(p.numel() for p in P), sum_grad_ce=sc, sum_grad_sigreg=ss, batches=recs,
               restored="training reloads init.pt (sha checked) and the prefill queue, starts at stream step 0; all loss RNG streams are seeded per step")
    (run / "calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(lambda_sigreg=lam)), flush=True)


def train(a, run, dev):
    out = run / "train"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        return
    vit, m = build(a.model, dev)
    m.load_state_dict(torch.load(run / "init.pt", map_location=dev, weights_only=False))
    init_sha = sha_state(m)
    assert init_sha == json.loads((run / "init.json").read_text())["sha256"]
    lam = json.loads((run / "calibration.json").read_text())["lambda_sigreg"]
    m.train()
    opt = make_opt(a.model, m)
    queue = TeacherQueue(torch.load(PREFILL, weights_only=False)["features"].to(dev))
    vit_sum0 = float(sum(p.double().sum() for p in vit.parameters()))
    frozen_sum0 = float(sum(p.double().sum() for p in m.parameters() if not p.requires_grad))
    ex = eval_exclusions()["raev2_stream"]
    start = 0
    rec = out / "recovery.pt"
    if rec.exists():
        ck = torch.load(rec, map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        queue.features.copy_(ck["queue"].to(dev))
        queue.pointer = ck["queue_pointer"]
        start = ck["step"]
    n_steps = 30 if a.smoke else STEPS
    params = m.trainable()
    cfg = dict(model=a.model, total_optimizer_updates=STEPS, images_per_update="16 stream slots minus excluded PACO duplicates (16 in almost every step)", rois_per_image=8,
               roi_parent_side=[1, 6], microbatch="whole update (all images and ROIs in one forward / backward)", grad_accumulation=1,
               sigreg_samples="min(1024, valid deduplicated children of the update)", lambda_sigreg=lam, tau=TAU, queue=4096, enqueue_per_image=32, lrs=LRS[a.model], betas=list(BETAS),
               wd=WD, warmup=WARM, schedule="linear warmup 600 then cosine to 0", clip=CLIP, amp="bf16 autocast; FP32 loss statistics with TF32 off", init_sha256=init_sha,
               trainable_params=sum(p.numel() for p in params), seed=0, smoke=a.smoke, torch=torch.__version__)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    dl = DataLoader(StreamDS(np.load(STREAM)[:STEPS], ex), sampler=range(start, n_steps), batch_size=None, num_workers=12, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    logs = {n: (out / f"{n}.jsonl").open("a") for n in ("train", "grads")}
    P = m.latent_params()
    names = {id(p): n for n, ps in m.groups().items() for p in ps}

    def save(step, final=False):
        tmp = out / "recovery.tmp"
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), step=step, queue=queue.features.cpu(), queue_pointer=queue.pointer), tmp)
        tmp.replace(rec)
        if final:
            torch.save(dict(kind=a.model, model=m.state_dict(), step=step, config=cfg), out / "model.pt")

    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.time()
    acc = {}
    for b in dl:
        step = b["step"]
        d = prepare(vit, b, dev)
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_scale(step)
        Y, Z, cid, valid, Tt = run_model(a.model, m, d)
        Qs = queue.snapshot()
        ce, st = patch_ce_roi(Y, Tt, valid, Qs)
        sg, n_s = sig_term(Z, cid, valid, d["rimg"], 1_000_000 + step, 2_000_000 + step)
        loss = ce + lam * sg
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        opt.zero_grad(set_to_none=True)
        diag = step < 3 or (step + 1) % 100 == 0
        grec = dict(step=step + 1, lambda_sigreg=lam)
        if diag:
            gc = torch.autograd.grad(ce, P, retain_graph=True, allow_unused=True)
            gs = torch.autograd.grad(lam * sg, P, retain_graph=True, allow_unused=True)
            fc, fs = flat_norm(gc), flat_norm(gs)
            grec.update(grad_ce=float(fc.norm()), grad_sigreg_weighted=float(fs.norm()), grad_cos=float(F.cosine_similarity(fc, fs, dim=0)), ratio_sig_over_ce=float(fs.norm() / fc.norm().clamp_min(1e-12)))
            assert all(p.grad is None for p in params)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"non-finite gradient at step {step + 1}")
        if diag:
            grec["total_before_clip"] = float(gn)
            for gname, ps in m.groups().items():
                gg = [p.grad.float().norm() for p in ps if p.grad is not None]
                grec[f"total/{gname}"] = float(torch.stack(gg).norm()) if gg else 0.0
            assert all(p.grad is None for p in vit.parameters()) and all(p.grad is None for p in m.parameters() if not p.requires_grad)
            logs["grads"].write(json.dumps(grec) + "\n")
            logs["grads"].flush()
        opt.step()
        queue.enqueue(enqueue_samples(Tt.reshape(len(Tt), -1, Tt.shape[-1]).float(), valid.reshape(len(Tt), -1), d["rimg"], cid.reshape(len(Tt), -1), per_image=32, seed=3_000_000 + step))
        if step < 3 or (step + 1) % 10 == 0:
            with torch.no_grad():
                v = valid
                Yf, Tf = Y.float(), Tt.float()
                vf = v.float()
                n = vf.sum()
                cos = (F.cosine_similarity(Yf, Tf, dim=-1) * vf).sum() / n
                full = v.all(-1)
                def pairc(X):
                    q = F.normalize(X, dim=-1)
                    gm = q @ q.transpose(-1, -2)
                    return ((gm.sum((-1, -2)) - 4) / 12)[full].mean()
                with fp32():
                    qn = F.normalize(Qs, dim=-1)
                    pq = F.softmax(F.normalize(Tf[v][:2048], dim=-1) @ qn.T / TAU, -1).mean(0)
                    queue_entropy = float(-(pq * pq.clamp_min(1e-12).log()).sum())
                stx = dict(cos=float(cos), norm_ratio=float((Yf.norm(dim=-1) * vf).sum() / (Tf.norm(dim=-1) * vf).sum()),
                           child_pair_cos_student=float(pairc(Yf)), child_pair_cos_teacher=float(pairc(Tf)), Z_norm=float(Z.float().norm(dim=-1)[v].mean()), Z_std=float(Z.float()[v].std(0).mean()),
                           queue_marginal_entropy=queue_entropy, sigreg_samples=n_s)
        else:
            stx = dict(sigreg_samples=n_s)
        vals = dict(loss=float(loss), sigreg=float(sg), **st, **stx)
        for k, vv in vals.items():
            acc.setdefault(k, []).append(vv)
        if step == start or (step + 1) % 50 == 0:
            now = time.time()
            r = dict(step=step + 1, lr_scale=lr_scale(step), elapsed_s=round(now - t0, 1), sec_per_step=round((now - last) / max(len(acc["loss"]), 1), 3), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                     **{k: float(np.nanmean(vv)) for k, vv in acc.items()})
            logs["train"].write(json.dumps(r) + "\n")
            logs["train"].flush()
            print(json.dumps({k: (round(vv, 5) if isinstance(vv, float) else vv) for k, vv in r.items() if k in ("step", "sec_per_step", "loss", "patch_ce", "patch_kl", "sigreg", "cos", "norm_ratio", "child_pair_cos_student", "Z_norm", "peak_gib")}), flush=True)
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
    (out / "done.json").write_text(json.dumps(dict(status="done", steps=n_steps, time_s=round(time.time() - t0), peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, frozen_unchanged=True, smoke=a.smoke), indent=1))
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("P1", "P2"), required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--init", action="store_true", help="write init.pt / init.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _stop)
    dev = torch.device("cuda")
    try:
        if a.init:
            _, m = build(a.model, dev)
            torch.save(m.state_dict(), run / "init.pt")
            (run / "init.json").write_text(json.dumps(dict(sha256=sha_state(m), trainable=sum(p.numel() for p in m.parameters() if p.requires_grad), frozen=sum(p.numel() for p in m.parameters() if not p.requires_grad)), indent=1))
            print((run / "init.json").read_text(), flush=True)
        elif a.calibrate:
            calibrate(a, run, dev)
        else:
            train(a, run, dev)
    except Exception:
        (run / f"failed_{'calib' if a.calibrate else 'train'}.json").write_text(json.dumps(dict(error=traceback.format_exc()), indent=1))
        raise


if __name__ == "__main__":
    main()
