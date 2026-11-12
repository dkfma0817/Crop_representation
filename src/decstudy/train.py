"""Decoder-only training on the FROZEN CLIP-L G1-cross encoder (2026-11-07 decoder study).

  F = frozen G1-cross EMA encoder (output/2026-11-07/g1clip/G1cross/ema_last.pt, key 'enc') on the full image (fp32)
  T = frozen ORIGINAL CLIP ViT-L/14@336 on the GT C0 crop (fp32) = the G1-cross crop teacher (unchanged)
  B = bilinear_roi(F, ROI);  Z = B + Delta,  Delta = ResidualDecoder(B, F, ROI)  (src.decstudy.model)
Conditions (each its OWN decoder, optimizer, EMA; same init (seed 7), same data stream, trained side by side in one
process so the teacher / encoder forward passes are computed once per step and shared; nothing else is shared):
  D1 : L = L_original                         (= lam_detail * L_detail, lam_detail = 1.11 from the G1-cross calibration)
  L1 : L = L_original + lam_cov * L_cov
  L2 : L = L_original + lam_rel * L_rel
Unchanged from G1-cross (src.g1clip.distill): COCO 12k stream (16 train2017 images x 4 ROIs / step, no flip) minus the
PACO val / test matches, 12,000 steps, world 1 (= the G1-cross run), decoder AdamW lr 3e-4 wd 0.05 (none on 1-D /
q_grid), betas (0.9, 0.999), warmup 600 + cosine, grad clip 1.0, EMA 0.999 -> 1 (cosine), decoder bf16 autocast,
encoders fp32 (TF32 off), final EMA weights are used downstream. The encoder never trains here.

  python -m src.decstudy.train --calib --out <run>/calib            (lam_cov / lam_rel: one-shot gradient-scale rule)
  python -m src.decstudy.train --conds D1,L1,L2 --weights <run>/calib/weights.json --out <run>/dec
"""
import argparse
import copy
import hashlib
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.config import FEAT, STREAM
from src.decstudy import losses as LS
from src.decstudy.model import build_residual
from src.g1clip.distill import data, run_enc, set_fp32
from src.g1clip.model import ROIDecoderD
from src.g1x.data import CocoDS
from src.config import COCO_TRAIN
from src.ivc.ft import ema_update
from src.maskemb import common as MC
from src.msd.common import StreamSampler

C = RC.C
S, P, D = MC.S, MC.P, MC.DIM
WARM, M0_EMA, DEC_LR, WD = 600, 0.999, 3e-4, 0.05
G1 = "output/2026-11-07/g1clip"
ENC_CKPT = f"{G1}/G1cross/ema_last.pt"
G1_WEIGHTS = f"{G1}/calib/weights.json"
CONDS = {"D1": {}, "L1": {"cov": True}, "L2": {"rel": True}}


def state_hash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_frozen(dev):
    """-> (G1-cross encoder (frozen), original CLIP teacher (frozen), info)."""
    ck = torch.load(ENC_CKPT, map_location="cpu", weights_only=False)
    assert ck["kind"] == "ema" and ck["cond"] == "cross" and ck["arch"] == MC.CLIP_ARCH
    vit, _ = MC.build_encoder("clip", dev)
    vit.load_state_dict(ck["enc"])
    enc0, _ = MC.build_encoder("clip", dev)
    for m in (vit, enc0):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    info = dict(enc_ckpt=ENC_CKPT, enc_ckpt_sha256=MC.sha256(ENC_CKPT), enc_state_sha256=state_hash(vit),
                teacher="original OpenAI CLIP ViT-L/14@336 (timm " + MC.CLIP_ARCH + ")", teacher_state_sha256=state_hash(enc0))
    return vit, enc0, info, ck


def lam_detail():
    return float(json.load(open(G1_WEIGHTS))["lam_detail"])


def cond_losses(name, dec, F_, T, b, lam_d, lam, row_all=None):
    """-> dict of scalar losses (step means) + tensors for diagnostics."""
    ri, cc = b["roi_img"], b["ccov"]
    with torch.autocast(F_.device.type, dtype=torch.bfloat16):
        Bf, dl = dec.parts(F_, ri, b["box_c"], b["qrel"], b["fvalid"])
    Z = Bf + dl
    tok, dp = LS.detail_terms(Z, T, cc)
    out = dict(Z=Z, B=Bf, Delta=dl, tok=tok, dpool=dp, orig=lam_d * (tok.mean() + dp.mean()))
    total = out["orig"]
    if CONDS[name].get("cov"):
        e, ok, _ = LS.cov_terms(Z, T, cc)
        out["cov"] = e[ok].mean() if ok.any() else Z.sum() * 0
        total = total + lam.get("lam_cov", 0.0) * out["cov"]
    if CONDS[name].get("rel"):
        kl = LS.rel_kl(LS.pool(Z, cc), LS.pool(T, cc))
        out["rel"] = kl.mean()
        total = total + lam.get("lam_rel", 0.0) * out["rel"]
    out["total"] = total
    return out


def gnorm(gs):
    sq = [(g.float() ** 2).sum() for g in gs if g is not None]
    return float(torch.sqrt(sum(sq))) if sq else 0.0


def per_loss_grads(o, dec, keys):
    ps = list(dec.parameters())
    r = {}
    for k in keys:
        if k not in o:
            continue
        g = torch.autograd.grad(o[k], ps, retain_graph=True, allow_unused=True)
        r[f"g_{k}"] = gnorm(g)
        r[f"loss_{k}"] = float(o[k])
        named = dict(zip([n for n, _ in dec.named_parameters()], g))
        r[f"g_{k}_out"] = gnorm([named["out.weight"], named["out.bias"]])
        r[f"g_{k}_body"] = gnorm([v for n, v in named.items() if not n.startswith("out.")])
    return r


# ------------------------------------------------------------------------------------------------ calibration
def calib(a, dev):
    """lam_cov, lam_rel = median over 16 stream steps at init of ||grad_dec L_original|| / ||grad_dec L_new|| (3 s.f.),
    the same one-shot gradient-scale rule that set lam_detail in G1-cross. No sweep. At init `out` is zero, so the
    decoder gradient sits on `out` (= ln_out(q)^T dL/dDelta); also logged: ||dL/dZ|| ratio (decoder-independent)."""
    set_fp32()
    MC.set_clip_norm()
    os.makedirs(a.out, exist_ok=True)
    vit, enc0, info, _ = load_frozen(dev)
    lam_d = lam_detail()
    ex, _, ds = data()
    stream = np.load(STREAM)
    rows = []
    for k in range(a.calib_steps):
        ids = [int(i) for i in stream[k] if int(i) not in set(ex["raev2_stream"])]
        b = RC.to_dev(ds[ids], dev)
        with torch.no_grad():
            T, _ = run_enc(enc0, b["crops"])
            F_, _ = run_enc(vit, b["full"])
        dec = build_residual(S, P, D).to(dev)
        lam = {}
        o1 = cond_losses("L1", dec, F_, T, b, lam_d, lam)
        o2 = cond_losses("L2", dec, F_, T, b, lam_d, lam)
        r = dict(step=k, n_roi=len(T))
        r.update({f"L1_{kk}": v for kk, v in per_loss_grads(o1, dec, ["orig", "cov"]).items()})
        r.update({f"L2_{kk}": v for kk, v in per_loss_grads(o2, dec, ["orig", "rel"]).items()})
        for nm, o, key in (("cov", o1, "cov"), ("rel", o2, "rel")):
            g_orig = torch.autograd.grad(o["orig"], o["Z"], retain_graph=True)[0]
            g_new = torch.autograd.grad(o[key], o["Z"], retain_graph=True)[0]
            r[f"dZ_orig_{nm}"] = float(g_orig.norm())
            r[f"dZ_{nm}"] = float(g_new.norm())
        st = LS.pool(T, b["ccov"])
        S_t = LS.rel_matrix(st)
        off = ~torch.eye(len(st), dtype=torch.bool, device=dev)
        pt = torch.softmax((S_t / LS.TAU).masked_fill(~off, float("-inf")), -1)
        r["teacher_rel_offdiag_mean"] = float(S_t[off].mean())
        r["teacher_rel_offdiag_std"] = float(S_t[off].std())
        r["teacher_softmax_entropy_over_log_n"] = float((-(pt * pt.clamp_min(1e-12).log()).sum(-1)).mean() / math.log(len(st) - 1))
        r["cov_valid_rois"] = int(((b["ccov"] > 0).sum(1) >= LS.MIN_TOK).sum())
        r["valid_tokens_min"] = int((b["ccov"] > 0).sum(1).min())
        rows.append(r)
        print(json.dumps(r), flush=True)
    df = pd.DataFrame(rows)
    rc = (df["L1_g_orig"] / df["L1_g_cov"]).values
    rr = (df["L2_g_orig"] / df["L2_g_rel"]).values
    w = dict(lam_detail=lam_d, lam_cov=float(f"{np.median(rc):.3g}"), lam_rel=float(f"{np.median(rr):.3g}"), tau=LS.TAU, min_tok=LS.MIN_TOK,
             rule=f"lam_x = median(||grad_dec L_original|| / ||grad_dec L_x||) over {a.calib_steps} stream steps at init (3 s.f.); fixed once, no sweep",
             ratio_cov_iqr=[float(np.percentile(rc, 25)), float(np.percentile(rc, 75))],
             ratio_rel_iqr=[float(np.percentile(rr, 25)), float(np.percentile(rr, 75))],
             ratio_cov_dZ_median=float(np.median(df.dZ_orig_cov / df.dZ_cov)), ratio_rel_dZ_median=float(np.median(df.dZ_orig_rel / df.dZ_rel)),
             loss_cov_init_median=float(df.L1_loss_cov.median()), loss_rel_init_median=float(df.L2_loss_rel.median()),
             loss_orig_init_median=float(df.L1_loss_orig.median()),
             teacher_rel_offdiag_mean=float(df.teacher_rel_offdiag_mean.mean()), teacher_rel_offdiag_std=float(df.teacher_rel_offdiag_std.mean()),
             teacher_softmax_entropy_over_log_n=float(df.teacher_softmax_entropy_over_log_n.mean()),
             body_grad_at_init=dict(L1_orig=float(df.L1_g_orig_body.max()), L2_orig=float(df.L2_g_orig_body.max())),
             encoder=info, time=time.strftime("%F %T"))
    df.to_csv(os.path.join(a.out, "calib_grads.csv"), index=False)
    json.dump(w, open(os.path.join(a.out, "weights.json"), "w"), indent=1)
    print(json.dumps(w), flush=True)


# ------------------------------------------------------------------------------------------------ COCO dev diagnostics
@torch.no_grad()
def zstats(Z, T, cc, Bf=None):
    """Per-chunk sums for Table C. Returns dict name -> (sum, count)."""
    tok, dp = LS.detail_terms(Z, T, cc)
    e, ok, sim = LS.cov_terms(Z, T, cc)
    pz, pt = LS.pool(Z, cc), LS.pool(T, cc)
    kl = LS.rel_kl(pz, pt)
    r = dict(tok_cos=(1 - tok).sum(), pool_cos=(1 - dp).sum(), l_detail=(tok + dp).sum(), cov_err=e[ok].sum(), cov_sim=sim[ok].sum(),
             sv_ratio=(LS.sv(Z, cc) / LS.sv(T, cc)).sum(), sv_student=LS.sv(Z, cc).sum(), sv_teacher=LS.sv(T, cc).sum(), rel_kl=kl.sum(),
             norm_ratio=((Z.float().norm(dim=-1) * (cc > 0)).sum(1) / (T.float().norm(dim=-1) * (cc > 0)).sum(1)).sum())
    n = dict(cov_err=int(ok.sum()), cov_sim=int(ok.sum()))
    out = {k: (float(v), n.get(k, len(Z))) for k, v in r.items()}
    out["rel_r"] = (LS.relation_preservation(pz, pt), 1)
    if Bf is not None:
        m = (cc > 0).float()[..., None]
        dn = ((Z - Bf).float() * m).flatten(1).norm(dim=1)
        bn = (Bf.float() * m).flatten(1).norm(dim=1)
        out["delta_over_b"] = (float((dn / bn.clamp_min(1e-12)).sum()), len(Z))
    return out


def merge(acc, st):
    for k, (s, n) in st.items():
        a = acc.setdefault(k, [0.0, 0])
        a[0] += s
        a[1] += n


def finish(acc):
    return {k: (s / n if n else None) for k, (s, n) in acc.items()}


def dev_set(dev_max=0):
    ex = data()[0]
    dr = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    dr = dr[~dr.image_id.isin(ex["raev2_dev"])]
    ids = sorted(int(i) for i in dr.image_id.unique())[: (dev_max or None)]
    return CocoDS(dr, COCO_TRAIN, S, P), [ids[i:i + 16] for i in range(0, len(ids), 16)], int(dr.image_id.isin(ids).sum())


@torch.no_grad()
def dev_diag(paths, vit, enc0, dev_ds, chunks, dev):
    """paths: name -> callable(F, b) -> (Z, B or None). Same dev ROIs / teacher for every path."""
    acc = {k: {} for k in paths}
    for ch in chunks:
        b = RC.to_dev(dev_ds[ch], dev)
        T, _ = run_enc(enc0, b["crops"])
        F_, _ = run_enc(vit, b["full"])
        for k, fn in paths.items():
            Z, Bf = fn(F_, b)
            merge(acc[k], zstats(Z, T, b["ccov"], Bf))
    return {k: finish(v) for k, v in acc.items()}


def path_residual(dec):
    def f(F_, b):
        with torch.autocast(F_.device.type, dtype=torch.bfloat16):
            Bf, dl = dec.parts(F_, b["roi_img"], b["box_c"], b["qrel"], b["fvalid"])
        return Bf + dl, Bf
    return f


def path_bilinear():
    from src.decstudy.model import bilinear_roi

    def f(F_, b):
        Bf = bilinear_roi(F_, b["roi_img"], b["box_c"], b["qrel"], S)
        return Bf, None
    return f


def path_existing(dec_old):
    def f(F_, b):
        with torch.autocast(F_.device.type, dtype=torch.bfloat16):
            Z = dec_old(F_, b["roi_img"], b["box_c"], b["qrel"], b["fvalid"]).float()
        return Z, None
    return f


# ------------------------------------------------------------------------------------------------ training
def train(a, dev):
    set_fp32()
    MC.set_clip_norm()
    os.makedirs(a.out, exist_ok=True)
    names = a.conds.split(",")
    for n in names:
        assert n in CONDS and not os.path.exists(os.path.join(a.out, n, "done.json")), f"{n}: run exists (never overwritten)"
        os.makedirs(os.path.join(a.out, n), exist_ok=True)
    w = json.load(open(a.weights))
    lam_d = lam_detail()
    assert abs(lam_d - w["lam_detail"]) < 1e-12
    lam = {"lam_cov": float(w["lam_cov"]), "lam_rel": float(w["lam_rel"])}
    vit, enc0, info, ck = load_frozen(dev)
    decs, emas, opts = {}, {}, {}
    for n in names:
        decs[n] = build_residual(S, P, D).to(dev).train()
        emas[n] = copy.deepcopy(decs[n]).eval()
        for p in emas[n].parameters():
            p.requires_grad_(False)
        dg = {}
        for pn, p in decs[n].named_parameters():
            k = p.ndim <= 1 or pn == "q_grid"
            dg.setdefault(k, dict(params=[], weight_decay=0.0 if k else WD, base=DEC_LR, lr=DEC_LR))
            dg[k]["params"].append(p)
        opts[n] = torch.optim.AdamW(list(dg.values()), betas=(0.9, 0.999))
    init_hash = {n: state_hash(decs[n]) for n in names}
    assert len(set(init_hash.values())) == 1, "all conditions must start from the same decoder state"
    ex, rois, ds = data()
    stream = np.load(STREAM)[: a.steps]
    total = len(stream)
    sampler = StreamSampler(stream, 0, 1, exclude=ex["raev2_stream"])
    dl = DataLoader(ds, sampler=sampler, batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dev_ds, chunks, n_dev_rois = dev_set(a.dev_max)
    cfg = dict(conds={n: CONDS[n] for n in names}, lam_detail=lam_d, **lam, tau=LS.TAU, min_tok=LS.MIN_TOK, steps=total, warmup=WARM,
               world=1, dec_lr=DEC_LR, wd=WD, ema_m0=M0_EMA, rois_per_step=64, excluded_stream_images=ex["raev2_stream"],
               n_stream_images_unique=int(len(set(int(i) for i in stream.ravel()) - set(ex["raev2_stream"]))),
               n_dev_images=sum(len(c) for c in chunks), n_dev_rois=n_dev_rois, encoder=info, decoder_init_sha256=init_hash[names[0]],
               decoder_params=sum(p.numel() for p in decs[names[0]].parameters()), S=S, P=P, D=D, encoder_trainable=False,
               encoder_precision="fp32 (TF32 off)", decoder_precision="bf16 autocast (= G1-cross); B and Z = B + Delta in fp32",
               args=vars(a), weights=w)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
    print(json.dumps(cfg, default=str), flush=True)
    logs = {n: open(os.path.join(a.out, n, "train_log.jsonl"), "a") for n in names}
    glog = {n: open(os.path.join(a.out, n, "grad_check.jsonl"), "a") for n in names}
    dlog = open(os.path.join(a.out, "diag.jsonl"), "a")
    t0 = time.time()

    def diag(step, extra=None):
        paths = {n: path_residual(emas[n]) for n in names}
        if extra:
            paths.update(extra)
        r = dev_diag(paths, vit, enc0, dev_ds, chunks, dev)
        for k, v in r.items():
            rec = dict(step=step, model=k, s=round(time.time() - t0), **v)
            dlog.write(json.dumps(rec) + "\n")
            dlog.flush()
            print(json.dumps(rec), flush=True)

    dec_old = ROIDecoderD(S, P, D).to(dev)
    dec_old.load_state_dict(ck["dec"])
    dec_old.eval()
    diag(0, {"B0_bilinear": path_bilinear(), "existing_g1cross_decoder": path_existing(dec_old)})
    del dec_old
    run = {n: np.zeros(6) for n in names}
    peak = 0.0
    for step, b in enumerate(dl):
        b = RC.to_dev(b, dev)
        with torch.no_grad():
            T, _ = run_enc(enc0, b["crops"])
            F_, _ = run_enc(vit, b["full"])
        mt = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
        for n in names:
            dec, opt = decs[n], opts[n]
            for g in opt.param_groups:
                g["lr"] = C.lr_at(step, g["base"], WARM, total)
            o = cond_losses(n, dec, F_, T, b, lam_d, lam)
            if step == 0:
                chk = dict(step=0, delta_max_abs=float(o["Delta"].abs().max()), z_equals_b=bool(torch.equal(o["Z"], o["B"])))
                assert chk["delta_max_abs"] == 0.0 and chk["z_equals_b"], chk
                glog[n].write(json.dumps(chk) + "\n")
            if step in (0, 1, 2, 10) or (step + 1) % 1000 == 0:
                gr = per_loss_grads(o, dec, ["orig", "cov", "rel", "total"])
                gr.update(step=step + 1, delta_rel=float(o["Delta"].norm() / o["B"].norm()))
                glog[n].write(json.dumps(gr) + "\n")
                glog[n].flush()
            opt.zero_grad(set_to_none=True)
            o["total"].backward()
            gd = float(torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0))
            opt.step()
            ema_update(emas[n], dec, mt)
            if not torch.isfinite(o["total"].detach()):
                raise FloatingPointError(f"{n}: non-finite loss at step {step}")
            run[n] += np.array([float(o["orig"]), float(o.get("cov", torch.zeros(()))), float(o.get("rel", torch.zeros(()))),
                                float(o["total"]), gd, 1.0])
        peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
        if (step + 1) % 100 == 0:
            for n in names:
                r = run[n] / run[n][5]
                rec = dict(step=step + 1, s=round(time.time() - t0), l_orig=round(r[0], 5), l_cov=round(r[1], 5), l_rel=round(r[2], 5),
                           l_total=round(r[3], 5), gnorm_dec=round(r[4], 4), ema_m=round(mt, 6),
                           lr=opts[n].param_groups[0]["lr"], peak_mem_gb=round(peak, 1))
                logs[n].write(json.dumps(rec) + "\n")
                logs[n].flush()
                run[n][:] = 0
            print(json.dumps(dict(step=step + 1, s=round(time.time() - t0), peak_mem_gb=round(peak, 1))), flush=True)
        if (step + 1) % a.diag_every == 0 and step + 1 < total:
            diag(step + 1)
    diag(total)
    enc_after = state_hash(vit)
    t_after = state_hash(enc0)
    for n in names:
        torch.save(dict(dec=emas[n].state_dict(), step=total, kind="ema", cond=n, arch="ResidualDecoder", D=D, enc_ckpt=ENC_CKPT,
                        lam=dict(lam_detail=lam_d, **lam)), os.path.join(a.out, n, "ema_last.pt"))
        torch.save(dict(dec=decs[n].state_dict(), step=total, kind="student", cond=n), os.path.join(a.out, n, "student_last.pt"))
        json.dump(dict(status="done", steps=total, time_s=round(time.time() - t0), time=time.strftime("%F %T"), peak_mem_gb=round(peak, 1),
                       gpu=torch.cuda.get_device_name(), shared_process_with=names, encoder_hash_before=info["enc_state_sha256"],
                       encoder_hash_after=enc_after, encoder_unchanged=enc_after == info["enc_state_sha256"],
                       teacher_unchanged=t_after == info["teacher_state_sha256"], encoder_grads_none=all(p.grad is None for p in vit.parameters())),
                  open(os.path.join(a.out, n, "done.json"), "w"), indent=1)
    assert enc_after == info["enc_state_sha256"] and t_after == info["teacher_state_sha256"]
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="D1,L1,L2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib_steps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--diag_every", type=int, default=3000)
    ap.add_argument("--dev_max", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.calib:
        return calib(a, dev)
    os.makedirs(a.out, exist_ok=True)
    try:
        train(a, dev)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


if __name__ == "__main__":
    main()
