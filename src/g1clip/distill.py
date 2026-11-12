"""G1-pool / G1-cross ported to OpenAI CLIP ViT-L/14@336 (2026-11-07) = src.g1x.distill (DINOv3-B/16 @384) with ONLY the
backbone-dependent pieces changed; losses, data, schedule, EMA and checkpoint rule are the existing ones (user decision:
existing code, not the prose of the request).

  teacher  E_0 = frozen ORIGINAL CLIP ViT-L/14@336 (timm vit_large_patch14_clip_quickgelu_336.openai == OpenAI .pt):
           E_0(x_crop) -> T (576 C0 crop-grid tokens, final-norm (ln_post) patch tokens, no projection, 1024-d),
           E_0(x_full) -> original tokens / CLS for L_global
  student  E_s = trainable copy of the same weights: E_s(x_full) -> F (576 full tokens), CLS (ln_post, no projection)
  G1-pool : L = L_roi + lam_g L_global
  G1-cross: L = L_roi + lam_d L_detail + lam_g L_global,  Z = ROIDecoderD(F, ROI box) (D = 1024), trained jointly
  L_roi    = mean_ROI [1 - cos(sum_k w_k F_k / sum_k w_k , pool_c(T))]
  L_detail = mean_ROI [sum_q c_q (1 - cos(Z_q, T_q)) / sum_q c_q] + mean_ROI [1 - cos(pool_c(Z), pool_c(T))]
  L_global = mean_img 1/2 [(1 - cos(CLS_s, CLS_0)) + (1 - cos(mean patch_s, mean patch_0))]
Unchanged: COCO 12k stream (16 train2017 images x 4 ROIs / step, no flip) minus images matching PACO val / test,
12,000 steps, encoder AdamW lr 5e-5 at the top with layer-wise decay 0.75, wd 0.05 (none on 1-D / tokens / pos), decoder
lr 3e-4 wd 0.05, betas (0.9, 0.999), warmup 600 + cosine, grad clip 1.0 per module, EMA m 0.999 -> 1 (cosine), the
PACO probes use the FINAL EMA weights; lam_detail calibrated by the same rule (median init gradient ratio over 16
stream steps), lam_global = 0.5.
Backbone changes (recorded in config.json):
  * canvases 336 / 14 (24 x 24 grid as before), OpenAI CLIP mean / std (src.maskemb.common.set_clip_norm)
  * encoder precision fp32 with TF32 OFF (bf16 / TF32 corrupt CLIP ViT-L tokens: sanity/precision.json of maskembed_paco);
    the decoder keeps bf16 autocast as in the original
  * width 1024 for the decoder (src.g1clip.model.ROIDecoderD)
  * layer-wise LR: `norm_pre` (CLIP ln_pre, at the INPUT) gets the layer-0 rate; src.ivc.ft.enc_param_groups would match
    it by its "norm" prefix and give it the top rate. 24 blocks -> lowest blocks get 5e-5 * 0.75^24 (recorded, not tuned)

  python -m src.g1clip.distill --calib --out <run>/calib
  torchrun --nproc_per_node 2 -m src.g1clip.distill --cond cross --weights <run>/calib/weights.json --out <run>/G1cross
  python -m src.g1clip.distill --smoke --out <run>/smoke     (CPU: shapes, gradient flow, detail loss -> encoder)
"""
import argparse
import copy
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT, STREAM
from src.e6atas.losses import relation_error_full
from src.g1x.data import CocoDS
from src.g1clip.model import ROIDecoderD
from src.ivc.ft import ema_update
from src.maskemb import common as MC
from src.msd.common import StreamSampler
from src.msd.pretrain import eval_exclusions

C = RC.C
S, P, D = MC.S, MC.P, MC.DIM
WARM, M0_EMA, ENC_LR, LLRD, DEC_LR, WD = 600, 0.999, 5e-5, 0.75, 3e-4, 0.05


def pool(x, w):
    w = w.float()
    return (x.float() * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)


def cosd(a, b):
    return 1 - F.cosine_similarity(a.float(), b.float(), dim=-1)


def sv(x, w):
    w = w.float()
    mu = pool(x, w)
    return ((x.float() - mu[:, None]).pow(2).mean(-1) * w).sum(1) / w.sum(1).clamp_min(1e-12)


def set_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_enc(vit, x, chunk=32):
    """fp32 (no autocast) final-norm tokens -> (patch tokens, CLS); = src.ivc.ft.run_enc without bf16."""
    out = [vit.forward_features(x[i:i + chunk]) for i in range(0, len(x), chunk)]
    t = torch.cat(out).float()
    return t[:, vit.num_prefix_tokens:], t[:, 0]


def enc_param_groups(vit, base, decay, wd):
    """= src.ivc.ft.enc_param_groups except `norm_pre` (input LayerNorm) -> layer 0."""
    nb = len(vit.blocks)
    g = {}
    for n, p in vit.named_parameters():
        if n.startswith("blocks."):
            lid = int(n.split(".")[1]) + 1
        elif n.startswith(("norm.", "fc_norm", "head")):
            lid = nb + 1
        else:
            lid = 0                                       # patch_embed, cls_token, pos_embed, norm_pre
        lr = base * decay ** (nb + 1 - lid)
        no_wd = p.ndim <= 1 or "token" in n or "pos_embed" in n
        k = (lid, no_wd)
        g.setdefault(k, dict(params=[], weight_decay=0.0 if no_wd else wd, base=lr, lr=lr, lid=lid))
        g[k]["params"].append(p)
    return list(g.values())


def build(cond, dev):
    enc, _ = MC.build_encoder("clip", dev)
    enc0, _ = MC.build_encoder("clip", dev)               # frozen original teacher (never trains)
    for p in enc.parameters():
        p.requires_grad_(True)
    enc.train()
    torch.manual_seed(7)                                  # decoder init seed (calibration == training init)
    dec = ROIDecoderD(S, P, D).to(dev) if cond == "cross" else None
    return enc, enc0, dec


def terms(b, enc, enc0, dec):
    with torch.no_grad():
        T, _ = run_enc(enc0, b["crops"])
        otok, ocls = run_enc(enc0, b["full"])
    ftok, fcls = run_enc(enc, b["full"])
    ri, wf, cc, fv = b["roi_img"], b["wfull"], b["ccov"], b["fvalid"].float()
    tpool = pool(T, cc)
    out = dict(T=T, tpool=tpool, ftok=ftok, fcls=fcls, otok=otok, ocls=ocls,
               roi=cosd(pool(ftok[ri], wf), tpool),
               glob=0.5 * (cosd(fcls, ocls) + cosd(pool(ftok, fv), pool(otok, fv))))
    if dec is not None:
        with torch.autocast(ftok.device.type, dtype=torch.bfloat16):
            Z = dec(ftok, ri, b["box_c"], b["qrel"], b["fvalid"]).float()
        out.update(Z=Z, tok=(cosd(Z, T) * cc).sum(1) / cc.sum(1), dpool=cosd(pool(Z, cc), tpool))
    return out


def gnorm(grads):
    sq = [(g.float() ** 2).sum() for g in grads if g is not None]
    return float(torch.sqrt(sum(sq))) if sq else 0.0


def per_loss_grads(o, enc, dec):
    ep = [p for p in enc.parameters() if p.requires_grad]
    r = {}
    ls = {"roi": o["roi"].mean(), "global": o["glob"].mean()}
    if dec is not None:
        ls["detail"] = o["tok"].mean() + o["dpool"].mean()
    for k, l in ls.items():
        g = torch.autograd.grad(l, ep + (list(dec.parameters()) if dec is not None else []), retain_graph=True, allow_unused=True)
        r[f"enc_{k}"] = gnorm(g[:len(ep)])
        if dec is not None:
            r[f"dec_{k}"] = gnorm(g[len(ep):])
        r[f"loss_{k}"] = float(l)
    return r


def data(dev_split=False):
    ex = eval_exclusions()
    rois = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    return ex, rois, CocoDS(rois, COCO_TRAIN, S, P)


# ------------------------------------------------------------------------------------------------ CPU smoke
def smoke(a):
    """CPU: 2 stream images (8 ROIs). Shapes, finite losses, gradient flow of EVERY loss into the student encoder,
    L_detail reaches the encoder (through the decoder keys), teacher untouched, decoder absent in the pool condition."""
    torch.set_num_threads(48)
    MC.set_clip_norm()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cpu")
    ex, rois, ds = data()
    ids = [int(i) for i in np.load(STREAM)[0] if int(i) not in set(ex["raev2_stream"])][:2]
    b = ds[ids]
    out = dict(images=ids, shapes={k: list(v.shape) for k, v in b.items() if torch.is_tensor(v)})
    for cond in ("pool", "cross"):
        enc, enc0, dec = build(cond, dev)
        t_sum0 = float(sum(p.double().sum() for p in enc0.parameters()))
        o = terms(b, enc, enc0, dec)
        r = dict(T=list(o["T"].shape), ftok=list(o["ftok"].shape), fcls=list(o["fcls"].shape), roi=o["roi"].tolist(), glob=o["glob"].tolist(),
                 decoder_present=dec is not None, teacher_requires_grad=any(p.requires_grad for p in enc0.parameters()))
        if dec is not None:
            r.update(Z=list(o["Z"].shape), tok=o["tok"].tolist(), dpool=o["dpool"].tolist())
        r["grads"] = per_loss_grads(o, enc, dec)
        if dec is not None:                                # detail loss -> encoder: blocks 23 / 0 and norm grads
            ep = dict(enc.named_parameters())
            names = ["norm.weight", "blocks.23.mlp.fc2.weight", "blocks.12.attn.qkv.weight", "blocks.0.attn.qkv.weight"]
            g = torch.autograd.grad(o["tok"].mean() + o["dpool"].mean(), [ep[n] for n in names], retain_graph=True, allow_unused=True)
            r["detail_grad_norm_by_param"] = {n: (float(x.norm()) if x is not None else None) for n, x in zip(names, g)}
        loss = o["roi"].mean() + 0.5 * o["glob"].mean() + (o["tok"].mean() + o["dpool"].mean() if dec is not None else 0)
        loss.backward()
        r["teacher_unchanged"] = float(sum(p.double().sum() for p in enc0.parameters())) == t_sum0
        r["teacher_grads_none"] = all(p.grad is None for p in enc0.parameters())
        r["param_groups"] = sorted({(g_["lid"], round(g_["base"], 12)) for g_ in enc_param_groups(enc, ENC_LR, LLRD, WD)})[:3] + \
            sorted({(g_["lid"], round(g_["base"], 12)) for g_ in enc_param_groups(enc, ENC_LR, LLRD, WD)})[-2:]
        npre = [g_ for g_ in enc_param_groups(enc, ENC_LR, LLRD, WD) if any(p is enc.norm_pre.weight for p in g_["params"])][0]
        r["norm_pre_lid"] = npre["lid"]
        out[cond] = r
        print(cond, json.dumps({k: v for k, v in r.items() if k in ("T", "ftok", "Z", "grads", "detail_grad_norm_by_param", "norm_pre_lid", "teacher_unchanged")}), flush=True)
    assert out["cross"]["grads"]["enc_detail"] > 0 and all(v and v > 0 for v in out["cross"]["detail_grad_norm_by_param"].values())
    assert out["pool"]["grads"]["enc_roi"] > 0 and not out["pool"]["decoder_present"]
    assert out["pool"]["teacher_unchanged"] and out["cross"]["teacher_unchanged"] and out["cross"]["norm_pre_lid"] == 0
    json.dump(out, open(os.path.join(a.out, "smoke_cpu.json"), "w"), indent=1)
    print("SMOKE OK", flush=True)


# ------------------------------------------------------------------------------------------------ calibration
def calib(a, dev):
    """= src.g1x.distill.calib (same rule) on CLIP-L."""
    set_fp32()
    MC.set_clip_norm()
    os.makedirs(a.out, exist_ok=True)
    enc, enc0, dec = build("cross", dev)
    ex, rois, ds = data()
    stream = np.load(STREAM)
    rows = []
    for k in range(a.calib_steps):
        ids = [int(i) for i in stream[k] if int(i) not in set(ex["raev2_stream"])]
        b = RC.to_dev(ds[ids], dev)
        o = terms(b, enc, enc0, dec)
        r = per_loss_grads(o, enc, dec)
        r["step"] = k
        rows.append(r)
        print(json.dumps(r), flush=True)
    df = pd.DataFrame(rows)
    ratio = (df.enc_roi / df.enc_detail).values
    lam_d = float(f"{np.median(ratio):.3g}")
    w = dict(lam_detail=lam_d, lam_global=0.5, rule="lam_detail = median(||g_enc L_roi|| / ||g_enc L_detail||) over "
             f"{a.calib_steps} stream steps at init (3 s.f.); lam_global = existing G1 value (init gradient of L_global = 0)",
             median_ratio=float(np.median(ratio)), ratio_iqr=[float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
             enc_global_max=float(df.enc_global.max()), loss_global_max=float(df.loss_global.max()),
             enc_detail_min=float(df.enc_detail.min()), dec_detail_median=float(df.dec_detail.median()), time=time.strftime("%F %T"))
    df.to_csv(os.path.join(a.out, "calib_grads.csv"), index=False)
    json.dump(w, open(os.path.join(a.out, "weights.json"), "w"), indent=1)
    print(json.dumps(w), flush=True)


# ------------------------------------------------------------------------------------------------ dev diagnostics
@torch.no_grad()
def dev_diag(enc, enc0, dec, dev_ds, chunks, dev):
    keys = ["roi_pool_cos", "full_cls_cos", "full_mean_cos", "full_tok_cos", "in_tok_cos", "out_tok_cos", "img_out_tok_cos",
            "dec_tok_cos", "dec_pool_cos", "sv_ratio", "rel_err"]
    acc = {k: torch.zeros(2, dtype=torch.float64, device=dev) for k in keys}
    was = enc.training
    enc.eval()
    if dec is not None:
        dec.eval()

    def add(k, v):
        acc[k] += torch.stack([v.double().sum(), torch.tensor(float(v.numel()), device=dev, dtype=torch.float64)])

    for ch in chunks:
        b = RC.to_dev(dev_ds[ch], dev)
        o = terms(b, enc, enc0, dec)
        ri, wf, fv = b["roi_img"], b["wfull"], b["fvalid"]
        add("roi_pool_cos", 1 - o["roi"])
        add("full_cls_cos", 1 - cosd(o["fcls"], o["ocls"]))
        add("full_mean_cos", 1 - cosd(pool(o["ftok"], fv), pool(o["otok"], fv)))
        tc = F.cosine_similarity(o["ftok"], o["otok"], dim=-1)
        add("full_tok_cos", (tc * fv).sum(1) / fv.sum(1))
        tr = tc[ri]
        inside, outside = (wf > 0).float(), ((wf == 0) & fv[ri]).float()
        add("in_tok_cos", (tr * inside).sum(1) / inside.sum(1).clamp_min(1))
        ok = outside.sum(1) > 0
        add("out_tok_cos", ((tr * outside).sum(1) / outside.sum(1).clamp_min(1))[ok])
        cover = torch.zeros(fv.shape, device=dev)
        cover.index_add_(0, ri, (wf > 0).float())
        im_out = (cover == 0) & fv
        okm = im_out.sum(1) > 0
        add("img_out_tok_cos", ((tc * im_out).sum(1) / im_out.sum(1).clamp_min(1))[okm])
        if dec is not None:
            cc = b["ccov"]
            add("dec_tok_cos", 1 - o["tok"])
            add("dec_pool_cos", 1 - o["dpool"])
            add("sv_ratio", sv(o["Z"], cc) / sv(o["T"], cc))
            add("rel_err", relation_error_full(o["Z"], o["T"], cc))
    for k in keys:
        RC.all_sum(acc[k])
    if was:
        enc.train()
        if dec is not None:
            dec.train()
    return {k: (float(v[0] / v[1]) if float(v[1]) > 0 else None) for k, v in acc.items()}


# ------------------------------------------------------------------------------------------------ training
def train(a, rank, world, dev):
    set_fp32()
    MC.set_clip_norm()
    assert not os.path.exists(os.path.join(a.out, "done.json")), "run exists (never overwritten)"
    w = json.load(open(a.weights))
    lam_d = float(w["lam_detail"]) if a.cond == "cross" else 0.0
    lam_g = float(w["lam_global"])
    enc, enc0, dec = build(a.cond, dev)
    enc_t = copy.deepcopy(enc).eval()
    dec_t = copy.deepcopy(dec).eval() if dec is not None else None
    for p in list(enc_t.parameters()) + (list(dec_t.parameters()) if dec is not None else []):
        p.requires_grad_(False)
    eg = enc_param_groups(enc, ENC_LR, LLRD, WD)
    for g in eg:
        g["mod"] = "enc"
    groups = eg
    if dec is not None:
        dg = {}
        for n, p in dec.named_parameters():
            k = p.ndim <= 1 or n == "q_grid"
            dg.setdefault(k, dict(params=[], weight_decay=0.0 if k else WD, base=DEC_LR, lr=DEC_LR, mod="dec"))
            dg[k]["params"].append(p)
        groups = eg + list(dg.values())
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    ex, rois, ds = data()
    dev_rois = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex["raev2_dev"])]
    dev_ids = sorted(int(i) for i in dev_rois.image_id.unique())[: (a.dev_max or None)]
    stream = np.load(STREAM)[: a.steps]
    total = len(stream)
    dev_ds = CocoDS(dev_rois, COCO_TRAIN, S, P)
    dev_chunks = [dev_ids[i:i + 8] for i in range(0, len(dev_ids), 8)][rank::world]
    sampler = StreamSampler(stream, rank, world, exclude=ex["raev2_stream"])
    dl = DataLoader(ds, sampler=sampler, batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    cfg = dict(cond=a.cond, lam_detail=lam_d, lam_global=lam_g, weights_file=a.weights, steps=total, warmup=WARM, world=world,
               enc_lr_top=ENC_LR, llrd=LLRD, enc_lr_lowest=ENC_LR * LLRD ** (len(enc.blocks) + 1), dec_lr=DEC_LR, wd=WD, ema_m0=M0_EMA,
               rois_per_step=64, excluded=dict(stream=ex["raev2_stream"], dev=ex["raev2_dev"]), n_dev_images=len(dev_ids),
               n_dev_rois=int(dev_rois.image_id.isin(dev_ids).sum()), encoder=MC.CLIP_ARCH, S=S, P=P, D=D,
               encoder_precision="fp32 (TF32 off)", decoder_precision="bf16 autocast (= original)", norm=dict(mean=MC.CLIP_MEAN, std=MC.CLIP_STD),
               norm_pre_layer="0 (input)", n_params_enc=sum(p.numel() for p in enc.parameters()),
               n_params_dec=sum(p.numel() for p in dec.parameters()) if dec is not None else 0, args=vars(a))
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
        print(json.dumps(cfg, default=str), flush=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a") if rank == 0 else None
    dlog = open(os.path.join(a.out, "diag.jsonl"), "a") if rank == 0 else None
    glog = open(os.path.join(a.out, "grad_check.jsonl"), "a") if rank == 0 else None
    t0 = time.time()

    def diag(step, which):
        for nm, (e, d) in which.items():
            r = dev_diag(e, enc0, d, dev_ds, dev_chunks, dev)
            if rank == 0:
                rec = dict(step=step, model=nm, s=round(time.time() - t0), **r)
                dlog.write(json.dumps(rec) + "\n")
                dlog.flush()
                print(json.dumps(rec), flush=True)

    diag(0, {"init": (enc, dec)})
    dec_params = list(dec.parameters()) if dec is not None else []
    trainable = list(enc.parameters()) + dec_params
    run = np.zeros(6)
    for step, b in enumerate(dl):
        b = RC.to_dev(b, dev)
        for g in opt.param_groups:
            g["lr"] = C.lr_at(step, g["base"], WARM, total)
        o = terms(b, enc, enc0, dec)
        n_roi = RC.all_sum(torch.tensor(float(len(o["roi"])), device=dev))
        n_img = RC.all_sum(torch.tensor(float(len(o["glob"])), device=dev))
        l_roi, l_glob = o["roi"].sum() / n_roi, o["glob"].sum() / n_img
        l_det = (o["tok"].sum() + o["dpool"].sum()) / n_roi if dec is not None else torch.zeros((), device=dev)
        loss = l_roi + lam_d * l_det + lam_g * l_glob
        if step in (0, 1, 2) or (step + 1) % 1000 == 0:
            gr = per_loss_grads(o, enc, dec)
            if rank == 0:
                glog.write(json.dumps(dict(step=step + 1, **gr)) + "\n")
                glog.flush()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        RC.allreduce_grads(trainable)
        g_enc = float(torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0))
        g_dec = float(torch.nn.utils.clip_grad_norm_(dec_params, 1.0)) if dec is not None else 0.0
        opt.step()
        mt = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
        ema_update(enc_t, enc, mt)
        if dec is not None:
            ema_update(dec_t, dec, mt)
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"non-finite loss at step {step}")
        run += np.array([float(l_roi), float(l_det), float(l_glob), g_enc, g_dec, 1.0])
        if (step + 1) % 100 == 0 and rank == 0:
            r = run / run[5]
            rec = dict(step=step + 1, s=round(time.time() - t0), l_roi=round(r[0], 5), l_detail=round(r[1], 5), l_global=round(r[2], 6),
                       gnorm_enc=round(r[3], 4), gnorm_dec=round(r[4], 4), ema_m=round(mt, 6),
                       enc_lr_top=max(g["lr"] for g in opt.param_groups if g["mod"] == "enc"),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run[:] = 0
        if (step + 1) % a.diag_every == 0 and step + 1 < total:
            diag(step + 1, {"ema": (enc_t, dec_t)})
    diag(total, {"ema": (enc_t, dec_t), "student": (enc, dec)})
    if rank == 0:
        torch.save(dict(enc=enc_t.state_dict(), dec=dec_t.state_dict() if dec is not None else None, step=total, kind="ema",
                        cond=a.cond, arch=MC.CLIP_ARCH, D=D), os.path.join(a.out, "ema_last.pt"))
        torch.save(dict(enc=enc.state_dict(), dec=dec.state_dict() if dec is not None else None, step=total, kind="student",
                        cond=a.cond, arch=MC.CLIP_ARCH, D=D), os.path.join(a.out, "student_last.pt"))
        json.dump(dict(status="done", steps=total, time_s=round(time.time() - t0), time=time.strftime("%F %T")),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
        print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("pool", "cross"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calib_steps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--diag_every", type=int, default=3000)
    ap.add_argument("--dev_max", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.smoke:
        return smoke(a)
    if a.calib:
        return calib(a, torch.device("cuda"))
    rank, world = RC.ddp_setup()
    os.makedirs(a.out, exist_ok=True)
    try:
        train(a, rank, world, torch.device("cuda"))
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if rank == 0:
            json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


if __name__ == "__main__":
    main()
