"""Coarse-Fine pilot training (P / M). Same recipe as src.g1x.distill (G1-pool), new only where the spec requires it.

Same for both: init = G1-pool EMA encoder; frozen ORIGINAL DINOv3 teacher (cached by src.hcf.rois); 12,000 steps of the existing COCO stream
(16 images / step, 8 per GPU, the 2 PACO-matching images dropped); encoder AdamW lr 5e-5 top + LLRD 0.75, wd 0.05, warmup 600 + cosine, grad clip
1.0, bf16; EMA 0.999 -> 1 (final EMA goes to the probe); L = L_crop + 0.5 L_global (G1 value, vs the ORIGINAL encoder).
New-module lr = 3e-4, wd 0.05, clip 1.0 (the G1-cross decoder group's values).

  L_crop   = mean_ROI [1 - cos(Q_r, pool_c(T_r))], Q_r = model.fuse(area-pooled coarse, area-pooled memory), over object + nested ROIs
  L_global = mean_img 1/2 [(1 - cos(CLS, CLS_0)) + (1 - cos(mean patch, mean patch_0))]

  python -m src.hcf.distill --calib --cond P --out <run>/P --shared <run>/shared     (one GPU; residual scales on real images + cache check)
  torchrun --nproc_per_node 2 -m src.hcf.distill --cond P --out <run>/P --shared <run>/shared
"""
import argparse
import copy
import glob
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT, STREAM
from src.g1x.data import CocoDS
from src.hcf.model import ML, build_student, checksum, load_vit, n_params
from src.ivc.ft import cosd, ema_update, enc_param_groups, run_enc
from src.msd.pretrain import eval_exclusions
from src.viz.c0_pca import LB

C = RC.C
S, P = 384, 16
INIT = "output/2026-11-06/g1_cross_distill/G1pool/ema_last.pt"
WARM, M0_EMA, ENC_LR, LLRD, NEW_LR, WD, LAM_G, TARGET = 600, 0.999, 5e-5, 0.75, 3e-4, 0.05, 0.5, 0.05


def pool(x, w):
    w = w.float()
    return (x.float() * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)


def cell_w(box_c, G, cell):
    """(R, 4) canvas boxes -> (R, G*G) box ∩ cell area in cell units (= src.g1x.data.roi_geom wfull at G=24, cell=16)."""
    e = torch.arange(G + 1, device=box_c.device, dtype=box_c.dtype) * cell
    X0, Y0, X1, Y1 = box_c[:, 0, None], box_c[:, 1, None], box_c[:, 2, None], box_c[:, 3, None]
    ox = (torch.minimum(e[1:][None], X1) - torch.maximum(e[:-1][None], X0)).clamp_min(0) / cell
    oy = (torch.minimum(e[1:][None], Y1) - torch.maximum(e[:-1][None], Y0)).clamp_min(0) / cell
    return (oy[:, :, None] * ox[:, None, :]).flatten(1)


class Slots:
    """rank r of `world` takes stream columns [8r, 8r + 8) of every step, minus excluded images (= src.msd StreamSampler + step / slot ids)."""

    def __init__(self, stream, rank, world, exclude, start=0):
        self.stream, self.rank, self.world, self.ex, self.start = stream, rank, world, set(int(x) for x in exclude), start

    def __iter__(self):
        k = self.stream.shape[1] // self.world
        for s in range(self.start, len(self.stream)):
            row = self.stream[s]
            n = sum(int(i) not in self.ex for i in row)
            yield (s, [(c, int(row[c])) for c in range(self.rank * k, (self.rank + 1) * k) if int(row[c]) not in self.ex], n)

    def __len__(self):
        return len(self.stream) - self.start


class TrainDS(Dataset):
    """Full canvases only (teacher values come from the cache) + ROI geometry; the same on-disk manifest for P and M."""

    def __init__(self, shared):
        self.d = shared
        self.ids = np.load(f"{shared}/image_ids.npy")
        self.pos = {int(i): k for k, i in enumerate(self.ids)}
        self.obj, self.NB = np.load(f"{shared}/object_boxes.npy"), np.load(f"{shared}/nested_boxes.npy")
        self.m = None

    def _open(self):
        self.m = {k: np.load(f"{self.d}/{k}.npy", mmap_mode="r") for k in ("t_object", "t_nested", "g_cls", "g_mean")}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if self.m is None:
            self._open()
        step, slots, n_glob = item
        full, fv, roi_img, box, tp, nest, gc, gm = [], [], [], [], [], [], [], []
        for k, (c, iid) in enumerate(slots):
            i = self.pos[iid]
            img = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
            W, H = img.size
            fl = LB(W, H, S, P)
            full.append(fl.canvas(img))
            fv.append(fl.coverage() > 0)
            gc.append(np.array(self.m["g_cls"][i]))
            gm.append(np.array(self.m["g_mean"][i]))

            def add(b, t, n):
                l, tt, r, bb = (float(v) for v in b)
                box.append([fl.ox + l * fl.nw / W, fl.oy + tt * fl.nh / H, fl.ox + r * fl.nw / W, fl.oy + bb * fl.nh / H])
                tp.append(np.array(t))
                nest.append(n)
                roi_img.append(k)
            for j in range(4):
                add(self.obj[i][j], self.m["t_object"][i, j], False)
            for j in range(4):
                b = self.NB[step, c, j]
                if b[2] > b[0]:
                    add(b, self.m["t_nested"][step, c, j], True)
        return dict(step=step, full=torch.stack(full), fvalid=torch.from_numpy(np.stack(fv)), roi_img=torch.as_tensor(roi_img),
                    box_c=torch.as_tensor(box, dtype=torch.float32), tpool=torch.from_numpy(np.stack(tp)).float(), nested=torch.as_tensor(nest),
                    gcls=torch.from_numpy(np.stack(gc)).float(), gmean=torch.from_numpy(np.stack(gm)).float(), n_img_global=torch.tensor(float(n_glob)))


def read_q(model, o, ri, w24, w48):
    A = pool(o["coarse"][ri], w24)
    mem = {k: pool(v[ri], w48 if o["grids"][k] == 2 else w24) for k, v in o["mem"].items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Q = model.fuse(A, mem)
    return A, Q.float()


def fwd(model, x, collect=False):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model.forward_all(x, collect=collect)
    return dict(cls=o["cls"].float(), coarse=o["coarse"].float(), mem={k: v.float() for k, v in o["mem"].items()}, grids=o["grids"])


def terms(b, model, collect=False):
    o = fwd(model, b["full"], collect)
    ri, fv = b["roi_img"], b["fvalid"].float()
    A, Q = read_q(model, o, ri, cell_w(b["box_c"], 24, 16), cell_w(b["box_c"], 48, 8))
    return dict(roi=cosd(Q, b["tpool"]), coarse=cosd(A, b["tpool"]), glob=0.5 * (cosd(o["cls"], b["gcls"]) + cosd(pool(o["coarse"], fv), b["gmean"])),
                nested=b["nested"], o=o, Q=Q, A=A)


def gnorm(gs):
    q = [(g.float() ** 2).sum() for g in gs if g is not None]
    return float(torch.sqrt(sum(q))) if q else 0.0


def submods(model):
    if model.kind == "P":
        return dict(fine_embed=model.fine_embed, fine_blocks=model.fine_blocks, inter=model.inter, Wd=model.Wd)
    return dict(ln=model.ln, W=model.W)


def grad_check(t, model):
    """per-loss gradient norms on the encoder / each new module (local batch), graph retained."""
    ep, r = list(model.vit.parameters()), {}
    sm = {k: list(v.parameters()) for k, v in submods(model).items()}
    for name, l in dict(roi=t["roi"].mean(), global_=t["glob"].mean()).items():
        g = torch.autograd.grad(l, ep + [p for v in sm.values() for p in v], retain_graph=True, allow_unused=True)
        r[f"enc_{name}"], o = gnorm(g[:len(ep)]), len(ep)
        for k, v in sm.items():
            r[f"{k}_{name}"] = gnorm(g[o:o + len(v)])
            o += len(v)
        r[f"loss_{name}"] = float(l)
    return r


@torch.no_grad()
def calibrate(a, dev):
    """Residual scales on real images (one fixed rule for both conditions): every interaction residual = 5% of its target's mean token norm and the
    readout residual ||Q - A|| = 5% of ||A|| (M: split evenly over the two layers), 3 fixed-point rounds; plus the teacher-cache reproduction check."""
    os.makedirs(a.out, exist_ok=True)
    model = build_student(a.cond, dev, a.init, 0.1).eval()
    stream, ids = np.load(STREAM), []
    ex = set(eval_exclusions()["raev2_stream"])
    ids = [int(i) for i in stream[0] if int(i) not in ex][:8]
    x = []
    for iid in ids:
        img = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
        x.append(LB(*img.size, S, P).canvas(img))
    x = torch.stack(x).to(dev)
    hist = []
    for rnd in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.forward_all(x, collect=True)
            A = o["coarse"]
            if model.kind == "P":
                B = F.avg_pool2d(o["mem"]["fine"].reshape(len(x), 48, 48, -1).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)
                q = model.fuse(A, dict(fine=B))
                ratio = float((q - A).norm(dim=-1).mean() / A.norm(dim=-1).mean())
                model.s_d.mul_(TARGET / ratio)
                for st, m in enumerate(model.inter):
                    m.s_f2c.mul_(TARGET / model.diag[f"stage{st + 1}_res_c"])
                    m.s_c2f.mul_(TARGET / model.diag[f"stage{st + 1}_res_f"])
                hist.append(dict(rnd=rnd, q_ratio=ratio, **model.diag))
            else:
                r = {}
                for l in ML[:-1]:
                    q = model.s[str(l)] * model.W[str(l)](o["mem"][f"l{l}"])
                    r[l] = float(q.norm(dim=-1).mean() / A.norm(dim=-1).mean())
                    model.s[str(l)].mul_(TARGET / math.sqrt(2) / r[l])
                hist.append(dict(rnd=rnd, **{f"res_l{l}": v for l, v in r.items()}))
    sc = {n: float(p) for n, p in model.named_parameters() if p.ndim == 0}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model.forward_all(x, collect=True)
    final = dict(model.diag)
    enc0, chk = load_vit(dev).eval(), {}
    d, shared = {int(v): k for k, v in enumerate(np.load(f"{a.shared}/image_ids.npy"))}, a.shared
    m = {k: np.load(f"{shared}/{k}.npy", mmap_mode="r") for k in ("t_object", "t_nested", "g_cls", "g_mean")}
    rois = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    b = RC.to_dev(CocoDS(rois, COCO_TRAIN, S, P)[ids[:4]], dev)
    T, _ = run_enc(enc0, b["crops"])
    cached = torch.from_numpy(np.stack([np.array(m["t_object"][d[i], k]) for i in ids[:4] for k in range(4)])).to(dev)
    chk["t_object_cos_min"] = float(F.cosine_similarity(pool(T, b["ccov"]), cached, dim=-1).min())
    tok, cls = run_enc(enc0, b["full"])
    chk["g_cls_cos_min"] = float(F.cosine_similarity(cls, torch.from_numpy(np.stack([np.array(m["g_cls"][d[i]]) for i in ids[:4]])).to(dev), dim=-1).min())
    chk["g_mean_cos_min"] = float(F.cosine_similarity(pool(tok, b["fvalid"]), torch.from_numpy(np.stack([np.array(m["g_mean"][d[i]]) for i in ids[:4]])).to(dev), dim=-1).min())
    assert min(chk.values()) > 0.9995, chk
    json.dump(dict(cond=a.cond, target=TARGET, scales=sc, history=hist, final_stage_ratios=final, cache_check=chk, n_params=n_params(model), n_params_vit=n_params(model.vit),
                   n_params_new=n_params(model) - n_params(model.vit), time=time.strftime("%F %T")), open(f"{a.out}/calib.json", "w"), indent=1)
    print(json.dumps(dict(cond=a.cond, final=final, scales=sc, chk=chk, new_params=n_params(model) - n_params(model.vit)), indent=1), flush=True)


@torch.no_grad()
def dev_diag(model, enc0, ds, chunks, dev):
    keys = ["q_roi_cos", "coarse_roi_cos", "cls_cos", "mean_cos", "tok_cos"]
    acc = {k: torch.zeros(2, dtype=torch.float64, device=dev) for k in keys}

    def add(k, v):
        acc[k] += torch.stack([v.double().sum(), torch.tensor(float(v.numel()), device=dev, dtype=torch.float64)])
    was = model.training
    model.eval()
    for ch in chunks:
        b = RC.to_dev(ds[ch], dev)
        T, _ = run_enc(enc0, b["crops"])
        tp = pool(T, b["ccov"])
        otok, ocls = run_enc(enc0, b["full"])
        o = fwd(model, b["full"])
        fv = b["fvalid"].float()
        A, Q = read_q(model, o, b["roi_img"], cell_w(b["box_c"], 24, 16), cell_w(b["box_c"], 48, 8))
        assert (cell_w(b["box_c"], 24, 16) - b["wfull"]).abs().max() < 1e-3, "cell_w != roi_geom wfull"
        add("q_roi_cos", 1 - cosd(Q, tp))
        add("coarse_roi_cos", 1 - cosd(A, tp))
        add("cls_cos", 1 - cosd(o["cls"], ocls))
        add("mean_cos", 1 - cosd(pool(o["coarse"], fv), pool(otok, fv)))
        tc = F.cosine_similarity(o["coarse"], otok, dim=-1)
        add("tok_cos", (tc * fv).sum(1) / fv.sum(1))
    for k in keys:
        RC.all_sum(acc[k])
    if was:
        model.train()
    return {k: float(v[0] / v[1]) for k, v in acc.items()}


def groups(model):
    eg = enc_param_groups(model.vit, ENC_LR, LLRD, WD)
    for g in eg:
        g["mod"] = "enc"
    dg = {}
    for n, p in model.new_named():
        k = p.ndim <= 1
        dg.setdefault(k, dict(params=[], weight_decay=0.0 if k else WD, base=NEW_LR, lr=NEW_LR, mod="new"))
        dg[k]["params"].append(p)
    return eg + list(dg.values())


def train(a, rank, world, dev):
    cal = json.load(open(f"{a.out}/calib.json"))
    model = build_student(a.cond, dev, a.init, 0.1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in cal["scales"]:
                p.copy_(torch.tensor(cal["scales"][n], device=dev))
    init_enc = checksum(model.vit.parameters())
    model.train()
    enc0 = load_vit(dev).eval()
    for p in enc0.parameters():
        p.requires_grad_(False)
    teacher_sum = checksum(enc0.parameters())
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(groups(model), betas=(0.9, 0.999))
    start = 0
    latest = sorted(glob.glob(f"{a.out}/checkpoints/ckpt_*.pt"), key=lambda s: int(s.rsplit("_", 1)[1][:-3]))
    if latest and not a.fresh:
        ck = torch.load(latest[-1], map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        if rank == 0:
            print(f"RESUME from {latest[-1]} step {start}", flush=True)
    ex = eval_exclusions()
    stream = np.load(STREAM)[: a.steps]
    total = len(stream)
    ds = TrainDS(a.shared)
    dev_rois = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex["raev2_dev"])]
    dev_ids = sorted(int(i) for i in dev_rois.image_id.unique())[: (a.dev_max or None)]
    dev_ds = CocoDS(dev_rois, COCO_TRAIN, S, P)
    dev_chunks = [dev_ids[i:i + 8] for i in range(0, len(dev_ids), 8)][rank::world]
    dl = DataLoader(ds, sampler=Slots(stream, rank, world, ex["raev2_stream"], start), batch_size=None, num_workers=a.workers, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    cfg = dict(cond=a.cond, init=a.init, init_enc_checksum=init_enc, teacher_checksum=teacher_sum, lam_global=LAM_G, steps=total, warmup=WARM, world=world,
               images_per_step=16, enc_lr_top=ENC_LR, llrd=LLRD, new_lr=NEW_LR, wd=WD, ema_m0=M0_EMA, target_residual=TARGET, excluded_stream=ex["raev2_stream"],
               n_params_total=n_params(model), n_params_vit=n_params(model.vit), n_params_new=n_params(model) - n_params(model.vit), n_dev_images=len(dev_ids),
               shared=a.shared, encoder="DINOv3-B/16 lvd1689m @384", args=vars(a), torch=torch.__version__, time=time.strftime("%F %T"))
    if rank == 0:
        for d in ("checkpoints", "logs", "results"):
            os.makedirs(f"{a.out}/{d}", exist_ok=True)
        yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), open(f"{a.out}/config.yaml", "w"), sort_keys=False)
        print(json.dumps(cfg, default=str), flush=True)
    log = open(f"{a.out}/logs/train_log.jsonl", "a") if rank == 0 else None
    dlog = open(f"{a.out}/logs/diag.jsonl", "a") if rank == 0 else None
    glog = open(f"{a.out}/logs/grad_check.jsonl", "a") if rank == 0 else None
    t0 = time.time()

    def diag(step, which):
        for nm, m in which.items():
            r = dev_diag(m, enc0, dev_ds, dev_chunks, dev)
            if rank == 0:
                rec = dict(step=step, model=nm, s=round(time.time() - t0), **r)
                dlog.write(json.dumps(rec) + "\n")
                dlog.flush()
                print(json.dumps(rec), flush=True)

    def save(step, name=None):
        if rank != 0:
            return
        if name is None:
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, cond=a.cond), f"{a.out}/checkpoints/ckpt_{step}.pt")
        else:
            m = ema if name == "ema_last" else model
            torch.save(dict(model=m.state_dict(), enc=m.vit.state_dict(), step=step, kind=name.split("_")[0], cond=a.cond, s0=0.1), f"{a.out}/checkpoints/{name}.pt")

    if start == 0:
        diag(0, dict(init=model))
    params = list(model.parameters())
    new_params = [p for _, p in model.new_named()]
    run = np.zeros(8)
    im_seen, last_t, last_step = 0.0, time.time(), start
    for k, b in enumerate(dl):
        step = start + k
        b = RC.to_dev(b, dev)
        for g in opt.param_groups:
            g["lr"] = C.lr_at(step, g["base"], WARM, total)
        check = step in (0, 1, 2) or (step + 1) % 1000 == 0
        t = terms(b, model, collect=check)
        n_roi = RC.all_sum(torch.tensor(float(len(t["roi"])), device=dev))
        n_img = RC.all_sum(torch.tensor(float(len(t["glob"])), device=dev))
        l_roi, l_glob = t["roi"].sum() / n_roi, t["glob"].sum() / n_img
        loss = l_roi + LAM_G * l_glob
        if check:
            gr = grad_check(t, model)
            if rank == 0:
                rec = dict(step=step + 1, **gr, **{f"res_{kk}": v for kk, v in model.diag.items()}, scales={n: float(p) for n, p in model.named_parameters() if p.ndim == 0})
                glog.write(json.dumps(rec) + "\n")
                glog.flush()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        RC.allreduce_grads(params)
        g_enc = float(torch.nn.utils.clip_grad_norm_(model.vit.parameters(), 1.0))
        g_new = float(torch.nn.utils.clip_grad_norm_(new_params, 1.0))
        opt.step()
        mt = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
        ema_update(ema, model, mt)
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"non-finite loss at step {step}")
        ns = t["nested"]
        lo = float(t["roi"][~ns].mean())
        ln = float(t["roi"][ns].mean()) if bool(ns.any()) else lo
        run += np.array([float(l_roi), lo, ln, float(t["coarse"].mean()), float(l_glob), g_enc, g_new, 1.0])
        im_seen += float(n_img)
        if (step + 1) % a.log_every == 0 and rank == 0:
            r, now = run / run[7], time.time()
            rec = dict(step=step + 1, s=round(now - t0), l_crop=round(r[0], 5), l_object=round(r[1], 5), l_nested=round(r[2], 5), l_coarse_only=round(r[3], 5), l_global=round(r[4], 6),
                       gnorm_enc=round(r[5], 4), gnorm_new=round(r[6], 4), ema_m=round(mt, 6), enc_lr_top=max(g["lr"] for g in opt.param_groups if g["mod"] == "enc"),
                       steps_per_s=round((step + 1 - last_step) / (now - last_t), 3), images_per_s=round(im_seen / (now - last_t), 1), mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run[:] = 0
            im_seen, last_t, last_step = 0.0, now, step + 1
        if (step + 1) % a.diag_every == 0 and step + 1 < total:
            save(step + 1)
            diag(step + 1, dict(ema=ema))
    diag(total, dict(ema=ema, student=model))
    save(total)
    save(total, "ema_last")
    save(total, "student_last")
    if rank == 0:
        assert checksum(enc0.parameters()) == teacher_sum, "teacher changed"
        json.dump(dict(status="done", steps=total, time_s=round(time.time() - t0), teacher_unchanged=True, time=time.strftime("%F %T")), open(f"{a.out}/done.json", "w"), indent=1)
        print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("P", "M"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--init", default=INIT)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--diag_every", type=int, default=3000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--dev_max", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.calib:
        return calibrate(a, torch.device("cuda"))
    rank, world = RC.ddp_setup()
    os.makedirs(a.out, exist_ok=True)
    try:
        train(a, rank, world, torch.device("cuda"))
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if rank == 0:
            json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(f"{a.out}/failed.json", "w"), indent=1)
        raise


if __name__ == "__main__":
    main()
