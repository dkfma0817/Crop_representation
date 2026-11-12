"""Teacher-student training of S1 / S2 / S3 token splitting on frozen DINOv3-B/16 (one process, one GPU, shared frozen forward).

Data   : the existing COCO stream (src.config.STREAM) rows 0..11,999, 16 images / step, PACO val/test duplicates removed
         (src.msd.pretrain.eval_exclusions: id / md5 / pixel md5 / dHash; 2 stream images). Same images for S1/S2/S3 every step.
Inputs : from ONE PIL image per slot (hflip p = 0.5, rng (step, slot)): student = LB 384 letterbox canvas; teacher = the original
         image resized directly to 2 x (nw, nh) and pasted at 2 x (ox, oy) on a 768 canvas (src.g1native.native.native_canvas).
         Student parent (i, j) (16 px at 384) <-> teacher cells (2i + a, 2j + b) (16 px at 768 = 8 px at 384).
ROIs   : 8 per image, parent rectangle H, W ~ U{1..6} (clipped to the content), uniform position inside the parents that overlap
         the letterbox content (rng (step, slot, 1)). Children that do not overlap the content are masked (loss and attention keys).
Loss   : L2 feature regression on every valid child: per-ROI mean over valid children of mean_d (S - T)^2, then mean over ROIs.
         Teacher and student encoders frozen (no grad); only the TokenSplit modules train.
Optim  : per model AdamW lr 3e-4, wd 0.05 (weights with ndim >= 2 only), betas (0.9, 0.999), warmup 600 + cosine to 0 over 12,000
         steps, grad clip 1.0, bf16 autocast, seed 0.
Logs   : train.jsonl (losses + B0 / nearest-copy references on the same ROIs every 50 steps), grads.jsonl (per-group gradient norms
         before clipping, every 100 steps), heldout.jsonl (500 COCO dev images, fixed ROIs, every 2,000 steps).

  python -m src.tsplit.train --out <run>/train [--smoke]
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import COCO_TRAIN, FEAT, STREAM
from src.g1native.native import native_canvas
from src.msd.pretrain import eval_exclusions
from src.tsplit.features import frozen_vit, student, teacher, up2, valid48
from src.tsplit.model import G, G2, TokenSplit, n_params, param_report, rect_index
from src.viz.c0_pca import LB

KINDS = ("S1", "S2", "S3")
STEPS, IMGS, ROIS_PER_IMG, MAXS = 12000, 16, 8, 6
LR, WD, WARM, CLIP = 3e-4, 0.05, 600, 1.0
STOP = False


def _stop(*_):
    global STOP
    STOP = True


def content_parents(lb):
    """inclusive parent ranges (y0, y1, x0, x1) that overlap the content rectangle."""
    return lb.oy // 16, (lb.oy + lb.nh - 1) // 16, lb.ox // 16, (lb.ox + lb.nw - 1) // 16


def sample_rects(rng, lb, n, smin=1, smax=MAXS):
    y0, y1, x0, x1 = content_parents(lb)
    ny, nx = y1 - y0 + 1, x1 - x0 + 1
    out = []
    for _ in range(n):
        H, W = min(int(rng.integers(smin, smax + 1)), ny), min(int(rng.integers(smin, smax + 1)), nx)
        out.append([y0 + int(rng.integers(0, ny - H + 1)), x0 + int(rng.integers(0, nx - W + 1)), H, W])
    return out


def load_pair(iid, flip):
    with Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")) as im:
        img = im.convert("RGB")
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    lb = LB(img.size[0], img.size[1], 384, 16)
    return lb, lb.canvas(img), native_canvas(img, lb)


class StreamDS(Dataset):
    def __init__(self, stream, exclude):
        self.stream, self.ex = stream, set(int(i) for i in exclude)

    def __len__(self):
        return len(self.stream)

    def __getitem__(self, s):
        xs, xt, content, rects, rimg, ids = [], [], [], [], [], []
        for c, iid in enumerate(self.stream[s]):
            iid = int(iid)
            if iid in self.ex:
                continue
            rng = np.random.default_rng([s, c])
            lb, a, t = load_pair(iid, bool(rng.random() < 0.5))
            k = len(xs)
            xs.append(a)
            xt.append(t)
            content.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
            for r in sample_rects(np.random.default_rng([s, c, 1]), lb, ROIS_PER_IMG):
                rects.append(r)
                rimg.append(k)
            ids.append(iid)
        return dict(step=s, x=torch.stack(xs), x768=torch.stack(xt), content=torch.tensor(content, dtype=torch.float32),
                    rects=torch.tensor(rects, dtype=torch.long), rimg=torch.tensor(rimg, dtype=torch.long), ids=torch.tensor(ids))


def dev_set():
    """Held-out COCO images (cache/feat dev_rois, 500 images, disjoint from the stream; PACO duplicates removed). Fixed ROIs, no flip:
    8 'train' rectangles (1..6), 2 'large' (7..12, clipped), 1 'whole' (all content parents)."""
    ex = set(eval_exclusions()["raev2_dev"])
    ids = sorted(int(i) for i in pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet")).image_id.unique() if int(i) not in ex)
    assert not set(ids) & set(int(i) for i in np.load(STREAM)[:STEPS].ravel())
    return ids


def dev_batch(ids):
    xs, xt, content, rects, rimg, bucket = [], [], [], [], [], []
    for k, iid in enumerate(ids):
        lb, a, t = load_pair(iid, False)
        xs.append(a)
        xt.append(t)
        content.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
        rng = np.random.default_rng([10 ** 6, iid])
        for r in sample_rects(rng, lb, 8) + sample_rects(rng, lb, 2, 7, 12):
            rects.append(r)
            rimg.append(k)
            bucket.append("train" if max(r[2], r[3]) <= MAXS else "large")
        y0, y1, x0, x1 = content_parents(lb)
        rects.append([y0, x0, y1 - y0 + 1, x1 - x0 + 1])
        rimg.append(k)
        bucket.append("whole")
    return dict(x=torch.stack(xs), x768=torch.stack(xt), content=torch.tensor(content, dtype=torch.float32),
                rects=torch.tensor(rects, dtype=torch.long), rimg=torch.tensor(rimg, dtype=torch.long), bucket=bucket)


def roi_mse(pred, target, mask):
    e = (pred.float() - target.float()).pow(2).mean(-1)
    w = mask.float()
    return (e * w).sum(1) / w.sum(1).clamp_min(1)


def lr_at(step):
    if step < WARM:
        return LR * (step + 1) / WARM
    return LR * 0.5 * (1 + math.cos(math.pi * (step - WARM) / (STEPS - WARM)))


def make_opt(m):
    decay = [p for p in m.parameters() if p.ndim >= 2]
    rest = [p for p in m.parameters() if p.ndim < 2]
    return torch.optim.AdamW([dict(params=decay, weight_decay=WD), dict(params=rest, weight_decay=0.)], lr=LR, betas=(.9, .999))


def gnorm(ps):
    gs = [p.grad.float().norm() for p in ps if p.grad is not None]
    return float(torch.stack(gs).norm()) if gs else 0.0


class Held:
    """Held-out feature metrics. Every child of a ROI rectangle is predicted (S2/S3: one transformer pass per ROI); metrics over
    valid children; within-parent residual = child - mean of its parent's 4 children (parents with all 4 children valid)."""

    def __init__(self, vit, dev, ids, chunk=50):
        self.parts = []
        for i in range(0, len(ids), chunk):
            b = dev_batch(ids[i:i + chunk])
            x, x768 = b["x"].to(dev), b["x768"].to(dev)
            fin, ear = student(vit, x)
            T = teacher(vit, x768)
            self.parts.append(dict(fin=fin.half(), ear=ear.half(), T=T.half(), v48=valid48(b["content"].to(dev)), rects=b["rects"].to(dev),
                                   rimg=b["rimg"].to(dev), bucket=np.array(b["bucket"])))
        self.n_img = len(ids)

    @torch.no_grad()
    def evaluate(self, models):
        """ROIs grouped by rectangle shape (H, W): no padding, one batched transformer pass per group."""
        acc = {}
        for p in self.parts:
            T, fin, ear = p["T"].float(), p["fin"].float(), p["ear"].float()
            shapes = {}
            for i, (H, W) in enumerate(p["rects"][:, 2:].tolist()):
                shapes.setdefault((H, W), []).append(i)
            for kind in ("B0", "NN") + tuple(models):
                model = models.get(kind)
                if kind == "B0":
                    cmap = up2(fin)
                elif kind == "NN":
                    cmap = fin.view(-1, G, 1, G, 1, 768).expand(-1, G, 2, G, 2, 768).reshape(-1, G2 * G2, 768)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        cmap = model.child_map(fin, ear).float()
                for (H, W), rows in shapes.items():
                    sel = torch.tensor(rows, device=T.device)
                    ri = p["rimg"][sel]
                    idx, cy, cx, ok = rect_index(p["rects"][sel])
                    key_ok = ok & p["v48"][ri[:, None], idx]
                    if kind in ("S2", "S3"):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            y = model.refine(cmap, ri, idx, cy, cx, key_ok).float()
                    else:
                        y = cmap[ri[:, None], idx]
                    t = T[ri[:, None], idx]
                    w = key_ok.float()
                    mse = ((y - t).pow(2).mean(-1) * w).sum(1) / w.sum(1)
                    cos = (torch.nn.functional.cosine_similarity(y, t, dim=-1) * w).sum(1) / w.sum(1)
                    n = len(rows)
                    yy, tt, oo = y.view(n, H, 2, W, 2, -1), t.view(n, H, 2, W, 2, -1), w.view(n, H, 2, W, 2)
                    full = (oo.sum((2, 4)) == 4).float()                                            # (n, H, W) parents with 4 valid children
                    ry, rt = yy - yy.mean((2, 4), keepdim=True), tt - tt.mean((2, 4), keepdim=True)
                    rs = ((ry - rt).pow(2).mean(-1).sum((2, 4)) / 4 * full).sum((1, 2))
                    re = (rt.pow(2).mean(-1).sum((2, 4)) / 4 * full).sum((1, 2))
                    pm = ((yy.mean((2, 4)) - tt.mean((2, 4))).pow(2).mean(-1) * full).sum((1, 2))
                    npar = full.sum((1, 2))
                    mse, cos, rs, re, pm, npar = (v.tolist() for v in (mse, cos, rs, re, pm, npar))
                    for j, i in enumerate(rows):
                        for bk in (p["bucket"][i], "all"):
                            a = acc.setdefault((kind, bk), dict(n=0, mse=0., cos=0., res_se=0., res_energy=0., pm_se=0., parents=0.))
                            a["n"] += 1
                            a["mse"] += float(mse[j])
                            a["cos"] += float(cos[j])
                            a["res_se"] += float(rs[j])
                            a["res_energy"] += float(re[j])
                            a["pm_se"] += float(pm[j])
                            a["parents"] += float(npar[j])
        out = {}
        for (kind, bk), a in acc.items():
            out.setdefault(kind, {})[bk] = dict(n_roi=a["n"], mse=a["mse"] / a["n"], cos=a["cos"] / a["n"],
                                                within_parent_residual_mse=a["res_se"] / max(a["parents"], 1),
                                                within_parent_teacher_energy=a["res_energy"] / max(a["parents"], 1),
                                                within_parent_r2=1 - a["res_se"] / max(a["res_energy"], 1e-12),
                                                parent_mean_mse=a["pm_se"] / max(a["parents"], 1))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
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
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    vit = frozen_vit(dev)
    sum0 = float(sum(p.double().sum() for p in vit.parameters()))
    models = {k: TokenSplit(k).to(dev).train() for k in KINDS}
    opts = {k: make_opt(m) for k, m in models.items()}
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:STEPS]
    start = 0
    rec = out / "recovery.pt"
    if rec.exists():
        ck = torch.load(rec, map_location=dev, weights_only=False)
        for k in KINDS:
            models[k].load_state_dict(ck["models"][k])
            opts[k].load_state_dict(ck["opts"][k])
        start = ck["step"]
        print(json.dumps(dict(resumed_from=start)), flush=True)
    n_steps = 30 if a.smoke else STEPS
    cfg = dict(steps=STEPS, images_per_step=IMGS, rois_per_image=ROIS_PER_IMG, roi_parent_side=[1, MAXS], lr=LR, wd=WD, warmup=WARM, clip=CLIP,
               schedule="linear warmup 600 then cosine to 0", seed=0, loss="per-ROI mean over valid children of mean_d (S - T)^2, then mean over ROIs",
               teacher="frozen DINOv3-B/16 final-norm patch tokens on the native 768 canvas (original image resized directly)",
               student="frozen DINOv3-B/16 final-norm patch tokens at 384 (+ patch_embed output for S3)", excluded_stream_images=ex,
               params=param_report(), n_params={k: n_params(m) for k, m in models.items()}, flip="hflip p=0.5 per image (rng (step, slot)), both canvases from the same flipped PIL image",
               smoke=a.smoke, torch=torch.__version__)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    held = None if a.smoke else Held(vit, dev, dev_set())
    ds = StreamDS(stream, ex)
    dl = DataLoader(ds, sampler=range(start, n_steps), batch_size=None, num_workers=a.workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    logs = {n: (out / f"{n}.jsonl").open("a") for n in ("train", "grads", "heldout")}

    def heldout(step):
        if held is None:
            return
        for m in models.values():
            m.eval()
        t = time.time()
        r = held.evaluate(models)
        for m in models.values():
            m.train()
        logs["heldout"].write(json.dumps(dict(step=step, seconds=round(time.time() - t, 1), metrics=r)) + "\n")
        logs["heldout"].flush()
        print(json.dumps(dict(heldout=step, all={k: dict(mse=round(v["all"]["mse"], 4), r2=round(v["all"]["within_parent_r2"], 4)) for k, v in r.items()})), flush=True)

    def save(step, final=False):
        tmp = out / "recovery.tmp"
        torch.save(dict(models={k: m.state_dict() for k, m in models.items()}, opts={k: o.state_dict() for k, o in opts.items()}, step=step), tmp)
        tmp.replace(rec)
        if final:
            for k, m in models.items():
                torch.save(dict(kind=k, model=m.state_dict(), step=step, config=cfg), out / f"{k}.pt")

    if start == 0:
        heldout(0)
    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.time()
    run_l = {k: [] for k in KINDS + ("B0", "NN")}
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
        with torch.no_grad():
            run_l["B0"].append(roi_mse(up2(fin)[rimg[:, None], idx], Tt, key_ok).mean())
            nn_map = fin.view(-1, G, 1, G, 1, 768).expand(-1, G, 2, G, 2, 768).reshape(-1, G2 * G2, 768)
            run_l["NN"].append(roi_mse(nn_map[rimg[:, None], idx], Tt, key_ok).mean())
        lr = lr_at(step)
        grec = dict(step=step + 1)
        for k in KINDS:
            m, o = models[k], opts[k]
            for g in o.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cmap = m.child_map(fin, ear)
                y = m.refine(cmap, rimg, idx, cy, cx, key_ok)
            loss = roi_mse(y, Tt, key_ok).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{k} loss at step {step + 1}")
            o.zero_grad(set_to_none=True)
            loss.backward()
            if step < 3 or (step + 1) % 100 == 0:
                for gname, ps in m.groups().items():
                    grec[f"{k}/{gname}"] = gnorm(ps)
                if k == "S3":
                    gw = m.split.lin.weight.grad
                    grec["S3/split_final_cols"] = float(gw[:, :768].float().norm())
                    grec["S3/split_early_cols"] = float(gw[:, 768:].float().norm())
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            grec[f"{k}/total"] = float(gn) if (step < 3 or (step + 1) % 100 == 0) else None
            o.step()
            run_l[k].append(loss.detach())
        if len(grec) > 1 and grec.get("S1/total") is not None:
            logs["grads"].write(json.dumps(grec) + "\n")
            logs["grads"].flush()
        if step == start or (step + 1) % 50 == 0:
            now = time.time()
            r = dict(step=step + 1, lr=lr, elapsed_s=round(now - t0, 1), sec_per_step=round((now - last) / max(len(run_l["S1"]), 1), 3),
                     n_img=len(x), n_roi=len(rects), mean_children=float(ok.sum(1).float().mean()), valid_child_frac=float(key_ok.sum() / ok.sum()),
                     peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                     **{f"loss_{k}": float(torch.stack(v).mean()) for k, v in run_l.items()})
            logs["train"].write(json.dumps(r) + "\n")
            logs["train"].flush()
            print(json.dumps(r), flush=True)
            run_l = {k: [] for k in run_l}
            last = now
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
    (out / "done.json").write_text(json.dumps(dict(status="done", steps=n_steps, time_s=round(time.time() - t0), peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, encoder_unchanged=True, smoke=a.smoke), indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
