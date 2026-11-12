"""G1-Uniform / G1-ROI: P384 Coarse-Fine + Upscaled Teacher spatial distillation.

Same start, COCO stream, ROIs, optimizer, schedule and EMA as r384_cosine (hcf_material_causal). Crop Cosine and Global Retention are unchanged; PixelDiT is not used.
  L = L_crop + lambda_grid L_grid + 0.5 L_global

Student F = fuse(bilinear(coarse 24 -> 48), fine 48) on the 48 x 48 grid.
Teacher T = frozen original DINOv3-B/16 on the student's own 384 canvas bilinearly upscaled to 768 (48 x 48 x 768, no_grad). No GT crop and no new pixels reach it.
L_grid = per image sum_j w_j (1 - cos(F_j, T_j)) / sum_j w_j, averaged over images.
  G1U  w_j = 1 on valid cells (touching the letterbox content)
  G1R  w_j = valid_j * max over the image's Object and Nested ROIs of (box area in cell j / cell area). Nested ROIs are COCO-derived boxes, not semantic parts.
lambda_grid is fixed once by --calibrate on train minibatches at the initial weights (median ||g_crop|| / ||g_grid||, both weightings, 2 significant figures) and read from lambda.json.

  python -m src.g1up.train --calibrate --out <run>/lambda.json
  python -m src.g1up.train --condition G1U --lambda-file <run>/lambda.json --out <run>/G1U
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.config import COCO_TRAIN, STREAM
from src.hcf.distill import LAM_G, M0_EMA, WARM, Slots, TrainDS, cell_w, fwd, groups, pool, read_q
from src.hcf.model import checksum, load_hcf, load_vit, n_params
from src.ivc.ft import cosd, ema_update, run_enc
from src.msd.pretrain import eval_exclusions
from src.pdit.crop_objectives import roi_tokens
from src.pdit.train import CropObjectiveDS
from src.viz.c0_pca import LB

STOP = False
INITIAL = "output/2026-11-08/hcf_material_causal/initial.pt"
SHARED = "output/2026-11-08/hcf_pilot/shared"
BASE_LOG = "output/2026-11-08/hcf_material_causal/r384_cosine/P/logs/train.jsonl"
CONDS = ("G1U", "G1R")


def stop(*_):
    global STOP
    STOP = True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class GridDS(TrainDS):
    """TrainDS + the letterbox content box of every image (canvas px)."""

    def __getitem__(self, item):
        b = super().__getitem__(item)
        content = []
        for _, iid in item[1]:
            with Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")) as im:
                W, H = im.size
            lb = LB(W, H, 384, 16)
            content.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
        b["content"] = torch.tensor(content, dtype=torch.float32)
        return b


def student48(model, o):
    n = len(o["coarse"])
    a = o["coarse"].transpose(1, 2).reshape(n, 768, 24, 24)
    a48 = F.interpolate(a, size=(48, 48), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = model.fuse(a48, dict(fine=o["mem"]["fine"]))
    return q.float(), a48


@torch.no_grad()
def teacher48(enc0, x):
    t, _ = run_enc(enc0, F.interpolate(x, size=(768, 768), mode="bilinear", align_corners=False), chunk=8)
    assert t.shape[1] == 2304, t.shape
    return t.float()


def grid_weights(condition, b):
    valid = (cell_w(b["content"], 48, 8) > 0).float()
    if condition == "G1U":
        return valid
    w = cell_w(b["box_c"], 48, 8)
    roi = torch.zeros_like(valid)
    for k in range(len(valid)):
        roi[k] = w[b["roi_img"] == k].amax(0)
    return roi * valid


def grid_loss(fuse48, t48, w):
    per = cosd(fuse48, t48)
    den = w.sum(1)
    keep = den > 0
    assert keep.any()
    return ((per * w).sum(1)[keep] / den[keep]).mean()


def base_losses(model, b):
    o = fwd(model, b["full"])
    _, Q = read_q(model, o, b["roi_img"], cell_w(b["box_c"], 24, 16), cell_w(b["box_c"], 48, 8))
    crop = cosd(Q, b["tpool"]).mean()
    glob = .5 * (cosd(o["cls"], b["gcls"]) + cosd(pool(o["coarse"], b["fvalid"].float()), b["gmean"])).mean()
    return o, crop, glob


def module_params(model):
    return dict(coarse=list(model.vit.parameters()), fine_transformer=list(model.fine_blocks.parameters()), fine_projection=list(model.fine_embed.parameters()) + list(model.Wd.parameters()) + list(model.fine_norm.parameters()) + [model.s_d], interaction=list(model.inter.parameters()))


def flat(gs):
    return torch.cat([g.float().flatten() for g in gs if g is not None]) if any(g is not None for g in gs) else torch.zeros(1)


def grads(loss, params):
    return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)


def diagnostics(model, b, crop, grid, F48, t48, a48, w):
    """Per-module gradient norms and the crop / grid conflict (diagnostic only; nothing is corrected)."""
    mods, out = module_params(model), {}
    all_params = [p for v in mods.values() for p in v]
    g_c, g_g = grads(crop, all_params), grads(grid, all_params)
    i = 0
    for name, v in mods.items():
        gc, gg = g_c[i:i + len(v)], g_g[i:i + len(v)]
        i += len(v)
        fc, fg = flat(gc), flat(gg)
        out[f"grad_crop_{name}"], out[f"grad_grid_{name}"] = float(fc.norm()), float(fg.norm())
        out[f"cos_crop_grid_{name}"] = float(F.cosine_similarity(fc, fg, dim=0)) if fc.numel() == fg.numel() else float("nan")
    fc, fg = flat(g_c), flat(g_g)
    out.update(grad_crop_total=float(fc.norm()), grad_grid_total=float(fg.norm()), cos_crop_grid_total=float(F.cosine_similarity(fc, fg, dim=0)))
    with torch.no_grad():
        out.update(fine_contribution_ratio=float((F48 - a48).norm(dim=-1).mean() / a48.norm(dim=-1).mean().clamp_min(1e-8)), s_d=float(model.s_d))
        valid = (cell_w(b["content"], 48, 8) > 0).float()
        match = F.cosine_similarity(F48, t48, dim=-1)
        shifted = F.cosine_similarity(F48, t48.reshape(-1, 48, 48, 768).roll(1, 2).flatten(1, 2), dim=-1)
        out.update(align_cos_matched=float((match * valid).sum() / valid.sum()), align_cos_shifted_1cell=float((shifted * valid).sum() / valid.sum()), valid_cell_fraction=float(valid.mean()), roi_weight_mean=float(w.mean()))
    return out


def build(a, dev):
    torch.manual_seed(0)
    np.random.seed(0)
    model, ck = load_hcf(a.initial, dev)
    assert ck["step"] == 0
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc0 = load_vit(dev).eval()
    for p in enc0.parameters():
        p.requires_grad_(False)
    return model, ema, enc0, torch.optim.AdamW(groups(model), betas=(.9, .999))


def stream_items(a, start=0):
    ex = eval_exclusions()["raev2_stream"]
    return ex, np.load(STREAM)[:a.steps]


def calibrate(a, dev):
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    a.condition = "G1U"
    model, _, enc0, _ = build(a, dev)
    ex, stream = stream_items(a)
    ds = GridDS(a.shared)
    params = list(model.parameters())
    records, ratios = [], []
    for s in range(a.calib_batches):
        b = RC.to_dev(ds[next(iter(Slots(stream, 0, 1, ex, s)))], dev)
        o, crop, _ = base_losses(model, b)
        F48, _ = student48(model, o)
        t48 = teacher48(enc0, b["full"])
        gc = flat(grads(crop, params)).norm()
        rec = dict(batch=s, grad_crop=float(gc))
        for c in CONDS:
            g = flat(grads(grid_loss(F48, t48, grid_weights(c, b)), params)).norm()
            rec[f"grad_grid_{c}"], rec[f"ratio_{c}"] = float(g), float(gc / g)
            ratios.append(rec[f"ratio_{c}"])
        records.append(rec)
    med = float(np.median(ratios))
    value = float(f"{med:.2g}")
    report = dict(lambda_grid=value, median_ratio=med, rule="median ||g_crop|| / ||g_grid|| at the initial weights, both weightings, all CoarseFine parameters, 2 significant figures", n_batches=a.calib_batches, records=records, initial_sha256=sha256(a.initial))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


def config(a, model, lam):
    return dict(args=vars(a), condition=a.condition, lambda_grid=lam, crop_weight=1., global_weight=LAM_G, steps=a.steps, world=1, images_per_step=16, warmup=WARM, full_size=384, teacher="frozen original DINOv3-B/16 on the student's 384 canvas bilinear-upscaled to 768", weighting=dict(G1U="uniform on valid cells", G1R="valid x max ROI overlap per cell (object + nested)")[a.condition], encoder_params=n_params(model), initial_sha256=sha256(a.initial), seed=0, source_sha256=sha256(__file__), torch=torch.__version__)


def batch_sha(b):
    return hashlib.sha1(b["box_c"].cpu().numpy().tobytes() + b["tpool"].cpu().numpy().tobytes() + b["full"].cpu().numpy().tobytes()).hexdigest()


def aux_connection(model, o, item, ds_aux, dev, F48, b):
    """Grid feature -> 24 x 24 ROI read vs the inference read of the same ROIs (cosine, must be close to 1)."""
    aux = RC.to_dev(ds_aux[item], dev)
    Q = roi_tokens(model, o, aux)
    maps = F48.transpose(1, 2).reshape(len(F48), 768, 48, 48)[aux["aux_ri"]]
    read = F.grid_sample(maps.float(), aux["aux_grid"].reshape(-1, 24, 24, 2), mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)
    v = (aux["aux_cov"] > 0).float()
    return float((F.cosine_similarity(read, Q, dim=-1) * v).sum() / v.sum())


def smoke(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lam = json.loads(Path(a.lambda_file).read_text())["lambda_grid"] if a.lambda_file else 1.
    model, ema, enc0, opt = build(a, dev)
    sum0 = checksum(enc0.parameters())
    ex, stream = stream_items(a)
    ds, ds_aux = GridDS(a.shared), CropObjectiveDS(a.shared)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    report = dict(status="running", cfg=config(a, model, lam), steps=[], device=dev.type)
    for k in range(2 if dev.type == "cpu" else 3):
        item = next(iter(Slots(stream, 0, 1, ex, k)))
        if a.smoke_images:
            item = (item[0], item[1][:a.smoke_images], item[2])
        b = RC.to_dev(ds[item], dev)
        for group in opt.param_groups:
            group["lr"] = RC.C.lr_at(k, group["base"], WARM, a.steps)
        o, crop, glob = base_losses(model, b)
        F48, a48 = student48(model, o)
        t48 = teacher48(enc0, b["full"])
        assert F48.shape == (len(b["full"]), 2304, 768) and not t48.requires_grad
        w = grid_weights(a.condition, b)
        grid = grid_loss(F48, t48, w)
        rec = dict(step=k + 1, crop=float(crop), glob=float(glob), grid=float(grid), weight_mean=float(w.mean()), **diagnostics(model, b, crop, grid, F48, t48, a48, w))
        rec["grid_vs_inference_read_cos"] = aux_connection(model, o, item, ds_aux, dev, F48, b)
        if k == 0:
            assert rec["align_cos_matched"] > rec["align_cos_shifted_1cell"], rec
            mods = module_params(model)
            assert all(rec[f"grad_grid_{n}"] > 0 for n in mods), rec
            if dev.type == "cuda" and not a.smoke_images:
                base = json.loads(open(BASE_LOG).readline())
                report["baseline_repro"] = dict(crop=float(crop), logged_crop=base["crop_cos_loss"], glob=float(glob), logged_glob=base["global_loss"])
                assert abs(float(crop) - base["crop_cos_loss"]) < 2e-3 and abs(float(glob) - base["global_loss"]) < 2e-3, report["baseline_repro"]
        report["steps"].append(rec)
        loss = crop + lam * grid + LAM_G * glob
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.vit.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)
        opt.step()
        ema_update(ema, model, .999)
    assert checksum(enc0.parameters()) == sum0, "teacher changed"
    path = out / "smoke_ckpt.pt"
    torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=0), path)
    load_hcf(path, dev)
    path.unlink()
    report.update(status="passed", peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None)
    (out / "smoke.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, default=str), flush=True)


def train(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        raise FileExistsError(out)
    for folder in ("logs", "checkpoints"):
        (out / folder).mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lam = json.loads(Path(a.lambda_file).read_text())["lambda_grid"]
    model, ema, enc0, opt = build(a, dev)
    sum0 = checksum(enc0.parameters())
    ex, stream = stream_items(a)
    total = len(stream)
    assert total == a.steps
    start = 0
    rec_path = out / "checkpoints/recovery.pt"
    if rec_path.exists():
        ck = torch.load(rec_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(json.dumps(dict(resumed_from=start)), flush=True)
    ds, ds_aux = GridDS(a.shared), CropObjectiveDS(a.shared)
    dl = DataLoader(ds, sampler=Slots(stream, 0, 1, ex, start), batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    cfg = config(a, model, lam)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps(cfg, default=str), flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.monotonic()
    values = []

    def save(step):
        tmp = out / "checkpoints/ema_last.tmp"
        torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=step, pretraining=dict(method="upscaled_teacher", condition=a.condition, lambda_grid=lam)), tmp)
        tmp.replace(out / "checkpoints/ema_last.pt")
        tmp = out / "checkpoints/recovery.tmp"
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, condition=a.condition, lambda_grid=lam), tmp)
        tmp.replace(rec_path)

    with (out / "logs/train.jsonl").open("a") as log, (out / "logs/diagnostics.jsonl").open("a") as checks:
        for k, b in enumerate(dl):
            step = start + k
            b = RC.to_dev(b, dev)
            for group in opt.param_groups:
                group["lr"] = RC.C.lr_at(step, group["base"], WARM, total)
            o, crop, glob = base_losses(model, b)
            F48, a48 = student48(model, o)
            t48 = teacher48(enc0, b["full"])
            w = grid_weights(a.condition, b)
            grid = grid_loss(F48, t48, w)
            if step in (0, 1, 2) or (step + 1) % a.check_every == 0:
                item = next(iter(Slots(stream, 0, 1, ex, step)))
                rec = dict(step=step + 1, batch_sha=batch_sha(b), crop=float(crop), grid=float(grid), glob=float(glob), **diagnostics(model, b, crop, grid, F48, t48, a48, w), grid_vs_inference_read_cos=aux_connection(model, o, item, ds_aux, dev, F48, b))
                checks.write(json.dumps(rec) + "\n")
                checks.flush()
                print(json.dumps(dict(check=rec)), flush=True)
            loss = crop + lam * grid + LAM_G * glob
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"loss at {step + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norms = [torch.nn.utils.clip_grad_norm_(model.vit.parameters(), 1.), torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)]
            if not all(bool(torch.isfinite(v)) for v in norms):
                raise FloatingPointError("gradient")
            opt.step()
            m = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
            ema_update(ema, model, m)
            values.append([float(crop), float(grid), float(glob), float(norms[0]), float(norms[1])])
            if step == 0 or (step + 1) % a.log_every == 0:
                now = time.monotonic()
                v = np.mean(values, axis=0)
                record = dict(step=step + 1, elapsed_s=round(now - t0, 2), crop_cos_loss=float(v[0]), grid_loss=float(v[1]), global_loss=float(v[2]), grad_encoder=float(v[3]), grad_new=float(v[4]), step_seconds=(now - last) / len(values), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                values, last = [], now
            if (step + 1) % a.save_every == 0:
                save(step + 1)
            if STOP:
                save(step + 1)
                (out / "interrupted.json").write_text(json.dumps(dict(step=step + 1, reason="signal")))
                return 143
    save(total)
    assert checksum(enc0.parameters()) == sum0, "teacher changed"
    result = dict(status="done", steps=total, time_s=round(time.monotonic() - t0), started_from_step=start, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, teacher_unchanged=True, ema_checksum=checksum(ema.parameters()))
    (out / "done.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", choices=CONDS)
    ap.add_argument("--initial", default=INITIAL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared", default=SHARED)
    ap.add_argument("--lambda-file")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--check-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-images", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu else "cuda")
    try:
        if a.calibrate:
            calibrate(a, dev)
        elif a.smoke:
            smoke(a, dev)
        else:
            raise SystemExit(train(a, dev))
    except Exception:
        path = Path(a.out)
        if not a.calibrate:
            path.mkdir(parents=True, exist_ok=True)
            (path / "failed.json").write_text(json.dumps(dict(status="failed", error=traceback.format_exc()), indent=1))
        raise


if __name__ == "__main__":
    main()
