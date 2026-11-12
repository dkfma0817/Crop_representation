"""H0/H1 training: P384's crop cosine + 0.5 global (src.hcf.causal_distill, centered weight 0), same P384 start state, stream, ROIs, optimizer, schedule and EMA.

Only additions: the Detail canvas in the batch and the Detail parameters in the new-module optimizer groups. H0 and H1 differ in the canvas only.
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
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.config import COCO_TRAIN, STREAM
from src.hcf.distill import LAM_G, M0_EMA, WARM, Slots, cell_w, groups, pool
from src.hcf.model import checksum, load_hcf, load_vit, n_params
from src.hier.data import HierDS
from src.hier.model import FGRID, Hier, align_to_fine, load_hier
from src.ivc.ft import cosd, ema_update, run_enc
from src.msd.pretrain import eval_exclusions
from src.viz.c0_pca import geometry

STOP = False
P384_LOG = "output/2026-11-08/hcf_material_causal/r384_cosine/P/logs/train.jsonl"


def stop(*_):
    global STOP
    STOP = True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fwd(model, b, collect=False):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model.forward_all(b["full"], b["detail"], collect=collect)
    return dict(cls=o["cls"].float(), coarse=o["coarse"].float(), mem={k: v.float() for k, v in o["mem"].items()}, grids=o["grids"])


def read_parts(model, o, b):
    w24, w48 = cell_w(b["box_c"], 24, 16), cell_w(b["box_c"], 48, 8)
    ri = b["roi_img"]
    A = pool(o["coarse"][ri], w24)
    mem = {k: pool(v[ri], w48) for k, v in o["mem"].items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        base = model.fuse_base(A, mem).float()
        residual = model.detail_residual(mem).float()
    return A, base, residual


def losses(model, b):
    o = fwd(model, b)
    A, base, residual = read_parts(model, o, b)
    Q = base + residual
    crop = cosd(Q, b["tpool"]).mean()
    glob = .5 * (cosd(o["cls"], b["gcls"]) + cosd(pool(o["coarse"], b["fvalid"].float()), b["gmean"])).mean()
    return dict(o=o, A=A, base=base, residual=residual, Q=Q, crop=crop, glob=glob)


def gnorm(grads):
    q = [g.float().square().sum() for g in grads if g is not None]
    return float(torch.stack(q).sum().sqrt()) if q else 0.


def module_grads(model):
    out = {}
    for name, ps in dict(vit=model.vit.parameters(), fine=[p for n, p in model.named_parameters() if n.startswith(("fine_", "inter.", "Wd", "s_d"))], detail=[p for _, p in model.detail_named()]).items():
        out[f"grad_{name}"] = gnorm([p.grad for p in ps])
    return out


def build(a, dev):
    torch.manual_seed(0)
    np.random.seed(0)
    model, ck = load_hier(a.initial, dev)
    assert ck["step"] == 0
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(groups(model), betas=(.9, .999))
    return model, ema, opt


def config(a, model):
    return dict(args=vars(a), condition=a.condition, steps=a.steps, world=1, images_per_step=16, warmup=WARM, global_weight=LAM_G, crop_weight=1., centered_weight=0., detail_canvas=640, global_input=384, seed=0,
                initial_sha256=sha256(a.initial), n_params=n_params(model), n_params_detail=sum(p.numel() for _, p in model.detail_named()), alpha0=.1, source_sha256=sha256(__file__), torch=torch.__version__)


def smoke(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ema, opt = build(a, dev)
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:a.steps]
    ds = HierDS(a.shared, a.condition)
    items = [next(iter(Slots(stream, 0, 1, ex, s))) for s in range(3)]
    report = dict(status="running", condition=a.condition, cfg=config(a, model))
    torch.cuda.reset_peak_memory_stats()
    vit0 = load_vit(dev).eval()
    b = RC.to_dev(ds[items[0]], dev)
    # Detail off reproduces the logged P384 first step (same start state, same batch).
    p384 = json.loads(open(P384_LOG).readline())
    with torch.no_grad():
        model.detail_off = True
        t = losses(model, b)
        cq = float(cosd(t["base"], b["tpool"]).mean())
        report["detail_off_step1"] = dict(crop_loss=cq, p384_crop_loss=p384["crop_cos_loss"], global_loss=float(t["glob"]), p384_global_loss=p384["global_loss"])
        assert abs(cq - p384["crop_cos_loss"]) < 2e-3 and abs(float(t["glob"]) - p384["global_loss"]) < 2e-3, report["detail_off_step1"]
        model.detail_off = False
        report["full_sha"] = hashlib.sha1(b["full"].cpu().numpy().tobytes()).hexdigest()
        report["batch_sha"] = hashlib.sha1(b["box_c"].cpu().numpy().tobytes() + b["tpool"].cpu().numpy().tobytes()).hexdigest()
    # Teacher correspondence: online frozen crops of the first image's object ROIs vs the cached means used by the loss.
    iid = items[0][1][0][1]
    img = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
    boxes = ds.obj[ds.pos[iid]]
    crops, covs = [], []
    for bx in boxes:
        bx = tuple(int(v) for v in bx)
        crop, _, _ = geometry(*img.size, bx, 384, 16)
        crops.append(crop.canvas(img.crop(bx)))
        covs.append(crop.coverage())
    with torch.no_grad():
        T, _ = run_enc(vit0, torch.stack(crops).to(dev))
    T = T[:, 5:] if T.shape[1] == 581 else T
    cov = torch.from_numpy(np.stack(covs)).to(dev).float()
    online = (T.float() * cov[..., None]).sum(1) / cov.sum(1, keepdim=True)
    cos = torch.nn.functional.cosine_similarity(online, b["tpool"][:4], dim=-1)
    assert float(cos.min()) > .9995, float(cos.min())
    area = cell_w(b["box_c"], 24, 16).sum(1)
    expected = (b["box_c"][:, 2] - b["box_c"][:, 0]) * (b["box_c"][:, 3] - b["box_c"][:, 1]) / 16 ** 2
    assert float((area - expected).abs().max()) < 1e-2
    report["teacher"] = dict(cached_vs_online_cos_min=float(cos.min()), dim=int(online.shape[1]), roi_area_mass_error_max=float((area - expected).abs().max()))
    # Alignment: a block of detail tokens must land on the same physical place of the 48 x 48 grid.
    with torch.no_grad():
        z = torch.zeros(1, 1, 80, 80, device=dev)
        z[:, :, 25:35, 40:60] = 1.
        grid = align_to_fine(z)[0, 0]
        ys, xs = torch.meshgrid(torch.arange(48., device=dev), torch.arange(48., device=dev), indexing="ij")
        cx, cy = float((grid * (xs + .5)).sum() / grid.sum() * 8), float((grid * (ys + .5)).sum() / grid.sum() * 8)
    expect = (50. * 8 * 3 / 5, 30. * 8 * 3 / 5)
    assert abs(cx - expect[0]) < 1. and abs(cy - expect[1]) < 1., (cx, cy, expect)
    report["align_centroid_px384"] = dict(measured=[cx, cy], expected=list(expect))
    for k, item in enumerate(items):
        b = RC.to_dev(ds[item], dev)
        for group in opt.param_groups:
            group["lr"] = RC.C.lr_at(k, group["base"], WARM, a.steps)
        t = losses(model, b)
        if k == 0:
            params = [p for _, p in model.detail_named()]
            crop_g = torch.autograd.grad(t["crop"], params, retain_graph=True, allow_unused=True)
            glob_g = torch.autograd.grad(t["glob"], params, retain_graph=True, allow_unused=True)
            names = [n for n, _ in model.detail_named()]
            assert all(g is not None and float(g.norm()) > 0 for g in crop_g), [n for n, g in zip(names, crop_g) if g is None or float(g.norm()) == 0]
            assert all(g is None for g in glob_g), "global loss reaches the detail branch"
            ratio = (t["residual"].norm(dim=-1) / t["base"].norm(dim=-1)).detach()
            report["init"] = dict(detail_grad_crop=gnorm(crop_g), crop_detail_on=float(t["crop"]), crop_detail_off=float(cosd(t["base"], b["tpool"]).mean()), global_loss=float(t["glob"]),
                                  residual_over_base_mean=float(ratio.mean()), residual_over_base_max=float(ratio.max()), alpha=float(model.alpha), detail_tokens=model.detail_tokens(b["detail"]).shape[1])
        loss = t["crop"] + LAM_G * t["glob"]
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if k == 0:
            bad = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            assert not bad, bad
            report["grads"] = module_grads(model)
        torch.nn.utils.clip_grad_norm_(model.vit.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)
        opt.step()
        ema_update(ema, model, .999)
        report.setdefault("steps", []).append(dict(step=k + 1, crop=float(t["crop"]), glob=float(t["glob"]), alpha=float(model.alpha)))
    path = out / "smoke_ckpt.pt"
    torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond=a.condition, kind="ema", s0=.1, step=0), path)
    load_hier(path, dev)
    path.unlink()
    report.update(status="passed", peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, detail_canvas_shape=list(b["detail"].shape), detail_mean_abs=float(b["detail"].abs().mean()))
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
    model, ema, opt = build(a, dev)
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:a.steps]
    total = len(stream)
    assert total == a.steps
    ds = HierDS(a.shared, a.condition)
    dl = DataLoader(ds, sampler=Slots(stream, 0, 1, ex), batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    cfg = config(a, model)
    cfg.update(excluded_stream=ex)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps(cfg, default=str), flush=True)

    def save(step):
        ck = dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond=a.condition, kind="ema", s0=.1, step=step)
        tmp = out / "checkpoints/ema_last.tmp"
        torch.save(ck, tmp)
        tmp.replace(out / "checkpoints/ema_last.pt")
        tmp = out / "checkpoints/recovery.tmp"
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step), tmp)
        tmp.replace(out / "checkpoints/recovery.pt")

    t0 = time.monotonic()
    last = t0
    run = []
    with (out / "logs/train.jsonl").open("a") as log, (out / "logs/grad_checks.jsonl").open("a") as checks:
        for step, b in enumerate(dl):
            b = RC.to_dev(b, dev)
            for group in opt.param_groups:
                group["lr"] = RC.C.lr_at(step, group["base"], WARM, total)
            t = losses(model, b)
            loss = t["crop"] + LAM_G * t["glob"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"loss at {step + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            logging = step == 0 or (step + 1) % a.log_every == 0
            grads = module_grads(model) if logging else None
            ge, gn = torch.nn.utils.clip_grad_norm_(model.vit.parameters(), 1.), torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)
            if not (torch.isfinite(ge) and torch.isfinite(gn)):
                raise FloatingPointError("gradient")
            opt.step()
            mt = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
            ema_update(ema, model, mt)
            with torch.no_grad():
                ratio = float((t["residual"].norm(dim=-1) / t["base"].norm(dim=-1)).mean())
                off = float(cosd(t["base"], b["tpool"]).mean())
                run.append([float(t["crop"]), off, float(t["glob"]), float(cosd(t["A"], b["tpool"]).mean()), ratio])
            if logging:
                now = time.monotonic()
                v = np.mean(run, axis=0)
                record = dict(step=step + 1, elapsed_s=round(now - t0, 2), crop_cos_loss=float(v[0]), crop_loss_detail_off=float(v[1]), global_loss=float(v[2]), coarse_cos_loss=float(v[3]), detail_residual_over_base=float(v[4]), alpha=float(model.alpha.detach()), s_d=float(model.s_d.detach()),
                              step_seconds=(now - last) / len(run), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, **grads)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                run = []
                last = now
            if step == 0 or (step + 1) % 3000 == 0:
                batch_sha = hashlib.sha1(b["box_c"].cpu().numpy().tobytes() + b["tpool"].cpu().numpy().tobytes() + b["full"].cpu().numpy().tobytes()).hexdigest()
                checks.write(json.dumps(dict(step=step + 1, batch_sha=batch_sha, alpha=float(model.alpha.detach()), residual_over_base=ratio, crop_detail_on=float(t["crop"]), crop_detail_off=off, **module_grads(model))) + "\n")
                checks.flush()
            if (step + 1) % 3000 == 0:
                save(step + 1)
            if STOP:
                save(step + 1)
                (out / "interrupted.json").write_text(json.dumps(dict(step=step + 1, reason="signal")))
                return 143
    save(total)
    result = dict(status="done", steps=total, time_s=round(time.monotonic() - t0), ema_checksum=checksum(ema.parameters()), peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    (out / "done.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    return 0


def prepare(a, dev):
    run, path = Path(a.run), Path(a.run) / "initial.pt"
    if path.exists():
        raise FileExistsError(path)
    p384, ck = load_hcf(a.p384_initial, dev)
    torch.manual_seed(9001)
    model = Hier(load_vit(dev), ck["s0"]).to(dev)
    missing, unexpected = model.load_state_dict(p384.state_dict(), strict=False)
    new = sorted(n for n in model.state_dict() if n.startswith(("detail_", "alpha")))
    assert not unexpected and sorted(missing) == new, (missing, unexpected)
    same = all(torch.equal(v, model.state_dict()[k]) for k, v in p384.state_dict().items())
    assert same, "shared weights differ from the P384 start state"
    torch.save(dict(model=model.state_dict(), enc=model.vit.state_dict(), cond="H0", kind="ema", s0=ck["s0"], step=0), path)
    meta = dict(p384_initial=a.p384_initial, p384_initial_sha256=sha256(a.p384_initial), initial_sha256=sha256(path), detail_seed=9001, shared_equal_to_p384_initial=True, n_params_total=n_params(model), n_params_p384=n_params(p384), n_params_detail=sum(p.numel() for _, p in model.detail_named()), new_params=new)
    (run / "design.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--p384-initial", default="output/2026-11-08/hcf_material_causal/initial.pt")
    ap.add_argument("--condition", choices=("H0", "H1"))
    ap.add_argument("--initial")
    ap.add_argument("--out")
    ap.add_argument("--shared", default="output/2026-11-08/hcf_pilot/shared")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.prepare:
        return prepare(a, dev)
    try:
        if a.smoke:
            smoke(a, dev)
        else:
            raise SystemExit(train(a, dev))
    except Exception:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "failed.json").write_text(json.dumps(dict(status="failed", error=traceback.format_exc()), indent=1))
        raise


if __name__ == "__main__":
    main()
