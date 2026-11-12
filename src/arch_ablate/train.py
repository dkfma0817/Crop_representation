"""A0 / A1 / A2 training: the existing P384 Crop Cosine loop (hcf_material_causal/source/src/hcf/causal_distill.py, centered weight 0, size 384) with only the model structure changed.

L = L_crop + 0.5 L_global. Common initial state, same COCO stream, optimizer groups, schedule, EMA, gradient clipping. A0 is used here only for the parity smoke (the existing A0 run is reused, not retrained).
Resumes from recovery.pt; SIGTERM saves and exits 143.

  python -m src.arch_ablate.train --condition A1 --out <dir> [--smoke]
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.arch_ablate.model import MODES, from_initial
from src.config import STREAM
from src.hcf.distill import LAM_G, M0_EMA, WARM, Slots, TrainDS, cell_w, fwd, groups, pool, read_q
from src.hcf.model import checksum
from src.ivc.ft import cosd, ema_update
from src.msd.pretrain import eval_exclusions

INITIAL = "output/2026-11-08/hcf_material_causal/initial.pt"
SHARED = "output/2026-11-08/hcf_pilot/shared"
A0_LOG = "output/2026-11-08/hcf_material_causal/r384_cosine/P/logs/train.jsonl"
SIZE = 384
_STOP = False


def stop(*_):
    global _STOP
    _STOP = True


def build(a, dev):
    torch.manual_seed(0)
    np.random.seed(0)
    model = from_initial(a.initial, a.condition, dev)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(groups(model), betas=(.9, .999))
    return model, ema, opt


def terms(model, b, collect=False):
    o = fwd(model, b["full"], collect)
    A, Q = read_q(model, o, b["roi_img"], cell_w(b["box_c"], SIZE // 16, 16), cell_w(b["box_c"], SIZE // 8, 8))
    target = b["tpool"]
    crop = cosd(Q, target).mean()
    glob = .5 * (cosd(o["cls"], b["gcls"]) + cosd(pool(o["coarse"], b["fvalid"].float()), b["gmean"])).mean()
    return dict(A=A, Q=Q, crop=crop, glob=glob, coarse_cos=cosd(A, target).mean())


def sub_norm(params):
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return float(torch.cat(grads).norm()) if grads else 0.0


def branch_norms(model):
    named = dict(model.new_named())
    pick = lambda *pre: [p for n, p in named.items() if n.startswith(pre)]  # noqa: E731
    return dict(grad_coarse=sub_norm(model.vit.parameters()), grad_fine=sub_norm(pick("fine_embed", "fine_blocks", "fine_norm")), grad_interaction=sub_norm(pick("inter")), grad_fusion=sub_norm(pick("Wd", "s_d")))


def batch_sha(b):
    return hashlib.sha1(b["full"].detach().cpu().numpy().tobytes() + b["box_c"].detach().cpu().numpy().tobytes()).hexdigest()


def config(a, model, total, ex):
    n_new = sum(p.numel() for _, p in model.new_named())
    return dict(args=vars(a), condition=a.condition, initial_sha256=hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(), world=1, images_per_step=16, steps=total, warmup=WARM, global_weight=LAM_G, input_size=SIZE, teacher_size=SIZE, excluded_stream=ex, seed=0, n_params_total=sum(p.numel() for p in model.parameters()), n_params_new=n_new, torch=torch.__version__)


def smoke(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ema, opt = build(a, dev)
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:a.steps]
    ds = TrainDS(a.shared)
    items = [next(iter(Slots(stream, 0, 1, ex, s))) for s in range(3)]
    report = dict(status="running", condition=a.condition, cfg=config(a, model, len(stream), ex), steps=[])
    torch.cuda.reset_peak_memory_stats()
    for k, item in enumerate(items):
        b = RC.to_dev(ds[item], dev)
        for g in opt.param_groups:
            g["lr"] = RC.C.lr_at(k, g["base"], WARM, len(stream))
        t = terms(model, b, collect=True)
        record = dict(step=k + 1, batch_sha=batch_sha(b), crop=float(t["crop"]), glob=float(t["glob"]), coarse_cos=float(t["coarse_cos"]))
        loss = t["crop"] + LAM_G * t["glob"]
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        zero = [n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) == 0]
        record.update(branch_norms(model), n_no_grad=len(missing), n_zero_grad=len(zero), diag=dict(model.diag))
        if k == 0:
            assert not missing, missing[:8]
            bad = [n for n in zero if not n.startswith("vit.")]
            assert not bad, bad[:8]
            assert record["grad_coarse"] > 0 and record["grad_fine"] > 0 and record["grad_fusion"] > 0, record
            assert (record["grad_interaction"] > 0) == (a.condition != "A2"), record
            log = json.loads(Path(A0_LOG).read_text().splitlines()[0])
            report["a0_logged_step1"] = dict(crop=log["crop_cos_loss"], glob=log["global_loss"], coarse_cos=log["coarse_cos_loss"])
            if a.condition == "A0":
                for key in ("crop", "glob", "coarse_cos"):
                    assert abs(record[key] - report["a0_logged_step1"][key]) < 2e-3, (key, record, report["a0_logged_step1"])
        torch.nn.utils.clip_grad_norm_(model.vit.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)
        opt.step()
        ema_update(ema, model, 1 - (1 - M0_EMA) * (math.cos(math.pi * k / len(stream)) + 1) / 2)
        report["steps"].append(record)
    report.update(status="passed", peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    (out / "smoke.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, default=str), flush=True)


def train(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        raise FileExistsError(out)
    for name in ("checkpoints", "logs"):
        (out / name).mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    model, ema, opt = build(a, dev)
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:a.steps]
    total = len(stream)
    start, rec = 0, out / "checkpoints/recovery.pt"
    if rec.exists():
        ck = torch.load(rec, map_location=dev, weights_only=False)
        assert ck["condition"] == a.condition
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        torch.set_rng_state(ck["rng"].cpu())
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        print(json.dumps(dict(resumed_from=start)), flush=True)
    cfg = config(a, model, total, ex)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg), flush=True)
    dl = DataLoader(TrainDS(a.shared), sampler=Slots(stream, 0, 1, ex, start), batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.monotonic()
    run = []

    def save(step):
        tmp = out / "checkpoints/ema_last.tmp"
        torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", interaction=a.condition, kind="ema", s0=.1, step=step), tmp)
        tmp.replace(out / "checkpoints/ema_last.pt")
        tmp = out / "checkpoints/recovery.tmp"
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, condition=a.condition, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), tmp)
        tmp.replace(rec)

    with (out / "logs/train.jsonl").open("a") as log:
        for k, b in enumerate(dl):
            step = start + k
            b = RC.to_dev(b, dev)
            for g in opt.param_groups:
                g["lr"] = RC.C.lr_at(step, g["base"], WARM, total)
            logging = (step + 1) % a.log_every == 0 or step == 0
            t = terms(model, b, collect=logging)
            loss = t["crop"] + LAM_G * t["glob"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norms = branch_norms(model) if logging else None
            ge = torch.nn.utils.clip_grad_norm_(model.vit.parameters(), 1.)
            gn = torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)
            if not torch.isfinite(ge) or not torch.isfinite(gn):
                raise FloatingPointError("non-finite gradient")
            opt.step()
            ema_update(ema, model, 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2)
            with torch.no_grad():
                contribution = float((t["Q"] - t["A"]).norm(dim=-1).mean() / t["A"].norm(dim=-1).mean().clamp_min(1e-8))
            run.append([float(t["crop"]), float(t["glob"]), float(t["coarse_cos"]), float(ge), float(gn), contribution])
            if logging:
                now = time.monotonic()
                v = np.mean(run, axis=0)
                r = dict(step=step + 1, elapsed_s=round(now - t0, 2), crop_cos_loss=float(v[0]), global_loss=float(v[1]), coarse_cos_loss=float(v[2]), grad_encoder=float(v[3]), grad_new=float(v[4]), fine_contribution=float(v[5]), **norms, diag=dict(model.diag), step_seconds=(now - last) / len(run), mem_gb=torch.cuda.max_memory_allocated() / 2**30)
                log.write(json.dumps(r) + "\n")
                log.flush()
                print(json.dumps(r), flush=True)
                run, last = [], now
            if (step + 1) % 3000 == 0:
                save(step + 1)
            if _STOP:
                save(step + 1)
                (out / "interrupted.json").write_text(json.dumps(dict(step=step + 1, reason="signal; saved EMA and optimizer")))
                return 143
    save(total)
    result = dict(status="done", condition=a.condition, steps=total, time_s=round(time.monotonic() - t0), started_from_step=start, peak_gib=torch.cuda.max_memory_allocated() / 2**30, ema_checksum=checksum(ema.parameters()))
    (out / "done.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=MODES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--initial", default=INITIAL)
    ap.add_argument("--shared", default=SHARED)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.smoke:
        return smoke(a, dev)
    assert a.condition != "A0", "A0 is the existing r384_cosine run; it is reused, not retrained"
    raise SystemExit(train(a, dev))


if __name__ == "__main__":
    main()
