"""PixelDiT x G1 follow-up on the A0 bidirectional Coarse-Fine student (DINOv3-B/16, Full RGB 384 only), COCO 12k steps, seed 0, the common initial.pt (G1-pool encoder + calib scales).

  J1       PixelDiT Hybrid (src.pdit.train condition 'baseline') + the G1U grid loss (src.g1up.train: Full384 canvas bilinear-upscaled to 768, frozen DINOv3, 48x48, uniform valid-cell weights):
           L = L_crop + 0.20 L_grid + 1.0 L_flow + 0.5 L_global   (crop / global computed once; no gradient routing)
  LowFlow  PixelDiT Hybrid with lambda_flow = 0.25:  L = L_crop + 0.25 L_flow + 0.5 L_global
  --mix    W1: theta = (1 - alpha) theta_Hybrid + alpha theta_E2L over every common encoder tensor of the two final EMA checkpoints (flow head is not in them), after compatibility checks.

Gradient logging (steps 1-3 and every 1000): each loss's gradient on the student parameters only (flow head excluded), computed independently on the step's own minibatch with autograd.grad before
the update. Norms are given unweighted and with the loss coefficient; cosines per group. Logging never touches .grad or the RNG (checked every time).

  python -m src.jointpg.train --condition J1|LowFlow --out <dir> [--smoke]
  python -m src.jointpg.train --mix --out <run> --alphas 0,0.25,0.5,0.75,1
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
import src.pdit.train as PD
from src.config import COCO_TRAIN, STREAM
from src.g1up.train import INITIAL, flat, grid_loss, grid_weights, student48, teacher48
from src.hcf.distill import LAM_G, M0_EMA, WARM, Slots
from src.hcf.model import checksum, load_hcf, load_vit, n_params
from src.ivc.ft import ema_update
from src.msd.pretrain import eval_exclusions
from src.overnight6.train import groups_map
from src.viz.c0_pca import LB

HYBRID = "output/2026-11-08/hcf_material_causal/r384_pixeldit_hybrid/P"
E2L = "results/overnight_six/E2L"
HYBRID_LOG = HYBRID + "/logs/train.jsonl"
G1U_DIAG = "output/2026-11-09/g1_upscaled_teacher_s0/G1U/logs/diagnostics.jsonl"
CONDS = dict(J1=dict(lam_grid=.20, lam_flow=1.), LowFlow=dict(lam_grid=0., lam_flow=.25))
STOP = False


def stop(*_):
    global STOP
    STOP = True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class JointDS(PD.CropObjectiveDS):
    """The PixelDiT Hybrid batch + the letterbox content box of every image (= src.g1up.train.GridDS)."""

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


def build(a, dev):
    spec = CONDS[a.condition]
    enc0 = load_vit(dev).eval().requires_grad_(False) if spec["lam_grid"] > 0 else None    # before PD.build, which seeds its own RNG
    model, ema, head, opt = PD.build(argparse.Namespace(**dict(vars(a), condition="baseline")), dev)
    return model, ema, head, opt, enc0


def terms(a, model, head, enc0, b, step):
    spec = CONDS[a.condition]
    o, crop, glob = PD.losses(model, b)
    features = PD.roi_tokens(model, o, b)
    memory, mcov = PD.conditioning("baseline", features, b["aux_cov"])
    flow, _ = PD.flow_loss(head, memory, mcov, b, step)
    t = dict(o=o, crop=crop, glob=glob, flow=flow)
    loss = crop + spec["lam_flow"] * flow + LAM_G * glob
    if spec["lam_grid"] > 0:
        F48, _ = student48(model, o)
        grid = grid_loss(F48, teacher48(enc0, b["full"]), grid_weights("G1U", b))
        t["grid"] = grid
        loss = loss + spec["lam_grid"] * grid
    t["loss"] = loss
    return t


def grad_log(a, model, t):
    """Independent per-loss gradients on the student only; nothing applied. Returns dict and asserts .grad / RNG untouched."""
    spec = CONDS[a.condition]
    coef = dict(crop=1., flow=spec["lam_flow"], glob=LAM_G, **({"grid": spec["lam_grid"]} if "grid" in t else {}))
    mods = groups_map(model)
    mods = dict(coarse=mods["coarse"], fine=[p for k, v in mods.items() if k != "coarse" for p in v], **{f"fine.{k}": v for k, v in mods.items() if k != "coarse"})
    every = mods["coarse"] + mods["fine"]
    assert len(every) == len(list(model.parameters()))
    rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
    assert all(p.grad is None for p in model.parameters())
    g = {k: torch.autograd.grad(t[k], every, retain_graph=True, allow_unused=True) for k in coef}
    index = {id(p): i for i, p in enumerate(every)}
    out = {}
    for name, ps in mods.items():
        sl = {k: torch.cat([(v[index[id(p)]] if v[index[id(p)]] is not None else torch.zeros_like(p)).float().flatten() for p in ps]) for k, v in g.items()}   # unused = zero gradient
        for k, v in sl.items():
            n = float(v.norm())
            out[f"{name}/grad_{k}_raw"], out[f"{name}/grad_{k}_weighted"] = n, n * coef[k]
        keys = [k for k in ("crop", "grid", "flow", "glob") if k in sl]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                out[f"{name}/cos_{keys[i]}_{keys[j]}"] = float(F.cosine_similarity(sl[keys[i]], sl[keys[j]], dim=0))
    out["logging_left_grad_none"] = all(p.grad is None for p in model.parameters())
    out["logging_left_rng_unchanged"] = bool(torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state()))
    assert out["logging_left_grad_none"] and out["logging_left_rng_unchanged"], out
    out.update({f"loss_{k}": float(t[k]) for k in coef}, coefficients=coef)
    return out


def clip(model, head):
    return [torch.nn.utils.clip_grad_norm_(list(model.vit.parameters()), 1.), torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.), torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)]


def config(a, model, head):
    return dict(args={k: str(v) for k, v in vars(a).items()}, condition=a.condition, spec=CONDS[a.condition], crop_weight=1., global_weight=LAM_G, steps=a.steps, world=1, images_per_step=16, warmup=WARM, full_size=384, crop_teacher="C0 Native Crop384 (output/2026-11-08/hcf_pilot/shared)", grid_teacher="G1U: frozen DINOv3 on the 384 canvas bilinear-upscaled to 768, 48x48, uniform valid cells" if CONDS[a.condition]["lam_grid"] else None,
                flow="PixelDiT Hybrid: CropPixelDiT head, 8x8 conditioning, 384 RGB crop flow target, head dropped for probes", initial_sha256=sha256(a.initial), seed=0, encoder_params=n_params(model), head_params=n_params(head), source_sha256=sha256(__file__), pdit_train_sha256=sha256(PD.__file__), torch=torch.__version__)


def smoke(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ema, head, opt, enc0 = build(a, dev)
    ex, stream = eval_exclusions()["raev2_stream"], np.load(STREAM)[:a.steps]
    ds = JointDS(a.shared)
    report = dict(status="running", cfg=config(a, model, head), steps=[])
    for k in range(3):
        b = RC.to_dev(ds[next(iter(Slots(stream, 0, 1, ex, k)))], dev)
        for group in opt.param_groups:
            group["lr"] = RC.C.lr_at(k, group["base"], WARM, a.steps)
        if k == 0:                                                  # logging must not change the update: same step with and without logging on two copies
            twin = [copy.deepcopy(x) for x in (model, head)]
            twin_opt = torch.optim.AdamW(PD.groups(twin[0]) + [dict(params=[p for p in twin[1].parameters() if p.ndim > 1], base=PD.NEW_LR, lr=opt.param_groups[-2]["lr"], weight_decay=PD.WD, mod="decoder"), dict(params=[p for p in twin[1].parameters() if p.ndim <= 1], base=PD.NEW_LR, lr=opt.param_groups[-1]["lr"], weight_decay=0., mod="decoder")], betas=(.9, .999))
            for g_src, g_dst in zip(opt.param_groups, twin_opt.param_groups):
                g_dst["lr"] = g_src["lr"]
            tt = terms(a, twin[0], twin[1], enc0, b, k)
            twin_opt.zero_grad(set_to_none=True)
            tt["loss"].backward()
            clip(*twin)
            twin_opt.step()
        t = terms(a, model, head, enc0, b, k)
        opt.zero_grad(set_to_none=True)
        rec = dict(step=k + 1, **grad_log(a, model, t))
        if k == 0:
            log0 = json.loads(open(HYBRID_LOG).readline())
            parity = dict(flow=float(t["flow"]), logged_flow=log0["method_loss"], crop=float(t["crop"]), logged_crop=log0["crop_cos_loss"], glob=float(t["glob"]), logged_glob=log0["global_loss"])
            if "grid" in t:
                parity.update(grid=float(t["grid"]), logged_grid_g1u=json.loads(open(G1U_DIAG).readline())["grid"])
            assert all(abs(parity[x] - parity[f"logged_{x}"]) < 2e-3 for x in ("flow", "crop", "glob")) and ("grid" not in t or abs(parity["grid"] - parity["logged_grid_g1u"]) < 2e-3), parity
            report["parity"] = parity
        assert torch.isfinite(t["loss"])
        t["loss"].backward()
        clip(model, head)
        opt.step()
        ema_update(ema, model, .999)
        if k == 0:
            num = sum(float((p - q).abs().max()) for p, q in zip(model.parameters(), twin[0].parameters()))
            upd = max(float((p.detach() - q.detach()).abs().max()) for p, q in zip(model.parameters(), ema.parameters()))
            report["logging_vs_no_logging_step1"] = dict(sum_of_max_abs_param_diff=num, max_abs_param_diff=max(float((p - q).abs().max()) for p, q in zip(model.parameters(), twin[0].parameters())), max_abs_update_size=upd)
            del twin, twin_opt
        report["steps"].append(rec)
    path = out / "smoke_ckpt.pt"
    torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=0), path)
    load_hcf(path, dev)
    path.unlink()
    report.update(status="passed", peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    (out / "smoke.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(dict(status="passed", condition=a.condition, parity=report["parity"], logging_check=report["logging_vs_no_logging_step1"], step1={k: round(v, 4) for k, v in report["steps"][0].items() if isinstance(v, float) and k.startswith(("coarse/", "fine/"))}), default=str), flush=True)


def train(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        raise FileExistsError(out)
    for folder in ("logs", "checkpoints"):
        (out / folder).mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    model, ema, head, opt, enc0 = build(a, dev)
    ex, stream = eval_exclusions()["raev2_stream"], np.load(STREAM)[:a.steps]
    total = len(stream)
    assert total == a.steps
    start = 0
    rec_path = out / "checkpoints/recovery.pt"
    if rec_path.exists():
        ck = torch.load(rec_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(json.dumps(dict(resumed_from=start)), flush=True)
    ds = JointDS(a.shared)
    dl = DataLoader(ds, sampler=Slots(stream, 0, 1, ex, start), batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    cfg = config(a, model, head)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps(cfg, default=str), flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.monotonic()
    values = []

    def save(step):
        tmp = out / "checkpoints/ema_last.tmp"
        torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=step, pretraining=dict(method="jointpg", condition=a.condition, **CONDS[a.condition])), tmp)
        tmp.replace(out / "checkpoints/ema_last.pt")
        tmp = out / "checkpoints/recovery.tmp"
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), head=head.state_dict(), opt=opt.state_dict(), step=step, condition=a.condition), tmp)
        tmp.replace(rec_path)

    with (out / "logs/train.jsonl").open("a") as log, (out / "logs/gradients.jsonl").open("a") as glog:
        for k, b in enumerate(dl):
            step = start + k
            b = RC.to_dev(b, dev)
            for group in opt.param_groups:
                group["lr"] = RC.C.lr_at(step, group["base"], WARM, total)
            t = terms(a, model, head, enc0, b, step)
            opt.zero_grad(set_to_none=True)
            if step in (0, 1, 2) or (step + 1) % a.check_every == 0:
                rec = dict(step=step + 1, **grad_log(a, model, t))
                glog.write(json.dumps(rec) + "\n")
                glog.flush()
            if not bool(torch.isfinite(t["loss"])):
                raise FloatingPointError(f"loss at {step + 1}")
            t["loss"].backward()
            norms = clip(model, head)
            if not all(bool(torch.isfinite(v)) for v in norms):
                raise FloatingPointError("gradient")
            opt.step()
            m = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
            ema_update(ema, model, m)
            values.append([float(t["crop"]), float(t["flow"]), float(t.get("grid", 0.)), float(t["glob"])] + [float(v) for v in norms])
            if step == 0 or (step + 1) % a.log_every == 0:
                now = time.monotonic()
                v = np.mean(values, axis=0)
                record = dict(step=step + 1, elapsed_s=round(now - t0, 2), crop_cos_loss=float(v[0]), flow_loss=float(v[1]), grid_loss=float(v[2]), global_loss=float(v[3]), grad_encoder=float(v[4]), grad_fine=float(v[5]), grad_head=float(v[6]), step_seconds=(now - last) / len(values), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
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
    result = dict(status="done", condition=a.condition, steps=total, time_s=round(time.monotonic() - t0), started_from_step=start, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, head_removed_for_probe=True, ema_checksum=checksum(ema.parameters()))
    tmp = out / "done.json.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out / "done.json")
    print(json.dumps(result), flush=True)
    return 0


def mix(a):
    """W1. Compatibility checks first; a failed check stops before any checkpoint is written."""
    run = Path(a.out)
    H = torch.load(f"{HYBRID}/checkpoints/ema_last.pt", map_location="cpu", weights_only=False)
    E = torch.load(f"{E2L}/checkpoints/ema_last.pt", map_location="cpu", weights_only=False)
    I = torch.load(a.initial, map_location="cpu", weights_only=False)
    ch = {}
    ch["meta"] = dict(hybrid={k: H[k] for k in ("cond", "kind", "s0", "step")}, e2l={k: E[k] for k in ("cond", "kind", "s0", "step")})
    assert all(H[k] == E[k] for k in ("cond", "kind", "s0", "step")) and H["kind"] == "ema" and H["step"] == 12000, ch["meta"]
    hk, ek = set(H["model"]), set(E["model"])
    ch["keys_equal"], ch["n_keys"] = hk == ek, len(hk)
    ch["shape_mismatch"] = [k for k in hk & ek if H["model"][k].shape != E["model"][k].shape or H["model"][k].dtype != E["model"][k].dtype]
    assert ch["keys_equal"] and not ch["shape_mismatch"], ch
    ch["initial_keys_equal"] = set(I["model"]) == hk
    ch["initial_sha256"] = dict(hybrid=json.load(open(f"{HYBRID}/config.json"))["initial_sha256"], e2l=json.load(open(f"{E2L}/config.json"))["initial_sha256"], file=sha256(a.initial))
    assert len(set(ch["initial_sha256"].values())) == 1 and ch["initial_keys_equal"], ch
    ch["calib_scales_at_init"] = {k: float(v) for k, v in I["model"].items() if v.ndim == 0}
    ch["scales_final"] = {k: dict(hybrid=float(H["model"][k]), e2l=float(E["model"][k])) for k in ch["calib_scales_at_init"]}
    probe = load_hcf(f"{HYBRID}/checkpoints/ema_last.pt", torch.device("cpu"))[0]
    params = {n for n, _ in probe.named_parameters()}
    buffers = {n for n, _ in probe.named_buffers()}
    ch["n_params"], ch["buffers"] = len(params), sorted(buffers)
    ch["state_not_parameter"] = sorted(hk - params)
    nonparam_equal = {k: bool(torch.equal(H["model"][k], E["model"][k]) and torch.equal(H["model"][k], I["model"][k])) for k in hk - params}
    ch["non_parameter_state_equal_to_each_other_and_initial"] = nonparam_equal
    assert all(nonparam_equal.values()), nonparam_equal
    ch["enc_is_model_vit"] = all(torch.equal(H["enc"][k], H["model"]["vit." + k]) and torch.equal(E["enc"][k], E["model"]["vit." + k]) for k in H["enc"])
    assert ch["enc_is_model_vit"]
    ch["layernorm_params"] = sum(1 for k in params if ".norm" in k or "_norm" in k)
    ch["distance"] = dict(rel_l2_hybrid_vs_e2l=float(math.sqrt(sum(float((H["model"][k].double() - E["model"][k].double()).pow(2).sum()) for k in params)) / math.sqrt(sum(float(H["model"][k].double().pow(2).sum()) for k in params))), rel_l2_hybrid_vs_initial=float(math.sqrt(sum(float((H["model"][k].double() - I["model"][k].double()).pow(2).sum()) for k in params)) / math.sqrt(sum(float(I["model"][k].double().pow(2).sum()) for k in params))), rel_l2_e2l_vs_initial=float(math.sqrt(sum(float((E["model"][k].double() - I["model"][k].double()).pow(2).sum()) for k in params)) / math.sqrt(sum(float(I["model"][k].double().pow(2).sum()) for k in params))))
    made = {}
    for alpha in [float(x) for x in a.alphas.split(",")]:
        name = f"W1a{int(round(alpha * 100)):03d}"
        state = {}
        for k in H["model"]:
            h, e = H["model"][k], E["model"][k]
            if k in params:
                state[k] = h.clone() if alpha == 0 else e.clone() if alpha == 1 else ((1 - alpha) * h.double() + alpha * e.double()).to(h.dtype)
            else:
                state[k] = h.clone()
        d = run / name / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        ck = dict(model=state, enc={k[4:]: v for k, v in state.items() if k.startswith("vit.")}, cond="P", kind="ema", s0=H["s0"], step=12000, pretraining=dict(method="weight_mix", alpha=alpha, theta0=f"{HYBRID}/checkpoints/ema_last.pt", theta1=f"{E2L}/checkpoints/ema_last.pt", rule="(1 - alpha) theta_PixelDiT + alpha theta_E2L, every parameter, float64 then cast; buffers / non-parameter state copied (equal in both)"))
        tmp = d / "ema_last.tmp"
        torch.save(ck, tmp)
        tmp.replace(d / "ema_last.pt")
        load_hcf(d / "ema_last.pt", torch.device("cpu"))
        endpoint = dict(alpha0_equals_hybrid=all(torch.equal(state[k], H["model"][k]) for k in state)) if alpha == 0 else dict(alpha1_equals_e2l=all(torch.equal(state[k], E["model"][k]) for k in state)) if alpha == 1 else {}
        assert all(endpoint.values())
        made[name] = dict(alpha=alpha, path=str(d / "ema_last.pt"), **endpoint)
        (run / name / "config.json").write_text(json.dumps(ck["pretraining"], indent=1))
        (run / name / "done.json").write_text(json.dumps(dict(status="done", kind="weight_mix", alpha=alpha), indent=1))
    ch["made"] = made
    (run / "verification").mkdir(exist_ok=True)
    (run / "verification/w1_mix_compatibility.json").write_text(json.dumps(ch, indent=1))
    print(json.dumps({k: v for k, v in ch.items() if k not in ("calib_scales_at_init", "scales_final", "non_parameter_state_equal_to_each_other_and_initial")}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", choices=tuple(CONDS))
    ap.add_argument("--initial", default=INITIAL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared", default="output/2026-11-08/hcf_pilot/shared")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--check-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    a = ap.parse_args()
    if a.mix:
        return mix(a)
    dev = torch.device("cuda")
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
