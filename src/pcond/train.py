"""P1/P2: train one conditioning variant of the DINOv3 Coarse-Fine P384 + PixelDiT hybrid. Port of hcf_material_causal/architecture_source/src/hcf/train_crop_objective.py (method pixeldit, crop weight 1).

Changes vs that file: (1) `memory_for` / `build_head` decide the conditioning (B = the existing 8x8 path, kept to reproduce the logged first step); (2) extra logging (conditioning loss with
zero / different-image memory, grad norms of encoder / fine / head). Dataset, sampler, optimizer, schedule, EMA, loss terms and weights, noise and timesteps are unchanged.

  python -m src.pcond.train --condition P1 --initial <initial.pt> --out <dir> [--smoke]
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
from src.hcf.distill import LAM_G, M0_EMA, NEW_LR, WARM, WD, Slots, TrainDS, cell_w, fwd, groups, pool, read_q
from src.hcf.model import load_hcf, n_params
from src.ivc.ft import cosd, ema_update
from src.msd.pretrain import eval_exclusions
from src.pcond.heads import build_head, memory_for
from src.siglip_pixel.crop_objectives import roi_tokens
from src.viz.c0_pca import MEAN_T, STD_T, geometry

STOP = False
BASE = "output/2026-11-08/hcf_material_causal/r384_pixeldit_hybrid/P"
CHECK_EVERY = 1000


def stop(*_):
    global STOP
    STOP = True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def selected_rois(ds, step, column, iid):
    """The baseline choice: one object ROI and one nested ROI, deterministically rotated with the step."""
    i = ds.pos[iid]
    nested = [box for box in ds.NB[step, column] if box[2] > box[0]]
    boxes = list(ds.obj[i]) + nested
    selected = [(step + column) % 4]
    if nested:
        selected.append(4 + (step + column) % len(nested))
    return boxes, selected


class CropObjectiveDS(TrainDS):
    def __getitem__(self, item):
        b = super().__getitem__(item)
        step, slots, _ = item
        crops, cov, grids, valid, indices, ri = [], [], [], [], [], []
        offset = 0
        for k, (column, iid) in enumerate(slots):
            boxes, selected = selected_rois(self, step, column, iid)
            image = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
            for j in selected:
                bx = tuple(int(x) for x in boxes[j])
                crop, _, grid = geometry(*image.size, bx, 384, 16)
                crops.append(crop.canvas(image.crop(bx)))
                cov.append(crop.coverage())
                grids.append(grid)
                mask = torch.zeros(1, 384, 384)
                mask[:, crop.oy:crop.oy + crop.nh, crop.ox:crop.ox + crop.nw] = 1.
                valid.append(mask)
                indices.append(offset + j)
                ri.append(k)
            offset += len(boxes)
        assert offset == len(b["tpool"])
        b.update(crops=torch.stack(crops), aux_cov=torch.from_numpy(np.stack(cov)), aux_grid=torch.from_numpy(np.stack(grids)), pixel_valid=torch.stack(valid), aux_index=torch.tensor(indices), aux_ri=torch.tensor(ri))
        return b


def flow_objective(head, memory, mcov, b, step, mode="matched"):
    """Baseline pixel-flow loss, same generators; `mode` swaps the conditioning only (shuffled = next image in the batch, as in the baseline check)."""
    if mode == "shuffled":
        memory, mcov = memory.roll(2, 0), mcov.roll(2, 0)
    elif mode == "zero":
        memory = torch.zeros_like(memory)
    r = len(memory)
    clean = (b["crops"] * STD_T.to(memory.device) + MEAN_T.to(memory.device)) * 2 - 1
    gen = torch.Generator(device=memory.device).manual_seed(200000 + step)
    t = .1 + .8 * torch.rand(r, device=memory.device, generator=gen)
    noise = torch.randn(clean.shape, device=memory.device, dtype=clean.dtype, generator=gen)
    noisy = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = head(noisy, t, memory, mcov)
    squared = (pred.float() - (noise - clean)).pow(2) * b["pixel_valid"]
    return (squared.flatten(1).sum(1) / (3 * b["pixel_valid"].flatten(1).sum(1)).clamp_min(1)).mean(), float(t.mean())


def losses(model, head, condition, b, step):
    o = fwd(model, b["full"])
    _, Q = read_q(model, o, b["roi_img"], cell_w(b["box_c"], 24, 16), cell_w(b["box_c"], 48, 8))
    crop = cosd(Q, b["tpool"]).mean()
    glob = .5 * (cosd(o["cls"], b["gcls"]) + cosd(pool(o["coarse"], b["fvalid"].float()), b["gmean"])).mean()
    features = roi_tokens(model, o, b)
    memory, mcov = memory_for(condition, features, b["aux_cov"])
    flow, time_mean = flow_objective(head, memory, mcov, b, step)
    return dict(crop=crop, glob=glob, flow=flow, time_mean=time_mean, features=features, memory=memory, mcov=mcov, o=o)


def sha(*tensors):
    return hashlib.sha1(b"".join(x.detach().cpu().numpy().tobytes() for x in tensors)).hexdigest()


def rounded_sha(x):
    return hashlib.sha1((x.detach().cpu() * 1000).round().numpy().tobytes()).hexdigest()


def build(a, dev):
    torch.manual_seed(0)
    np.random.seed(0)
    model, _ = load_hcf(a.initial, dev)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    torch.manual_seed(700)
    head = build_head(a.condition).to(dev).train()
    pg = groups(model)
    for no_decay in (False, True):
        params = [p for p in head.parameters() if (p.ndim <= 1) == no_decay]
        pg.append(dict(params=params, base=NEW_LR, lr=NEW_LR, weight_decay=0. if no_decay else WD, mod="decoder"))
    return model, ema, head, torch.optim.AdamW(pg, betas=(.9, .999))


def config(a, model, head, total):
    return dict(args=vars(a), condition=a.condition, method="pixeldit", crop_weight=1., method_weight=1., global_weight=LAM_G, steps=total, world=1, images_per_step=16, aux_crops_per_image="1 object + 1 nested if present", full_size=384, crop_size=384,
                memory_tokens=dict(B=64, P1=576, P2=576)[a.condition], head_params=n_params(head), encoder_params=n_params(model), initial_sha256=sha256(a.initial), new_lr=NEW_LR, wd=WD, warmup=WARM, ema_m0=M0_EMA, seed=0, source_sha256=sha256(__file__), heads_sha256=sha256(Path(__file__).with_name("heads.py")), teacher="Frozen original DINOv3 crop / global values from the shared cache; RGB crop pixels for L_flow")


def conditioning_check(model, head, condition, b, step, t):
    """L_flow per conditioning, same x_t and timestep; aux gradient into the encoder (no assertion at step 1 for P2: its adaLN last layer is zero-initialised, so the ROI path is open only after one update)."""
    grads = torch.autograd.grad(t["flow"], [model.vit.patch_embed.proj.weight, model.fine_embed.weight, model.Wd.weight, model.s_d], retain_graph=True, allow_unused=True)
    norms = [0. if g is None else float(g.float().norm()) for g in grads]
    if step > 0 and not sum(norms[:2]) > 0:
        raise AssertionError("L_flow has no gradient into the encoder")
    with torch.no_grad():
        shuffled, _ = flow_objective(head, t["memory"].detach(), t["mcov"], b, step, "shuffled")
        zero, _ = flow_objective(head, t["memory"].detach(), t["mcov"], b, step, "zero")
        memory = t["features"]
        spatial = float(memory.std(1).mean())
    return dict(step=step + 1, condition=condition, time_mean=t["time_mean"], aux_grad_coarse_embed=norms[0], aux_grad_fine_embed=norms[1], aux_grad_Wd=norms[2], aux_grad_s_d=norms[3], flow_matched=float(t["flow"]), flow_shuffled=float(shuffled), flow_zero=float(zero),
                gap_shuffled=float(shuffled - t["flow"]), gap_zero=float(zero - t["flow"]), roi_feature_position_std=spatial, batch_sha=sha(b["box_c"], b["tpool"], b["full"]), features_rounded_sha=rounded_sha(t["features"]))


def smoke(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ema, head, opt = build(a, dev)
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:a.steps]
    ds = CropObjectiveDS(a.shared)
    items = [next(iter(Slots(stream, 0, 1, ex, s))) for s in range(3)]
    report = dict(status="running", condition=a.condition, cfg=config(a, model, head, len(stream)))
    torch.cuda.reset_peak_memory_stats()
    for k, item in enumerate(items):
        b = RC.to_dev(ds[item], dev)
        for group in opt.param_groups:
            group["lr"] = RC.C.lr_at(k, group["base"], WARM, len(stream))
        t = losses(model, head, a.condition, b, k)
        record = dict(step=k + 1, flow=float(t["flow"]), crop=float(t["crop"]), glob=float(t["glob"]))
        if k == 0:
            record.update(conditioning_check(model, head, a.condition, b, k, t))
            log = [json.loads(x) for x in Path(BASE, "logs/train.jsonl").read_text().splitlines()][0]
            old = json.loads(Path(BASE, "logs/objective_checks.jsonl").read_text().splitlines()[0])
            report["baseline_step1"] = dict(crop=log["crop_cos_loss"], glob=log["global_loss"], flow=old["matched_condition_loss"], shuffled=old["shuffled_condition_loss"], t=old["time_mean"])
            assert abs(record["crop"] - log["crop_cos_loss"]) < 2e-3 and abs(record["glob"] - log["global_loss"]) < 2e-3, ("encoder start or batch differs from the baseline", record, report["baseline_step1"])
            if a.condition == "B":  # the existing 8x8 path must reproduce the logged pixel-flow loss of the baseline
                assert abs(record["flow"] - old["matched_condition_loss"]) < 2e-3 and abs(record["flow_shuffled"] - old["shuffled_condition_loss"]) < 2e-3, (record, report["baseline_step1"])
            if a.condition == "P2":
                assert t["memory"].shape[1] == 576 and float(t["features"].std(1).mean()) > 1e-3
        loss = t["flow"] + t["crop"] + LAM_G * t["glob"]
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        bad = [n for n, p in head.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not bad, bad
        if k == 1:
            flow_grad = torch.autograd.grad(losses(model, head, a.condition, b, k)["flow"], model.vit.patch_embed.proj.weight)[0]
            record.update(flow_grad_coarse_embed_step2=float(flow_grad.float().norm()))
            assert record["flow_grad_coarse_embed_step2"] > 0, "L_flow does not reach the encoder after one update"
        norms = [torch.nn.utils.clip_grad_norm_(list(model.vit.parameters()), 1.), torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.), torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)]
        record.update(grad_encoder=float(norms[0]), grad_fine=float(norms[1]), grad_head=float(norms[2]))
        opt.step()
        ema_update(ema, model, .999)
        report.setdefault("steps", []).append(record)
    path = out / "smoke_ckpt.pt"
    torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=0, pretraining=dict(method="pixeldit", crop_weight=1., head_params=n_params(head))), path)
    load_hcf(path, dev)
    path.unlink()
    report.update(status="passed", peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, head_params=n_params(head))
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
    model, ema, head, opt = build(a, dev)
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:a.steps]
    total = len(stream)
    assert total == a.steps
    ds = CropObjectiveDS(a.shared)
    dl = DataLoader(ds, sampler=Slots(stream, 0, 1, ex), batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    cfg = config(a, model, head, total)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps(cfg, default=str), flush=True)

    def save(step):
        ck = dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=step, pretraining=dict(method="pixeldit", crop_weight=1., head_params=n_params(head), condition=a.condition))
        path = out / "checkpoints/ema_last.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(ck, tmp)
        tmp.replace(path)
        path = out / "checkpoints/recovery.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), head=head.state_dict(), opt=opt.state_dict(), step=step, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), tmp)
        tmp.replace(path)

    t0 = time.monotonic()
    last = t0
    values = []
    with (out / "logs/train.jsonl").open("a") as log, (out / "logs/objective_checks.jsonl").open("a") as checks:
        for step, b in enumerate(dl):
            b = RC.to_dev(b, dev)
            for group in opt.param_groups:
                group["lr"] = RC.C.lr_at(step, group["base"], WARM, total)
            t = losses(model, head, a.condition, b, step)
            if step == 0 or (step + 1) % CHECK_EVERY == 0:
                check = conditioning_check(model, head, a.condition, b, step, t)
                checks.write(json.dumps(check) + "\n")
                checks.flush()
                print(json.dumps(dict(objective_check=check)), flush=True)
            loss = t["flow"] + t["crop"] + LAM_G * t["glob"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"loss at {step + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norms = [torch.nn.utils.clip_grad_norm_(list(model.vit.parameters()), 1.), torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.), torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)]
            if not all(bool(torch.isfinite(v)) for v in norms):
                raise FloatingPointError("gradient")
            opt.step()
            m = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
            ema_update(ema, model, m)
            values.append([float(t["flow"]), float(t["crop"]), float(t["glob"]), float(norms[0]), float(norms[1]), float(norms[2])])
            if step == 0 or (step + 1) % a.log_every == 0:
                now = time.monotonic()
                v = np.mean(values, axis=0)
                record = dict(step=step + 1, elapsed_s=round(now - t0, 2), flow_mse=float(v[0]), crop_cos_loss=float(v[1]), global_loss=float(v[2]), grad_encoder=float(v[3]), grad_fine=float(v[4]), grad_head=float(v[5]), step_seconds=(now - last) / len(values), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                values = []
                last = now
            if (step + 1) % 3000 == 0:
                save(step + 1)
            if STOP:
                save(step + 1)
                (out / "interrupted.json").write_text(json.dumps(dict(step=step + 1, reason="signal")))
                return 143
    save(total)
    assert not any(p.grad is not None or p.requires_grad for p in ema.parameters())
    result = dict(status="done", steps=total, time_s=round(time.monotonic() - t0), peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, ema_target_stop_gradient=True, head_removed_for_probe=True)
    (out / "done.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=("B", "P1", "P2"))
    ap.add_argument("--initial", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared", default="output/2026-11-08/hcf_pilot/shared")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
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
