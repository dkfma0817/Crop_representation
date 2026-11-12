"""P1 / P2 training: port of hcf_material_causal/architecture_source/src/hcf/train_crop_objective.py (method pixeldit, crop weight 1).

Conditions (only the ROI conditioning of the PixelDiT head differs):
  baseline  existing path, pooled_grid(M_r, cov, 8): 64 memory tokens, CropPixelDiT     (kept to reproduce the logged first step)
  P1        M_r 24 x 24 as 576 memory tokens, the same CropPixelDiT                     (8 x 8 pooling removed)
  P2        M_r 24 x 24, position-wise AdaLN, SpatialPixelDiT                           (ROI cross-attention removed)
M_r = roi_tokens(): coarse and fine maps bilinearly sampled on aux_grid (crop-canvas cell centre -> original -> Full384) and fused.

L = L_flow + L_crop + 0.5 L_global. Start = initial.pt, the same COCO stream, optimizer, schedule and EMA as the baseline.
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
from src.pdit.crop_objectives import CropPixelDiT, pooled_grid, roi_tokens
from src.pdit.heads import SpatialPixelDiT
from src.viz.c0_pca import MEAN_T, STD_T, geometry

STOP = False
BASE_LOG = "output/2026-11-08/hcf_material_causal/r384_pixeldit_hybrid/P/logs/train.jsonl"
CONDS = ("baseline", "P1", "P2")


def stop(*_):
    global STOP
    STOP = True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class CropObjectiveDS(TrainDS):
    """Unchanged from train_crop_objective.py: 1 object + 1 nested crop per image, deterministically rotated."""

    def __getitem__(self, item):
        b = super().__getitem__(item)
        step, slots, _ = item
        crops, cov, grids, valid, indices, ri = [], [], [], [], [], []
        offset = 0
        for k, (column, iid) in enumerate(slots):
            i = self.pos[iid]
            nested = [box for box in self.NB[step, column] if box[2] > box[0]]
            boxes = list(self.obj[i]) + nested
            selected = [(step + column) % 4]
            if nested:
                selected.append(4 + (step + column) % len(nested))
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


def build_head(condition):
    return SpatialPixelDiT() if condition == "P2" else CropPixelDiT()


def conditioning(condition, features, cov):
    """(M_r, coverage) -> (head memory, memory coverage)."""
    return pooled_grid(features, cov, 8) if condition == "baseline" else (features, cov)


def flow_loss(head, memory, mcov, b, step, shuffle=False, zero=False):
    """objective(method='pixeldit') of the baseline. t and noise depend only on the step, so every call on a step sees the same x_t."""
    if shuffle:
        memory, mcov = memory.roll(2, 0), mcov.roll(2, 0)
    if zero:
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
    loss = (squared.flatten(1).sum(1) / (3 * b["pixel_valid"].flatten(1).sum(1)).clamp_min(1)).mean()
    return loss, dict(time_mean=float(t.mean()))


def batch_sha(b):
    return hashlib.sha1(b["box_c"].cpu().numpy().tobytes() + b["tpool"].cpu().numpy().tobytes() + b["full"].cpu().numpy().tobytes() + b["crops"].cpu().numpy().tobytes()).hexdigest()


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


def config(a, model, head):
    return dict(args=vars(a), condition=a.condition, method="pixeldit", crop_weight=1., method_weight=1., global_weight=LAM_G, steps=a.steps, world=1, images_per_step=16, aux_crops_per_image="1 object + 1 nested if present", full_size=384, crop_size=384,
                memory_tokens=dict(baseline=64, P1=576, P2=576)[a.condition], conditioning=dict(baseline="8x8 cov-pooled memory + ROI cross-attention", P1="24x24 memory + ROI cross-attention", P2="24x24 position-wise AdaLN, ROI cross-attention removed")[a.condition],
                head_params=n_params(head), encoder_params=n_params(model), initial_sha256=sha256(a.initial), seed=0, source_sha256=sha256(__file__), heads_sha256=sha256(Path(__file__).with_name("heads.py")), crop_objectives_sha256=sha256(Path(__file__).with_name("crop_objectives.py")), torch=torch.__version__)


def losses(model, b):
    o = fwd(model, b["full"])
    _, Q = read_q(model, o, b["roi_img"], cell_w(b["box_c"], 24, 16), cell_w(b["box_c"], 48, 8))
    crop = cosd(Q, b["tpool"]).mean()
    glob = .5 * (cosd(o["cls"], b["gcls"]) + cosd(pool(o["coarse"], b["fvalid"].float()), b["gmean"])).mean()
    return o, crop, glob


def flow_encoder_grads(model, auxiliary):
    watched = [model.vit.patch_embed.proj.weight, model.fine_embed.weight]
    grads = torch.autograd.grad(auxiliary, watched, retain_graph=True, allow_unused=True)
    return [0. if g is None else float(g.float().norm()) for g in grads]


def smoke(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ema, head, opt = build(a, dev)
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:a.steps]
    ds = CropObjectiveDS(a.shared)
    items = [next(iter(Slots(stream, 0, 1, ex, s))) for s in range(3)]
    torch.cuda.reset_peak_memory_stats()
    report = dict(status="running", cfg=config(a, model, head), steps=[])
    for k, item in enumerate(items):
        b = RC.to_dev(ds[item], dev)
        for group in opt.param_groups:
            group["lr"] = RC.C.lr_at(k, group["base"], WARM, a.steps)
        o, crop, glob = losses(model, b)
        features = roi_tokens(model, o, b)
        memory, mcov = conditioning(a.condition, features, b["aux_cov"])
        aux, _ = flow_loss(head, memory, mcov, b, k)
        norms = flow_encoder_grads(model, aux)
        rec = dict(step=k + 1, flow=float(aux), crop=float(crop), glob=float(glob), memory_shape=list(memory.shape), aux_grad_coarse_embed=norms[0], aux_grad_fine_embed=norms[1], batch_sha=batch_sha(b))
        if k == 0:
            assert features.shape == (len(b["crops"]), 576, 768), features.shape
            rec["padding_cell_fraction"] = float((b["aux_cov"] == 0).float().mean())
            if a.condition == "baseline":
                base = json.loads(open(BASE_LOG).readline())
                report["baseline_repro"] = dict(flow=float(aux), logged_flow=base["method_loss"], crop=float(crop), logged_crop=base["crop_cos_loss"], glob=float(glob), logged_glob=base["global_loss"])
                for x, y in ((float(aux), base["method_loss"]), (float(crop), base["crop_cos_loss"]), (float(glob), base["global_loss"])):
                    assert abs(x - y) < 2e-3, report["baseline_repro"]
            if a.condition == "P2":
                assert sum(norms) == 0., "zero-initialised AdaLN must give no flow gradient into the encoder at step 1"
                c = head.last_c
                rec["cond_std_across_positions"] = float(c.float().std(1).mean())
                assert rec["cond_std_across_positions"] > 0
        elif a.condition == "P2" and k == 1:
            assert sum(norms) > 0, "flow gradient must reach the encoder after the first update"
        if a.condition != "P2" or k > 0:
            assert sum(norms) > 0, "no flow gradient into the encoder"
        report["steps"].append(rec)
        loss = aux + crop + LAM_G * glob
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.vit.parameters()), 1.)
        torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        opt.step()
        ema_update(ema, model, .999)
    path = out / "smoke_ckpt.pt"
    torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=0), path)
    load_hcf(path, dev)
    path.unlink()
    report.update(status="passed", peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
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
    cfg = config(a, model, head)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps(cfg, default=str), flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.monotonic()
    values = []

    def save(step):
        ck = dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=step, pretraining=dict(method="pixeldit", condition=a.condition, crop_weight=1., head_params=n_params(head)))
        path = out / "checkpoints/ema_last.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(ck, tmp)
        tmp.replace(path)
        path = out / "checkpoints/recovery.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), head=head.state_dict(), opt=opt.state_dict(), step=step, condition=a.condition), tmp)
        tmp.replace(path)

    with (out / "logs/train.jsonl").open("a") as log, (out / "logs/objective_checks.jsonl").open("a") as checks:
        for step, b in enumerate(dl):
            b = RC.to_dev(b, dev)
            for group in opt.param_groups:
                group["lr"] = RC.C.lr_at(step, group["base"], WARM, total)
            o, crop_loss, global_loss = losses(model, b)
            features = roi_tokens(model, o, b)
            memory, mcov = conditioning(a.condition, features, b["aux_cov"])
            auxiliary, diagnostics = flow_loss(head, memory, mcov, b, step)
            if step in (0, 1) or (step + 1) % 3000 == 0:
                norms = flow_encoder_grads(model, auxiliary)
                if not sum(norms) > 0 and not (a.condition == "P2" and step == 0):
                    raise AssertionError("Auxiliary objective has no gradient into encoder embeddings")
                with torch.no_grad():
                    shuffled, _ = flow_loss(head, memory.detach(), mcov, b, step, shuffle=True)
                    zeroed, _ = flow_loss(head, memory.detach(), mcov, b, step, zero=True)
                diagnostics.update(aux_grad_coarse_embed=norms[0], aux_grad_fine_embed=norms[1], matched_condition_loss=float(auxiliary), shuffled_condition_loss=float(shuffled), zero_condition_loss=float(zeroed),
                                   conditioning_gap_shuffled=float(shuffled - auxiliary), conditioning_gap_zero=float(zeroed - auxiliary), batch_sha=batch_sha(b))
                checks.write(json.dumps(dict(step=step + 1, **diagnostics)) + "\n")
                checks.flush()
                print(json.dumps(dict(step=step + 1, objective_check=diagnostics)), flush=True)
            loss = auxiliary + crop_loss + LAM_G * global_loss
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
            values.append([float(auxiliary), float(crop_loss), float(global_loss)] + [float(v) for v in norms])
            if step == 0 or (step + 1) % a.log_every == 0:
                now = time.monotonic()
                v = np.mean(values, axis=0)
                record = dict(step=step + 1, elapsed_s=round(now - t0, 2), method_loss=float(v[0]), crop_cos_loss=float(v[1]), global_loss=float(v[2]), grad_encoder=float(v[3]), grad_fine=float(v[4]), grad_head=float(v[5]), step_seconds=(now - last) / len(values), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
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
    ap.add_argument("--condition", required=True, choices=CONDS)
    ap.add_argument("--initial", default="output/2026-11-08/hcf_material_causal/initial.pt")
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
