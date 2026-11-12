"""Six fixed encoder pilots on the A0 bidirectional Coarse-Fine student (DINOv3-B/16, Full RGB 384 only), COCO 12k steps, seed 0, the common initial.pt.

Everything not named here is src.g1up.train (G1U) or the A0 loop it extends: stream, ROIs, optimizer groups, schedule, clipping, EMA, the crop and global losses.

  E2    G1U with the grid-loss gradient applied to Fine parameters only (L = L_crop + 0.61 L_grid + 0.5 L_global)
  E2L   G1U with lambda_grid = 0.20, no routing
  E1a   A0 with the Native Crop768 pooled teacher (L = L_crop768 + 0.5 L_global)
  E1b   E1a + lambda_spatial L_spatial (cells from src.overnight6.teacher768; lambda fixed by the gradient ratio before training)
  F1    A0 + an extra crop-loss gradient on Fine parameters: g_F = grad_F(L_crop + 0.5 L_global) + 1.0 grad_F(L_crop)
  S1    A0 with crop weight 2.0 for ROIs of area 4 <= a < 16 tokens (Full384, patch 16), normalised by the sum of weights

Fine parameters: fine_embed, fine_blocks, fine_norm, Wd, s_d, inter[*].c2f, inter[*].s_c2f.
Blocked from the extra term: vit (coarse backbone), inter[*].f2c, inter[*].s_f2c.

  python -m src.overnight6.train --condition E2 --out <dir> [--smoke]
  python -m src.overnight6.train --calibrate-spatial --shared768 <dir> --out <json>
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.g1up.train import BASE_LOG, INITIAL, SHARED, GridDS, aux_connection, batch_sha, build, diagnostics, flat, grads, grid_loss, grid_weights, sha256, stream_items, student48, teacher48
from src.hcf.distill import LAM_G, M0_EMA, WARM, Slots, TrainDS, cell_w, fwd, pool, read_q
from src.hcf.model import checksum, load_hcf
from src.ivc.ft import cosd, ema_update
from src.overnight6.teacher768 import edges, overlap, pool_grid
from src.pdit.train import CropObjectiveDS

G1U_DIAG = "output/2026-11-09/g1_upscaled_teacher_s0/G1U/logs/diagnostics.jsonl"
CONDS = dict(
    E2=dict(base="G1U", lam_grid=.61, route="grid"),
    E2L=dict(base="G1U", lam_grid=.20),
    E1a=dict(base="A0", teacher768=True),
    E1b=dict(base="A0", teacher768=True, spatial=True),
    F1=dict(base="A0", route="crop"),
    S1=dict(base="A0", small_weight=2.),
)
STOP = False


def stop(*_):
    global STOP
    STOP = True


def fine_params(model):
    names, params = [], []
    for n, p in model.named_parameters():
        fine = n.startswith(("fine_embed.", "fine_blocks.", "fine_norm.", "Wd.")) or n == "s_d" or (n.startswith("inter.") and (".c2f." in n or n.endswith(".s_c2f")))
        if fine:
            names.append(n)
            params.append(p)
    return names, params


def blocked_params(model):
    names, params = [], []
    for n, p in model.named_parameters():
        if n.startswith("vit.") or (n.startswith("inter.") and (".f2c." in n or n.endswith(".s_f2c"))):
            names.append(n)
            params.append(p)
    return names, params


def check_partition(model):
    fine, blocked = fine_params(model)[0], blocked_params(model)[0]
    every = [n for n, _ in model.named_parameters()]
    assert sorted(fine + blocked) == sorted(every) and not set(fine) & set(blocked), "fine / blocked lists must partition the parameters"
    return dict(fine=fine, blocked_prefixes=sorted({n.split(".")[0] if n.startswith("vit.") else n for n in blocked}), n_fine=len(fine), n_blocked=len(blocked))


def groups_map(model):
    inter_c2f, inter_f2c = [], []
    for m in model.inter:
        inter_c2f += list(m.c2f.parameters()) + [m.s_c2f]
        inter_f2c += list(m.f2c.parameters()) + [m.s_f2c]
    return dict(coarse=list(model.vit.parameters()), fine_embed=list(model.fine_embed.parameters()), fine_transformer=list(model.fine_blocks.parameters()), fine_projection=list(model.Wd.parameters()) + list(model.fine_norm.parameters()) + [model.s_d], coarse_to_fine=inter_c2f, fine_to_coarse=inter_f2c)


def loss_diag(model, losses):
    """Per-group gradient norms of each loss and pairwise cosine on the coarse and Fine groups. Nothing is applied."""
    mods = groups_map(model)
    params = [p for v in mods.values() for p in v]
    gs = {k: grads(v, params) for k, v in losses.items()}
    out, i = {}, 0
    for name, v in mods.items():
        slices = {k: flat(g[i:i + len(v)]) for k, g in gs.items()}
        i += len(v)
        for k, g in slices.items():
            out[f"grad_{k}_{name}"] = float(g.norm())
        keys = list(slices)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                if slices[keys[x]].numel() == slices[keys[y]].numel():
                    out[f"cos_{keys[x]}_{keys[y]}_{name}"] = float(F.cosine_similarity(slices[keys[x]], slices[keys[y]], dim=0))
    for k, g in gs.items():
        out[f"grad_{k}_total"] = float(flat(g).norm())
    return out


def backward_routed(base, extra, fine, blocked, verify=False):
    """p.grad = grad(base) for every parameter, + grad(extra) on the Fine list only. verify: measures that blocked parameters received exactly grad(base)."""
    gx = torch.autograd.grad(extra, fine, retain_graph=True, allow_unused=True)
    base.backward()
    before = None
    if verify:                                                      # snapshots, not a second autograd pass: repeated backward on one graph differs by ~0.5% on this GPU
        before = dict(blocked=[None if p.grad is None else p.grad.detach().clone() for p in blocked], fine=[None if p.grad is None else p.grad.detach().clone() for p in fine])
    for p, g in zip(fine, gx):
        if g is not None:
            p.grad = g.detach() if p.grad is None else p.grad + g.detach()
    if not verify:
        return None
    blocked_diff = [(p.grad - g).flatten() for p, g in zip(blocked, before["blocked"]) if p.grad is not None and g is not None]
    fine_diff = [(p.grad - (b if b is not None else 0) - x).flatten() for p, b, x in zip(fine, before["fine"], gx) if x is not None]
    report = dict(blocked_changed_by_extra_norm=float(torch.cat(blocked_diff).norm()), blocked_grad_is_none=sum(p.grad is None for p in blocked), fine_applied_minus_extra_norm=float(torch.cat(fine_diff).norm()), fine_extra_norm=float(flat(gx).norm()), fine_base_norm=float(flat(before["fine"]).norm()), blocked_base_norm=float(flat(before["blocked"]).norm()), fine_extra_missing=sum(g is None for g in gx))
    assert report["blocked_changed_by_extra_norm"] == 0, report
    assert report["fine_extra_norm"] > 0 and report["fine_extra_missing"] == 0 and report["blocked_grad_is_none"] == 0, report
    assert report["fine_applied_minus_extra_norm"] <= 1e-4 * max(report["fine_extra_norm"] + report["fine_base_norm"], 1e-6), report
    return report


class SpatialDS(TrainDS):
    """TrainDS + the ROI's spatial Teacher cells (src.overnight6.teacher768), concatenated in the TrainDS ROI order."""

    def __init__(self, shared):
        super().__init__(shared)
        self.cells = None

    def __getitem__(self, item):
        if self.cells is None:
            self.cells = np.load(f"{self.d}/sp_cells.npy", mmap_mode="r")
            self.io, self.ineg = np.load(f"{self.d}/sp_index_object.npy"), np.load(f"{self.d}/sp_index_nested.npy")
        b = super().__getitem__(item)
        step, slots, _ = item
        roi, gw, gh, cells = [], [], [], []
        r = 0

        def add(index):
            nonlocal r
            off, w, h = (int(v) for v in index)
            if off >= 0:
                roi.append(r)
                gw.append(w)
                gh.append(h)
                cells.append(np.array(self.cells[off:off + w * h], dtype=np.float32))
            r += 1
        for c, iid in slots:
            i = self.pos[iid]
            for j in range(4):
                add(self.io[i, j])
            for j in range(4):
                if self.NB[step, c, j][2] > self.NB[step, c, j][0]:
                    add(self.ineg[step, c, j])
        assert r == len(b["box_c"]), (r, len(b["box_c"]))
        b["sp_roi"], b["sp_gw"], b["sp_gh"] = torch.as_tensor(roi, dtype=torch.long), torch.as_tensor(gw, dtype=torch.long), torch.as_tensor(gh, dtype=torch.long)
        b["sp_cells"] = torch.from_numpy(np.concatenate(cells, 0)) if cells else torch.zeros(0, 768)
        return b


def spatial_loss(model, o, b):
    """L_spatial on the ROI's cells: student cell = fuse(area-pooled coarse, area-pooled Fine) over the ROI sub-rectangle; teacher cell from the cache. Mean over cells of a ROI, then over eligible ROIs."""
    dev, A, B, sizes = b["full"].device, [], [], []
    for r, gw, gh in zip(b["sp_roi"].tolist(), b["sp_gw"].tolist(), b["sp_gh"].tolist()):
        k, box = int(b["roi_img"][r]), b["box_c"][r]
        ex, ey = edges(box[0], box[2], gw, dev), edges(box[1], box[3], gh, dev)
        A.append(pool_grid(o["coarse"][k].reshape(24, 24, -1), overlap(ey, 24, 16), overlap(ex, 24, 16)).reshape(gw * gh, -1))
        B.append(pool_grid(o["mem"]["fine"][k].reshape(48, 48, -1), overlap(ey, 48, 8), overlap(ex, 48, 8)).reshape(gw * gh, -1))
        sizes.append(gw * gh)
    assert sizes, "no eligible ROI in the batch"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Q = model.fuse(torch.cat(A), dict(fine=torch.cat(B)))
    per = cosd(Q.float(), b["sp_cells"].float())
    return torch.stack([v.mean() for v in per.split(sizes)]).mean(), sizes


def forward_terms(model, b, spec):
    o = fwd(model, b["full"])
    _, Q = read_q(model, o, b["roi_img"], cell_w(b["box_c"], 24, 16), cell_w(b["box_c"], 48, 8))
    per = cosd(Q, b["tpool"])
    plain = per.mean()
    crop, sw = plain, None
    if spec.get("small_weight"):
        area = (b["box_c"][:, 2] - b["box_c"][:, 0]) * (b["box_c"][:, 3] - b["box_c"][:, 1]) / 256
        sw = torch.where((area >= 4) & (area < 16), torch.full_like(area, spec["small_weight"]), torch.ones_like(area))
        crop = (sw * per).sum() / sw.sum()
    glob = .5 * (cosd(o["cls"], b["gcls"]) + cosd(pool(o["coarse"], b["fvalid"].float()), b["gmean"])).mean()
    return o, crop, plain, glob, sw


def teacher_dir(a, spec):
    return a.shared768 if spec.get("teacher768") else a.shared


def dataset(a, spec):
    if spec.get("spatial"):
        return SpatialDS(a.shared768)
    if spec["base"] == "G1U":
        return GridDS(a.shared)
    return TrainDS(teacher_dir(a, spec))


def calibrate_spatial(a, dev):
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    a.condition = "E1b"
    model, _, _, _ = build(a, dev)
    ex, stream = stream_items(a)
    ds = SpatialDS(a.shared768)
    params = list(model.parameters())
    records = []
    for s in range(a.calib_batches):
        b = RC.to_dev(ds[next(iter(Slots(stream, 0, 1, ex, s)))], dev)
        o, crop, *_ = forward_terms(model, b, CONDS["E1b"])
        sp, _ = spatial_loss(model, o, b)
        gc, gs = flat(grads(crop, params)).norm(), flat(grads(sp, params)).norm()
        records.append(dict(batch=s, grad_crop=float(gc), grad_spatial=float(gs), ratio=float(gc / gs), spatial_loss=float(sp), crop_loss=float(crop)))
    med = float(np.median([r["ratio"] for r in records]))
    report = dict(lambda_spatial=float(f"{med:.2g}"), median_ratio=med, rule="median ||g_crop768|| / ||g_spatial|| at the initial weights, all CoarseFine parameters, 8 batches, 2 significant figures", records=records, initial_sha256=sha256(a.initial))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


def config(a, spec, lam_spatial, model):
    return dict(args={k: str(v) for k, v in vars(a).items()}, condition=a.condition, spec=spec, lambda_grid=spec.get("lam_grid"), lambda_spatial=lam_spatial, global_weight=LAM_G, steps=a.steps, world=1, images_per_step=16, warmup=WARM, full_size=384, initial_sha256=sha256(a.initial), seed=0, source_sha256=sha256(__file__), partition=check_partition(model), torch=torch.__version__, teacher_shared=str(teacher_dir(a, spec)))


def gather_step(a, spec, model, enc0, b, lam_spatial):
    """All loss terms of one step. Returns (terms, base, extra); base + extra is the objective (extra = None when no routing)."""
    o, crop, plain, glob, sw = forward_terms(model, b, spec)
    t = dict(o=o, crop=crop, plain=plain, glob=glob, sw=sw)
    base = crop + LAM_G * glob
    extra = None
    if spec["base"] == "G1U":
        F48, a48 = student48(model, o)
        t48 = teacher48(enc0, b["full"])
        w = grid_weights("G1U", b)
        grid = grid_loss(F48, t48, w)
        t.update(grid=grid, F48=F48, a48=a48, t48=t48, w=w)
        if spec.get("route") == "grid":
            extra = spec["lam_grid"] * grid
        else:
            base = base + spec["lam_grid"] * grid
    if spec.get("route") == "crop":
        extra = crop
    if spec.get("spatial"):
        sp, sizes = spatial_loss(model, o, b)
        t.update(spatial=sp, sizes=sizes)
        base = base + lam_spatial * sp
    return t, base, extra


def check_record(a, spec, model, b, t, step):
    if spec["base"] == "G1U":
        rec = dict(step=step + 1, batch_sha=batch_sha(b), crop=float(t["crop"]), grid=float(t["grid"]), glob=float(t["glob"]), **diagnostics(model, b, t["crop"], t["grid"], t["F48"], t["t48"], t["a48"], t["w"]))
        rec["coarse_grid_over_crop_grad"] = rec["grad_grid_coarse"] / max(rec["grad_crop_coarse"], 1e-12)
        rec.update({f"group_{k}": v for k, v in loss_diag(model, dict(crop=t["crop"], grid=t["grid"])).items()})
        return rec
    losses = dict(crop=t["crop"], glob=t["glob"])
    if spec.get("spatial"):
        losses["spatial"] = t["spatial"]
    rec = dict(step=step + 1, batch_sha=batch_sha(b), crop=float(t["crop"]), crop_plain=float(t["plain"]), glob=float(t["glob"]), **loss_diag(model, losses))
    if spec.get("spatial"):
        rec.update(spatial=float(t["spatial"]), spatial_rois=len(t["sizes"]), spatial_roi_fraction=len(t["sizes"]) / len(b["box_c"]), spatial_cells_mean=float(np.mean(t["sizes"])))
    if t["sw"] is not None:
        rec.update(small_roi_fraction=float((t["sw"] > 1).float().mean()), small_weight_sum_over_n=float(t["sw"].mean()))
    return rec


def smoke(a, dev):
    spec = CONDS[a.condition]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lam_spatial = json.loads(Path(a.lambda_spatial_file).read_text())["lambda_spatial"] if a.lambda_spatial_file else (.1 if spec.get("spatial") else None)
    model, ema, enc0, opt = build(a, dev)
    sum0 = checksum(enc0.parameters())
    ex, stream = stream_items(a)
    ds = dataset(a, spec)
    aux_ds = CropObjectiveDS(a.shared) if spec["base"] == "G1U" else None
    fine, blocked = fine_params(model)[1], blocked_params(model)[1]
    report = dict(status="running", cfg=config(a, spec, lam_spatial, model), steps=[], routing=[])
    for k in range(3):
        item = next(iter(Slots(stream, 0, 1, ex, k)))
        b = RC.to_dev(ds[item], dev)
        for group in opt.param_groups:
            group["lr"] = RC.C.lr_at(k, group["base"], WARM, a.steps)
        t, base, extra = gather_step(a, spec, model, enc0, b, lam_spatial)
        rec = check_record(a, spec, model, b, t, k)
        if spec["base"] == "G1U":
            rec["grid_vs_inference_read_cos"] = aux_connection(model, t["o"], item, aux_ds, dev, t["F48"], b)
        if k == 0:
            log0 = json.loads(open(BASE_LOG).readline())
            parity = dict(crop=float(t["plain"]), logged_crop=log0["crop_cos_loss"], glob=float(t["glob"]), logged_glob=log0["global_loss"])
            assert abs(parity["glob"] - parity["logged_glob"]) < 2e-3, parity
            if not spec.get("teacher768"):
                assert abs(parity["crop"] - parity["logged_crop"]) < 2e-3, parity
            if spec["base"] == "G1U":
                g0 = json.loads(open(G1U_DIAG).readline())
                parity.update(grid=float(t["grid"]), logged_grid=g0["grid"], batch_sha=rec["batch_sha"], logged_sha=g0["batch_sha"])
                assert abs(parity["grid"] - parity["logged_grid"]) < 2e-3 and parity["batch_sha"] == parity["logged_sha"], parity
            report["parity"] = parity
        report["steps"].append(rec)
        loss = base if extra is None else base + extra
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        if extra is None:
            base.backward()
        else:
            report["routing"].append(backward_routed(base, extra, fine, blocked, verify=True))
        torch.nn.utils.clip_grad_norm_(model.vit.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)
        opt.step()
        ema_update(ema, model, .999)
    assert checksum(enc0.parameters()) == sum0, "teacher changed"
    path = out / "smoke_ckpt.pt"
    torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=0), path)
    load_hcf(path, dev)
    path.unlink()
    report.update(status="passed", peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    (out / "smoke.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(dict(status="passed", condition=a.condition, parity=report.get("parity"), routing=report["routing"][:1], peak_gib=report["peak_gib"], last=report["steps"][-1]), default=str), flush=True)


def train(a, dev):
    spec = CONDS[a.condition]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        raise FileExistsError(out)
    for folder in ("logs", "checkpoints"):
        (out / folder).mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lam_spatial = json.loads(Path(a.lambda_spatial_file).read_text())["lambda_spatial"] if spec.get("spatial") else None
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
    ds = dataset(a, spec)
    aux_ds = CropObjectiveDS(a.shared) if spec["base"] == "G1U" else None
    dl = DataLoader(ds, sampler=Slots(stream, 0, 1, ex, start), batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    cfg = config(a, spec, lam_spatial, model)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps(cfg, default=str), flush=True)
    fine, blocked = fine_params(model)[1], blocked_params(model)[1]
    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.monotonic()
    values = []
    sp_hist, sp_rois, all_rois, small_rois = np.zeros(145, np.int64), 0, 0, 0

    def save(step):
        tmp = out / "checkpoints/ema_last.tmp"
        torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=step, pretraining=dict(method="overnight6", condition=a.condition, lambda_grid=spec.get("lam_grid"), lambda_spatial=lam_spatial)), tmp)
        tmp.replace(out / "checkpoints/ema_last.pt")
        tmp = out / "checkpoints/recovery.tmp"
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, condition=a.condition), tmp)
        tmp.replace(rec_path)

    with (out / "logs/train.jsonl").open("a") as log, (out / "logs/diagnostics.jsonl").open("a") as checks:
        for k, b in enumerate(dl):
            step = start + k
            b = RC.to_dev(b, dev)
            for group in opt.param_groups:
                group["lr"] = RC.C.lr_at(step, group["base"], WARM, total)
            t, base, extra = gather_step(a, spec, model, enc0, b, lam_spatial)
            check = step in (0, 1, 2) or (step + 1) % a.check_every == 0
            if check:
                rec = check_record(a, spec, model, b, t, step)
                if spec["base"] == "G1U":
                    item = next(iter(Slots(stream, 0, 1, ex, step)))
                    rec["grid_vs_inference_read_cos"] = aux_connection(model, t["o"], item, aux_ds, dev, t["F48"], b)
            loss = base if extra is None else base + extra
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"loss at {step + 1}")
            opt.zero_grad(set_to_none=True)
            if extra is None:
                base.backward()
            else:
                routing = backward_routed(base, extra, fine, blocked, verify=check)
                if check:
                    rec["routing"] = routing
            if check:
                checks.write(json.dumps(rec) + "\n")
                checks.flush()
                print(json.dumps(dict(check=rec)), flush=True)
            norms = [torch.nn.utils.clip_grad_norm_(model.vit.parameters(), 1.), torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)]
            if not all(bool(torch.isfinite(v)) for v in norms):
                raise FloatingPointError("gradient")
            opt.step()
            m = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
            ema_update(ema, model, m)
            all_rois += len(b["box_c"])
            if spec.get("spatial"):
                sp_rois += len(t["sizes"])
                for s in t["sizes"]:
                    sp_hist[s] += 1
            if t["sw"] is not None:
                small_rois += int((t["sw"] > 1).sum())
            values.append([float(t["crop"]), float(t["plain"]), float(t["glob"]), float(t.get("grid", 0.)), float(t.get("spatial", 0.)), float(norms[0]), float(norms[1])])
            if step == 0 or (step + 1) % a.log_every == 0:
                now = time.monotonic()
                v = np.mean(values, axis=0)
                record = dict(step=step + 1, elapsed_s=round(now - t0, 2), crop_cos_loss=float(v[0]), crop_cos_plain=float(v[1]), global_loss=float(v[2]), grid_loss=float(v[3]), spatial_loss=float(v[4]), grad_encoder=float(v[5]), grad_new=float(v[6]), step_seconds=(now - last) / len(values), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
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
    if spec.get("spatial"):
        total_sp = max(sp_rois, 1)
        cdf = sp_hist.cumsum() / total_sp
        (out / "spatial_stats.json").write_text(json.dumps(dict(roi_seen=all_rois, roi_with_spatial=sp_rois, applied_fraction=sp_rois / max(all_rois, 1), cells_p01=int(np.searchsorted(cdf, .01)), cells_p25=int(np.searchsorted(cdf, .25)), cells_p50=int(np.searchsorted(cdf, .5)), cells_p75=int(np.searchsorted(cdf, .75)), cells_p99=int(np.searchsorted(cdf, .99)), histogram={str(i): int(v) for i, v in enumerate(sp_hist) if v}, note="counted over the steps of this process; on resume, only the steps after the resume point"), indent=1))
    if spec.get("small_weight"):
        (out / "small_roi_stats.json").write_text(json.dumps(dict(roi_seen=all_rois, roi_weighted=small_rois, weighted_fraction=small_rois / max(all_rois, 1), weight=spec["small_weight"], rule="4 <= area / 16^2 < 16 on the 384 canvas"), indent=1))
    result = dict(status="done", condition=a.condition, steps=total, time_s=round(time.monotonic() - t0), started_from_step=start, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, teacher_unchanged=True, ema_checksum=checksum(ema.parameters()), checkpoint_sha1=hashlib.sha1((out / "checkpoints/ema_last.pt").read_bytes()).hexdigest())
    tmp = out / "done.json.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out / "done.json")
    print(json.dumps(result), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", choices=tuple(CONDS))
    ap.add_argument("--initial", default=INITIAL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared", default=SHARED)
    ap.add_argument("--shared768", default="")
    ap.add_argument("--lambda-spatial-file", default="")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--check-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--calibrate-spatial", action="store_true")
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    try:
        if a.calibrate_spatial:
            calibrate_spatial(a, dev)
        elif a.smoke:
            smoke(a, dev)
        else:
            raise SystemExit(train(a, dev))
    except Exception:
        if not a.calibrate_spatial:
            path = Path(a.out)
            path.mkdir(parents=True, exist_ok=True)
            (path / "failed.json").write_text(json.dumps(dict(status="failed", error=traceback.format_exc()), indent=1))
        raise


if __name__ == "__main__":
    main()
