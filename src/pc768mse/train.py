"""Two encoder pilots on the A0 bidirectional Coarse-Fine student (DINOv3-B/16, Full RGB 384 only), COCO 12k steps, seed 0, the common initial.pt.

  PC768  PixelDiT Hybrid (src.pdit.train condition 'baseline' = the ported r384_pixeldit_hybrid recipe) with ONLY the crop-feature teacher changed:
         tpool = Native Crop768 pooled teacher (output/2026-11-10/overnight_six_teacher768) instead of Native Crop384.
         L = L_crop768 + 1.0 L_flow + 0.5 L_global; flow head, 8x8 conditioning, RGB flow target (384 crops), head removal, optimizer, EMA unchanged.
  M1     A0 Crop Cosine with the crop loss replaced by raw-feature MSE on the same vectors (Q = fuse(area-pooled coarse, area-pooled Fine), T = pooled C0 Crop384 teacher; no L2 normalisation):
         L = lambda_mse mean((Q - T)^2) + 0.5 L_global; lambda_mse = median ||g_cos|| / ||g_mse|| over 8 batches at the initial weights, all CoarseFine parameters, 2 significant figures, fixed before training.

  python -m src.pc768mse.train --verify-teacher --out <json>
  python -m src.pc768mse.train --calibrate-mse --out <json>
  python -m src.pc768mse.train --condition PC768|M1 --out <dir> [--lambda-mse-file <json>] [--smoke]
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
from src.g1up.train import INITIAL, flat, grads
from src.hcf.distill import LAM_G, M0_EMA, WARM, Slots, TrainDS, cell_w, fwd, groups, pool, read_q
from src.hcf.model import checksum, load_hcf, load_vit, n_params
from src.ivc.ft import cosd, ema_update
from src.msd.pretrain import eval_exclusions
from src.overnight6.teacher768 import S_T, cov48
from src.overnight6.train import groups_map, loss_diag
from src.viz.c0_pca import LB

SHARED384 = "output/2026-11-08/hcf_pilot/shared"
SHARED768 = "output/2026-11-10/overnight_six_teacher768"
HYBRID_LOG = "output/2026-11-08/hcf_material_causal/r384_pixeldit_hybrid/P/logs/train.jsonl"
A0_LOG = "output/2026-11-08/hcf_material_causal/r384_cosine/P/logs/train.jsonl"
E1A_SMOKE_CROP = 0.16176007688045502                                          # E1a (same 768 teacher, same ROIs) step-1 crop loss
CONDS = dict(PC768=dict(base="Hybrid", teacher="Native Crop768", crop_loss="1 - cos", flow=True), M1=dict(base="A0", teacher="Native Crop384 (C0)", crop_loss="raw MSE", flow=False))
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


class PixelDS(PD.CropObjectiveDS):
    """The PixelDiT Hybrid batch (CropObjectiveDS) on the Crop768 teacher dir, + the Crop384 teacher of the same ROIs (diagnostics only)."""

    def __init__(self, shared):
        super().__init__(shared)
        self.t384 = None

    def __getitem__(self, item):
        if self.t384 is None:
            self.t384 = {k: np.load(f"{SHARED384}/{k}.npy", mmap_mode="r") for k in ("t_object", "t_nested")}
        b = super().__getitem__(item)
        step, slots, _ = item
        rows = []
        for c, iid in slots:
            i = self.pos[iid]
            rows += [np.array(self.t384["t_object"][i, j]) for j in range(4)]
            rows += [np.array(self.t384["t_nested"][step, c, j]) for j in range(4) if self.NB[step, c, j][2] > self.NB[step, c, j][0]]
        b["tpool384"] = torch.from_numpy(np.stack(rows)).float()
        assert b["tpool384"].shape == b["tpool"].shape
        return b


def stream_items(a):
    return eval_exclusions()["raev2_stream"], np.load(STREAM)[:a.steps]


def build(a, dev):
    if a.condition == "PC768":
        a.condition_pdit = "baseline"
        ns = argparse.Namespace(**dict(vars(a), condition="baseline"))
        model, ema, head, opt = PD.build(ns, dev)
        return model, ema, head, opt
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
    return model, ema, None, torch.optim.AdamW(groups(model), betas=(.9, .999))


def dataset(a):
    return PixelDS(a.shared768) if a.condition == "PC768" else TrainDS(a.shared384)


def feature_stats(Q, T, T384=None):
    with torch.no_grad():
        qn, tn = Q.norm(dim=-1), T.norm(dim=-1)
        out = dict(student_norm_mean=float(qn.mean()), student_norm_std=float(qn.std()), teacher_norm_mean=float(tn.mean()), teacher_norm_std=float(tn.std()), norm_diff_mean=float((qn - tn).mean()), norm_absdiff_mean=float((qn - tn).abs().mean()),
                   cos_student_teacher=float(F.cosine_similarity(Q, T, dim=-1).mean()), raw_mse=float((Q - T).pow(2).mean()))
        if T384 is not None:
            out.update(teacher384_norm_mean=float(T384.norm(dim=-1).mean()), cos_teacher768_teacher384=float(F.cosine_similarity(T, T384, dim=-1).mean()), cos_student_teacher384=float(F.cosine_similarity(Q, T384, dim=-1).mean()))
    return out


def step_terms(a, model, head, b, step, lam):
    """-> dict of loss terms and the objective."""
    o = fwd(model, b["full"])
    _, Q = read_q(model, o, b["roi_img"], cell_w(b["box_c"], 24, 16), cell_w(b["box_c"], 48, 8))
    T = b["tpool"]
    glob = .5 * (cosd(o["cls"], b["gcls"]) + cosd(pool(o["coarse"], b["fvalid"].float()), b["gmean"])).mean()
    cos = cosd(Q, T).mean()
    t = dict(o=o, Q=Q, glob=glob, cos=cos)
    if a.condition == "PC768":
        features = PD.roi_tokens(model, o, b)
        memory, mcov = PD.conditioning("baseline", features, b["aux_cov"])
        flow, _ = PD.flow_loss(head, memory, mcov, b, step)
        t.update(flow=flow, memory=memory, mcov=mcov, loss=flow + cos + LAM_G * glob)
    else:
        mse = (Q - T).pow(2).mean()
        t.update(mse=mse, loss=lam * mse + LAM_G * glob)
    return t


def check_record(a, model, head, b, t, step, lam):
    if a.condition == "PC768":
        rec = loss_diag(model, dict(crop=t["cos"], flow=t["flow"], glob=t["glob"]))
        rec.update(feature_stats(t["Q"], b["tpool"], b["tpool384"]))
        with torch.no_grad():
            shuffled, _ = PD.flow_loss(head, t["memory"].detach(), t["mcov"], b, step, shuffle=True)
        rec.update(crop=float(t["cos"]), flow=float(t["flow"]), glob=float(t["glob"]), conditioning_gap_shuffled=float(shuffled - t["flow"]))
    else:
        rec = loss_diag(model, dict(mse=lam * t["mse"], cos=t["cos"], glob=t["glob"]))
        rec.update(feature_stats(t["Q"], b["tpool"]))
        rec.update(mse=float(t["mse"]), weighted_mse=float(lam * t["mse"]), cos_loss=float(t["cos"]), glob=float(t["glob"]))
    rec.update(step=step + 1, batch_sha=hashlib.sha1(b["box_c"].cpu().numpy().tobytes() + b["tpool"].cpu().numpy().tobytes() + b["full"].cpu().numpy().tobytes()).hexdigest())
    return rec


def clip(model, head):
    norms = [torch.nn.utils.clip_grad_norm_(list(model.vit.parameters()), 1.), torch.nn.utils.clip_grad_norm_([p for _, p in model.new_named()], 1.)]
    if head is not None:
        norms.append(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.))
    return norms


@torch.no_grad()
def verify_teacher(a, dev):
    """Crop768 cache vs the PixelDiT Hybrid training data: same ROI manifest files, same ROI order, same crop/letterbox rule, same layer (final-norm patch tokens), same pooling.
    Live re-encoding of sampled object AND nested ROIs at 384 (must reproduce the Hybrid teacher) and at 768 (must reproduce the cache)."""
    out = dict(manifest={})
    for name in ("image_ids.npy", "object_boxes.npy", "nested_boxes.npy", "g_cls.npy", "g_mean.npy"):
        p768, p384 = Path(SHARED768) / name, Path(SHARED384) / name
        out["manifest"][name] = dict(same_file=os.path.realpath(p768) == os.path.realpath(p384), sha_equal=sha256(p768) == sha256(p384) if name.endswith("boxes.npy") or name == "image_ids.npy" else None)
        assert out["manifest"][name]["same_file"], name
    enc0 = load_vit(dev).eval().requires_grad_(False)
    ds768, ex = PixelDS(SHARED768), eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)
    rng = np.random.default_rng(0)
    rec = dict(cos_live768_vs_cache=[], cos_live384_vs_hybrid_teacher=[], cos_768_vs_384=[], norm768=[], norm384=[], kinds=[])
    for s in sorted(rng.choice(12000, 12, replace=False).tolist()):
        item = next(iter(Slots(stream, 0, 1, ex, s)))
        b = ds768[item]
        r = 0
        for c, iid in item[1]:
            i = ds768.pos[iid]
            img = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
            boxes = [(0, ds768.obj[i][j]) for j in range(4)] + [(1, ds768.NB[s, c, j]) for j in range(4) if ds768.NB[s, c, j][2] > ds768.NB[s, c, j][0]]
            for kind, bx in boxes:
                bx = tuple(int(v) for v in bx)
                if rng.random() < .25:
                    for S, key, ref in ((S_T, "cos_live768_vs_cache", b["tpool"][r]), (384, "cos_live384_vs_hybrid_teacher", b["tpool384"][r])):
                        lb = LB(bx[2] - bx[0], bx[3] - bx[1], S, 16)
                        x = lb.canvas(img.crop(bx))[None].to(dev)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            tok = enc0.forward_features(x)[:, enc0.num_prefix_tokens:].float()[0]
                        cov = torch.from_numpy(cov48(lb) if S == S_T else lb.coverage()).to(dev)
                        live = (tok * cov[:, None]).sum(0) / cov.sum()
                        rec[key].append(float(F.cosine_similarity(live, ref.to(dev), dim=0)))
                    rec["kinds"].append(kind)
                rec["cos_768_vs_384"].append(float(F.cosine_similarity(b["tpool"][r], b["tpool384"][r], dim=0)))
                rec["norm768"].append(float(b["tpool"][r].norm()))
                rec["norm384"].append(float(b["tpool384"][r].norm()))
                r += 1
        assert r == len(b["tpool"])
    res = dict(manifest=out["manifest"], n_live_checked=len(rec["kinds"]), n_live_nested=int(sum(rec["kinds"])), min_cos_live768_vs_cache=min(rec["cos_live768_vs_cache"]), min_cos_live384_vs_hybrid_teacher=min(rec["cos_live384_vs_hybrid_teacher"]),
               n_rois=len(rec["norm768"]), cos_768_vs_384_mean=float(np.mean(rec["cos_768_vs_384"])), cos_768_vs_384_p05=float(np.percentile(rec["cos_768_vs_384"], 5)), norm768_mean=float(np.mean(rec["norm768"])), norm384_mean=float(np.mean(rec["norm384"])),
               norm768_over_384_mean=float(np.mean(np.array(rec["norm768"]) / np.array(rec["norm384"]))), layer="final-norm patch tokens (forward_features, prefix dropped), both", pooling="content-coverage-weighted mean (48x48 at 768, 24x24 at 384)", flow_target="unchanged: the 384 RGB crops of CropObjectiveDS")
    assert res["min_cos_live768_vs_cache"] > .999 and res["min_cos_live384_vs_hybrid_teacher"] > .999 and res["n_live_nested"] > 0, res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def calibrate_mse(a, dev):
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    a.condition = "M1"
    model, _, _, _ = build(a, dev)
    ex, stream = stream_items(a)
    ds = TrainDS(a.shared384)
    mods = groups_map(model)
    params = [p for v in mods.values() for p in v]
    coarse = len(mods["coarse"])
    records = []
    for s in range(a.calib_batches):
        b = RC.to_dev(ds[next(iter(Slots(stream, 0, 1, ex, s)))], dev)
        t = step_terms(a, model, None, b, s, 1.)
        gc, gm = grads(t["cos"], params), grads(t["mse"], params)
        rec = dict(batch=s, grad_cos_total=float(flat(gc).norm()), grad_mse_total=float(flat(gm).norm()), grad_cos_coarse=float(flat(gc[:coarse]).norm()), grad_mse_coarse=float(flat(gm[:coarse]).norm()), grad_cos_fine_and_interaction=float(flat(gc[coarse:]).norm()), grad_mse_fine_and_interaction=float(flat(gm[coarse:]).norm()), cos_between_grads=float(F.cosine_similarity(flat(gc), flat(gm), dim=0)), mse=float(t["mse"]), cos_loss=float(t["cos"]), **feature_stats(t["Q"], b["tpool"]))
        rec["ratio_total"] = rec["grad_cos_total"] / rec["grad_mse_total"]
        rec["ratio_coarse"] = rec["grad_cos_coarse"] / rec["grad_mse_coarse"]
        rec["ratio_fine"] = rec["grad_cos_fine_and_interaction"] / rec["grad_mse_fine_and_interaction"]
        records.append(rec)
    med = float(np.median([r["ratio_total"] for r in records]))
    lam = float(f"{med:.2g}")
    report = dict(lambda_mse=lam, median_ratio=med, rule="median ||g_cos|| / ||g_mse|| at the initial weights over 8 batches, all CoarseFine parameters, 2 significant figures; fixed for 12k steps", median_ratio_coarse=float(np.median([r["ratio_coarse"] for r in records])), median_ratio_fine=float(np.median([r["ratio_fine"] for r in records])),
                  scaled_grad_coarse_over_fine=dict(cos=float(np.median([r["grad_cos_coarse"] / r["grad_cos_fine_and_interaction"] for r in records])), mse_scaled=float(np.median([r["grad_mse_coarse"] / r["grad_mse_fine_and_interaction"] for r in records]))), median_cos_between_grads=float(np.median([r["cos_between_grads"] for r in records])), records=records, initial_sha256=sha256(a.initial))
    assert report["median_cos_between_grads"] < .999, "MSE and cosine gradients are identical"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


def lam_of(a):
    return json.loads(Path(a.lambda_mse_file).read_text())["lambda_mse"] if a.condition == "M1" else None


def config(a, model, head, lam):
    return dict(args={k: str(v) for k, v in vars(a).items()}, condition=a.condition, spec=CONDS[a.condition], lambda_mse=lam, lambda_flow=1. if a.condition == "PC768" else None, global_weight=LAM_G, steps=a.steps, world=1, images_per_step=16, warmup=WARM, full_size=384,
                teacher_dir=a.shared768 if a.condition == "PC768" else a.shared384, initial_sha256=sha256(a.initial), seed=0, encoder_params=n_params(model), head_params=n_params(head) if head is not None else 0, source_sha256=sha256(__file__), pdit_train_sha256=sha256(PD.__file__), torch=torch.__version__)


def smoke(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lam = lam_of(a)
    model, ema, head, opt = build(a, dev)
    ex, stream = stream_items(a)
    ds = dataset(a)
    sum0 = checksum(model.parameters())
    report = dict(status="running", cfg=config(a, model, head, lam), steps=[])
    for k in range(3):
        b = RC.to_dev(ds[next(iter(Slots(stream, 0, 1, ex, k)))], dev)
        for group in opt.param_groups:
            group["lr"] = RC.C.lr_at(k, group["base"], WARM, a.steps)
        t = step_terms(a, model, head, b, k, lam)
        rec = check_record(a, model, head, b, t, k, lam)
        if k == 0:
            if a.condition == "PC768":
                log0 = json.loads(open(HYBRID_LOG).readline())
                parity = dict(flow=float(t["flow"]), logged_flow=log0["method_loss"], glob=float(t["glob"]), logged_glob=log0["global_loss"], crop768=float(t["cos"]), e1a_crop768=E1A_SMOKE_CROP, crop384_same_batch=float(cosd(t["Q"], b["tpool384"]).mean()), logged_crop384=log0["crop_cos_loss"])
                assert abs(parity["flow"] - parity["logged_flow"]) < 2e-3 and abs(parity["glob"] - parity["logged_glob"]) < 2e-3 and abs(parity["crop768"] - E1A_SMOKE_CROP) < 2e-3 and abs(parity["crop384_same_batch"] - parity["logged_crop384"]) < 2e-3, parity
                assert PD.flow_encoder_grads(model, t["flow"])[0] > 0
            else:
                log0 = json.loads(open(A0_LOG).readline())
                parity = dict(cos=float(t["cos"]), logged_crop=log0["crop_cos_loss"], glob=float(t["glob"]), logged_glob=log0["global_loss"])
                assert abs(parity["cos"] - parity["logged_crop"]) < 2e-3 and abs(parity["glob"] - parity["logged_glob"]) < 2e-3, parity
                assert abs(rec["student_norm_mean"] - 1) > .1 and abs(rec["teacher_norm_mean"] - 1) > .1, "features look unit-normalised"
                assert rec["cos_mse_cos_coarse"] < .999 and rec["cos_mse_cos_fine_transformer"] < .999, "MSE and cosine give the same gradient"
            report["parity"] = parity
        report["steps"].append(rec)
        assert torch.isfinite(t["loss"])
        opt.zero_grad(set_to_none=True)
        t["loss"].backward()
        clip(model, head)
        opt.step()
        ema_update(ema, model, .999)
    report["encoder_updated"] = dict(checksum_before=sum0, checksum_after=checksum(model.parameters()), vit_changed=True)
    assert report["encoder_updated"]["checksum_before"] != report["encoder_updated"]["checksum_after"]
    path = out / "smoke_ckpt.pt"
    torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=0), path)
    load_hcf(path, dev)
    path.unlink()
    report.update(status="passed", peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    (out / "smoke.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(dict(status="passed", condition=a.condition, parity=report["parity"], lam=lam, step1={k: v for k, v in report["steps"][0].items() if "norm" in k or "cos_" in k[:4] or "mse" in k}), default=str), flush=True)


def train(a, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        raise FileExistsError(out)
    for folder in ("logs", "checkpoints"):
        (out / folder).mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lam = lam_of(a)
    model, ema, head, opt = build(a, dev)
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
        if head is not None:
            head.load_state_dict(ck["head"])
        start = ck["step"]
        print(json.dumps(dict(resumed_from=start)), flush=True)
    ds = dataset(a)
    dl = DataLoader(ds, sampler=Slots(stream, 0, 1, ex, start), batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    cfg = config(a, model, head, lam)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps(cfg, default=str), flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = last = time.monotonic()
    values = []

    def save(step):
        tmp = out / "checkpoints/ema_last.tmp"
        torch.save(dict(model=ema.state_dict(), enc=ema.vit.state_dict(), cond="P", kind="ema", s0=.1, step=step, pretraining=dict(method="pc768mse", condition=a.condition, lambda_mse=lam)), tmp)
        tmp.replace(out / "checkpoints/ema_last.pt")
        tmp = out / "checkpoints/recovery.tmp"
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), head=head.state_dict() if head is not None else None, step=step, condition=a.condition), tmp)
        tmp.replace(rec_path)

    with (out / "logs/train.jsonl").open("a") as log, (out / "logs/diagnostics.jsonl").open("a") as checks:
        for k, b in enumerate(dl):
            step = start + k
            b = RC.to_dev(b, dev)
            for group in opt.param_groups:
                group["lr"] = RC.C.lr_at(step, group["base"], WARM, total)
            t = step_terms(a, model, head, b, step, lam)
            if step in (0, 1, 2) or (step + 1) % a.check_every == 0:
                rec = check_record(a, model, head, b, t, step, lam)
                checks.write(json.dumps(rec) + "\n")
                checks.flush()
                print(json.dumps(dict(check=rec)), flush=True)
            if not bool(torch.isfinite(t["loss"])):
                raise FloatingPointError(f"loss at {step + 1}")
            opt.zero_grad(set_to_none=True)
            t["loss"].backward()
            norms = clip(model, head)
            if not all(bool(torch.isfinite(v)) for v in norms):
                raise FloatingPointError("gradient")
            opt.step()
            m = 1 - (1 - M0_EMA) * (math.cos(math.pi * step / total) + 1) / 2
            ema_update(ema, model, m)
            fs = feature_stats(t["Q"], b["tpool"], b.get("tpool384"))
            values.append([float(t["cos"]), float(t.get("flow", 0.)), float(t.get("mse", 0.)), float(t["glob"]), fs["student_norm_mean"], fs["teacher_norm_mean"], fs["norm_absdiff_mean"], fs.get("cos_teacher768_teacher384", float("nan"))] + [float(v) for v in norms])
            if step == 0 or (step + 1) % a.log_every == 0:
                now = time.monotonic()
                v = np.mean(values, axis=0)
                record = dict(step=step + 1, elapsed_s=round(now - t0, 2), crop_cos_loss=float(v[0]), flow_loss=float(v[1]), raw_mse=float(v[2]), global_loss=float(v[3]), student_norm=float(v[4]), teacher_norm=float(v[5]), norm_absdiff=float(v[6]), cos_teacher768_teacher384=float(v[7]), grad_encoder=float(v[8]), grad_new=float(v[9]), grad_head=float(v[10]) if len(v) > 10 else None, step_seconds=(now - last) / len(values), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
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
    result = dict(status="done", condition=a.condition, steps=total, time_s=round(time.monotonic() - t0), started_from_step=start, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, head_removed_for_probe=a.condition == "PC768", ema_checksum=checksum(ema.parameters()))
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
    ap.add_argument("--shared384", default=SHARED384)
    ap.add_argument("--shared768", default=SHARED768)
    ap.add_argument("--lambda-mse-file", default="")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--check-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-teacher", action="store_true")
    ap.add_argument("--calibrate-mse", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    try:
        if a.verify_teacher:
            verify_teacher(a, dev)
        elif a.calibrate_mse:
            calibrate_mse(a, dev)
        elif a.smoke:
            smoke(a, dev)
        else:
            raise SystemExit(train(a, dev))
    except Exception:
        if a.condition and not (a.verify_teacher or a.calibrate_mse):
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / "failed.json").write_text(json.dumps(dict(status="failed", error=traceback.format_exc()), indent=1))
        raise


if __name__ == "__main__":
    main()
