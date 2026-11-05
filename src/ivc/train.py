"""IVC (interpolation vs C0) probe job = copy of src/readout/train.py (11-05 MSD probe protocol) with three additions:
  * job key `interp` (mode dec, decoder bilinear): 'bilinear' (= the MSD probe_bil call) or 'bicubic' (same grid, mode bicubic)
  * reader kind 'r0' (src.ivc.common.R0); its per-level z-score stats come from THIS condition's train features
    (4,000 train images, rng 0, no flip; = paco_region_reader mean_stats) computed before training (no RNG side effects
    on the data order / flips / R2 init)
  * every epoch's reader state is kept (<out>/<reader>/ep<k>.pt) so the dev-selected epoch has a checkpoint
Original docstring follows.

One queue job: train every reader listed in a job YAML on ONE shared frozen-encoder forward per step (GPU 2+3 DDP).

Fixed protocol (identical for every reader of every job = equal comparison budget):
  PACO train minus data_check exclusions; image packs of ~128 ROIs (C.full_epoch_batches, data seed 0), 10 epochs,
  image hflip p=0.5, masked BCE / #known labels in the global step, AdamW (reader lr 3e-4, head lr 1e-3, wd 0.05),
  1 epoch warmup + cosine, grad clip 1.0 per reader, reader init torch.manual_seed(7) (= 1000*seed+7, seed 0).
Every epoch: PACO VAL scores of every reader -> loss / M0 / M2 (curves.jsonl, scores_val_ep*.npy). Test is never read.
Primary number = val M2 Overall at the LAST epoch (fixed budget, no selection on val); best-val-epoch is logged too.
Final epoch extras: train-subset fit (2,000 fixed images, no flip), per-reader direct-check arrays (extra_val.npz),
full-train scores for readers listed in `score_train` (teachers for E12).

  torchrun --nproc_per_node 2 -m src.readout.train --job configs/readout_queue/jobs/ctrl_f336.yaml --out <dir>
"""
import argparse
import json
import os
import time
import traceback

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.readout.common import C, LEVELS
from src.readout.models import Reader, n_params, param_groups
from src.ivc.common import R0, sample_roi


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def build_readers(job, dev):
    readers = {}
    for spec in job["readers"]:
        torch.manual_seed(1000 * job.get("seed", 0) + 7)
        kw = {k: spec[k] for k in ("T", "K") if k in spec}
        layers = tuple(spec.get("layers", RC.ML_LAYERS))
        readers[spec["name"]] = (R0() if spec["kind"] == "r0" else Reader(spec["kind"], layers=layers, **kw)).to(dev)
    return readers


def bind_loss(logits, Y, lv, group):
    """E05: within-image ROI binding. Triplets (i, j, a): same image, same level, i != j, Y[i,a] = 1, Y[j,a] = 0 (explicit)
    -> softplus(s_j,a - s_i,a). Returns (sum, count)."""
    same = (group[:, None] == group[None, :]) & (lv[:, None] == lv[None, :])
    same.fill_diagonal_(False)
    pos, neg = (Y == 1), (Y == 0)
    m = (same[:, :, None] & pos[:, None, :] & neg[None, :, :]).float()          # (i, j, a)
    d = logits[None, :, :] - logits[:, None, :]                                # d[i,j,a] = s_j,a - s_i,a
    return (F_softplus(d) * m).sum(), m.sum()                                  # (no boolean indexing -> no host sync)


@torch.no_grad()
def r0_stats(readers, dtr, Ytr, rank, world, dev, encode, prep, mode, msd, a):
    """Per-level mean / std (unbiased, + 1e-6) of the coverage-weighted crop-grid ROI mean of THIS condition on 4,000
    train images (rng 0, no flip) = paco_region_reader common_rr.mean_stats. Images split over ranks, sums all-reduced
    in float64. Uses its own torch.Generator-free DataLoader with num_workers=0 -> no draw from the global RNG."""
    groups = C.image_groups(dtr)
    idx = np.sort(np.random.default_rng(0).choice(len(groups), min(4000, len(groups)), replace=False))
    if mode == "dec":
        ds = msd["MS"].MSDImageDS(dtr, Ytr, flip=False, want_crops=False)
    else:
        ds = RC.PacoImageDS(dtr, Ytr, flip=False, mode=mode)
    chunks = [list(map(int, idx[i:i + 16])) for i in range(0, len(idx), 16)][rank::world]
    acc = torch.zeros(2, 2, 768, dtype=torch.float64, device=dev)
    cnt = torch.zeros(2, dtype=torch.float64, device=dev)
    for ch in chunks:
        b = prep(ds[ch])
        f = encode(b)
        m = R0.pooled(f[12][b["img_index"]], b["cov"]).double()
        for k in (0, 1):
            sel = b["lv"] == k
            acc[k, 0] += m[sel].sum(0)
            acc[k, 1] += m[sel].pow(2).sum(0)
            cnt[k] += sel.sum()
    RC.all_sum(acc)
    RC.all_sum(cnt)
    mu = acc[:, 0] / cnt[:, None]
    var = (acc[:, 1] - cnt[:, None] * mu.pow(2)) / (cnt[:, None] - 1)
    sd = var.clamp_min(0).sqrt() + 1e-6
    for r in readers.values():
        if r.kind == "r0":
            r.set_stats(mu.float(), sd.float())
    if rank == 0:
        torch.save(dict(mu=mu.float().cpu(), sd=sd.float().cpu(), n=cnt.cpu(), n_images=len(idx)), os.path.join(a.out, "r0_stats.pt"))
        print(json.dumps(dict(r0_stats=dict(n_roi=cnt.tolist(), n_images=len(idx), mu_norm=mu.norm(dim=1).tolist(), sd_mean=sd.mean(1).tolist()))), flush=True)


def F_softplus(x):
    return torch.nn.functional.softplus(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images per split")
    ap.add_argument("--epochs", type=int, default=0, help="override (smoke test only)")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: steps per epoch")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache", default="", help="uint8 canvas cache (src.readout.fastdata); empty = on-the-fly PIL")
    a = ap.parse_args()
    job = yaml.safe_load(open(a.job).read().replace("__RUN__", os.environ.get("READOUT_RUN", "__RUN__")))
    rank, world = RC.ddp_setup()
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    try:
        run(job, a, rank, world, dev)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if rank == 0:
            json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


def run(job, a, rank, world, dev):
    epochs = a.epochs or job.get("epochs", 10)
    mode = job.get("mode", "full")
    lr, head_lr, wd = job.get("lr", 3e-4), job.get("head_lr", 1e-3), job.get("wd", 0.05)
    data = RC.load_data(a.exclude or None, a.limit)
    Ytr_by = RC.ytr_by_level(data)          # unit inclusion from the FULL train labels (same for every probe)
    dtr, Ytr = data["train"]
    dva, Yva = data["val"]
    ddev = Ydev = None
    if job.get("paco_dev_images"):          # fixed PACO dev = train images held out for R2 epoch selection (rng 0)
        ids = np.array(sorted(dtr.image_id.unique()))
        dev_ids = set(np.random.default_rng(0).choice(ids, int(job["paco_dev_images"]), replace=False).tolist())
        m = dtr.image_id.isin(dev_ids).values
        ddev, Ydev = dtr[m].reset_index(drop=True), Ytr[m]
        dtr, Ytr = dtr[~m].reset_index(drop=True), Ytr[~m]
    readers = build_readers(job, dev)
    specs = {s["name"]: s for s in job["readers"]}
    opts = {n: torch.optim.AdamW(param_groups(r, lr, head_lr, wd), lr=lr, betas=(0.9, 0.999)) for n, r in readers.items()}
    need_layers = sorted({l for r in readers.values() for l in r.fuse.layers})
    vit = RC.load_vit().to(dev)
    msd = None
    if mode == "dec":                        # MSD probe: student encoder (+ adapters) and ROI decoder, FROZEN
        import src.msd.common as MS
        from models.roi_reader import bilinear as roi_bilinear
        student = MS.Student(adapters=False).to(dev)
        decoder = None
        if job.get("decoder", "bilinear") != "bilinear":
            ck = torch.load(job["decoder"], map_location="cpu")
            if ck["adapters"] is not None:
                student = MS.Student(adapters=True).to(dev)
                student.adapters.load_state_dict(ck["adapters"])
            decoder = MS.load_decoder(ck, dev)
        student.eval()
        frozen = [p for p in student.parameters()] + ([p for p in decoder.parameters()] if decoder is not None else [])
        for p in frozen:
            p.requires_grad_(False)
        sum0 = float(sum(p.double().sum() for p in frozen))
        msd = dict(student=student, decoder=decoder, frozen=frozen, sum0=sum0, bil=roi_bilinear, MS=MS)
    # E12 teachers: train-row logits (manifest order of THIS train set)
    teachers = {}
    for n, s in specs.items():
        if "kd" in s:
            t = np.load(s["kd"]["teacher"]).astype(np.float32)
            assert t.shape == (len(dtr), 59), (t.shape, len(dtr))
            teachers[n] = torch.from_numpy(t).to(dev)
    if mode == "dec":
        tr_ds = msd["MS"].MSDImageDS(dtr, Ytr, flip=True, want_crops=False)
        sampler = RC.RankStepSampler(tr_ds.n_roi, rank, world, seed=job.get("data_seed", 0))
        prep = lambda b: msd["MS"].to_dev(b, dev)  # noqa: E731
    elif a.cache:
        import src.readout.fastdata as FD
        tr_ds = FD.CachedDS(dtr, Ytr, a.cache, "train", flip=True, mode=mode)
        sampler = FD.GlobalStepSampler(tr_ds.n_roi, rank, world, Ytr=Ytr, groups=tr_ds.groups, seed=job.get("data_seed", 0))
        prep = lambda b: FD.gpu_prepare(b, dev, mode)  # noqa: E731
    else:
        tr_ds = RC.PacoImageDS(dtr, Ytr, flip=True, mode=mode)
        sampler = RC.RankStepSampler(tr_ds.n_roi, rank, world, seed=job.get("data_seed", 0))
        prep = lambda b: RC.to_dev(b, dev)  # noqa: E731
    torch.manual_seed(job.get("seed", 0) * 100 + rank)
    dl = DataLoader(tr_ds, batch_sampler=None, sampler=sampler, batch_size=1, num_workers=a.workers, pin_memory=True,
                    collate_fn=RC.collate_one, persistent_workers=True, prefetch_factor=4)
    steps_per_ep = [len(C.full_epoch_batches(tr_ds.n_roi, e, job.get("data_seed", 0))) for e in range(1, epochs + 1)]
    if a.max_steps:
        steps_per_ep = [min(s, a.max_steps) for s in steps_per_ep]
    total, warm = sum(steps_per_ep), steps_per_ep[0]
    cfg = dict(job=job, args=vars(a), world=world, n_train_rows=len(dtr), n_train_images=int(dtr.image_id.nunique()),
               n_val_rows=len(dva), n_val_images=int(dva.image_id.nunique()), steps_per_epoch=steps_per_ep, total_steps=total,
               n_params={n: n_params(r) for n, r in readers.items()},
               n_params_reader_only={n: n_params(r) - n_params(r.heads) for n, r in readers.items()},
               encoder=C.TIMM_B, layers=need_layers, torch=torch.__version__)
    if rank == 0:
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
        for n in readers:
            os.makedirs(os.path.join(a.out, n), exist_ok=True)
    log(rank, json.dumps(cfg, default=str))

    def encode(b):
        if mode == "dec":
            with torch.no_grad():
                feats, _ = msd["student"](b["full"])
                if msd["decoder"] is None:
                    Z = sample_roi(feats[12], b["geom"], job.get("interp", "bilinear"))
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        Z = msd["decoder"](feats, b["geom"])
            b["geo"], b["box4"], b["cov"] = b["geo_c"], b["box4_c"], b["cov_c"]
            b["img_index"] = torch.arange(len(Z), device=Z.device)
            b["side"], b["x"] = 336, b["full"]
            return {12: Z.float().detach()}
        x = RC.TU.upsample(b["x"], "f672") if mode == "f672" else b["x"]
        if len(x) > 32:                                   # c0: many crops per step
            parts = [RC.encode_layers(vit, x[i:i + 32], need_layers) for i in range(0, len(x), 32)]
            return {l: torch.cat([p[l] for p in parts]) for l in need_layers}
        return RC.encode_layers(vit, x, need_layers)

    @torch.no_grad()
    def score(df, Y, final=False, ctx_swap=False, rows_subset=None, split="val"):
        """-> {reader: (len(df),59)} logits (+ extras) for every row; images split over ranks, summed back."""
        if mode == "dec":
            ds = msd["MS"].MSDImageDS(df, Y, flip=False, want_crops=False)
        else:
            ds = FD.CachedDS(df, Y, a.cache, split, flip=False, mode=mode) if a.cache else RC.PacoImageDS(df, Y, flip=False, mode=mode)
        n_img = len(ds)
        idx = np.arange(n_img) if rows_subset is None else rows_subset
        es = RC.EvalSampler(len(idx), {"full": 24, "f672": 8, "c0": 6, "dec": 12}[mode], rank, world)
        chunks = [[int(idx[i]) for i in ch] for ch in es]
        dle = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
        Sc = {n: torch.zeros(len(df), 59, device=dev) for n in readers}
        ext = {}
        for r in readers.values():
            r.eval()
        for b in dle:
            b = prep(b)
            f = encode(b)
            rows = b["rows"]
            for n, r in readers.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = r(f, b)
                Sc[n][rows] = o["logits"].float()
                if final:
                    for k, v in o["extra"].items():
                        key = f"{n}/{k}"
                        if key not in ext:
                            ext[key] = torch.zeros((len(df),) + tuple(v.shape[1:]), device=dev)
                        ext[key][rows] = v.float()
                    if ctx_swap and r.kind in ("gate", "concat", "ctx"):
                        for cm in ("remove", "swap"):
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                o2 = r(f, b, ctx_mode=cm)
                            for k2, v2 in [("logits", o2["logits"])] + list(o2["extra"].items()):
                                key = f"{n}/ctx_{cm}_{k2}"
                                if key not in ext:
                                    ext[key] = torch.zeros((len(df),) + tuple(v2.shape[1:]), device=dev)
                                ext[key][rows] = v2.float()
        for r in readers.values():
            r.train()
        Sc = {n: RC.all_sum(v).cpu().numpy() for n, v in Sc.items()}
        ext = {k: RC.all_sum(v).cpu().numpy() for k, v in ext.items()}
        return Sc, ext

    if any(r.kind == "r0" for r in readers.values()):
        r0_stats(readers, dtr, Ytr, rank, world, dev, encode, prep, mode, msd, a)
    best = {n: -1.0 for n in readers}
    step, t0 = 0, time.time()

    def evaluate(ep, run_stats, final=False):
        Sv, ext = score(dva, Yva, final=final, ctx_swap=final)
        Sd = score(ddev, Ydev, split="train")[0] if ddev is not None else None
        if rank != 0:
            return
        if Sd is not None:
            for n in readers:
                rd_ = RC.m2_eval(Sd[n], ddev, Ydev, Ytr_by)
                with open(os.path.join(a.out, n, "curves_dev.jsonl"), "a") as fh:
                    fh.write(json.dumps(dict(epoch=ep, dev=dict(loss=rd_["loss"], M0=rd_["M0"], M2=rd_["M2"]))) + "\n")
                np.save(os.path.join(a.out, n, f"scores_dev_ep{ep}.npy"), Sd[n])
        line = {}
        for n in readers:
            r = RC.m2_eval(Sv[n], dva, Yva, Ytr_by)
            if not np.isfinite(r["loss"]):
                raise FloatingPointError(f"{n}: non-finite val loss at epoch {ep}")
            rec = dict(epoch=ep, step=step, elapsed_s=round(time.time() - t0, 1), **run_stats.get(n, {}),
                       val=dict(loss=r["loss"], object_loss=r["object_loss"], part_loss=r["part_loss"], M0=r["M0"], M2=r["M2"]))
            with open(os.path.join(a.out, n, "curves.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            np.save(os.path.join(a.out, n, f"scores_val_ep{ep}.npy"), Sv[n])
            if r["M2"]["overall"] > best[n]:
                best[n] = r["M2"]["overall"]
                torch.save(dict(reader=readers[n].state_dict(), epoch=ep, val_M2=r["M2"]), os.path.join(a.out, n, "best_val.pt"))
            line[n] = dict(M2=round(r["M2"]["overall"], 2), loss=round(r["loss"], 4))
        if final:
            np.savez(os.path.join(a.out, "extra_val.npz"), **{k.replace("/", "__"): v for k, v in ext.items()})
        print(json.dumps(dict(epoch=ep, val=line)), flush=True)

    evaluate(0, {})
    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        run_l = {n: torch.zeros((), device=dev) for n in readers}
        run_aux = {n: torch.zeros((), device=dev) for n in readers}
        bad = torch.zeros((), device=dev)
        run_n = 0.0
        all_params = [p for r in readers.values() for p in r.parameters()]
        for k, b in enumerate(dl):
            if a.max_steps and k >= a.max_steps:
                break
            b = prep(b)
            f = encode(b)
            lv, Y = b["lv"], b["Y"]
            if "nt_global" in b:      # global counts from the sampler (CPU) -> no all-reduce, no host sync
                nt, nroi = b["nt_global"].float(), b["nroi_global"].float()
            else:
                nt = RC.all_sum(torch.tensor(float((Y >= 0).sum()), device=dev))
                nroi = RC.all_sum(torch.tensor(float(len(Y)), device=dev))
            for n, r in readers.items():          # every reader: forward + backward (grads kept)
                s, opt = specs[n], opts[n]
                for g in opt.param_groups:
                    g["lr"] = C.lr_at(step, g["base"], warm, total)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = r(f, b)
                logits = o["logits"].float()
                ls, _ = C.masked_bce_sum(logits, Y)
                loss = ls / nt
                aux_val = torch.zeros((), device=dev)
                if "bind" in s:
                    bs, bc = bind_loss(logits, Y, lv, b["group"])
                    bc = RC.all_sum(bc.detach().clone())
                    aux_val = bs / bc.clamp(min=1)
                    loss = loss + s["bind"] * aux_val
                if n in teachers:
                    tl = teachers[n][b["rows"]]
                    kd = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.sigmoid(tl), reduction="sum")
                    aux_val = kd / (nroi * 59)
                    loss = loss + s["kd"]["lam"] * aux_val
                bad = bad + (~torch.isfinite(loss.detach())).float()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                run_l[n] += ls.detach()
                run_aux[n] += aux_val.detach()
            RC.allreduce_grads(all_params)             # ONE fused all-reduce for all readers (same sum as per reader)
            for n, r in readers.items():
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
                opts[n].step()
            run_n += (b["nt_cpu"] if "nt_cpu" in b else float(nt)) / world
            step += 1
            if step % 100 == 0:
                if float(bad) > 0:
                    raise FloatingPointError(f"non-finite train loss before step {step}")
                log(rank, json.dumps(dict(step=step, epoch=ep, s=round(time.time() - t0),
                                          loss_run={n: round(float(run_l[n]) / max(run_n, 1), 4) for n in readers})))   # rank-0 estimate
        if float(bad) > 0:
            raise FloatingPointError(f"non-finite train loss in epoch {ep}")
        run_l = {n: float(v) for n, v in run_l.items()}
        run_aux = {n: float(v) for n, v in run_aux.items()}
        tot_l = {n: float(RC.all_sum(torch.tensor(run_l[n], device=dev))) for n in readers}
        tot_n = float(RC.all_sum(torch.tensor(run_n, device=dev)))
        stats = {n: dict(train_running_loss=tot_l[n] / max(tot_n, 1), train_running_aux=run_aux[n] / max(k + 1, 1)) for n in readers}
        evaluate(ep, stats, final=(ep == epochs))
        if rank == 0:
            torch.save(dict(readers={n: r.state_dict() for n, r in readers.items()}, epoch=ep), os.path.join(a.out, "last.pt"))
            for n, r in readers.items():
                torch.save(dict(reader=r.state_dict(), epoch=ep), os.path.join(a.out, n, f"ep{ep}.pt"))
    # ---------------------------------------------------------------- final: train fit + teacher scores
    rng = np.random.default_rng(0)
    n_img_tr = len(C.image_groups(dtr))
    sub = np.sort(rng.choice(n_img_tr, min(2000, n_img_tr), replace=False))
    St, _ = score(dtr, Ytr, rows_subset=sub, split="train")
    if rank == 0:
        groups = C.image_groups(dtr)
        rows = np.concatenate([groups[i] for i in sub])
        fit = {}
        for n in readers:
            dsub, Ysub = dtr.iloc[rows].reset_index(drop=True), Ytr[rows]
            r = RC.m2_eval(St[n][rows], dsub, Ysub, Ytr_by)
            fit[n] = dict(train_subset_loss=r["loss"], train_subset_M2=r["M2"], n_images=len(sub), n_rows=len(rows))
        json.dump(fit, open(os.path.join(a.out, "train_fit.json"), "w"), indent=1)
    want = job.get("score_train", [])
    if want:
        St, _ = score(dtr, Ytr, split="train")
        if rank == 0:
            for n in want:
                np.save(os.path.join(a.out, n, "scores_train_final.npy"), St[n].astype(np.float16))
    if msd is not None and rank == 0:     # encoder / decoder must be bit-identical after probe training
        sum1 = float(sum(p.double().sum() for p in msd["frozen"]))
        json.dump(dict(requires_grad_any=any(p.requires_grad for p in msd["frozen"]), grads_present=any(p.grad is not None for p in msd["frozen"]),
                       checksum_before=msd["sum0"], checksum_after=sum1, unchanged=sum1 == msd["sum0"], decoder=job.get("decoder", "bilinear")),
                  open(os.path.join(a.out, "frozen_check.json"), "w"), indent=1)
        assert sum1 == msd["sum0"], "frozen encoder / decoder changed during probe training"
    if rank == 0:
        json.dump(dict(status="done", epochs=epochs, steps=step, time_s=round(time.time() - t0), time=time.strftime("%F %T")),
                  open(os.path.join(a.out, "done.json"), "w"), indent=1)
    log(rank, "DONE")


if __name__ == "__main__":
    main()
