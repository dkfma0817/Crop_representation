"""Frozen SigLIP2 P/M/B0/C0 independent R2 training. Existing PACO split, ROI geometry, masked BCE, optimizer and epoch budget. No DDP at world1."""
import argparse
import json
import os
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.readout.common import C
from src.readout.models import Reader, n_params as reader_params, param_groups
from src.siglip_pm.model import MultiLayer, checksum, load_hcf, load_vit
from src.siglip_pm.bbdata import BBImageDS

S, P, G = 384, 16, 24


def load_frozen(job, dev):
    if job["cond"] in ("M", "P"):
        m, ck = load_hcf(job["ckpt"], dev)
        assert ck["cond"] == job["cond"] and ck["kind"] == "ema"
    else:
        m = MultiLayer(load_vit(dev, job.get("ckpt")), 0.1).to(dev).eval()     # L4 / L8 / L12 extraction only; its fuse is never called
        for p in m.parameters():
            p.requires_grad_(False)
    m.roi_mode = job["cond"]
    return m


def roi_align(tok, mult, b):
    g = G * mult
    maps = tok.transpose(1, 2).reshape(tok.shape[0], -1, g, g)[b["img_index"]]
    return F.grid_sample(maps.float(), b["bil_grid"].view(-1, G, G, 2), mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)


@torch.no_grad()
def encode(model, b, keys, drop_fine=False):
    x, out = b["x"], {}
    if getattr(model, 'roi_mode', '') == 'C0':
        toks=[]
        for i in range(0,len(x),64):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                toks.append(model.forward_features(x[i:i+64]).float())
        return {12: torch.cat(toks)}
    parts = []
    for i in range(0, len(x), 64):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            parts.append(model.forward_all(x[i:i + 64]))
    cat = lambda f: torch.cat([f(p).float() for p in parts])  # noqa: E731
    A = roi_align(cat(lambda p: p["coarse"]), 1, b)
    f = {12: A}
    if "Q" in keys:
        if drop_fine:
            f["Q"] = A
        else:
            mem = {k: roi_align(cat(lambda p: p["mem"][k]), parts[0]["grids"][k], b) for k in parts[0]["mem"]}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f["Q"] = model.fuse(A, mem).float()
    for l in (4, 8):
        if l in keys:
            f[l] = roi_align(cat(lambda p: p["raw"][l]), 1, b)
    b["img_index"] = torch.arange(len(A), device=A.device)
    return f


def build_readers(job, dev):
    out = {}
    for s in job["readers"]:
        torch.manual_seed(job.get("seed", 0))
        out[s["name"]] = Reader("r2", layers=tuple(s["layers"])).to(dev)
    return out


def run(job, a, rank, world, dev):
    epochs = a.epochs or job.get("epochs", 10)
    lr, head_lr, wd = 3e-4, 1e-3, 0.05
    data = RC.load_data(a.exclude or None, a.limit, splits=("train",))
    dtr, Ytr = data["train"]
    ids = np.array(sorted(dtr.image_id.unique()))
    nd = min(int(job.get("paco_dev_images", 1000)), max(1, len(ids) // 10)) if a.limit else int(job.get("paco_dev_images", 1000))
    dev_ids = set(np.random.default_rng(0).choice(ids, nd, replace=False).tolist())
    keep = ~dtr.image_id.isin(dev_ids).values
    dtr, Ytr = dtr[keep].reset_index(drop=True), Ytr[keep]
    readers = build_readers(job, dev)
    opts = {n: torch.optim.AdamW(param_groups(r, lr, head_lr, wd), lr=lr, betas=(0.9, 0.999)) for n, r in readers.items()}
    keys = sorted({l for r in readers.values() for l in r.fuse.layers}, key=str)
    model = load_frozen(job, dev)
    frozen = list(model.parameters())
    assert not any(p.requires_grad for p in frozen)
    sum0 = checksum(frozen)
    ds = BBImageDS(dtr, Ytr, True, "c0" if job["cond"] == "C0" else "bil", S, P)
    sampler = RC.RankStepSampler(ds.n_roi, rank, world, seed=job.get("data_seed", 0))
    torch.manual_seed(job.get("seed", 0) * 100 + rank)
    dl = DataLoader(ds, batch_sampler=None, sampler=sampler, batch_size=1, num_workers=a.workers, pin_memory=True, collate_fn=RC.collate_one, persistent_workers=True, prefetch_factor=4)
    sp = [len(C.full_epoch_batches(ds.n_roi, e, job.get("data_seed", 0))) for e in range(1, epochs + 1)]
    if a.max_steps:
        sp = [min(s, a.max_steps) for s in sp]
    total, warm = sum(sp), sp[0]
    cfg = dict(job=job, args=vars(a), world=world, n_train_rows=len(dtr), n_train_images=int(dtr.image_id.nunique()), steps_per_epoch=sp, total_steps=total,
               reader_params={n: reader_params(r) for n, r in readers.items()}, keys=[str(k) for k in keys], roialign_grid=G, encoder_checksum=sum0, time=time.strftime("%F %T"))
    if rank == 0:
        for n in readers:
            os.makedirs(f"{a.out}/{n}", exist_ok=True)
        json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=1, default=str)
        print(json.dumps(cfg, default=str), flush=True)
    step, t0 = 0, time.time()
    params = [p for r in readers.values() for p in r.parameters()]
    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        run_l = {n: torch.zeros((), device=dev) for n in readers}
        run_n = 0.0
        for k, b in enumerate(dl):
            if a.max_steps and k >= a.max_steps:
                break
            b = RC.to_dev(b, dev)
            f = encode(model, b, keys)
            Y, lv = b["Y"], b["lv"]
            nt = RC.all_sum(torch.tensor(float((Y >= 0).sum()), device=dev))
            for n, r in readers.items():
                for g in opts[n].param_groups:
                    g["lr"] = C.lr_at(step, g["base"], warm, total)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = r(f, b)
                ls, _ = C.masked_bce_sum(o["logits"].float(), Y)
                loss = ls / nt
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError(f"{n} non-finite loss at step {step}")
                opts[n].zero_grad(set_to_none=True)
                loss.backward()
                run_l[n] += ls.detach()
            RC.allreduce_grads(params)
            for n, r in readers.items():
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
                opts[n].step()
            run_n += float(nt) / world
            step += 1
            if step % 100 == 0 and rank == 0:
                print(json.dumps(dict(step=step, epoch=ep, s=round(time.time() - t0), loss={n: round(float(v) / max(run_n, 1), 4) for n, v in run_l.items()}, mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))), flush=True)
        if rank == 0:
            for n, r in readers.items():
                torch.save(dict(reader=r.state_dict(), epoch=ep, layers=[str(l) for l in r.fuse.layers]), f"{a.out}/{n}/ep{ep}.pt")
            print(json.dumps(dict(epoch=ep, s=round(time.time() - t0))), flush=True)
    if rank == 0:
        sum1 = checksum(frozen)
        json.dump(dict(checksum_before=sum0, checksum_after=sum1, unchanged=sum1 == sum0, requires_grad_any=any(p.requires_grad for p in frozen)), open(f"{a.out}/frozen_check.json", "w"), indent=1)
        assert sum1 == sum0, "frozen encoder changed"
        json.dump(dict(status="done", epochs=epochs, steps=step, time_s=round(time.time() - t0), time=time.strftime("%F %T")), open(f"{a.out}/done.json", "w"), indent=1)
        print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    job = yaml.safe_load(open(a.job))
    assert "RANK" not in os.environ, "This experiment is single-GPU only"
    rank, world = RC.ddp_setup()
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    try:
        run(job, a, rank, world, dev)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if rank == 0:
            json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(f"{a.out}/failed.json", "w"), indent=1)
        raise


if __name__ == "__main__":
    main()
