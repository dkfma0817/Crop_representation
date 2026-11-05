"""Step-time breakdown of the readout training loop (no eval, no checkpoint): data wait / H2D / encoder / readers fwd+bwd /
gradient all-reduce / optimizer, with cuda.synchronize between phases (--sync) or plain throughput (no --sync).

  torchrun --nproc_per_node 2 -m src.readout.bench --job configs/readout_queue/jobs/E11.yaml --steps 150
"""
import argparse
import json
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.readout.common import C
from src.readout.models import param_groups
from src.readout.train import bind_loss, build_readers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--exclude", default="output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warm", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--cache", default="", help="canvas cache dir (fast pipeline)")
    a = ap.parse_args()
    job = yaml.safe_load(open(a.job).read().replace("__RUN__", "output/2026-11-04/readout_queue"))
    for s in job["readers"]:
        s.pop("kd", None)
    rank, world = RC.ddp_setup()
    dev = torch.device("cuda")
    mode = job.get("mode", "full")
    data = RC.load_data(a.exclude)
    dtr, Ytr = data["train"]
    readers = build_readers(job, dev)
    specs = {s["name"]: s for s in job["readers"]}
    opts = {n: torch.optim.AdamW(param_groups(r, 3e-4, 1e-3, 0.05), lr=3e-4) for n, r in readers.items()}
    layers = sorted({l for r in readers.values() for l in r.fuse.layers})
    vit = RC.load_vit().to(dev)
    if a.cache:
        import src.readout.fastdata as FD
        tr_ds = FD.CachedDS(dtr, Ytr, a.cache, "train", flip=True, mode=mode)
        sampler = FD.GlobalStepSampler(tr_ds.n_roi, rank, world, Ytr=Ytr, groups=tr_ds.groups)
        prep = lambda b: FD.gpu_prepare(b, dev, mode)  # noqa: E731
    else:
        tr_ds = RC.PacoImageDS(dtr, Ytr, flip=True, mode=mode)
        sampler = RC.RankStepSampler(tr_ds.n_roi, rank, world)
        prep = lambda b: RC.to_dev(b, dev)  # noqa: E731
    sampler.set_epoch(1)
    dl = DataLoader(tr_ds, sampler=sampler, batch_size=1, num_workers=a.workers, pin_memory=True, collate_fn=RC.collate_one,
                    persistent_workers=True, prefetch_factor=4)
    T = {k: 0.0 for k in ("data", "h2d", "enc", "readers", "comm", "opt")}
    sync = (lambda: torch.cuda.synchronize()) if a.sync else (lambda: None)  # noqa: E731
    it = iter(dl)
    t_start = None
    n_img = n_roi = 0
    for step in range(a.warm + a.steps):
        if step == a.warm:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            T = {k: 0.0 for k in T}
        t = time.perf_counter()
        b = next(it)
        t1 = time.perf_counter()
        b = prep(b)
        sync()
        t2 = time.perf_counter()
        x = RC.TU.upsample(b["x"], "f672") if mode == "f672" else b["x"]
        if len(x) > 32:
            parts = [RC.encode_layers(vit, x[i:i + 32], layers) for i in range(0, len(x), 32)]
            f = {l: torch.cat([p_[l] for p_ in parts]) for l in layers}
        else:
            f = RC.encode_layers(vit, x, layers)
        sync()
        t3 = time.perf_counter()
        Y, lv = b["Y"], b["lv"]
        nt = b["nt_global"] if "nt_global" in b else RC.all_sum(torch.tensor(float((Y >= 0).sum()), device=dev))
        for n, r in readers.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = r(f, b)
            ls, _ = C.masked_bce_sum(o["logits"].float(), Y)
            loss = ls / nt
            if "bind" in specs[n]:
                bs, bc = bind_loss(o["logits"].float(), Y, lv, b["group"])
                loss = loss + 0.5 * bs / RC.all_sum(bc.detach().clone()).clamp(min=1)
            opts[n].zero_grad(set_to_none=True)
            loss.backward()
        sync()
        t4 = time.perf_counter()
        if a.cache:
            RC.allreduce_grads([p for r in readers.values() for p in r.parameters()])
        else:
            for r in readers.values():
                RC.allreduce_grads(list(r.parameters()))
        sync()
        t5 = time.perf_counter()
        for n, r in readers.items():
            torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            opts[n].step()
        sync()
        t6 = time.perf_counter()
        if step >= a.warm:
            for k, d in zip(T, (t1 - t, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)):
                T[k] += d
            n_img += int(b["x"].shape[0]) if mode != "c0" else int(b["group"].max()) + 1
            n_roi += len(Y)
    torch.cuda.synchronize()
    el = time.perf_counter() - t_start
    if rank == 0:
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        print(json.dumps(dict(job=job["name"], world=world, cache=bool(a.cache), sync=a.sync, steps_per_s=round(a.steps / el, 2),
                              ms_per_step=round(1e3 * el / a.steps, 1), ms={k: round(1e3 * v / a.steps, 1) for k, v in T.items()},
                              rois_per_rank_step=round(n_roi / a.steps, 1), max_mem_gb=round(mem, 1))), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
