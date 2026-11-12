"""Frozen BRIXEL comparison: unchanged project R2/loss/sampler and PACO official evaluator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.readout.models import Reader, param_groups
from src.brixel_eval.backbone import FrozenBackbone
from src.brixel_eval.data import PacoDS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output/2026-11-07/paco_oracle_attr/data"
EXCL = ROOT / "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
EPY = "/home/irteam/envs/paco_eval/bin/python"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def ds(df, labels, flip, condition):
    return PacoDS(df, labels, flip=flip, mode=condition)


def roi_chunks(features, b, size):
    """Unchanged R2: discard only keys already masked to -inf; pad within ROI chunks."""
    n = len(b["rows"])
    for start in range(0, n, size):
        stop = min(start + size, n)
        bb = {k: (v[start:stop] if torch.is_tensor(v) and k in
                  ("geo", "cov", "box4", "img_index", "rows", "lv", "Y", "group") else v)
              for k, v in b.items() if k != "x"}
        mask = bb["cov"] > 0
        width = int(mask.sum(1).max())
        assert width > 0 and mask.any(1).all()
        tokens = features[12]
        indices = torch.arange(mask.shape[1], device=mask.device).expand(len(mask), -1)
        indices = indices.masked_fill(~mask, mask.shape[1]).sort(1).values[:, :width]
        valid = indices < mask.shape[1]
        indices = indices.clamp_max(mask.shape[1] - 1)
        local = torch.arange(len(mask), device=mask.device)[:, None]
        tok = tokens[bb["img_index"][:, None], indices]
        bb["geo"] = bb["geo"][local, indices]
        bb["cov"] = bb["cov"][local, indices] * valid
        bb["img_index"] = torch.arange(len(mask), device=mask.device)
        yield {12: tok}, bb


@torch.no_grad()
def encode(model, b, chunk):
    maps = []
    for start in range(0, len(b["x"]), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            maps.append(model.features(b["x"][start:start + chunk]).float())
    f = torch.cat(maps)
    assert f.shape[1] == 768 and f.shape[2] == f.shape[3]
    assert f.shape[2] ** 2 == b["cov"].shape[1], (f.shape, b["cov"].shape)
    return {12: f.flatten(2).transpose(1, 2)}


def official(run, condition, epoch, split):
    out = run / (f"eval_val/{condition}/ep{epoch}" if split == "val" else f"eval_test/{condition}")
    if (out / "official_results.json").exists():
        return
    pred = run / f"preds/{condition}/ep{epoch}/probs_{split}.npz"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1", PYTHONPATH=f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}")
    with (out / "log.txt").open("w") as log:
        subprocess.run([EPY, "-u", "-m", "src.pacoattr.official_eval", "--split", split,
                        "--data", str(DATA), "--pred", str(pred), "--out", str(out)],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    h = json.loads((out / "official_results.json").read_text())["hierarchical"]
    print(json.dumps(dict(event="official", condition=condition, epoch=epoch, split=split,
                          object=h["obj-attr-AP"], part=h["obj-attr-part-AP"])), flush=True)


@torch.no_grad()
def predict(model, reader, args, epoch, split):
    run, condition = Path(args.root), args.condition
    out = run / f"preds/{condition}/ep{epoch}/probs_{split}.npz"
    if out.exists():
        return
    # Test manifest is loaded only after selection.json has been fixed.
    if split == "test":
        assert json.loads((run / "selection.json").read_text())[condition]["epoch"] == epoch
    manifest = pd.read_parquet(DATA / f"manifest_{split}.parquet")
    dataset = ds(manifest, np.zeros((len(manifest), 59), np.int8), False, condition)
    batches = [list(range(i, min(len(dataset), i + 4))) for i in range(0, len(dataset), 4)]
    loader = DataLoader(dataset, sampler=batches, batch_size=1, num_workers=args.workers,
                        collate_fn=RC.collate_one, pin_memory=True)
    logits = torch.full((len(manifest), 59), float("nan"))
    reader.eval()
    torch.cuda.reset_peak_memory_stats()
    canvases, seconds, shapes = 0, time.perf_counter(), set()
    for batch in loader:
        b = RC.to_dev(batch, "cuda")
        f = encode(model, b, args.encoder_chunk)
        canvases += len(b["x"])
        shapes.add(tuple(f[12].shape[1:]))
        for ff, bb in roi_chunks(f, b, args.roi_chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                score = reader(ff, bb)["logits"].float()
            logits[bb["rows"].cpu()] = score.cpu()
    assert torch.isfinite(logits).all()
    expected = len(manifest) if condition == "B2" else manifest.image_id.nunique()
    assert canvases == expected, (canvases, expected)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ann_id=manifest.ann_id.values, image_id=manifest.image_id.values,
             category_id=manifest.category_id.values, logits=logits.numpy(), probs=logits.sigmoid().numpy())
    dump(run / f"verify/infer_{condition}_{split}_ep{epoch}.json",
         dict(condition=condition, split=split, epoch=epoch, rows=len(manifest),
              images=int(manifest.image_id.nunique()), encoder_canvases=canvases,
              feature_token_shapes=sorted(shapes), seconds=time.perf_counter() - seconds,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              peak_scope="prediction pass with resident extractor and reader", image_once=condition != "B2"))


def benchmark(model, args):
    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            y = model.features(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(10):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            y = model.features(x)
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end))
    dump(Path(args.root) / f"verify/cost_{args.condition}.json",
         dict(condition=args.condition, input_shape=list(x.shape), feature_shape=list(y.shape),
              dtype=str(y.dtype), latency_ms=times, median_latency_ms=float(np.median(times)),
              mean_latency_ms=float(np.mean(times)), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              device=torch.cuda.get_device_name(), concurrent_other_jobs=True,
              unit="one image for B0/B1; one ROI crop for B2", torch=torch.__version__))


def train(args, model, reader):
    run, condition = Path(args.root), args.condition
    output = run / "readers" / condition
    output.mkdir(parents=True, exist_ok=True)
    if (output / "done.json").exists():
        return
    df, labels = RC.load_data(str(EXCL), splits=("train",))["train"]
    ids = np.array(sorted(df.image_id.unique()))
    heldout = set(np.random.default_rng(0).choice(ids, 1000, replace=False).tolist())
    keep = ~df.image_id.isin(heldout).values
    df, labels = df[keep].reset_index(drop=True), labels[keep]
    dataset = ds(df, labels, True, condition)
    sampler = RC.RankStepSampler(dataset.n_roi, 0, 1, seed=0)
    torch.manual_seed(0)
    loader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=args.workers,
                        collate_fn=RC.collate_one, pin_memory=True, persistent_workers=args.workers > 0,
                        **({"prefetch_factor": 2} if args.workers else {}))
    opt = torch.optim.AdamW(param_groups(reader, 3e-4, 1e-3, 0.05), lr=3e-4, betas=(0.9, 0.999))
    steps = [len(RC.C.full_epoch_batches(dataset.n_roi, ep, 0)) for ep in range(1, 11)]
    total, warm, step, start_epoch = sum(steps), steps[0], 0, 1
    init_hash = state_hash(reader)
    frozen_hash = state_hash(model)
    resume = output / "resume.pt"
    if resume.exists():
        ck = torch.load(resume, map_location="cuda", weights_only=False)
        reader.load_state_dict(ck["reader"])
        opt.load_state_dict(ck["optimizer"])
        step, start_epoch = ck["step"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng"].cpu())
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        # Every train worker uses torch's persistent-worker RNG; resume only at epoch boundaries.
        # Save/recreate explicit worker RNG policy below before enabling production resumes.
        if start_epoch <= 10:
            raise RuntimeError("Interrupted training cannot silently resume augmentation RNG; rerun in a fresh output root")
    config = dict(condition=condition, seed=0, reader_init_seed=7, reader_initial_hash=init_hash,
                  epochs=10, steps_per_epoch=steps, n_train_images=len(dataset), n_train_rois=len(df),
                  heldout_train_image_ids=sorted(heldout), exclude=str(EXCL), lr=3e-4, head_lr=1e-3,
                  weight_decay=0.05, betas=[0.9, 0.999], warmup_epochs=1, schedule="cosine",
                  grad_clip=1.0, loss="unchanged C.masked_bce_sum / global known-label count",
                  hflip=0.5, roi_chunk=args.roi_chunk, encoder_chunk=args.encoder_chunk,
                  reader="unchanged src.readout.models.Reader(r2,layers=(12,))",
                  selection="unchanged src.siglip_pm.select_epoch; official Val Overall; earlier tie",
                  preprocessing="official BRIXEL square256 uint8 antialiased bilinear then float ImageNet normalization")
    dump(output / "config.json", config)
    pool, pending = ThreadPoolExecutor(max_workers=2), []
    started = time.perf_counter()
    try:
        for ep in range(1, start_epoch):
            predict(model, reader, args, ep, "val")
            pending.append(pool.submit(official, run, condition, ep, "val"))
        for ep in range(start_epoch, 11):
            sampler.set_epoch(ep)
            reader.train()
            sum_loss, known = 0.0, 0.0
            for k, batch in enumerate(loader):
                b = RC.to_dev(batch, "cuda")
                f = encode(model, b, args.encoder_chunk)
                nt = (b["Y"] >= 0).sum()
                opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:
                    group["lr"] = RC.C.lr_at(step, group["base"], warm, total)
                for ff, bb in roi_chunks(f, b, args.roi_chunk):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = reader(ff, bb)["logits"].float()
                    ls, _ = RC.C.masked_bce_sum(logits, bb["Y"])
                    if not torch.isfinite(ls):
                        raise FloatingPointError(f"{condition} ep{ep} step{step}")
                    (ls / nt).backward()
                    sum_loss += float(ls.detach())
                known += float(nt)
                torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 50 == 0:
                    print(json.dumps(dict(condition=condition, epoch=ep, step=step, total_steps=total,
                                          loss=sum_loss / known, seconds=round(time.perf_counter() - started))), flush=True)
                if args.smoke_steps and k + 1 >= args.smoke_steps:
                    dump(output / "smoke.json", dict(steps=step, loss=sum_loss / known, frozen_unchanged=state_hash(model) == frozen_hash))
                    return
            torch.save(dict(reader=reader.state_dict(), epoch=ep), output / f"ep{ep}.pt")
            torch.save(dict(reader=reader.state_dict(), optimizer=opt.state_dict(), epoch=ep, step=step,
                            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), resume)
            predict(model, reader, args, ep, "val")
            pending.append(pool.submit(official, run, condition, ep, "val"))
        for future in pending:
            future.result()
        frozen_after = state_hash(model)
        assert frozen_hash == frozen_after
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        dump(run / f"verify/frozen_{condition}.json", dict(before=frozen_hash, after=frozen_after, unchanged=True))
        dump(output / "done.json", dict(epochs=10, steps=step, seconds=time.perf_counter() - started))
    finally:
        pool.shutdown(wait=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--condition", required=True, choices=("B0", "B1", "B2"))
    ap.add_argument("--stage", choices=("train", "test", "verify"), default="train")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--roi-chunk", type=int, default=64)
    ap.add_argument("--encoder-chunk", type=int, default=64)
    ap.add_argument("--smoke-steps", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = FrozenBackbone(args.condition, device="cuda")
    dump(Path(args.root) / f"verify/checkpoints_{args.condition}.json", model.audit)
    torch.manual_seed(7)
    reader = Reader("r2", layers=(12,)).cuda()
    benchmark(model, args)
    if args.stage == "verify":
        return
    if args.stage == "train":
        train(args, model, reader)
    else:
        epoch = json.loads((Path(args.root) / "selection.json").read_text())[args.condition]["epoch"]
        reader.load_state_dict(torch.load(Path(args.root) / f"readers/{args.condition}/ep{epoch}.pt", weights_only=False)["reader"])
        predict(model, reader, args, epoch, "test")
        official(Path(args.root), args.condition, epoch, "test")


if __name__ == "__main__":
    main()
