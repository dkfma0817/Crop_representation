"""ADE20K B0/B1 at square256; reuse the project's frozen linear-segmentation recipe."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import src.g1eval.ade as A
import src.readout.common as RC
from src.brixel_eval.backbone import FrozenBackbone, state_dict_sha256
from src.brixel_eval.data import make_transform

MODELS = ("B0", "B1")
PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "src/g1eval/ade.py"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


class TrainDS(A.TrainDS):
    """Unchanged training augmentation, deterministic per sample visit for exact resume."""
    def __getitem__(self, index):
        item, visit = index
        previous = random.getstate()
        try:
            random.seed(int(visit))
            return super().__getitem__(int(item))
        finally:
            random.setstate(previous)


class Sampler:
    def __init__(self, n, rank, world, iters, batch, start=0):
        self.base = A.Sampler(n, rank, world, iters, batch)
        self.rank, self.world, self.iters, self.batch, self.start = rank, world, iters, batch, start

    def __iter__(self):
        bs = self.batch // self.world
        for position, item in enumerate(self.base):
            iteration, offset = divmod(position, bs)
            if iteration >= self.start:
                yield int(item), iteration * self.batch + self.rank + offset * self.world

    def __len__(self):
        return (self.iters - self.start) * self.batch // self.world


class ValDS(Dataset):
    def __init__(self, limit=0):
        self.items = A.listing("val")[:limit or None]
        self.transform = make_transform(256)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img, lab = A.load_pair(*self.items[index])
        return index, self.transform(img), torch.from_numpy(lab)


def collate_val(items):
    assert len(items) == 1
    return items[0]


@torch.no_grad()
def features(encoder, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = encoder.features(x)
    return f.float()


@torch.no_grad()
def evaluate(encoders, heads, rank, world, workers, limit=0):
    ds = ValDS(limit)
    loader = DataLoader(ds, sampler=list(range(rank, len(ds), world)), batch_size=1,
                        num_workers=workers, collate_fn=collate_val, pin_memory=True)
    inter = {m: np.zeros((len(ds), A.NC), np.int64) for m in MODELS}
    union = {m: np.zeros_like(inter[m]) for m in MODELS}
    target = np.zeros_like(inter["B0"])
    for head in heads.values():
        head.eval()
    start = time.perf_counter()
    for k, (index, x, y) in enumerate(loader):
        x, y = x.unsqueeze(0).cuda(non_blocking=True), y.cuda(non_blocking=True)
        valid = y != A.IGN
        t = y[valid]
        counts = torch.bincount(t, minlength=A.NC)[:A.NC]
        target[index] = counts.cpu().numpy()
        for m in MODELS:
            lo = A.logits(heads[m], features(encoders[m], x))
            # The existing two-stage logits resize is retained; GT stays at original resolution.
            lo = F.interpolate(lo, size=(256, 256), mode="bilinear", align_corners=False)
            pred = F.interpolate(lo, size=y.shape, mode="bilinear", align_corners=False).argmax(1)[0]
            p = pred[valid]
            hit = torch.bincount(t[p == t], minlength=A.NC)[:A.NC]
            pc = torch.bincount(p, minlength=A.NC)[:A.NC]
            inter[m][index] = hit.cpu().numpy()
            union[m][index] = (pc + counts - hit).cpu().numpy()
        if rank == 0 and (k + 1) % 100 == 0:
            print(json.dumps(dict(event="val", rank0_images=k + 1, total_rank0=len(loader), seconds=round(time.perf_counter() - start))), flush=True)
    for m in MODELS:
        inter[m] = RC.all_sum(torch.from_numpy(inter[m]).cuda()).cpu().numpy()
        union[m] = RC.all_sum(torch.from_numpy(union[m]).cuda()).cpu().numpy()
    target = RC.all_sum(torch.from_numpy(target).cuda()).cpu().numpy()
    return ds, inter, union, target, time.perf_counter() - start


def report(root, info, results, inter, union, target, verification, train_seconds, eval_seconds):
    rows = []
    for m in MODELS:
        i, u, t = inter[m].sum(0), union[m].sum(0), target.sum(0)
        results[m] = dict(mIoU=A.miou(inter[m], union[m]), pixel_accuracy=100 * float(i.sum() / t.sum()),
                          mean_class_accuracy=100 * float(np.mean(i[t > 0] / t[t > 0])), n_classes=int((u > 0).sum()))
        for c in range(A.NC):
            rows.append(dict(condition=m, class_id=c, ade_label=c + 1, intersection=int(i[c]), union=int(u[c]),
                             gt_pixels=int(t[c]), IoU=100 * float(i[c] / u[c]) if u[c] else None))
    delta = {k: results["B1"][k] - results["B0"][k] for k in ("mIoU", "pixel_accuracy", "mean_class_accuracy")}
    np.savez_compressed(root / "val_inter_union.npz", **{f"I_{m}": inter[m] for m in MODELS},
                        **{f"U_{m}": union[m] for m in MODELS}, target=target)
    with (root / "per_class.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status="done", results=results, B1_minus_B0=delta, config=info,
                  verification=verification, train_seconds=train_seconds, eval_seconds=eval_seconds)
    save_json(root / "results.json", result)
    lines = ["# ADE20K B0/B1 frozen linear segmentation, square256", "",
             "Both conditions use identical pretrained DINOv3-B weights and the same input tensor. B1 adds only pretrained BRIXEL ViT-B/256.",
             "Existing src.g1eval.ade head, augmentation recipe, sampler order, AdamW, warmup/poly and mIoU are reused. Both linear heads are freshly trained from identical seed-0 initialization.",
             "Resolution: train crop 256, validation whole image resized to 256×256 (BRIXEL official transform); logits resized to original GT resolution. This differs from the old 384-crop / short-side-384 ADE run, so old mIoU is not a matched baseline.",
             "Final training iteration is evaluated; no validation checkpoint selection or test evaluation. Label 0 is ignored; ADE labels 1–150 map to 0–149.", "",
             "| Condition | mIoU | Pixel accuracy | Mean class accuracy |", "|---|---:|---:|---:|"]
    for m in MODELS:
        v = results[m]
        lines.append(f"| {m} | {v['mIoU']:.2f} | {v['pixel_accuracy']:.2f} | {v['mean_class_accuracy']:.2f} |")
    lines += [f"| B1−B0 (pp) | {delta['mIoU']:+.2f} | {delta['pixel_accuracy']:+.2f} | {delta['mean_class_accuracy']:+.2f} |", "",
              f"Train images: {info['n_train']}; Val images: {info['n_val']}; global batch: {info['cfg']['batch']}; iterations: {info['cfg']['iters']}; seed: 0.",
              "B0 feature grid: 16×16×768; B1 feature grid: 64×64×768. Extractors stay eval/frozen; no backbone or BRIXEL training.",
              "Augmentations are seeded per global sample visit for reproducible resume; both conditions see exactly the same augmented batches. Native feature grids go through the same 768→150 linear head; only logits are bilinearly interpolated.",
              f"Verification: {verification['status']}; both frozen extractors unchanged, identical head initialization and shared inputs; original ADE source unchanged. See verification.json for full evidence.",
              f"Common backbone tensor SHA256: `{verification['backbone_sha256']}`.",
              f"Train wall time: {train_seconds:.1f}s; validation wall time: {eval_seconds:.1f}s.", "",
              "Artifacts: config.json, checkpoint.pt, head_B0.pt, head_B1.pt, val_inter_union.npz, per_class.csv, results.json, verification.json."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    save_json(root / "done.json", dict(status="done", results=results, B1_minus_B0=delta,
                                      iterations=info["cfg"]["iters"], unchanged=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    A.S = 256
    A.CFG = dict(A.CFG, crop=256, eval_short=256, iters=args.iters,
                 warmup=min(1500, max(1, args.iters // 4)))
    rank, world = RC.ddp_setup()
    assert 16 % world == 0
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    encoders = {m: FrozenBackbone(m, device="cuda") for m in MODELS}
    assert len({e.audit["backbone_sha256"] for e in encoders.values()}) == 1
    assert encoders["B0"].audit["backbone_sha256"] == "6c08abfbf25c36717e3aefa68cf74b08be47a0ef271488881569d30e8a03d8de"
    heads, opts = {}, {}
    for m in MODELS:
        torch.manual_seed(0)
        heads[m] = torch.nn.Linear(768, A.NC).cuda()
        opts[m] = torch.optim.AdamW([dict(params=[heads[m].weight], weight_decay=A.CFG["wd"]),
                                    dict(params=[heads[m].bias], weight_decay=0.)],
                                   lr=A.CFG["lr"], betas=A.CFG["betas"])
    initial_hash = {m: state_dict_sha256(heads[m].state_dict()) for m in MODELS}
    assert initial_hash["B0"] == initial_hash["B1"]
    start, elapsed_before = 0, 0.
    checkpoint = root / "checkpoint.pt"
    if args.resume and checkpoint.exists():
        ck = torch.load(checkpoint, map_location="cuda", weights_only=False)
        assert ck["cfg"] == A.CFG
        for m in MODELS:
            heads[m].load_state_dict(ck["heads"][m])
            opts[m].load_state_dict(ck["optimizers"][m])
        start, elapsed_before = ck["iteration"], ck["train_seconds"]
    train = TrainDS()
    info = dict(cfg=A.CFG, world=world, n_train=len(train), n_val=len(ValDS(args.val_limit)),
                evaluation="square256 -> native logits -> 256 square logits -> original GT resolution; final iteration",
                sampler="existing global permutation; stateless augmentation seed=global sample visit",
                head_initial_hash=initial_hash, source_ade=str(SOURCE), source_ade_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                torch=torch.__version__, checkpoints={m: e.audit for m, e in encoders.items()})
    if rank == 0:
        save_json(root / "config.json", info)
        print(json.dumps(dict(event="config", rank=rank, cfg=A.CFG, n_train=len(train), n_val=info["n_val"], resume_iteration=start)), flush=True)
    loader = DataLoader(train, sampler=Sampler(len(train), rank, world, args.iters, 16, start),
                        batch_size=16 // world, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, **({"prefetch_factor": 2} if args.workers else {}))
    timer, running = time.perf_counter(), {m: 0. for m in MODELS}
    for iteration, (x, y) in enumerate(loader, start=start):
        if iteration == start:
            input_sha = hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
            save_json(root / f"input_rank{rank}.json", dict(shared_input=True, sha256=input_sha, shape=list(x.shape), label_shape=list(y.shape)))
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        nvalid = RC.all_sum((y != A.IGN).sum().float())
        assert nvalid > 0
        for m in MODELS:
            f = features(encoders[m], x)
            if iteration == start and rank == 0:
                print(json.dumps(dict(event="feature", condition=m, shape=list(f.shape), frozen=True)), flush=True)
            lo = F.interpolate(A.logits(heads[m], f), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(lo, y, ignore_index=A.IGN, reduction="sum") / nvalid
            assert torch.isfinite(loss)
            for group in opts[m].param_groups:
                group["lr"] = A.lr_at(iteration)
            opts[m].zero_grad(set_to_none=True)
            loss.backward()
            RC.allreduce_grads(list(heads[m].parameters()))
            opts[m].step()
            running[m] += float(loss.detach()) * world
        step = iteration + 1
        if step % 100 == 0 and rank == 0:
            item = dict(iteration=step, total=args.iters, seconds=round(elapsed_before + time.perf_counter() - timer),
                        lr=A.lr_at(iteration), **{f"loss_{m}": running[m] / 100 for m in MODELS})
            with (root / "train_log.jsonl").open("a") as log:
                log.write(json.dumps(item) + "\n")
            print(json.dumps(item), flush=True)
            running = {m: 0. for m in MODELS}
        if rank == 0 and (step % args.save_every == 0 or step == args.iters):
            tmp = checkpoint.with_suffix(".tmp")
            torch.save(dict(iteration=step, cfg=A.CFG, train_seconds=elapsed_before + time.perf_counter() - timer,
                            heads={m: h.state_dict() for m, h in heads.items()},
                            optimizers={m: o.state_dict() for m, o in opts.items()}), tmp)
            tmp.replace(checkpoint)
    train_seconds = elapsed_before + time.perf_counter() - timer
    if rank == 0:
        for m in MODELS:
            torch.save(heads[m].state_dict(), root / f"head_{m}.pt")
    ds, inter, union, target, eval_seconds = evaluate(encoders, heads, rank, world, args.workers, args.val_limit)
    if rank == 0:
        unchanged = {m: state_dict_sha256(e.model.state_dict()) == e.audit["model_sha256"] for m, e in encoders.items()}
        assert all(unchanged.values())
        assert all(not p.requires_grad and p.grad is None for e in encoders.values() for p in e.parameters())
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == info["source_ade_sha256"]
        verification = dict(status="passed", unchanged=unchanged,
                            backbone_sha256=encoders["B0"].audit["backbone_sha256"],
                            original_ade_source_unchanged=True, n_val=len(ds), shared_input=True,
                            head_initial_hash=initial_hash, native_feature_shapes={"B0": [768, 16, 16], "B1": [768, 64, 64]})
        save_json(root / "verification.json", verification)
        results = {}
        report(root, info, results, inter, union, target, verification, train_seconds, eval_seconds)
        print("DONE", json.dumps(results), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
