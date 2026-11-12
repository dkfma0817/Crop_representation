"""Checkpoint directories under <run>/checkpoints:  step_00012000/{state.pt, rng_rank<r>.pt, meta.json}
Written to .tmp_step_00012000 first; rank 0 publishes it with an atomic directory rename after every rank has written its RNG file, so a
step_* directory is always complete (meta.json is written last before the rename). --resume auto takes the newest step_* that loads."""
import glob
import json
import os
import random
import shutil

import numpy as np
import torch


def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_save(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(s["cuda"])


def clean_stale(root):
    for d in glob.glob(os.path.join(root, ".tmp_step_*")) + glob.glob(os.path.join(root, ".old_step_*")):
        shutil.rmtree(d, ignore_errors=True)


def save(root, step, rank, build_state, meta, keep_last, sync):
    name = f"step_{step:08d}"
    tmp, final, old = (os.path.join(root, p + name) for p in (".tmp_", "", ".old_"))
    os.makedirs(tmp, exist_ok=True)
    atomic_save(rng_state(), os.path.join(tmp, f"rng_rank{rank}.pt"))
    sync()                                      # every rank has written its RNG file
    if rank == 0:
        atomic_save(build_state(), os.path.join(tmp, "state.pt"))
        atomic_json(meta, os.path.join(tmp, "meta.json"))
        if os.path.exists(final):               # same step saved twice (a resume stopped immediately)
            os.rename(final, old)
        os.rename(tmp, final)
        _fsync_dir(root)
        shutil.rmtree(old, ignore_errors=True)
        for d in sorted(glob.glob(os.path.join(root, "step_*")))[:-max(1, keep_last)]:
            shutil.rmtree(d, ignore_errors=True)
    sync()                                      # rank != 0 continues only after the publish


def list_ckpts(root):
    return sorted((d for d in glob.glob(os.path.join(root, "step_*")) if os.path.exists(os.path.join(d, "meta.json")) and os.path.exists(os.path.join(d, "state.pt"))), reverse=True)


def load_latest(root, log=print):
    """-> (dir, state dict) of the newest loadable checkpoint, or (None, None). An unreadable checkpoint is reported and skipped."""
    for d in list_ckpts(root):
        try:
            return d, torch.load(os.path.join(d, "state.pt"), map_location="cpu", weights_only=False)
        except Exception as e:                  # noqa: BLE001
            log(f"checkpoint {d} is not readable ({e!r}); trying the previous one")
    return None, None


def load_rng(ckpt_dir, rank):
    p = os.path.join(ckpt_dir, f"rng_rank{rank}.pt")
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None
