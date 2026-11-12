"""YAML config with dotted overrides (--set a.b=c). The effective config is saved to <run>/config_effective.yaml."""
import copy
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT = os.path.join(ROOT, "configs", "crop_aware_pretrain", "default.yaml")


def _coerce(s):
    return yaml.safe_load(s)


def load(path=None, overrides=()):
    cfg = yaml.safe_load(open(path or DEFAULT))
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _coerce(v)
    cfg["run_dir"] = os.path.abspath(os.path.join(ROOT, cfg["run_dir"])) if not os.path.isabs(cfg["run_dir"]) else cfg["run_dir"]
    return cfg


def world_setup(cfg, world):
    """-> (accum, micro) so that world * accum * micro = global batch. The global batch never changes with the world size."""
    B, micro = int(cfg["train"]["global_batch_images"]), int(cfg["train"]["micro_batch_images"])
    accum = max(1, B // (world * micro) + (1 if B % (world * micro) else 0))
    assert world * accum * micro == B, f"global batch {B} not divisible into world {world} x micro {micro} (accum {accum})"
    return accum, micro


def snapshot(cfg):
    return copy.deepcopy(cfg)


def dirs(cfg):
    r = cfg["run_dir"]
    return dict(run=r, ckpt=os.path.join(r, "checkpoints"), logs=os.path.join(r, "logs"), control=os.path.join(r, "control"),
                data=os.path.join(r, "data"), evals=os.path.join(r, "evals"))
