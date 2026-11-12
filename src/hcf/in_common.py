"""ImageNet-1K frozen CLS probe constants for the Coarse-Fine pilot: Original + M + P. Copy of the src.g1in.common interface; protocol unchanged."""
import os

from src.g1in.common import DATA, MEAN, RESIZE, S, STD, checksum, parquet_files  # noqa: F401
from src.hcf.model import load_hcf, load_vit

ROOT = "/home/irteam/Crop_representation"
G1 = os.path.join(ROOT, "output/2026-11-08/hcf_pilot")
MODELS = ("original", "m", "p")
PACO_PROBE = {"original": "../../2026-11-06/g1_cross_distill/probe_frozen_full", "m": "probe_M", "p": "probe_P"}


def load_encoder(name, dev):
    if name == "original":
        v = load_vit(dev).eval()
        for p in v.parameters():
            p.requires_grad_(False)
        return v
    m, _ = load_hcf(os.path.join(G1, name.upper(), "checkpoints", "ema_last.pt"), dev)
    return m
