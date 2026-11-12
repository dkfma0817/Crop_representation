"""Student / teacher construction and which parameters train. No projection module: the ROI readout is geometry only, so the path used
at evaluation is exactly `shared encoder -> tokens -> R2 reader` (src.g1x.probe mode 'full')."""
import copy

import torch

from src.ivc.ft import enc_param_groups, ema_update  # noqa: F401
from src.viz.c0_pca import BACKBONES, load_encoder


def build(cfg, dev):
    enc = load_encoder(cfg["backbone"], dev)
    enc0 = load_encoder(cfg["backbone"], dev)
    for p in enc0.parameters():
        p.requires_grad_(False)
    enc0.eval()
    enc.eval()                                  # no dropout / drop-path in this model; asserted in assert_no_stochastic
    assert_no_stochastic(enc)
    n = int(cfg["model"]["trainable_last_blocks"])
    nb = len(enc.blocks)
    assert 0 <= n <= nb
    for p in enc.parameters():
        p.requires_grad_(False)
    for blk in enc.blocks[nb - n:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    if cfg["model"]["train_norm"]:
        for p in enc.norm.parameters():
            p.requires_grad_(True)
    return enc, enc0


def assert_no_stochastic(m):
    for x in m.modules():
        for attr in ("drop_prob", "p"):
            if hasattr(x, attr) and isinstance(getattr(x, attr), float) and getattr(x, attr) > 0 and x.__class__.__name__.lower().startswith(("drop", "patchdrop")):
                raise AssertionError(f"stochastic module active: {x}")


def trainable_named(enc):
    return [(n, p) for n, p in enc.named_parameters() if p.requires_grad]


def param_groups(enc, cfg):
    tc = cfg["train"]
    groups = enc_param_groups(enc, float(tc["enc_lr"]), float(tc["llrd"]), float(tc["weight_decay"]))
    out = []
    for g in groups:
        g["params"] = [p for p in g["params"] if p.requires_grad]
        if g["params"]:
            out.append(g)
    return out


def describe(enc):
    tr = trainable_named(enc)
    frozen = [(n, p) for n, p in enc.named_parameters() if not p.requires_grad]
    blocks = sorted({int(n.split(".")[1]) for n, _ in tr if n.startswith("blocks.")})
    return dict(n_params_total=sum(p.numel() for p in enc.parameters()), n_params_trainable=sum(p.numel() for _, p in tr),
                n_params_frozen=sum(p.numel() for _, p in frozen), trainable_blocks=blocks,
                trainable_top_names=sorted({n.split(".")[0] for n, _ in tr}), encoder=BACKBONES["dinov3"]["name"])


@torch.no_grad()
def ema_step(ema, enc, m):
    """EMA of the trainable parameters only (frozen ones are identical by construction)."""
    for (_, pe), (_, ps) in zip(ema.named_parameters(), enc.named_parameters()):
        if ps.requires_grad:
            pe.mul_(m).add_(ps.detach(), alpha=1 - m)


def make_ema(enc):
    ema = copy.deepcopy(enc).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema
