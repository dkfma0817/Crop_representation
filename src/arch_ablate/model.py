"""Coarse-Fine interaction ablation. Everything except the mid-network exchange is the existing src.hcf.model.CoarseFine.

  A0  Coarse <-> Fine, 4 stages (the existing model, unchanged)
  A1  Coarse -> Fine only, 4 stages: the Fine -> Coarse residual and its parameters are removed
  A2  No exchange: all Interaction modules are removed; Coarse and Fine meet only in the final ROI fusion Q = A + s_d Wd(B)

All weights come from the common initial checkpoint (A1 and A2 use the subset of its tensors they still own), never from a trained model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401

from src.hcf.model import D, FD, CoarseFine, _block, _prefix, load_vit, sincos_2d

MODES = ("A0", "A1", "A2")


class CoarseToFine(nn.Module):
    """The c <- ... half of Interaction removed: f <- f + s_c2f L_c2f(broadcast_2x2(c)). Parameter names match Interaction, so the common initial checkpoint loads directly."""

    def __init__(self, cd, fd, s0):
        super().__init__()
        self.c2f = nn.Linear(cd, fd)
        self.s_c2f = nn.Parameter(torch.tensor(float(s0)))

    def forward(self, c, f):
        cu = c.repeat_interleave(2, 1).repeat_interleave(2, 2)
        rf = self.s_c2f * self.c2f(cu)
        with torch.no_grad():
            d = dict(f=float(rf.norm(dim=-1).mean() / f.norm(dim=-1).mean().clamp_min(1e-8)))
        return f + rf, d


class AblationCoarseFine(CoarseFine):
    def __init__(self, vit, interaction, s0=0.1):
        super().__init__(vit, s0)
        assert interaction in MODES, interaction
        self.interaction = interaction
        if interaction == "A1":
            self.inter = nn.ModuleList([CoarseToFine(D, FD, s0) for _ in range(4)])
        elif interaction == "A2":
            self.inter = nn.ModuleList()

    def forward_all(self, x, collect=False):
        if self.interaction == "A0":
            return super().forward_all(x, collect)
        v, npt = self.vit, self.npt
        t, rope, H, W = _prefix(v, x)
        f = self.fine_embed(x).permute(0, 2, 3, 1)
        f = f + sincos_2d(2 * H, 2 * W, FD, x.device)[None]
        d = {}
        for st in range(4):
            for i in range(3 * st, 3 * st + 3):
                t = _block(v, i, t, rope)
            f = self.fine_blocks[st](f)
            if self.interaction == "A1":
                f, r = self.inter[st](t[:, npt:].reshape(len(x), H, W, D), f)
                if collect:
                    d[f"stage{st + 1}_res_f"] = r["f"]
        t = v.norm(t)
        f = self.fine_norm(f)
        if collect:
            self.diag = d
        return dict(cls=t[:, 0], coarse=t[:, npt:], H=H, W=W, mem=dict(fine=f.reshape(len(x), 4 * H * W, FD)), grids=dict(fine=2), reg=t[:, 1:npt])


def from_initial(path, interaction, dev):
    """Common initial state -> a model of the requested structure. Every tensor the structure owns must exist in the checkpoint; only removed Interaction tensors may be left over."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["cond"] == "P" and ck["step"] == 0, (ck["cond"], ck["step"])
    model = AblationCoarseFine(load_vit(dev), interaction, ck["s0"]).to(dev)
    own, saved = set(model.state_dict()), set(ck["model"])
    assert own <= saved, sorted(own - saved)[:5]
    left = saved - own
    allowed = {"A0": (), "A1": ("f2c",), "A2": ("f2c", "c2f")}[interaction]
    assert all(k.startswith("inter.") and any(tag in k for tag in allowed) for k in left), sorted(left)[:5]
    model.load_state_dict({k: ck["model"][k] for k in own}, strict=True)
    return model


def load_ablation(path, dev):
    """Saved EMA checkpoint -> frozen eval model. Checkpoints without the key are existing A0 (bidirectional) checkpoints."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = AblationCoarseFine(load_vit(dev), ck.get("interaction", "A0"), ck["s0"]).to(dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck
