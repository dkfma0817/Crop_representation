"""Spatial-variance amplification (V1-V3) and C0 spatial-pattern oracle (O2) on the xdec Bicubic tensor. No training.

B = src.xdec.geometry.bicubic_roi(feats[12], geom)  (= the xdec Bicubic-only baseline = IVC 'bicubic' call, fp32)
T = C0 teacher: frozen DINOv2-B/14 final-LN on the 336-letterbox GT crop (same encoder call as xdec / IVC C0)
w = C0 valid coverage (geom['valid_w']); valid tokens M = w > 0 (letterbox padding cells are NOT ROI).
  mu_B = sum_i w_i B_i / sum_i w_i   (token axis; all 768 channels kept), mu_T likewise
  V_k = mu_B + k (B - mu_B)    on M,   B on invalid cells (unchanged)        k = 1 (V0), 1.25 (V1), 1.5 (V2), 2.0 (V3)
  O2  = mu_B + (T - mu_T)      on M,   B on invalid cells (unchanged)        ORACLE: uses the C0 crop of the same image
Spatial variance (= xdec recon 'sv'): sum_i w_i mean_c (x_i - mu_x)^2 / sum_i w_i ; ratio to the same quantity of T.
"""
import torch
import torch.nn as nn

import src.readout.common as RC
from src.xdec import geometry as XG

CONDS = {"V0": ("amp", 1.0), "V1": ("amp", 1.25), "V2": ("amp", 1.5), "V3": ("amp", 2.0), "O2": ("oracle", None)}


def wmean(x, w):
    return (x * w[..., None]).sum(1) / w.sum(1, keepdim=True)


def transform(B, w, kind, k=None, T=None):
    """B, T (R, 576, D) fp32, w (R, 576) -> (R, 576, D) fp32."""
    B, w = B.float(), w.float()
    mu = wmean(B, w)[:, None]
    if kind == "amp":
        out = mu + k * (B - mu)
    else:
        Tf = T.float()
        out = mu + (Tf - wmean(Tf, w)[:, None])
    return torch.where((w > 0)[..., None], out, B)


@torch.no_grad()
def teacher(vit, crops, chunk=48):
    return torch.cat([RC.encode_layers(vit, crops[i:i + chunk], (12,))[12] for i in range(0, len(crops), chunk)])


class SVTransform(nn.Module):
    """Used in place of a decoder by the PACO probe (src.ivc.train mode 'dec'): forward(feats, geom) -> (R, 576, 768).
    No trainable parameters. O2 holds the frozen encoder (same weights as the student) to encode geom['crops']."""

    def __init__(self, cond):
        super().__init__()
        self.cond = cond
        self.kind, self.k = CONDS[cond]
        self.arch = f"svamp_{cond}"
        self.vit = RC.load_vit() if self.kind == "oracle" else None

    def load_state_dict(self, sd, strict=True):        # nothing to load (no learned weights); checkpoint = {}
        assert not sd, "SVTransform has no learned state"
        return torch.nn.modules.module._IncompatibleKeys([], [])

    @torch.no_grad()
    def forward(self, feats, g):
        B = XG.bicubic_roi(feats[12], g)
        T = None
        if self.kind == "oracle":
            T = teacher(self.vit, g["crops"])
        with torch.autocast("cuda", enabled=False):
            return transform(B, g["valid_w"], self.kind, self.k, T)
