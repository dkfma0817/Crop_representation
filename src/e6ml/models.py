"""E6 with a 3-projection K/V memory (2026-11-06 pilot). Everything is xdec E6 (src.xdec.models.XDec('E6'): Bicubic L12
ROI query, 2-block full cross-attention, Bicubic L12 skip, zero-init head, d 256, same canonical seed-0 init) EXCEPT the
memory token before the positional encoding:

  E6-original : m(p) = P12(F12_p)                                   P12 = mem_in = LN(768) -> Linear(768, 256)
  A (matched) : m(p) = P12(F12_p) + P4(F12_p) + P8(F12_p)
  B (multi)   : m(p) = P12(F12_p) + P4(F4_p)  + P8(F8_p)
  kv(p) = m(p) + pe(ROI-relative position of p)    (then per-block ln_kv and the K / V projections, unchanged)

P4, P8: the same structure as P12 (own LN(768) + Linear(768, 256)) with the Linear weight and bias ZERO-initialised, so at
step 0 A == B == E6-original exactly (checked). Their gradient is not blocked: dL/dW_P = g x LN(x) != 0 from step 1, and the
LN parameters receive gradient once W_P != 0 (step 2). F4 / F8 = raw residual stream after blocks 4 / 8, F12 = final-LN
output (src.msd.common.Student, the same layer definition as the Full Multi-layer + R2 baseline); one F336 forward per image.
A and B have identical parameter counts (E6 + 2 x 197,632).
"""
import torch
import torch.nn as nn

from src.xdec import geometry as XG
from src.xdec.models import D, DH, XDec


class E6Mem(XDec):
    def __init__(self, mode):
        assert mode in ("A", "B")
        super().__init__("E6")                       # canonical seed-0 init of every E6 module (incl. mem_in = P12)
        self.mode = mode
        self.arch = f"e6ml_{mode}"
        self.mem_in4 = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, DH))
        self.mem_in8 = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, DH))
        for m in (self.mem_in4, self.mem_in8):       # deterministic (no RNG): LN (1, 0), Linear zero
            nn.init.zeros_(m[1].weight)
            nn.init.zeros_(m[1].bias)
        self._feats = None

    def forward(self, feats, g, return_parts=False):
        self._feats = feats
        try:
            return super().forward(feats, g, return_parts)
        finally:
            self._feats = None

    def memory(self, F12, g):
        f = self._feats
        x4, x8 = (f[4], f[8]) if self.mode == "B" else (F12, F12)
        kv = (self.mem_in(F12) + self.mem_in4(x4) + self.mem_in8(x8))[g["img_index"]]
        uv, size = XG.full_uv(g)
        return kv + self.pe(uv, size), None

    def groups(self):
        gr = super().groups()
        gr["P12 (mem_in)"] = list(self.mem_in.parameters())
        gr["P4"] = list(self.mem_in4.parameters())
        gr["P8"] = list(self.mem_in8.parameters())
        gr["P4P8_linear"] = list(self.mem_in4[1].parameters()) + list(self.mem_in8[1].parameters())
        gr["P4P8_ln"] = list(self.mem_in4[0].parameters()) + list(self.mem_in8[0].parameters())
        return gr


def build(cond):
    return E6Mem(cond)


def build_from_arch(arch):
    assert arch.startswith("e6ml_"), arch
    return E6Mem(arch[len("e6ml_"):])
