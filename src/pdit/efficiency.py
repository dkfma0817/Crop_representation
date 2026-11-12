"""Head parameters and MACs for baseline / P1 / P2 on one crop (r = 1). No training.

Hook counts: Linear = N x in x out, LayerNorm = 2 x elements, MultiheadAttention = B (Nq d^2 + 2 Nk d^2 + Nq Nk d + Nq d^2) (q / k, v projections, attention, output projection).
Activations are not counted, so the numbers compare the heads rather than give an exact FLOP count.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.hcf.model import n_params
from src.pdit.train import build_head


def macs(head, noisy, t, memory, cov):
    n, hooks = [0.], []

    def linear(mod, i, o):
        n[0] += i[0].numel() / i[0].shape[-1] * mod.in_features * mod.out_features

    def norm(mod, i, o):
        n[0] += 2 * i[0].numel()

    def attn(mod, i, o):
        B, Nq, d = i[0].shape
        Nk = i[1].shape[1]
        n[0] += B * (Nq * d * d + 2 * Nk * d * d + Nq * Nk * d + Nq * d * d)

    for m in head.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear))
        elif isinstance(m, nn.LayerNorm):
            hooks.append(m.register_forward_hook(norm))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(attn))
    with torch.no_grad():
        head(noisy, t, memory, cov)
    for h in hooks:
        h.remove()
    return n[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {}
    for c in ("baseline", "P1", "P2"):
        head = build_head(c).eval()
        g = 8 if c == "baseline" else 24
        report[c] = dict(head_params=n_params(head), memory_tokens=g * g, macs_per_crop=macs(head, torch.zeros(1, 3, 384, 384), torch.tensor([.5]), torch.zeros(1, g * g, 768), torch.ones(1, g * g)))
    for c in ("P1", "P2"):
        report[c]["macs_ratio_to_baseline"] = report[c]["macs_per_crop"] / report["baseline"]["macs_per_crop"]
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
