"""Head cost: parameters and MACs of the PixelDiT head for B (64 memory tokens), P1 (576 tokens), P2 (position-wise modulation). One 384 crop."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.hcf.model import n_params
from src.pcond.heads import build_head, memory_for


def macs(model, *args):
    n = [0.]
    hooks = []

    def linear(mod, i, o):
        n[0] += i[0].numel() / i[0].shape[-1] * mod.in_features * mod.out_features

    def norm(mod, i, o):
        n[0] += 2 * i[0].numel()

    def attention(mod, i, o):
        q, k = i[0], i[1]
        B, L, S, E = q.shape[0], q.shape[1], k.shape[1], mod.embed_dim
        n[0] += B * ((L + 2 * S) * E * E + 2 * L * S * E + L * E * E)  # q, k, v, out projections; q.k and attn.v over heads
    for mod in model.modules():
        if isinstance(mod, nn.MultiheadAttention):
            hooks.append(mod.register_forward_hook(attention))
        elif isinstance(mod, nn.Linear) and "NonDynamically" not in type(mod).__name__:
            hooks.append(mod.register_forward_hook(linear))
        elif isinstance(mod, nn.LayerNorm):
            hooks.append(mod.register_forward_hook(norm))
    with torch.no_grad():
        model(*args)
    for h in hooks:
        h.remove()
    return n[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {}
    noisy, t, features, cov = torch.randn(1, 3, 384, 384, device=dev), torch.tensor([.5], device=dev), torch.randn(1, 576, 768, device=dev), torch.ones(1, 576, device=dev)
    for c in ("B", "P1", "P2"):
        head = build_head(c).to(dev).eval()
        memory, mcov = memory_for(c, features, cov)
        out[c] = dict(head_parameters=n_params(head), memory_tokens=memory.shape[1], head_gmacs_per_crop=macs(head, noisy, t, memory, mcov) / 1e9)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
