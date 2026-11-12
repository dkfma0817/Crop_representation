"""Frozen H0/H1 -> ROI read (the P384 hcf_fused path): one ROI-free encode per image, then bilinear reads on the C0 crop grid + fuse."""
import torch

from src.hcf.probe import roi_align


@torch.no_grad()
def encode(model, b, keys, count=None):
    x, detail, parts = b["x"], b["detail"], []
    for i in range(0, len(x), 64):
        if count is not None:
            count[0] += len(x[i:i + 64])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            parts.append(model.forward_all(x[i:i + 64], detail[i:i + 64]))
    cat = lambda f: torch.cat([f(p).float() for p in parts])  # noqa: E731
    A = roi_align(cat(lambda p: p["coarse"]), 1, b)
    mem = {k: roi_align(cat(lambda p: p["mem"][k]), parts[0]["grids"][k], b) for k in parts[0]["mem"]}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Q = model.fuse(A, mem).float()
    f = {12: A, "Q": Q}
    b["img_index"] = torch.arange(len(A), device=A.device)
    return f
