"""P1 / P2 read path for R2 (src.tsplit.features.RoiEncoder modes 'P1' / 'P2'); the feature-input part only, the R2 pipeline is unchanged.

Per image (once): the original frozen DINOv3 384 forward -> final patch tokens (P1 input) and the blocks[7] output (P2 input).
Per ROI (GT box + category from the official manifest): the S3 read rectangle (every bilinear corner of every valid C0 read point)
-> its parents (no cap: large PACO boxes use their full rectangle, up to 24 x 24) -> decoder children -> local map -> the C0 read points
(continuous GT-box geometry) in local coordinates -> grid_sample -> 24 x 24 tokens. ROIs are processed in chunks sorted by parent count so
padding stays small (P2 chunks of <= 40,000 padded tokens incl. the 581 context tokens per ROI).
"""
import torch

from src.fte.common import local_read
from src.fte.train import student_h8
from src.tsplit.features import read_rects


@torch.no_grad()
def fte_tokens(kind, model, vit, b, budget=None):
    x, grid, ii = b["x"], b["bil_grid"], b["img_index"]
    R = len(ii)
    final, h8 = student_h8(vit, x)
    rects = read_rects(grid, b["cov"])
    K = (rects[:, 2] * rects[:, 3]).tolist()
    per = (lambda k: 581 + 4 * k) if kind == "P2" else (lambda k: k)
    budget = budget or (40000 if kind == "P2" else 200000)
    order = sorted(range(R), key=lambda r: K[r])
    out = [None] * R
    i = 0
    while i < R:
        j = i + 1
        while j < R and per(K[order[j]]) * (j + 1 - i) <= budget:
            j += 1
        sel = torch.tensor(order[i:j], device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Y, _, _, _ = model(final if kind == "P1" else h8, ii[sel], rects[sel])
        tok = local_read(Y.float(), rects[sel], grid[sel])
        for k, r in enumerate(order[i:j]):
            out[r] = tok[k]
        i = j
    b["rects"] = rects
    return torch.stack(out)
