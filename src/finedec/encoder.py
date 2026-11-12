"""FA read path for R2 (used by src.tsplit.features.RoiEncoder mode 'FA').

Per image (once): DINOv3 384 -> final + X_pre; FineAware.image_side -> fine queries + per-block memory k/v.
Per ROI: the S3 read rectangle (src.tsplit.features.read_rects: every bilinear corner of every valid read point) -> one decoder pass over
ALL children of the rectangle (ROIs sorted by length and chunked to <= 40,000 padded tokens; the output depends on the rectangle).
Map per ROI (48 x 48): decoder output at the rectangle's valid children; elsewhere (never touched by a valid read) the model's own
zero-correction value = F384 parent token repeated 2 x 2. No zeros and no teacher features are inserted. Then the standard 24 x 24 read.
"""
import torch

from src.finedec.feats import student_pre
from src.tsplit.features import read_rects, sample_maps, valid48
from src.tsplit.model import G, G2, rect_index


@torch.no_grad()
def fa_tokens(model, vit, b, roi_chunk=96, budget=40000):
    x, grid, ii = b["x"], b["bil_grid"], b["img_index"]
    R = len(ii)
    final, xpre = student_pre(vit, x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fine, kvs = model.image_side(x, final)
    v48 = valid48(b["content"])
    rects = read_rects(grid, b["cov"])
    base = final.view(-1, G, 1, G, 1, 768).expand(-1, G, 2, G, 2, 768).reshape(-1, G2 * G2, 768)
    maps = torch.empty(R * G2 * G2 + 1, 768, device=x.device)
    maps[:-1] = base[ii].reshape(-1, 768)
    n = (4 * rects[:, 2] * rects[:, 3]).tolist()
    order = sorted(range(R), key=lambda r: n[r])
    i = 0
    while i < R:
        j = i + 1
        while j < R and n[order[j]] * (j + 1 - i) <= budget:
            j += 1
        sel = torch.tensor(order[i:j], device=x.device)
        rs = rects[sel]
        idx, cy, cx, ok = rect_index(rs, max(n[r] for r in order[i:j]))
        key_ok = ok & v48[ii[sel][:, None], idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Y, _, _, _ = model.decode(fine, kvs, xpre, ii[sel], idx, cy, cx, ok, key_ok, rs[:, 0], rs[:, 1], rs[:, 3])
        flat = torch.where(key_ok, sel[:, None] * (G2 * G2) + idx, R * G2 * G2)
        maps.index_put_((flat.reshape(-1),), Y.float().reshape(-1, 768))
        i = j
    maps = maps[:-1].view(R, G2 * G2, 768)
    b["rects"] = rects
    return torch.cat([sample_maps(maps[k:k + roi_chunk], grid[k:k + roi_chunk]) for k in range(0, R, roi_chunk)])
