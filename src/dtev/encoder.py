"""R2 read of the dense teacher (mode 'DT'; job key split_ckpt = a JSON config {"pt": 8|4, "attn": "masked"|"full", "margin": 1}).

Per batch: the frozen DINOv3 384 forward (coarse stream; for 'masked' its per-layer K/V are reused by every ROI of the image).
Per GT box: parent rectangle = parent cells covered by the canvas box + 1 cell margin (DT.parent_rects); the teacher's fine tokens of
the whole rectangle are laid out on a local fine map and read with the standard 24 x 24 bilinear ROIAlign read at b['bil_grid']
(= grid_sample on the full (384 / p_T)^2 fine map, align_corners False, border; checked in src.dtev.checks). Every bilinear corner of
every valid (cov > 0) read point is asserted to lie inside the rectangle. The R2 reader masks cov = 0 tokens; those keep the F384 read.
"""
import json

import torch
import torch.nn.functional as F

from src.dtev.teacher import DT, chunks_by_budget
from src.rsp.common import PREFIX
from src.tsplit.features import RoiEncoder, frozen_vit, sample_maps

BUDGET = {"masked": 49152, "full": 49152}


def corner_rects(grid, cov, gf):
    """(R, 576, 2) read grid, cov -> (lo (R, 2) [y, x], hi (R, 2)) fine-cell range of all bilinear corners of valid points; any (R,)."""
    u = ((grid.float() + 1) * gf / 2 - 0.5).clamp(0, gf - 1)
    ok = (cov > 0)[..., None]
    big = torch.tensor(1e9, device=u.device)
    lo = torch.where(ok, u, big).amin(1).floor().clamp(0, gf - 1).long()
    hi = (torch.where(ok, u, -big).amax(1).floor() + 1).clamp(0, gf - 1).long()
    return lo.flip(-1), hi.flip(-1), (cov > 0).any(1)


def sample_local(maps, frects, grid, gf):
    """maps (R, D, hm, wm) local fine maps (zero padded to the chunk max) -> (R, 576, D) = grid_sample on the full gf x gf map."""
    R, D, hm, wm = maps.shape
    u = ((grid.float() + 1) * gf / 2 - 0.5).clamp(0, gf - 1)
    lx = u[..., 0] - frects[:, 1:2].float()
    ly = u[..., 1] - frects[:, 0:1].float()
    g = torch.stack([(lx + .5) / wm * 2 - 1, (ly + .5) / hm * 2 - 1], -1).view(R, 24, 24, 2)
    return F.grid_sample(maps.float(), g, mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)


@torch.no_grad()
def dt_tokens(dt, attn, b, margin=1, budget=None, stats=None):
    x, grid, ii, cov, boxc = b["x"], b["bil_grid"], b["img_index"], b["cov"], b["boxc"]
    R = len(ii)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if attn == "masked":
            final, kv = dt.coarse(x)
        else:
            final = dt.vit.forward_features(x)
            h0, _ = dt.embed0(x)
        femb = dt.fine_embed(x)
    final = final.float()
    tok = torch.cat([sample_maps(final[ii[k:k + 96], PREFIX:], grid[k:k + 96]) for k in range(0, R, 96)])
    prects = dt.parent_rects(boxc, margin)
    fr = dt.fine_rects(prects)
    lo, hi, anyv = corner_rects(grid, cov, dt.gf)
    inside = (lo >= fr[:, :2]) & (hi <= fr[:, :2] + fr[:, 2:] - 1)
    assert bool(inside.all(1)[anyv].all()), "a bilinear corner of a valid read point lies outside the ROI rectangle"
    n = (fr[:, 2] * fr[:, 3]).tolist()
    extra = 0 if attn == "masked" else PREFIX + 576
    for sel in chunks_by_budget([k + extra for k in n], budget or BUDGET[attn]):
        sel = torch.tensor(sel, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if attn == "masked":
                T, ok = dt.fine_masked(femb, kv, prects[sel], ii[sel])
            else:
                T, ok = dt.fine_full(h0, femb, prects[sel], ii[sel])
        frs = fr[sel]
        hm, wm = int(frs[:, 2].max()), int(frs[:, 3].max())
        Rc, N = ok.shape
        t = torch.arange(N, device=x.device)[None].expand(Rc, -1)
        ly, lx = torch.div(t, frs[:, 3:4], rounding_mode="floor"), t % frs[:, 3:4]
        maps = torch.zeros(Rc, hm, wm, T.shape[-1], device=x.device)
        rr = torch.arange(Rc, device=x.device)[:, None].expand(-1, N)
        maps[rr[ok], ly[ok], lx[ok]] = T[ok].float()
        rd = sample_local(maps.permute(0, 3, 1, 2), frs, grid[sel], dt.gf)
        tok[sel] = torch.where((cov[sel] > 0)[..., None], rd, tok[sel])
    if stats is not None:
        stats["n_fine"] = n
        stats["prects"] = prects
    return tok


class DTRoiEncoder(RoiEncoder):
    MODES = RoiEncoder.MODES + ("DT",)

    def __init__(self, mode, dev, ckpt=None, layer=None, **kw):
        if mode != "DT":
            super().__init__(mode, dev, ckpt=ckpt, layer=layer, **kw)
            return
        self.mode, self.layer, self.dev, self.model = mode, layer, dev, None
        self.cfg = json.load(open(ckpt))
        assert self.cfg["attn"] in ("masked", "full") and self.cfg["pt"] in (4, 8), self.cfg
        self.vit = frozen_vit(dev)
        self.dt = DT(self.vit, self.cfg["pt"]).to(dev)
        self.ckpt_step = 0

    @torch.no_grad()
    def __call__(self, b, roi_chunk=96):
        if self.mode != "DT":
            return super().__call__(b, roi_chunk)
        tok = dt_tokens(self.dt, self.cfg["attn"], b, self.cfg.get("margin", 1))
        b["img_index"] = torch.arange(len(tok), device=tok.device)
        return tok
