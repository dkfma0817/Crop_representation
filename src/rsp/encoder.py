"""RSP read path for the R2 reader (frozen predictor; used by the R2 probe and official inference through RSPRoiEncoder).

Per image (once): frozen DINOv3 384 -> final tokens F (581); predictor context stream (L0).
Per GT box: L1 stream (evidence from the ORIGINAL image GT box resized to 64 x 64 [RSP-Masked] and z from the continuous GT box).
Queries: the fine (4 px = teacher spacing) cells of the smallest fine rectangle that contains every bilinear corner of every valid
(cov > 0) R2 read point (src.rsp.common.read_frects). Only these real fine-grid positions are predicted (a 1-parent box asks for
about 5 x 5 cells, never a forced 24 x 24 query grid). Predictions are de-normalised (x sd + mu) into DINOv3 final-token space,
laid out on a local fine map and read with the standard 24 x 24 bilinear ROIAlign read (= grid_sample on the full 96 x 96 map,
align_corners False, border; src.rsp.common.sample_local, checked equal). Read points with cov = 0 (outside the box, never used for
content) keep the F384 bilinear read value.
"""
import torch

from src.rsp.common import PREFIX, PT, fine_rect_index, read_frects, sample_local
from src.tsplit.features import RoiEncoder, frozen_vit, sample_maps


def load_rsp(ckpt, dev):
    from src.rsp.model import RSPPredictor
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["kind"] == "RSP", ck.get("kind")
    m = RSPPredictor(pix=ck["pix"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck["mu"].to(dev), ck["sd"].to(dev), ck


@torch.no_grad()
def rsp_tokens(model, vit, b, mu, sd, qbudget=65536, roi_chunk=96, return_stats=False):
    x, grid, ii, cov = b["x"], b["bil_grid"], b["img_index"], b["cov"]
    R = len(ii)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        final = vit.forward_features(x).float()
    fin = final[:, PREFIX:]
    tok = torch.cat([sample_maps(fin[ii[k:k + roi_chunk]], grid[k:k + roi_chunk]) for k in range(0, R, roi_chunk)])
    ev = b["ev"] if model.pix else None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cache = model.context(final)
        l1, z = model.level1(cache, ii, b["boxc"], ev)
    fr = read_frects(grid, cov)
    n = (fr[:, 2] * fr[:, 3]).tolist()
    order = sorted(range(R), key=lambda r: n[r])
    i = 0
    while i < R:
        j = i + 1
        while j < R and n[order[j]] * (j + 1 - i) <= qbudget:
            j += 1
        sel = torch.tensor(order[i:j], device=x.device)
        frs = fr[sel]
        idx, uy, ux, ok = fine_rect_index(frs)
        cen = torch.stack([(uy.float() + .5) * PT, (ux.float() + .5) * PT], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pn = model.queries(cache, [(k[sel], v[sel]) for k, v in l1], ii[sel], z[sel], cen)
        pr = pn.float() * sd + mu
        hm, wm = int(frs[:, 2].max()), int(frs[:, 3].max())
        Rc, N = idx.shape
        t = torch.arange(N, device=x.device)[None].expand(Rc, -1)
        ly, lx = torch.div(t, frs[:, 3:4], rounding_mode="floor"), t % frs[:, 3:4]
        maps = torch.zeros(Rc, hm, wm, 768, device=x.device)
        rr = torch.arange(Rc, device=x.device)[:, None].expand(-1, N)
        maps[rr[ok], ly[ok], lx[ok]] = pr[ok]
        rd = sample_local(maps.permute(0, 3, 1, 2), frs, grid[sel])
        v = (cov[sel] > 0)[..., None]
        tok[sel] = torch.where(v, rd, tok[sel])
        i = j
    b["rsp_frects"] = fr
    return tok


class RSPRoiEncoder(RoiEncoder):
    """RoiEncoder + mode 'RSP' (job key split_ckpt = the RSP model.pt). Other modes are untouched."""
    MODES = RoiEncoder.MODES + ("RSP",)

    def __init__(self, mode, dev, ckpt=None, layer=None, **kw):
        if mode != "RSP":
            super().__init__(mode, dev, ckpt=ckpt, layer=layer, **kw)
            return
        self.mode, self.layer, self.dev = mode, layer, dev
        self.vit = frozen_vit(dev)
        self.model, self.mu, self.sd, ck = load_rsp(ckpt, dev)
        self.ckpt_step = ck["step"]

    @torch.no_grad()
    def __call__(self, b, roi_chunk=96):
        if self.mode != "RSP":
            return super().__call__(b, roi_chunk)
        tok = rsp_tokens(self.model, self.vit, b, self.mu, self.sd, roi_chunk=roi_chunk)
        b["img_index"] = torch.arange(len(tok), device=tok.device)
        return tok
