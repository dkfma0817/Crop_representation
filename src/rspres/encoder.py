"""R2 read for RSP-Residual (mode 'RSPRES') and its Step-0 control (mode 'RSPRES0'); split_ckpt = the RSP-Residual model.pt.

Identical to src.rsp.encoder.rsp_tokens (same fine rectangles = every bilinear corner of every valid read point, same query budget,
same local-map layout and 24 x 24 ROIAlign read, cov = 0 points keep the F384 read) except the value written at each fine cell:
  RSPRES : Y_hat = F384_bil(cell centre) + sigma * Delta_n(cell)        (trained Delta)
  RSPRES0: Y_hat = F384_bil(cell centre)                                (Delta = 0, i.e. the step-0 model; no predictor call)
Step-0 is NOT the F384 baseline read: it is the F384 map sampled at the 4 px cell centres and then bilinearly re-read at the R2 points
(src.rspres.checks reports the difference); it is therefore evaluated as its own control.
"""
import torch

from src.rsp.common import PREFIX, PT, fine_rect_index, read_frects, sample_local
from src.rspres.model import RSPResidual, f384_bil
from src.tsplit.features import RoiEncoder, frozen_vit, sample_maps


def load_res(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["kind"] == "RSPRES", ck.get("kind")
    m = RSPResidual(ck["mu"], ck["sd"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck


@torch.no_grad()
def res_tokens(model, vit, b, step0=False, qbudget=65536, roi_chunk=96):
    x, grid, ii, cov = b["x"], b["bil_grid"], b["img_index"], b["cov"]
    R = len(ii)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        final = vit.forward_features(x).float()
    fin = final[:, PREFIX:]
    tok = torch.cat([sample_maps(fin[ii[k:k + roi_chunk]], grid[k:k + roi_chunk]) for k in range(0, R, roi_chunk)])
    if not step0:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cache = model.context(final)
            l1, z = model.level1(cache, ii, b["boxc"], b["ev"])
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
        pr = f384_bil(final, ii[sel], cen)
        if not step0:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dn = model.queries(cache, [(k[sel], v[sel]) for k, v in l1], ii[sel], z[sel], cen)
            pr = pr + dn.float() * model.sd
        hm, wm = int(frs[:, 2].max()), int(frs[:, 3].max())
        Rc, N = idx.shape
        t = torch.arange(N, device=x.device)[None].expand(Rc, -1)
        ly, lx = torch.div(t, frs[:, 3:4], rounding_mode="floor"), t % frs[:, 3:4]
        maps = torch.zeros(Rc, hm, wm, 768, device=x.device)
        rr = torch.arange(Rc, device=x.device)[:, None].expand(-1, N)
        maps[rr[ok], ly[ok], lx[ok]] = pr[ok]
        rd = sample_local(maps.permute(0, 3, 1, 2), frs, grid[sel])
        tok[sel] = torch.where((cov[sel] > 0)[..., None], rd, tok[sel])
        i = j
    return tok


class ResRoiEncoder(RoiEncoder):
    MODES = RoiEncoder.MODES + ("RSPRES", "RSPRES0")

    def __init__(self, mode, dev, ckpt=None, layer=None, **kw):
        if mode not in ("RSPRES", "RSPRES0"):
            super().__init__(mode, dev, ckpt=ckpt, layer=layer, **kw)
            return
        self.mode, self.layer, self.dev = mode, layer, dev
        self.vit = frozen_vit(dev)
        self.model, ck = load_res(ckpt, dev)
        self.ckpt_step = ck["step"]

    @torch.no_grad()
    def __call__(self, b, roi_chunk=96):
        if self.mode not in ("RSPRES", "RSPRES0"):
            return super().__call__(b, roi_chunk)
        tok = res_tokens(self.model, self.vit, b, step0=self.mode == "RSPRES0", roi_chunk=roi_chunk)
        b["img_index"] = torch.arange(len(tok), device=tok.device)
        return tok
