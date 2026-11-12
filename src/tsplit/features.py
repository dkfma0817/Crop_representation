"""Frozen DINOv3-B/16 features and the ROI read used by every R2 condition of the token-splitting experiment.

Encoder: timm vit_base_patch16_dinov3.lvd1689m (src.hcf.model.load_vit), frozen, bf16 autocast -> fp32. 5 prefix tokens (CLS + 4
registers) are dropped from every output. No absolute position embedding (pos_embed None), norm_pre = Identity, RoPE inside attention.
  final  = forward_features patch tokens (after the final LayerNorm)                         (B, 576, 768) at 384
  early  = patch_embed output = the patch part of the INPUT of block 0 (before any block)    (B, 576, 768) at 384, raw (no norm)
  teacher= forward_features patch tokens of the native 768 canvas                           (B, 2304, 768)

ROI read (all conditions): the R2 reader always gets 24 x 24 tokens sampled with F.grid_sample(bilinear, border, align_corners=False)
at the crop-grid points b['bil_grid'] (C0 cell centres mapped into the 384 canvas; same points for every condition):
  F384  from the 24 x 24 final map                       (= the existing 'bil' / G0-Full384 read)
  T768  from the 48 x 48 teacher map                     (= the existing 'native768' / G0-Native768 read)
  B0    from up2(final) = bilinear x2 of the final map  (24 -> 48, align_corners=False)
  S1    from the S1 child map (letterbox-invalid children fall back to B0)
  S2/S3 from a per-ROI 48 x 48 map = B0 everywhere except the ROI's parent rectangle, which holds the decoder output for all
        children of that rectangle (one transformer pass per ROI over all its children together)
ROI parent rectangle = the smallest parent rectangle containing every bilinear corner child of every valid (cov > 0) read point,
so every valid read of S2/S3 uses decoder children only (checked in src.tsplit.checks).
T768M (T768-MeanOnly): the T768 map with every child set to the mean of its 2 x 2 parent block, read like T768.
Layer diagnostics: 'tap' = layer-L 384 tokens (L = 0 patch embed, 1..11 block outputs, parameter-free LayerNorm) read like F384;
'rerun' = layer-L 384 patch tokens bilinear x2 -> remaining frozen blocks L+1..12 at 48 x 48 (RoPE for 48 x 48, prefix tokens
carried over) -> final norm -> read like T768.
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.hcf.model import load_vit
from src.g1native.bbdata import BBImageDS
from src.tsplit.model import G, G2, TokenSplit, rect_index
from src.viz.c0_pca import LB

PREFIX = 5


def frozen_vit(dev):
    vit = load_vit(dev).eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    assert vit.pos_embed is None and isinstance(vit.norm_pre, torch.nn.Identity) and vit.num_prefix_tokens == PREFIX
    return vit


@torch.no_grad()
def student(vit, x, chunk=64):
    """-> final (B, 576, 768), early (B, 576, 768) fp32 (early = patch_embed output, captured by a forward hook)."""
    fin, ear = [], []
    for i in range(0, len(x), chunk):
        store = []
        h = vit.patch_embed.register_forward_hook(lambda m, inp, out: store.append(out))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = vit.forward_features(x[i:i + chunk])
        finally:
            h.remove()
        assert len(store) == 1
        fin.append(t[:, PREFIX:].float())
        ear.append(store[0].flatten(1, 2).float())
    return torch.cat(fin), torch.cat(ear)


@torch.no_grad()
def teacher(vit, x768, chunk=16):
    out = []
    for i in range(0, len(x768), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(vit.forward_features(x768[i:i + chunk])[:, PREFIX:].float())
    t = torch.cat(out)
    assert t.shape[1] == G2 * G2, t.shape
    return t


def up2(tok):
    B, N, D = tok.shape
    m = tok.transpose(1, 2).reshape(B, D, G, G)
    return F.interpolate(m, size=(G2, G2), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)


def mean_only(t48):
    """T768-MeanOnly: (B, 2304, D) teacher map -> every child replaced by the mean of its parent's 2 x 2 children (same 48 x 48 layout;
    no interpolation, no normalisation). Removes the within-parent sub-patch residual, keeps the parent means exactly."""
    B, N, D = t48.shape
    m = t48.view(B, G, 2, G, 2, D).mean((2, 4), keepdim=True)
    return m.expand(B, G, 2, G, 2, D).reshape(B, N, D)


def valid48(content):
    """(B, 4) content rect on the 384 canvas -> (B, 2304) bool: 8-px child cell overlaps the content (= cell_w(.., 48, 8) > 0)."""
    from src.hcf.distill import cell_w
    return cell_w(content.float(), G2, 8) > 0


def read_rects(bil_grid, cov):
    """(R, 576, 2) grid, (R, 576) coverage -> (R, 4) long [py, px, H, W] parent rectangle covering every bilinear corner of the
    valid read points on the 48 x 48 child grid (align_corners=False: child coordinate u = (g + 1) * 24 - 0.5)."""
    u = (bil_grid.float() + 1) * G2 / 2 - 0.5
    ok = (cov > 0)[..., None]
    big = torch.tensor(1e9, device=u.device)
    lo = torch.where(ok, u, big).amin(1).floor().clamp(0, G2 - 1).long()
    hi = (torch.where(ok, u, -big).amax(1).floor() + 1).clamp(0, G2 - 1).long()
    plo, phi = torch.div(lo, 2, rounding_mode="floor"), torch.div(hi, 2, rounding_mode="floor")
    return torch.stack([plo[:, 1], plo[:, 0], phi[:, 1] - plo[:, 1] + 1, phi[:, 0] - plo[:, 0] + 1], 1)


def sample_maps(maps, grid):
    """maps (R, N, D) with N = g * g -> (R, 576, D) grid_sample at the read points."""
    R, N, D = maps.shape
    g = int(round(N ** .5))
    m = maps.transpose(1, 2).reshape(R, D, g, g)
    return F.grid_sample(m.float(), grid.view(R, 24, 24, 2).float(), mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)


@torch.no_grad()
def refine_rois(model, cmap, img, rects, val48, budget=40000):
    """Run the S2/S3 transformer on every ROI (all children of its rectangle together). Chunks of ROIs sorted by length so the
    padded token count stays <= budget. -> list over ROIs of (idx (n,), out (n, 768)) for valid children."""
    n = 4 * rects[:, 2] * rects[:, 3]
    order = torch.argsort(n).tolist()
    res = [None] * len(rects)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and int(n[order[j]]) * (j + 1 - i) <= budget:
            j += 1
        sel = torch.tensor(order[i:j], device=rects.device)
        idx, cy, cx, ok = rect_index(rects[sel])
        key_ok = ok & val48[img[sel][:, None], idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.refine(cmap, img[sel], idx, cy, cx, key_ok).float()
        for k, r in enumerate(sel.tolist()):
            m = key_ok[k]
            res[r] = (idx[k][m], out[k][m])
        i = j
    return res


@torch.no_grad()
def refine_maps(model, cmap, base, img, rects, val48, budget=40000):
    """Vectorised refine_rois + scatter (same ROI order, chunking and padding -> identical decoder inputs; one host sync).
    -> (R, 2304, 768): base[img] with every valid child of each ROI's rectangle replaced by that ROI's decoder output."""
    R = len(img)
    n_cpu = (4 * rects[:, 2] * rects[:, 3]).tolist()
    order = sorted(range(R), key=lambda r: n_cpu[r])                         # stable; chunks may differ from refine_rois on ties (padding only)
    maps = torch.empty(R * G2 * G2 + 1, base.shape[-1], device=base.device, dtype=torch.float32)
    maps[:-1] = base[img].reshape(-1, base.shape[-1])
    i = 0
    while i < R:
        j = i + 1
        while j < R and n_cpu[order[j]] * (j + 1 - i) <= budget:
            j += 1
        sel = torch.tensor(order[i:j], device=rects.device)
        idx, cy, cx, ok = rect_index(rects[sel], max(n_cpu[r] for r in order[i:j]))
        key_ok = ok & val48[img[sel][:, None], idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.refine(cmap, img[sel], idx, cy, cx, key_ok).float()
        flat = torch.where(key_ok, sel[:, None] * (G2 * G2) + idx, R * G2 * G2)
        maps.index_put_((flat.reshape(-1),), out.reshape(-1, out.shape[-1]))
        i = j
    return maps[:-1].view(R, G2 * G2, -1)


class ContentDS(BBImageDS):
    """BBImageDS + 'content' (B_img, 4): letterbox content rect of every full canvas on the 384 canvas (flip-invariant)."""

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        c = []
        for gi in idx:
            W, H = self.WH[self.groups[gi][0]]
            lb = LB(W, H, 384, 16)
            c.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
        out["content"] = torch.tensor(c, dtype=torch.float32)
        return out


class RoiEncoder:
    MODES = ("F384", "T768", "T768M", "B0", "S1", "S2", "S3", "tap", "rerun", "FA", "P1", "P2")

    def __init__(self, mode, dev, ckpt=None, layer=None, fa_model=None):
        assert mode in self.MODES, mode
        self.mode, self.layer, self.dev = mode, layer, dev
        self.vit = frozen_vit(dev)
        self.model = None
        if mode in ("P1", "P2"):                         # fine-token expansion decoders (src.fte)
            from src.fte.models import P1, P2
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            assert ck["kind"] == mode, ck.get("kind")
            m = (P1 if mode == "P1" else P2)(self.vit).to(dev)
            m.load_state_dict(ck["model"])
            self.model = m.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.ckpt_step = ck["step"]
        if mode == "FA":                                 # Fine-aware decoder (src.finedec); fa_model: an in-memory model (checks only)
            from src.finedec.model import FineAware
            if fa_model is None:
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                assert ck["kind"] == "FA", ck.get("kind")
                fa_model = FineAware(self.vit).to(dev)
                fa_model.load_state_dict(ck["model"])
                self.ckpt_step = ck["step"]
            self.model = fa_model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        if mode in ("S1", "S2", "S3"):
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            assert ck["kind"] == mode, (ck["kind"], mode)
            self.model = TokenSplit(mode).to(dev).eval()
            self.model.load_state_dict(ck["model"])
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.ckpt_step = ck["step"]
        if mode in ("tap", "rerun"):
            assert layer is not None and 0 <= layer <= 11

    def frozen_params(self):
        return list(self.vit.parameters()) + (list(self.model.parameters()) if self.model is not None else [])

    @torch.no_grad()
    def hidden(self, x):
        """Manual forward_features: -> list of hidden states after embedding (index 0) and after each block (1..12), + rope."""
        vit = self.vit
        h = vit.patch_embed(x)
        h, rope = vit._pos_embed(h)
        h = vit.norm_pre(h)
        hs = [h]
        for blk in vit.blocks:
            h = blk(h, rope=rope)
            hs.append(h)
        return hs

    @torch.no_grad()
    def rerun_final(self, x, L):
        vit = self.vit
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = vit.patch_embed(x)
            h, _ = vit._pos_embed(h)
            h = vit.norm_pre(h)
            rope = vit.rope.get_embed(shape=(G, G))
            for blk in vit.blocks[:L]:
                h = blk(h, rope=rope)
            pre, pt = h[:, :PREFIX], h[:, PREFIX:]
            pt = up2(pt.float()).to(h.dtype)
            h = torch.cat([pre, pt], 1)
            rope48 = vit.rope.get_embed(shape=(G2, G2))
            for blk in vit.blocks[L:]:
                h = blk(h, rope=rope48)
            h = vit.norm(h)
        return h[:, PREFIX:].float()

    @torch.no_grad()
    def __call__(self, b, roi_chunk=96):
        x, grid, ii = b["x"], b["bil_grid"], b["img_index"]
        R = len(ii)
        mode = self.mode
        if mode in ("P1", "P2"):
            from src.fte.encoder import fte_tokens
            tok = fte_tokens(mode, self.model, self.vit, b)
        elif mode == "FA":
            from src.finedec.encoder import fa_tokens
            tok = fa_tokens(self.model, self.vit, b)
        elif mode == "F384":
            fin, _ = student(self.vit, x)
            tok = sample_maps(fin[ii], grid)
        elif mode == "tap":
            outs = []
            for i in range(0, len(x), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hs = self.hidden(x[i:i + 64])
                outs.append(F.layer_norm(hs[self.layer][:, PREFIX:].float(), (768,)))
            tok = sample_maps(torch.cat(outs)[ii], grid)
        elif mode in ("T768", "T768M", "rerun"):
            if mode in ("T768", "T768M"):
                assert x.shape[-1] == 768
                full = teacher(self.vit, x, chunk=24)
                if mode == "T768M":
                    full = mean_only(full)
            else:
                assert x.shape[-1] == 384
                full = torch.cat([self.rerun_final(x[i:i + 24], self.layer) for i in range(0, len(x), 24)])
            tok = torch.cat([sample_maps(full[ii[i:i + roi_chunk]], grid[i:i + roi_chunk]) for i in range(0, R, roi_chunk)])
        else:
            fin, ear = student(self.vit, x)
            base = up2(fin)
            if mode == "B0":
                src = base
            else:
                v48 = valid48(b["content"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cmap = self.model.child_map(fin, ear).float()
                src = torch.where(v48[..., None], cmap, base) if mode == "S1" else base
            if mode in ("S2", "S3"):
                rects = read_rects(grid, b["cov"])
                maps = refine_maps(self.model, cmap, base, ii, rects, v48)
                tok = torch.cat([sample_maps(maps[i:i + roi_chunk], grid[i:i + roi_chunk]) for i in range(0, R, roi_chunk)])
            else:
                tok = torch.cat([sample_maps(src[ii[i:i + roi_chunk]], grid[i:i + roi_chunk]) for i in range(0, R, roi_chunk)])
        b["img_index"] = torch.arange(R, device=tok.device)
        return tok


def coverage_check(rects, grid, cov):
    """Fraction of valid read-point bilinear corners that fall inside the ROI rectangle (must be 1)."""
    u = (grid.float() + 1) * G2 / 2 - 0.5
    lo = u.floor().clamp(0, G2 - 1)
    hi = (u.floor() + 1).clamp(0, G2 - 1)
    py, px, H, W = rects.unbind(1)
    y0, x0, y1, x1 = 2 * py, 2 * px, 2 * (py + H) - 1, 2 * (px + W) - 1
    inside = ((lo[..., 0] >= x0[:, None]) & (hi[..., 0] <= x1[:, None]) & (lo[..., 1] >= y0[:, None]) & (hi[..., 1] <= y1[:, None]))
    m = cov > 0
    return float(inside[m].float().mean())


def np_rng(*keys):
    return np.random.default_rng(list(keys))
