"""RAEv2 (DINOv2-B, K=1, ImageNet stage-1) ROI crop-latent reconstruction pilot: shared pieces.

Official pair (nyu-visionx/RAEv2-models, repo nanovisionx/RAEv2 @ 8a0d238):
  encoder  = RAEv2 create_encoder("dinov2-vit-b", resolution=256)  -> torch.hub facebookresearch/dinov2 dinov2_vitb14_reg
             (registers, final LayerNorm WITHOUT affine, pos-embed resampled to 16x16 with timm), preprocess =
             x/255 -> ImageNet mean/std -> F.interpolate(224, bicubic). Output = 16x16x768 patch tokens ("raw latent").
  decoder  = GeneralDecoder(configs/decoder/ViTXL, hidden 768, patch 16, 256 patches) + stage1/imagenet/dinov2b-k1/decoder.pt
             -> 256x256 RGB (pixel space [0,1]; official sample.py clamps for display).
  stats    = stage1/imagenet/dinov2b-k1/stats.pt (mean, var (768,16,16)); RAE.encode normalises z = (raw-mean)/sqrt(var+1e-5),
             RAE.decode un-normalises first, so decoder(raw) == RAE.decode(normalise(raw)) (checked in sanity).

Geometry (all coordinates in the 256 RGB frame = RAE resolution; one encoder token (14 px at 224) == one 16x16 cell at 256):
  Letterbox(w, h) at S=256 exactly as dinov2_oracle_region_decoder/data/preprocessing (s = S/max(w,h), round, centred),
  PIL BICUBIC resize, padding = ImageNet mean colour (normalised value ~0, as the existing pipeline's 0-padding).
  Full image and GT crop use the same function; the encoder then applies the official 256 -> 224 bicubic.
  roi_geometry = the existing function with S=256, PATCH=16, G=16 (teacher grid = 16x16 crop-canvas cells).
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(HERE, "code")
CACHE = os.path.join(HERE, "perception_cache")
RES = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
RAEV2 = os.path.join(HERE, "third_party", "RAEv2")
MODELS = os.path.join(CACHE, "raev2_models", "stage1", "imagenet", "dinov2b-k1")
DEC_CFG = os.path.join(RAEV2, "configs", "decoder", "ViTXL")
ENC_NAME = "dinov2-vit-b"
RESOLUTION = 256
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE, "torch_hub"))
sys.path.insert(0, os.path.join(RAEV2, "src"))
sys.path.insert(0, os.path.join(RAEV2, "src", "stage1"))     # 'decoders' package without stage1/__init__ (vae deps)

S = 256
PATCH = 16
G = S // PATCH          # 16
NCELL = G * G           # 256
D = 768
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
PAD_RGB = tuple(int(round(255 * m)) for m in (0.485, 0.456, 0.406))   # (124, 116, 103)

# ROI lists rebuilt by code/make_rois.py (originals: dinov2_oracle_region_decoder coco_rois.parquet, rae_crop_pilot configs/rois.parquet)
COCO_ROIS = os.path.join(CACHE, "data", "coco_rois.parquet")
COCO_TRAIN = os.environ.get("COCO_TRAIN", "/home/irteam/datasets/coco/train2017")   # was /dataset/Cambrian-10M/coco/train2017
RAE_ROIS = os.path.join(CACHE, "data", "rae_rois.parquet")
STREAM = os.path.join(CACHE, "stream", "stream_ids_12000.npy")   # COCO image ids (12000,16); make_stream.py (PACO val/test excluded)
GEOM_KEYS = ("tok_idx", "inside", "coords", "n_in", "n_ctx", "valid_w", "full_cov", "bil_grid")


def set_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ------------------------------------------------------------------------------------------------ geometry (S=256)
class Letterbox:
    def __init__(self, w, h, side=S):
        self.w, self.h, self.side = int(w), int(h), side
        s = side / max(self.w, self.h)
        self.nw = min(side, max(1, int(round(self.w * s))))
        self.nh = min(side, max(1, int(round(self.h * s))))
        self.ox, self.oy = (side - self.nw) // 2, (side - self.nh) // 2
        self.sx, self.sy = self.nw / self.w, self.nh / self.h

    def to_canvas(self, x, y):
        return self.ox + np.asarray(x, np.float64) * self.sx, self.oy + np.asarray(y, np.float64) * self.sy

    def from_canvas(self, X, Y):
        return (np.asarray(X, np.float64) - self.ox) / self.sx, (np.asarray(Y, np.float64) - self.oy) / self.sy

    def coverage(self):
        e = np.arange(G + 1) * PATCH
        cx = np.clip(np.minimum(e[1:], self.ox + self.nw) - np.maximum(e[:-1], self.ox), 0, None) / PATCH
        cy = np.clip(np.minimum(e[1:], self.oy + self.nh) - np.maximum(e[:-1], self.oy), 0, None) / PATCH
        return (cy[:, None] * cx[None, :]).reshape(-1).astype(np.float32)

    def rect(self):
        return (self.ox, self.oy, self.nw, self.nh)


def u8_canvas(img, lb: Letterbox):
    """PIL image (lb.w x lb.h) -> (3,S,S) uint8 letterbox canvas, BICUBIC, padding = ImageNet mean colour."""
    img = img.convert("RGB")
    assert img.size == (lb.w, lb.h), (img.size, lb.w, lb.h)
    r = np.asarray(img.resize((lb.nw, lb.nh), resample=Image.BICUBIC), dtype=np.uint8)
    x = np.empty((lb.side, lb.side, 3), np.uint8)
    x[:] = PAD_RGB
    x[lb.oy:lb.oy + lb.nh, lb.ox:lb.ox + lb.nw] = r
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def content_mask(rects, dev, side=S):
    """(n,4) [ox, oy, nw, nh] -> (n,1,side,side) float content mask."""
    rects = torch.as_tensor(np.asarray(rects), device=dev)
    a = torch.arange(side, device=dev)
    ox, oy, nw, nh = (rects[:, k, None] for k in range(4))
    mx = (a[None] >= ox) & (a[None] < ox + nw)
    my = (a[None] >= oy) & (a[None] < oy + nh)
    return (my[:, :, None] & mx[:, None, :])[:, None].float()


def roi_geometry(W, H, ltrb):
    """= dinov2_oracle_region_decoder data.preprocessing.roi_geometry with S=256, PATCH=16, G=16 (code identical)."""
    l, t, r, b = map(int, ltrb)
    if not (0 <= l < r <= W and 0 <= t < b <= H):
        raise ValueError(f"invalid ROI {ltrb} for image {W}x{H}")
    full, crop = Letterbox(W, H), Letterbox(r - l, b - t)
    X0, Y0 = full.to_canvas(l, t)
    X1, Y1 = full.to_canvas(r, b)
    j0, j1 = int(math.floor(X0 / PATCH)), int(math.ceil(X1 / PATCH)) - 1
    i0, i1 = int(math.floor(Y0 / PATCH)), int(math.ceil(Y1 / PATCH)) - 1
    j0, i0, j1, i1 = max(j0, 0), max(i0, 0), min(j1, G - 1), min(i1, G - 1)
    if j1 < j0 or i1 < i0:
        raise ValueError(f"ROI {ltrb} has no token in the full grid")
    n_in = (i1 - i0 + 1) * (j1 - j0 + 1)
    ci0, ci1, cj0, cj1 = max(i0 - 1, 0), min(i1 + 1, G - 1), max(j0 - 1, 0), min(j1 + 1, G - 1)
    ii, jj = np.meshgrid(np.arange(ci0, ci1 + 1), np.arange(cj0, cj1 + 1), indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    inside = (ii >= i0) & (ii <= i1) & (jj >= j0) & (jj <= j1)
    xo, yo = full.from_canvas((jj + 0.5) * PATCH, (ii + 0.5) * PATCH)
    Xc, Yc = crop.to_canvas(xo - l, yo - t)
    pw = PATCH / full.sx * crop.sx / S
    ph = PATCH / full.sy * crop.sy / S
    coords = np.stack([Xc / S, Yc / S, np.full_like(Xc, pw), np.full_like(Yc, ph)], -1).astype(np.float32)
    ci, cj = np.divmod(np.arange(NCELL), G)
    xo2, yo2 = crop.from_canvas((cj + 0.5) * PATCH, (ci + 0.5) * PATCH)
    Xf, Yf = full.to_canvas(xo2 + l, yo2 + t)
    grid = np.stack([Xf / (S / 2) - 1, Yf / (S / 2) - 1], -1).astype(np.float32)
    return dict(ltrb=(l, t, r, b), W=W, H=H, full=full, crop=crop,
                tok_idx=(ii * G + jj).astype(np.int64), inside=inside, coords=coords,
                n_in=int(n_in), n_ctx=int(len(ii)), sub=(i0, i1, j0, j1),
                valid_w=crop.coverage(), full_cov=full.coverage(), bil_grid=grid)


def query_coords():
    ci, cj = np.divmod(np.arange(NCELL), G)
    return np.stack([(cj + 0.5) / G, (ci + 0.5) / G], -1).astype(np.float32)


def collate_geoms(geoms, img_index):
    """= models/roi_reader.collate_geoms (grid-size generic)."""
    R = len(geoms)
    n = max(g["n_ctx"] for g in geoms)
    tok = np.zeros((R, n), np.int64)
    coords = np.zeros((R, n, 4), np.float32)
    pad = np.ones((R, n), bool)
    inside = np.zeros((R, n), bool)
    for k, g in enumerate(geoms):
        m = g["n_ctx"]
        tok[k, :m], coords[k, :m], pad[k, :m], inside[k, :m] = g["tok_idx"], g["coords"], False, g["inside"]
    t = torch.from_numpy
    return dict(img_index=torch.as_tensor(img_index, dtype=torch.long), tok=t(tok), coords=t(coords), pad=t(pad), inside=t(inside),
                valid_w=t(np.stack([g["valid_w"] for g in geoms])), full_cov=t(np.stack([g["full_cov"] for g in geoms])),
                bil_grid=t(np.stack([g["bil_grid"] for g in geoms])),
                n_in=torch.as_tensor([g["n_in"] for g in geoms]), n_ctx=torch.as_tensor([g["n_ctx"] for g in geoms]))


def to_device(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def gather_roi(full_tokens, b):
    F_r = full_tokens[b["img_index"][:, None], b["tok"]]
    return F_r.masked_fill(b["pad"][..., None], 0.0)


def bilinear(full_tokens, b):
    """= models/roi_reader.bilinear with G=16: full 16x16 map sampled at the teacher cell centres (align_corners=False, border)."""
    maps = full_tokens.transpose(1, 2).reshape(full_tokens.shape[0], -1, G, G)[b["img_index"]]
    out = F.grid_sample(maps.float(), b["bil_grid"].view(-1, G, G, 2), mode="bilinear", padding_mode="border", align_corners=False)
    return out.flatten(2).transpose(1, 2)


# ------------------------------------------------------------------------------------------------ official RAEv2 pair
def official_encoder(dev, name=ENC_NAME):
    from encoders.vision_encoder import create_encoder
    enc = create_encoder(name, device=torch.device(dev), resolution=RESOLUTION)
    for p in enc.model.parameters():
        p.requires_grad_(False)
    enc.model.eval()
    return enc


def official_decoder(dev, path=None):
    """= stage1/rae.py _load_decoder (copied: importing stage1 pulls in VAE deps), but strict about keys."""
    from transformers import AutoConfig
    from decoders import GeneralDecoder
    cfg = AutoConfig.from_pretrained(DEC_CFG)
    cfg.hidden_size, cfg.patch_size = D, 16
    cfg.image_size = int(16 * math.sqrt(NCELL))
    dec = GeneralDecoder(cfg, num_patches=NCELL)
    sd = torch.load(path or os.path.join(MODELS, "decoder.pt"), map_location="cpu", weights_only=False)
    keys = dec.load_state_dict(sd, strict=False)
    assert not keys.missing_keys and not keys.unexpected_keys, keys
    dec = dec.to(dev).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


def load_stats(dev):
    s = torch.load(os.path.join(MODELS, "stats.pt"), map_location="cpu", weights_only=False)
    return s["mean"].to(dev), s["var"].to(dev)


class FrozenRAE(torch.nn.Module):
    """encode(u8 canvases (B,3,256,256)) -> raw latent (B,256,768) fp32 [official preprocess + forward_features];
    decode(raw (B,256,768)) -> (B,3,256,256) pixel values (unclamped) = official decoder(...).logits -> unpatchify."""

    def __init__(self, dev="cuda", decoder=True):
        super().__init__()
        self.enc = official_encoder(dev)
        self.dec = official_decoder(dev) if decoder else None
        self.mean, self.var = load_stats(dev)
        self.dev = dev

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, x_u8, chunk=128):
        outs = []
        for i in range(0, len(x_u8), chunk):
            x = x_u8[i:i + chunk].to(self.dev, non_blocking=True).float()
            outs.append(self.enc(x).float())        # official VisionEncoder.forward: preprocess + x_norm_patchtokens
        return torch.cat(outs)

    def decode(self, raw, chunk=0):
        """Gradient flows to `raw` (decoder params are frozen, requires_grad False). Call under autocast for BF16."""
        if chunk and len(raw) > chunk:
            return torch.cat([self.decode(raw[i:i + chunk]) for i in range(0, len(raw), chunk)])
        out = self.dec(raw, drop_cls_token=False).logits
        return self.dec.unpatchify(out)

    def normalise(self, raw):
        """raw (B,256,768) -> official normalised latent (B,768,16,16)."""
        z = raw.transpose(1, 2).reshape(raw.shape[0], D, G, G)
        return (z - self.mean) / torch.sqrt(self.var + 1e-5)


def param_hash(module):
    s = [t.detach().double() for t in list(module.parameters()) + list(module.buffers())]
    return float(sum(x.sum() for x in s)), float(sum((x * x).sum() for x in s))
