"""Multi-depth Crop-to-Shared Distillation (MSD) pilot: data, encoders, losses. H0-H3 share everything here.

Student (inference path, encode once per image):
  full image 336 letterbox -> DINOv2-B/14 (H2/H3: + zero-init residual adapters after blocks 4/8/12, backbone frozen)
  -> block 4 / 8 raw residual tokens, block 12 final-LN tokens -> ROI(+ring) tokens -> ROI decoder -> Z (576 x 768)
Teacher (training supervision only): original-pixel GT crop -> 336 letterbox (C0) -> frozen DINOv2 -> block 4/8/12 tokens.
Geometry / crops / letterbox / ROI+ring token selection = dinov2_oracle_region_decoder data.preprocessing.roi_geometry
and models.roi_reader.collate_geoms (unchanged); flip = image hflip + C.flip_box, teacher crop taken from the SAME
flipped image. Batches: image packs of ~64 ROIs (C.full_epoch_batches target 64, seed 0), 2 GPUs x 2 shards.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

import src.readout.common as RC
from src.config import RESEARCH_ROOT

C = RC.C
MLR_CODE = os.path.join(RESEARCH_ROOT, "2026-10-26", "multilevel_relation_decoder", "code")
sys.path.insert(0, MLR_CODE)
import mlr_common as MC  # noqa: E402  (data.preprocessing / models.roi_reader of dinov2_oracle_region_decoder)
import models_mlr as MM  # noqa: E402  (CoordMLP, Adapter: verbatim existing decoder pieces)
from data.preprocessing import G, PATCH, S, Letterbox, query_coords, roi_geometry, to_tensor_canvas  # noqa: E402
from models.roi_reader import collate_geoms, gather_roi  # noqa: E402

DEPTHS = (4, 8, 12)
D = 768
GEOM_KEYS = ("tok_idx", "inside", "coords", "n_in", "n_ctx", "valid_w", "full_cov", "bil_grid")


# ------------------------------------------------------------------------------------------------ data
class MSDImageDS(Dataset):
    """Item = list of image indices (one rank's share of a step). Returns full canvases, decoder geometry (ROI+ring
    tokens, crop-grid coords, valid_w = teacher-cell content coverage), optional C0 teacher crops, and the crop-grid
    R2 geometry (geo_c / box4_c / cov_c, C0 convention) used by the frozen-decoder R2 probe."""

    def __init__(self, df, Y, flip, want_crops):
        self.groups = C.image_groups(df)
        self.n_roi = np.array([len(g) for g in self.groups])
        self.p, self.box = df.path.values, df[["l", "t", "r", "b"]].values.astype(int)
        self.WH = df[["W", "H"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.Y, self.flip, self.want_crops = torch.as_tensor(Y), flip, want_crops

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, item):
        idx, n_img_global = (item if isinstance(item, tuple) else (item, None))
        full, crops, geoms, img_index, rows_all, gc = [], [], [], [], [], {"geo": [], "box4": [], "cov": []}
        for k, gi in enumerate(idx):
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[rows[0]]), self.p[rows[0]]
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            if f:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            full.append(to_tensor_canvas(img, Letterbox(W, H)))
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                g = roi_geometry(W, H, bx)
                geoms.append({kk: g[kk] for kk in GEOM_KEYS})
                if self.want_crops:
                    crops.append(to_tensor_canvas(img.crop(g["ltrb"]), g["crop"]))
                geo, box4, cov = RC.crop_canvas_geo(g["crop"])
                gc["geo"].append(geo)
                gc["box4"].append(box4)
                gc["cov"].append(cov)
                img_index.append(k)
            rows_all.append(rows)
        rows_all = np.concatenate(rows_all)
        out = dict(full=torch.stack(full), geom=collate_geoms(geoms, img_index), rows=torch.from_numpy(rows_all),
                   lv=torch.from_numpy(self.lv[rows_all]), Y=self.Y[rows_all],
                   geo_c=torch.from_numpy(np.stack(gc["geo"])), box4_c=torch.from_numpy(np.stack(gc["box4"])),
                   cov_c=torch.from_numpy(np.stack(gc["cov"])))
        if crops:
            out["crops"] = torch.stack(crops)
        if n_img_global is not None:
            out["n_img_global"] = torch.tensor(float(n_img_global))
        return out


class CocoStreamDS(Dataset):
    """COCO decoder pre-training (existing RAEv2 / R00 pipeline, attribute labels never touched): item = list of COCO
    train2017 image ids; every image brings its 4 ROIs of cache/feat/stream_rois.parquet (object boxes, random-filled
    when an image has < 4 objects, as the existing coco_rois pool). No flip (existing recipe). -> full canvases,
    decoder geometry, C0 teacher crops (original pixels)."""

    def __init__(self, rois, image_dir):
        self.by = {int(i): g.sort_values("k") for i, g in rois.groupby("image_id")}
        self.dir = image_dir

    def __getitem__(self, item):
        ids, n_img_global = (item if isinstance(item, tuple) else (item, None))
        full, crops, geoms, img_index, kinds = [], [], [], [], []
        for k, iid in enumerate(ids):
            g = self.by[int(iid)]
            img = Image.open(os.path.join(self.dir, f"{int(iid):012d}.jpg")).convert("RGB")
            W, H = img.size
            r0 = g.iloc[0]
            assert (W, H) == (int(r0.W), int(r0.H)), iid
            full.append(to_tensor_canvas(img, Letterbox(W, H)))
            for r in g.itertuples():
                gg = roi_geometry(W, H, (int(r.l), int(r.t), int(r.r), int(r.b)))
                geoms.append({kk: gg[kk] for kk in GEOM_KEYS})
                crops.append(to_tensor_canvas(img.crop(gg["ltrb"]), gg["crop"]))
                img_index.append(k)
                kinds.append(r.kind == "gt")
        out = dict(full=torch.stack(full), geom=collate_geoms(geoms, img_index), crops=torch.stack(crops),
                   is_gt=torch.as_tensor(kinds), n_in=torch.as_tensor([gg["n_in"] for gg in geoms]))
        if n_img_global is not None:
            out["n_img_global"] = torch.tensor(float(n_img_global))
        return out

    def __len__(self):
        return len(self.by)


class StreamSampler:
    """Existing 12k stream (cache/stream/stream_ids_12000.npy, (12000, 16) COCO image ids, seed 2027) minus excluded
    images; rank r of 2 takes stream columns [8r, 8r + 8) of every step (same 64-ROI optimizer step as 1 GPU)."""

    def __init__(self, stream, rank, world, exclude=(), start=0):
        self.stream, self.rank, self.world, self.ex, self.start = stream, rank, world, set(int(x) for x in exclude), start

    def __iter__(self):
        k = self.stream.shape[1] // self.world
        for row in self.stream[self.start:]:
            keep = [int(i) for i in row if int(i) not in self.ex]
            mine = [int(i) for i in row[self.rank * k:(self.rank + 1) * k] if int(i) not in self.ex]
            yield (mine, len(keep))

    def __len__(self):
        return len(self.stream) - self.start


class StepSampler:
    """Global steps = C.full_epoch_batches(n_roi, epoch, seed, target=64, shards=4) over epochs 1, 2, ...; rank r takes
    shards [2r, 2r+2) (world 2) -> one item per rank per step, plus the global image count of the step."""

    def __init__(self, n_roi, rank, world, steps, seed=0, target=64, start=0):
        self.n_roi, self.rank, self.world, self.steps, self.seed, self.target, self.start = n_roi, rank, world, steps, seed, target, start

    def all_steps(self):
        out, ep = [], 1
        while len(out) < self.steps:
            out += C.full_epoch_batches(self.n_roi, ep, self.seed, target=self.target, shards=4)
            ep += 1
        return out[:self.steps]

    def __iter__(self):
        k = 4 // self.world
        for st in self.all_steps()[self.start:]:
            mine = [i for sh in st[self.rank * k:(self.rank + 1) * k] for i in sh]
            yield (mine, sum(len(sh) for sh in st))

    def __len__(self):
        return self.steps - self.start


def collate_one(items):
    assert len(items) == 1
    return items[0]


def to_dev(b, dev):
    out = {}
    for k, v in b.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.to(dev, non_blocking=True) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=True)
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------------------------------------ encoders
def enc_adapter():
    """Residual adapter on the encoder residual stream (= models_mlr.Adapter: LN -> 768x192 -> GELU -> 192x768, last zero)."""
    return MM.Adapter()


class Student(nn.Module):
    """DINOv2-B/14 (timm, frozen weights) + optional residual adapters after blocks 4/8/12 (zero-init -> identity at start)."""

    def __init__(self, adapters):
        super().__init__()
        self.vit = RC.load_vit()
        self.adapters = nn.ModuleDict({str(d): enc_adapter() for d in DEPTHS}) if adapters else None

    def forward(self, x, use_adapters=True):
        """-> dict {4: (B,576,D) raw, 8: raw, 12: final-LN}, cls (B,D) final-LN CLS. bf16 autocast."""
        vit = self.vit
        npt = vit.num_prefix_tokens
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x))))
            out = {}
            for i, blk in enumerate(vit.blocks, 1):
                h = blk(h)
                if self.adapters is not None and use_adapters and str(i) in self.adapters:
                    h = h + self.adapters[str(i)](h)
                if i in (4, 8):
                    out[i] = h[:, npt:]
            hn = vit.norm(h)
            out[12] = hn[:, npt:]
        return {k: v.float() for k, v in out.items()}, hn[:, 0].float()


@torch.no_grad()
def teacher_feats(vit, crops, chunk=48):
    """Frozen original DINOv2 on C0 crops -> {4, 8, 12} (n,576,D) (same layer definition as the student, no adapters)."""
    parts = [RC.encode_layers(vit, crops[i:i + chunk], DEPTHS) for i in range(0, len(crops), chunk)]
    return {d: torch.cat([p[d] for p in parts]) for d in DEPTHS}


# ------------------------------------------------------------------------------------------------ decoder
class MSDDecoder(nn.Module):
    """Identical for H0-H3. Input F = F12 + A4(F4) + A8(F8) on the ROI(+ring) tokens (R10 MLRDecoder rule extended to two
    lower depths; A = models_mlr.Adapter, zero-init last layer) -> existing CoordMLP body -> Z (R,576,768) in block-12
    space (body output * std12 + mean12). Auxiliary heads (removed after training): z_hat_d = Linear_d((Z - mu12)/sd12)
    predicts the normalised teacher block d (d = 4, 8); both heads exist in every condition."""

    def __init__(self):
        super().__init__()
        self.in_adapt = nn.ModuleDict({"4": MM.Adapter(), "8": MM.Adapter()})
        self.body = MM.CoordMLP()
        self.aux = nn.ModuleDict({"4": nn.Linear(D, D), "8": nn.Linear(D, D)})
        self.register_buffer("P", query_coords_t())

    def forward(self, feats, g):
        pad = g["pad"][..., None]
        F12 = gather_roi(feats[12], g)
        Fin = F12 + sum(self.in_adapt[str(d)](gather_roi(feats[d], g)).masked_fill(pad, 0.0) for d in (4, 8))
        return self.body(Fin, g, self.P)

    def aux_pred(self, Zn):
        return {d: self.aux[str(d)](Zn) for d in (4, 8)}

    def set_stats(self, mu, sd):
        self.body.set_stats(mu, sd)

    def trunk_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("aux.")]


def build_decoder(arch="coordmlp"):
    if arch == "coordmlp":
        return MSDDecoder()
    import src.msd.sdec as SD
    return SD.build(arch)


def load_decoder(ck, dev):
    dec = build_decoder(ck.get("arch", "coordmlp")).to(dev)
    dec.load_state_dict(ck["decoder"])
    return dec.eval()


def query_coords_t():
    return torch.as_tensor(query_coords())


# ------------------------------------------------------------------------------------------------ losses
def feat_loss(pred_n, targ_n, w):
    """Per ROI-cell L_feat = mean_c (p - t)^2 + (1 - cos(p, t)) on normalised features, weighted by teacher-cell content
    coverage w (padding cells w = 0). -> (weighted sum, weight sum)."""
    mse = (pred_n - targ_n).pow(2).mean(-1)
    cos = F.cosine_similarity(pred_n, targ_n, dim=-1)
    l = mse + (1 - cos)
    return (l * w).sum(), w.sum()


def global_loss(cls_s, gap_s, cls_o, gap_o):
    """H2/H3 global preservation vs the frozen original encoder: 1/2 [(1 - cos CLS) + (1 - cos mean patch token)], per image (sum)."""
    return 0.5 * ((1 - F.cosine_similarity(cls_s, cls_o, dim=-1)) + (1 - F.cosine_similarity(gap_s, gap_o, dim=-1))).sum()


class Norm:
    """Per-depth per-channel teacher statistics (mu_d, sd_d) over valid teacher cells of a fixed train-ROI sample."""

    def __init__(self, path, dev):
        z = np.load(path)
        self.mu = {d: torch.as_tensor(z[f"mean_{d}"], device=dev) for d in DEPTHS}
        self.sd = {d: torch.as_tensor(z[f"std_{d}"], device=dev) for d in DEPTHS}

    def n(self, x, d):
        return (x - self.mu[d]) / self.sd[d]
