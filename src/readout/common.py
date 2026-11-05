"""Shared pieces of the shared-feature readout queue (frozen DINOv2-B/14, 336 letterbox, PACO GT ROIs).

Everything that decides comparability is reused unchanged from the earlier projects (third_party/research):
  * letterbox / ROI geometry / box-cell coverage        : paco_full_vs_crop_ft common.py (C), dinov2 data.preprocessing
  * R2 attention reader pieces (CrossBlock, Heads, geo) : paco_region_reader common_rr.py (RR)
  * image-pack batch order (~128 ROIs / step, seed 0)   : C.full_epoch_batches
  * loss = masked BCE / #known labels in the step       : C.masked_bce_sum
  * M2 = PACO category-macro AP, M0, loss              : paco_crop_finetune eval_tables.evaluate (ET)
New here: image-exclusion list from src.readout.data_check, multi-layer encoder output, DDP over GPUs, val-only eval.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset

from src.config import CACHE, PACO_READER_CODE, RESEARCH_ROOT
from src.rae.common import coco_image_path

os.environ.setdefault("HF_HUB_CACHE", os.path.join(CACHE, "hf_hub"))      # timm DINOv2 weights (old code defaults to /dataset)
os.environ["HF_HUB_CACHE"] = os.environ.get("READOUT_HF_HUB", os.path.join(CACHE, "hf_hub"))
sys.path.insert(0, PACO_READER_CODE)
import common_rr as RR  # noqa: E402
C = RR.C
from common import LEVELS, Letterbox, G, PATCH, S, roi_geometry, to_tensor_canvas  # noqa: E402
sys.path.insert(0, os.path.join(RESEARCH_ROOT, "2026-10-28", "paco_crop_finetune", "code"))
import eval_tables as ET  # noqa: E402
sys.path.insert(0, os.path.join(RESEARCH_ROOT, "2026-10-30", "paco_token_upsample", "code"))
import common_tu as TU  # noqa: E402  (F672: canvas_geo, full_box_canvas, upsample = the earlier token-count diagnostic)

D = 768
ML_LAYERS = (4, 8, 12)               # queue baseline: 1-based block outputs 4/8/12; 12 = final-LN output (= forward_features)
SIDE = {"full": 336, "c0": 336, "f672": 672}
NIN_SG = os.path.join(RESEARCH_ROOT, "2026-10-29", "crop_benefit_by_size", "perception_cache", "feat", "n_in_siglip2_{}.npy")


# ------------------------------------------------------------------------------------------------ data
def load_data(exclude_json=None, limit=0, splits=("train", "val")):
    """-> {split: (df, Y)} with local COCO paths; train images listed in exclude_json are dropped (before batching).
    Test is never loaded unless asked for explicitly (the queue never does)."""
    assert "test" not in splits or os.environ.get("READOUT_ALLOW_TEST") == "1", "PACO test is not used by the queue"
    data = {sp: C.load_split(sp) for sp in splits}
    out = {}
    for sp, (df, Y) in data.items():
        df = df.copy()
        df["path"] = df.path.map({p: coco_image_path(p) for p in df.path.unique()})
        if sp == "train" and exclude_json:
            ex = set(json.load(open(exclude_json)))
            keep = ~df.image_id.isin(ex).values
            df, Y = df[keep].reset_index(drop=True), Y[keep]
        if limit:
            ids = df.image_id.unique()[:limit]
            keep = df.image_id.isin(set(ids)).values
            df, Y = df[keep].reset_index(drop=True), Y[keep]
        out[sp] = (df, Y)
    if "train" in out and "val" in out:
        assert not (set(out["train"][0].image_id) & set(out["val"][0].image_id)), "train / val image overlap"
    return out


def crop_canvas_geo(cl: Letterbox):
    """C0 crop-grid geometry (probe.py convention): box = crop content rect on the 336 canvas."""
    X0, Y0, X1, Y1 = cl.ox, cl.oy, cl.ox + cl.nw, cl.oy + cl.nh
    bw, bh = X1 - X0, Y1 - Y0
    ci, cj = np.divmod(np.arange(G * G), G)
    cx, cy = (cj + 0.5) * PATCH, (ci + 0.5) * PATCH
    cov = cl.coverage()
    geo = np.stack([(cx - X0) / bw, (cy - Y0) / bh, cov, np.full(G * G, math.log(bw / PATCH)), np.full(G * G, math.log(bh / PATCH))], 1)
    return geo.astype(np.float32), np.array([X0 / S, Y0 / S, bw / S, bh / S], np.float32), cov


class PacoImageDS(Dataset):
    """One item = one image with all its ROIs (indexable by a LIST of image indices = one rank's share of a step).
    mode 'full': full-image canvas + full-canvas ROI geometry (common_rr.roi_geo, R2 convention)
    mode 'c0'  : one 336-letterbox crop per ROI from the original pixels + crop-grid geometry (C0 convention)
    mode 'f672': the SAME 336 canvas (upsampled x2 bicubic on the GPU, no new pixels) + geometry on the 672 canvas
                 (common_tu.canvas_geo / full_box_canvas, = the earlier F672 condition)"""

    def __init__(self, df, Y, flip, mode="full"):
        self.groups = C.image_groups(df)
        self.p, self.box = df.path.values, df[["l", "t", "r", "b"]].values.astype(int)
        self.WH = df[["W", "H"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.Y, self.flip, self.mode = torch.as_tensor(Y), flip, mode
        self.n_roi = np.array([len(g) for g in self.groups])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        xs, per, img_index = [], {k: [] for k in ("cov", "geo", "box4", "box_ltrb", "WH")}, []
        rows_all = []
        for k, gi in enumerate(idx):
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[rows[0]]), self.p[rows[0]]
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            if f:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lb = Letterbox(W, H)
            if self.mode in ("full", "f672"):
                xs.append(to_tensor_canvas(img, lb))
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                if self.mode == "full":
                    geo, box4, cov = RR.roi_geo(lb, bx)
                elif self.mode == "f672":
                    geo, box4, cov = TU.canvas_geo(672, *TU.full_box_canvas(lb, bx, "f672"))
                else:
                    g = roi_geometry(W, H, bx)
                    xs.append(to_tensor_canvas(img.crop(g["ltrb"]), g["crop"]))
                    geo, box4, cov = crop_canvas_geo(g["crop"])
                per["geo"].append(geo)
                per["box4"].append(box4)
                per["cov"].append(cov)
                per["box_ltrb"].append(np.array(bx, np.float32))
                per["WH"].append(np.array([W, H], np.float32))
                img_index.append(len(xs) - 1)
            rows_all.append(rows)
        rows_all = np.concatenate(rows_all)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items()}
        out.update(side=SIDE[self.mode], x=torch.stack(xs), img_index=torch.as_tensor(img_index, dtype=torch.long), rows=torch.from_numpy(rows_all),
                   lv=torch.from_numpy(self.lv[rows_all]), Y=self.Y[rows_all],
                   group=torch.as_tensor(np.concatenate([np.full(len(self.groups[gi]), k) for k, gi in enumerate(idx)]), dtype=torch.long))
        return out


class RankStepSampler:
    """Global steps = C.full_epoch_batches(n_roi, epoch, seed) (~128 ROIs, 4 ROI-balanced shards, identical to the earlier
    reader / Full-FT runs). Rank r of `world` gets shards [r*4/world, (r+1)*4/world) of every step, merged into ONE
    list -> one DataLoader item per rank per step (world must divide 4)."""

    def __init__(self, n_roi, rank, world, seed=0):
        assert 4 % world == 0
        self.n_roi, self.rank, self.world, self.seed, self.epoch = n_roi, rank, world, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def steps(self):
        k = 4 // self.world
        return [[i for sh in st[self.rank * k:(self.rank + 1) * k] for i in sh]
                for st in C.full_epoch_batches(self.n_roi, self.epoch, self.seed)]

    def __iter__(self):
        yield from self.steps()

    def __len__(self):
        return len(self.steps())


class EvalSampler:
    """Images in manifest order, chunked by `bs`, every world-th chunk -> this rank."""

    def __init__(self, n, bs, rank, world):
        self.ch = [list(range(i, min(n, i + bs))) for i in range(0, n, bs)][rank::world]

    def __iter__(self):
        yield from self.ch

    def __len__(self):
        return len(self.ch)


def collate_one(items):
    assert len(items) == 1
    return items[0]


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


# ------------------------------------------------------------------------------------------------ encoder
def load_vit():
    import timm
    vit = timm.create_model(C.TIMM_B, pretrained=True, img_size=336, dynamic_img_size=True)   # = common_tu.load_encoder
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit.eval()


@torch.no_grad()
def encode_layers(vit, x, layers=ML_LAYERS):
    """-> dict {layer: (B, 576, D) float32}; layer 12 = final-LN tokens (identical to vit.forward_features), others raw."""
    npt = vit.num_prefix_tokens
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = vit.patch_embed(x)
        h = vit._pos_embed(h)
        h = vit.patch_drop(h)
        h = vit.norm_pre(h)
        out = {}
        last = len(vit.blocks)
        for i, blk in enumerate(vit.blocks, 1):
            h = blk(h)
            if i in layers and i != last:
                out[i] = h[:, npt:].float()
        if last in layers:
            out[last] = vit.norm(h)[:, npt:].float()
    return out


# ------------------------------------------------------------------------------------------------ DDP helpers
def ddp_setup():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    return rank, world


def all_sum(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t)
    return t


def allreduce_grads(params):
    """Sum gradients over ranks (each rank's loss is already divided by the GLOBAL label count)."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    ps = [p for p in params if p.requires_grad]
    for p in ps:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.reshape(-1) for p in ps])
    dist.all_reduce(flat)
    o = 0
    for p in ps:
        n = p.numel()
        p.grad.copy_(flat[o:o + n].view_as(p))
        o += n


# ------------------------------------------------------------------------------------------------ metrics
def m2_eval(S, df, Y, Ytr_by):
    """ET.evaluate on the full split (val: no decoder-pool mask needed, the queue trains on PACO train only)."""
    try:
        return ET.evaluate([S.astype(np.float32)], df, Y, np.ones(len(df), bool), Ytr_by)
    except ValueError:            # no eligible M2 unit (only happens on tiny smoke-test subsets)
        r = ET.evaluate.__globals__["EM"]  # noqa: F841
        nan = float("nan")
        Sx, out = torch.as_tensor(S), {}
        for lv in LEVELS:
            m = (df.level == lv).values
            y = torch.as_tensor(Y[m])
            k = (y >= 0).float()
            out[f"{lv}_loss"] = float((torch.nn.functional.binary_cross_entropy_with_logits(Sx[m], (y == 1).float(), reduction="none") * k).sum() / k.sum())
        out["loss"] = float(np.mean([out[f"{lv}_loss"] for lv in LEVELS]))
        out.update(M0={"overall": nan}, M2={"overall": nan}, n_orig=0, n_cat=0, n_rows=len(df))
        return out


def ytr_by_level(data):
    dtr, Ytr = data["train"]
    return {lv: Ytr[(dtr.level == lv).values] for lv in LEVELS}
