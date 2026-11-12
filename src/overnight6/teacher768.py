"""Native Crop768 teacher cache for E1a / E1b (GT ROI cropped from the ORIGINAL COCO RGB, letterboxed on a 768 x 768 canvas, frozen DINOv3-B/16, 48 x 48 tokens).

Same ROIs, order and crop rule as src.hcf.rois (C0 Native Crop384), only the canvas differs: LB(w, h, 768, 16). Not the Full384 crop (C1) and not the full-image 768 of G1-Native.

  t_object.npy (N, 4, 768), t_nested.npy (12000, 16, 4, 768; NaN = absent): content-coverage-weighted mean of the 48 x 48 tokens (the existing pooled-teacher rule, on 48 cells)
  sp_cells.npy (M, 768) float16 + sp_index_object.npy (N, 4, 3), sp_index_nested.npy (12000, 16, 4, 3): [offset, gw, gh], offset -1 = ROI gets no spatial loss

Spatial grid of a ROI, fixed before training:
  the ROI's size on the student's Full384 canvas in Fine cells (8 px): gw0 = round(w_canvas / 8), gh0 = round(h_canvas / 8).
  Spatial loss only when gw0 >= 3 and gh0 >= 3 (smaller ROIs do not reference enough distinct Fine tokens). Grid = (min(gw0, 12), min(gh0, 12)), so a cell is never smaller than one Fine token.
  Teacher cell = area-weighted mean of the 48 x 48 tokens over the matching sub-rectangle of the letterbox CONTENT, so padding cells never enter; student cell = the same sub-rectangle of the ROI box on the 384 canvas.

  python -m src.overnight6.teacher768 alloc --out <dir> [--only-steps 3]
  python -m src.overnight6.teacher768 cache --out <dir> [--only-steps 3]
  python -m src.overnight6.teacher768 verify --out <dir> --report <json>
"""
import argparse
import json
import os
import queue
import threading
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import COCO_TRAIN, STREAM
from src.hcf.model import load_vit
from src.viz.c0_pca import LB

SRC = "output/2026-11-08/hcf_pilot/shared"
LINKS = ("image_ids.npy", "object_boxes.npy", "nested_boxes.npy", "nested_nadj.npy", "g_cls.npy", "g_mean.npy", "manifest.json")
ANN = "/home/irteam/datasets/coco/annotations/instances_train2017.json"
S_T, P, G_T, CAP, MIN_CELLS, FINE_CELL, NEST = 768, 16, 48, 12, 3, 8, 4


def round_half(x):
    return int(np.floor(x + .5))


def roi_grid(box, W, H):
    """Spatial grid (gw, gh, eligible) of a ROI from its size on the student's 384 canvas."""
    fl = LB(W, H, 384, 16)
    l, t, r, b = box
    gw0, gh0 = round_half((r - l) * fl.nw / W / FINE_CELL), round_half((b - t) * fl.nh / H / FINE_CELL)
    return min(gw0, CAP), min(gh0, CAP), bool(gw0 >= MIN_CELLS and gh0 >= MIN_CELLS)


def cov48(lb):
    e = np.arange(G_T + 1) * P
    cx = np.clip(np.minimum(e[1:], lb.ox + lb.nw) - np.maximum(e[:-1], lb.ox), 0, None) / P
    cy = np.clip(np.minimum(e[1:], lb.oy + lb.nh) - np.maximum(e[:-1], lb.oy), 0, None) / P
    return (cy[:, None] * cx[None, :]).reshape(-1).astype(np.float32)


def overlap(edges, n, cell):
    """edges (g + 1) px -> (g, n) overlap of each target interval with each grid cell, in cell units."""
    e = torch.arange(n + 1, device=edges.device, dtype=edges.dtype) * cell
    return (torch.minimum(e[1:][None], edges[1:][:, None]) - torch.maximum(e[:-1][None], edges[:-1][:, None])).clamp_min(0) / cell


def pool_grid(m, Wy, Wx):
    """m (H, W, D), Wy (gh, H), Wx (gw, W) -> (gh, gw, D) area-weighted mean."""
    x = torch.einsum("jy,yxd,kx->jkd", Wy.to(m.dtype), m, Wx.to(m.dtype))
    return x / (Wy.sum(1)[:, None] * Wx.sum(1)[None, :]).to(m.dtype).clamp_min(1e-12)[..., None]


def edges(a, b, g, dev):
    return a + (b - a) * torch.arange(g + 1, device=dev, dtype=torch.float32) / g


def image_sizes():
    return {im["id"]: (im["width"], im["height"]) for im in json.load(open(ANN))["images"]}


def visits(ids, stream):
    pos = {int(v): k for k, v in enumerate(ids)}
    out = {}
    for s, row in enumerate(stream):
        for c, iid in enumerate(row):
            out.setdefault(pos[int(iid)], []).append((s, c))
    return out


def alloc(a):
    os.makedirs(a.out, exist_ok=True)
    for name in LINKS:
        link = os.path.join(a.out, name)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(os.path.join(SRC, name)), link)
    ids, obj, NB = np.load(f"{a.out}/image_ids.npy"), np.load(f"{a.out}/object_boxes.npy"), np.load(f"{a.out}/nested_boxes.npy")
    size = image_sizes()
    stream = np.load(STREAM)
    N = len(ids)
    pos = {int(v): k for k, v in enumerate(ids)}
    io, ineg = np.full((N, 4, 3), -1, np.int64), np.full((12000, 16, NEST, 3), -1, np.int64)
    off, hist = 0, {}

    def place(index, box, W, H):
        nonlocal off
        gw, gh, ok = roi_grid(box, W, H)
        hist.setdefault("rois", 0)
        hist["rois"] += 1
        index[1], index[2] = gw, gh
        if ok:
            index[0] = off
            off += gw * gh
            hist["eligible"] = hist.get("eligible", 0) + 1
    for i in range(N):
        W, H = size[int(ids[i])]
        for k in range(4):
            place(io[i, k], obj[i, k], W, H)
    for s in range(12000):
        for c in range(16):
            W, H = size[int(stream[s, c])]
            for j in range(NEST):
                b = NB[s, c, j]
                if b[2] > b[0]:
                    place(ineg[s, c, j], b, W, H)
    np.save(f"{a.out}/sp_index_object.npy", io)
    np.save(f"{a.out}/sp_index_nested.npy", ineg)
    for name, shape in dict(t_object=(N, 4, 768), t_nested=(12000, 16, NEST, 768)).items():
        m = np.lib.format.open_memmap(f"{a.out}/{name}.npy", "w+", np.float32, shape)
        m[...] = np.nan
        m.flush()
    np.lib.format.open_memmap(f"{a.out}/sp_cells.npy", "w+", np.float16, (off, 768)).flush()
    wanted = sorted(set(pos[int(v)] for row in (stream[:a.only_steps] if a.only_steps else stream) for v in row))
    json.dump(dict(cells=int(off), bytes_float16=int(off) * 768 * 2, cap=CAP, min_cells=MIN_CELLS, only_steps=a.only_steps, images=len(wanted), **hist, time=time.strftime("%F %T")), open(f"{a.out}/alloc.json", "w"), indent=1)
    print(open(f"{a.out}/alloc.json").read(), flush=True)


class Crops(Dataset):
    """item = image index -> every crop canvas of that image (4 object ROIs + the nested ROIs of each stream visit)."""

    def __init__(self, ids, obj, NB, stream, vis, io, ineg):
        self.ids, self.obj, self.NB, self.stream, self.vis, self.io, self.ineg = ids, obj, NB, stream, vis, io, ineg

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = int(self.ids[i])
        img = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
        W, H = img.size
        crops, tag = [], []
        for k, b in enumerate(self.obj[i]):
            crops.append((LB(b[2] - b[0], b[3] - b[1], S_T, P), tuple(int(v) for v in b)))
            tag.append((0, k, -1, -1, *self.io[i, k]))
        for s, c in self.vis.get(i, []):
            for j in range(NEST):
                b = self.NB[s, c, j]
                if b[2] <= b[0]:
                    continue
                crops.append((LB(b[2] - b[0], b[3] - b[1], S_T, P), tuple(int(v) for v in b)))
                tag.append((1, j, s, c, *self.ineg[s, c, j]))
        X = torch.stack([lb.canvas(img.crop(box)) for lb, box in crops])
        cov = torch.from_numpy(np.stack([cov48(lb) for lb, _ in crops]))
        rect = torch.tensor([[lb.ox, lb.oy, lb.nw, lb.nh] for lb, _ in crops], dtype=torch.float32)
        return dict(i=i, X=X, cov=cov, rect=rect, tag=torch.tensor(tag, dtype=torch.int64), W=W, H=H)


@torch.no_grad()
def encode(enc0, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = enc0.forward_features(x)[:, enc0.num_prefix_tokens:].float()
    assert t.shape[1] == G_T * G_T, t.shape
    return t


def spatial_cells(tok, rect, gw, gh, dev):
    ox, oy, nw, nh = rect
    ex, ey = edges(ox, ox + nw, gw, dev), edges(oy, oy + nh, gh, dev)
    return pool_grid(tok.reshape(G_T, G_T, -1), overlap(ey, G_T, P), overlap(ex, G_T, P))


def cache(a):
    dev = torch.device("cuda")
    ids, obj, NB = np.load(f"{a.out}/image_ids.npy"), np.load(f"{a.out}/object_boxes.npy"), np.load(f"{a.out}/nested_boxes.npy")
    stream = np.load(STREAM)
    io, ineg = np.load(f"{a.out}/sp_index_object.npy"), np.load(f"{a.out}/sp_index_nested.npy")
    tobj, tnest = np.load(f"{a.out}/t_object.npy", mmap_mode="r+"), np.load(f"{a.out}/t_nested.npy", mmap_mode="r+")
    cells = np.load(f"{a.out}/sp_cells.npy", mmap_mode="r+")
    vis = visits(ids, stream[:a.only_steps] if a.only_steps else stream)
    enc0 = load_vit(dev).eval().requires_grad_(False)
    mine = sorted(vis) if a.only_steps else list(range(len(ids)))
    if a.only_steps:
        ids = ids[:]                                                         # same indices; only visited images are processed
    mine = mine[a.shard::a.nshards]
    ds = Crops(ids, obj, NB, stream, vis, io, ineg)
    dl = DataLoader(ds, sampler=mine, batch_size=None, num_workers=a.workers, prefetch_factor=4, persistent_workers=False)
    buf, n_buf = [], 0                                                       # GPU tensors; one .cpu() per batch, disk writes in their own thread
    t0, done_crops = time.time(), 0
    writes, failure = queue.Queue(maxsize=16), []

    def writer():
        try:
            while True:
                item = writes.get()
                if item is None:
                    return
                G, z, v = item
                at = 0
                for r in range(len(G)):
                    kind, j, s, c, off, gw, gh = (int(x) for x in G[r])
                    if kind == 0:
                        tobj[s, j] = z[r]                                    # object ROIs: column 2 (s) holds the image index
                    else:
                        tnest[s, c, j] = z[r]
                    if off >= 0:
                        cells[off:off + gw * gh] = v[at:at + gw * gh]
                        at += gw * gh
                assert at == len(v)
        except Exception:  # noqa: BLE001
            failure.append(traceback.format_exc())

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()

    def flush(force=False):
        nonlocal buf, n_buf, done_crops
        while n_buf >= a.batch or (force and n_buf):
            assert not failure, failure[0]
            X, C, R, G = (torch.cat([b[k] for b in buf]) for k in range(4))
            m = min(a.batch, len(X))
            T = encode(enc0, X[:m])
            z = (T * C[:m][..., None]).sum(1) / C[:m].sum(1, keepdim=True).clamp_min(1e-12)
            spatial = []
            for r in range(m):
                off, gw, gh = (int(v) for v in G[r, 4:7])
                if off >= 0:
                    spatial.append(spatial_cells(T[r], R[r], gw, gh, dev).reshape(-1, 768))
            v = torch.cat(spatial).cpu().numpy().astype(np.float16) if spatial else np.zeros((0, 768), np.float16)
            writes.put((G[:m].cpu().numpy(), z.cpu().numpy().astype(np.float32), v))
            buf, n_buf = [(X[m:], C[m:], R[m:], G[m:])], len(X) - m
            done_crops += m

    for it, b in enumerate(dl):
        tag = b["tag"].clone()
        tag[tag[:, 0] == 0, 2] = int(b["i"])                                 # object ROIs: column 2 carries the image index (unused -1 before)
        buf.append((b["X"].to(dev, non_blocking=True), b["cov"].to(dev), b["rect"].to(dev), tag.to(dev)))
        n_buf += len(b["X"])
        flush()
        if it % 500 == 0:
            s = time.time() - t0
            print(json.dumps(dict(it=it, n=len(mine), crops=done_crops, s=round(s), cps=round(done_crops / max(s, 1e-6), 1))), flush=True)
    flush(force=True)
    writes.put(None)
    thread.join()
    assert not failure, failure[0]
    tobj.flush(), tnest.flush(), cells.flush()
    print(json.dumps(dict(status="done", crops=done_crops, s=round(time.time() - t0))), flush=True)


@torch.no_grad()
def verify(a):
    dev = torch.device("cuda")
    ids, obj, NB = np.load(f"{a.out}/image_ids.npy"), np.load(f"{a.out}/object_boxes.npy"), np.load(f"{a.out}/nested_boxes.npy")
    stream = np.load(STREAM)
    io, ineg = np.load(f"{a.out}/sp_index_object.npy"), np.load(f"{a.out}/sp_index_nested.npy")
    tobj, tnest, cells = np.load(f"{a.out}/t_object.npy", mmap_mode="r"), np.load(f"{a.out}/t_nested.npy", mmap_mode="r"), np.load(f"{a.out}/sp_cells.npy", mmap_mode="r")
    t384 = np.load(f"{SRC}/t_object.npy", mmap_mode="r")
    enc0 = load_vit(dev).eval().requires_grad_(False)
    pos = {int(v): k for k, v in enumerate(ids)}
    pick = sorted(set(pos[int(v)] for v in stream[:a.only_steps].ravel())) if a.only_steps else sorted(np.random.default_rng(0).choice(len(ids), 100, replace=False).tolist())
    r = dict(res768_tokens=None, cached_vs_live_cos=[], pooled_768_vs_384_cos=[], cells_pool_vs_pooled_cos=[], matched_cell_cos=[], shifted_cell_cos=[], letterbox_token_valid_fraction=[])
    rng = np.random.default_rng(0)
    for i in pick:
        iid = int(ids[i])
        img = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
        W, H = img.size
        fl = LB(W, H, 384, 16)
        full = fl.canvas(img)[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ftok = enc0.forward_features(full)[:, enc0.num_prefix_tokens:].float()[0]
        for k in rng.permutation(4)[:2]:
            b = obj[i, k]
            lb = LB(b[2] - b[0], b[3] - b[1], S_T, P)
            x = lb.canvas(img.crop(tuple(int(v) for v in b)))[None].to(dev)
            T = encode(enc0, x)[0]
            r["res768_tokens"] = int(T.shape[0])
            cov = torch.from_numpy(cov48(lb)).to(dev)
            live = (T * cov[..., None]).sum(0) / cov.sum().clamp_min(1e-12)
            cached = torch.from_numpy(np.array(tobj[i, k])).to(dev)
            assert not cached.isnan().any(), (i, k)
            r["cached_vs_live_cos"].append(float(torch.cosine_similarity(live, cached, dim=0)))
            r["pooled_768_vs_384_cos"].append(float(torch.cosine_similarity(cached, torch.from_numpy(np.array(t384[i, k])).to(dev), dim=0)))
            r["letterbox_token_valid_fraction"].append(float((cov > 0).float().mean()))
            off, gw, gh = (int(v) for v in io[i, k])
            if off < 0:
                continue
            c = torch.from_numpy(np.array(cells[off:off + gw * gh], dtype=np.float32)).to(dev).reshape(gh, gw, 768)
            r["cells_pool_vs_pooled_cos"].append(float(torch.cosine_similarity(c.mean((0, 1)), cached, dim=0)))
            box = torch.tensor([fl.ox + b[0] * fl.nw / W, fl.oy + b[1] * fl.nh / H, fl.ox + b[2] * fl.nw / W, fl.oy + b[3] * fl.nh / H], device=dev, dtype=torch.float32)
            ex, ey = edges(box[0], box[2], gw, dev), edges(box[1], box[3], gh, dev)
            s = pool_grid(ftok.reshape(24, 24, -1), overlap(ey, 24, 16), overlap(ex, 24, 16))
            r["matched_cell_cos"].append(float(torch.cosine_similarity(s, c, dim=-1).mean()))
            r["shifted_cell_cos"].append(float(torch.cosine_similarity(s, c.roll(1, 1), dim=-1).mean()))
    out = dict(res768_tokens=r["res768_tokens"], n_images=len(pick), **{k: float(np.mean(v)) for k, v in r.items() if isinstance(v, list) and v}, min_cached_vs_live_cos=float(np.min(r["cached_vs_live_cos"])))
    if not a.only_steps:
        out["t_object_nan_rows"] = int(np.isnan(tobj).any(-1).sum())
        pres = NB[..., 2] > NB[..., 0]
        out["t_nested_present_nan"] = int(np.isnan(np.asarray(tnest)[pres]).any(-1).sum())
        out["t_nested_absent_all_nan"] = bool(np.isnan(np.asarray(tnest)[~pres]).all())
        el = (np.concatenate([io.reshape(-1, 3), ineg.reshape(-1, 3)])[:, 0] >= 0)
        out["eligible_fraction"] = float(el.sum() / (io.reshape(-1, 3).shape[0] + pres.sum()))
    assert out["min_cached_vs_live_cos"] > .999, out
    assert out["matched_cell_cos"] > out["shifted_cell_cos"], out
    assert out["res768_tokens"] == G_T * G_T
    if not a.only_steps:
        assert out["t_object_nan_rows"] == 0 and out["t_nested_present_nan"] == 0 and out["t_nested_absent_all_nan"], out
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(out, open(a.report, "w"), indent=1)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("alloc", "cache", "verify"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--only-steps", type=int, default=0, help="smoke: only images of the first N stream steps")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    dict(alloc=alloc, cache=cache, verify=verify)[a.cmd](a)


if __name__ == "__main__":
    main()
