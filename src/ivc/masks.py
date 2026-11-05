"""PACO instance / part masks of every VAL ROI on the C0 crop token grid (CPU).

For ROI row r (ann_id = PACO annotation id, object or part): polygon/RLE segmentation -> binary mask (H, W) (pycocotools)
-> crop with the integer ROI box (l, t, r, b) -> resize to the C0 letterbox content (nw, nh) with PIL BOX (area average)
-> paste on the 336 canvas exactly where the C0 crop content sits (ox, oy) -> per 14 x 14 cell:
  frac[r, i] = (mask area in cell) / (content area in cell)   (NaN where the cell has no content = pure padding)
Region rule (fixed before looking at any result): over cells with C0 coverage > 0,
  interior = frac >= 0.9, boundary = 0.1 < frac < 0.9, background (inside the ROI box, outside the mask) = frac <= 0.1.

  python -m src.ivc.masks --out output/2026-11-05/ivc/features
"""
import argparse
import json
import os
from multiprocessing import get_context

import numpy as np
from PIL import Image

import src.readout.common as RC
from src.config import PACO_DIR
from src.ivc.common import EXCL
from src.msd.common import roi_geometry

G_ = {}
S, P, G = 336, 14, 24


def _job(r):
    ann = G_["anns"].get(int(G_["ann_id"][r]))
    W, H = int(G_["W"][r]), int(G_["H"][r])
    ltrb = tuple(int(x) for x in G_["box"][r])
    if ann is None or not ann.get("segmentation"):
        return r, None, "no segmentation"
    from pycocotools import mask as mu
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rle = mu.merge(mu.frPyObjects(seg, H, W))
    elif isinstance(seg["counts"], list):
        rle = mu.frPyObjects(seg, H, W)
    else:
        rle = seg
    m = mu.decode(rle).astype(np.uint8)
    if m.ndim == 3:
        m = m.max(2)
    g = roi_geometry(W, H, ltrb)
    cl = g["crop"]
    l, t, rr, bb = ltrb
    mc = Image.fromarray(m[t:bb, l:rr] * 255).resize((cl.nw, cl.nh), resample=Image.BOX)
    canvas = np.zeros((S, S), np.float64)
    content = np.zeros((S, S), np.float64)
    canvas[cl.oy:cl.oy + cl.nh, cl.ox:cl.ox + cl.nw] = np.asarray(mc, np.float64) / 255.0
    content[cl.oy:cl.oy + cl.nh, cl.ox:cl.ox + cl.nw] = 1.0
    ms = canvas.reshape(G, P, G, P).sum((1, 3)).reshape(-1)
    cs = content.reshape(G, P, G, P).sum((1, 3)).reshape(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(cs > 0, ms / np.maximum(cs, 1e-9), np.nan)
    inside_box = float(m[t:bb, l:rr].mean())
    return r, frac.astype(np.float32), f"ok {inside_box:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dva, _ = RC.load_data(EXCL, splits=("val",))["val"]
    js = json.load(open(os.path.join(PACO_DIR, "paco_lvis_v1_val.json")))
    G_["anns"] = {an["id"]: an for an in js["annotations"]}
    G_.update(ann_id=dva.ann_id.values, W=dva.W.values, H=dva.H.values, box=dva[["l", "t", "r", "b"]].values)
    imgs = {im["id"]: im for im in js["images"]}
    # the manifest image size must equal the annotation's image size (else the mask would be misaligned)
    bad = [int(i) for i, w, h in zip(dva.image_id.values, dva.W.values, dva.H.values) if (imgs[int(i)]["width"], imgs[int(i)]["height"]) != (int(w), int(h))]
    assert not bad, f"image size mismatch for {len(bad)} ROIs"
    N = len(dva)
    frac = np.full((N, 576), np.nan, np.float32)
    status, fill = [""] * N, np.full(N, np.nan, np.float32)
    with get_context("fork").Pool(a.workers) as pool:
        for r, f, st in pool.imap_unordered(_job, range(N), chunksize=32):
            status[r] = st
            if f is not None:
                frac[r] = f
                fill[r] = float(st.split()[1])
    np.savez(os.path.join(a.out, "mask_frac.npz"), ann_id=dva.ann_id.values, frac=frac, box_fill=fill)
    ok = np.array([s.startswith("ok") for s in status])
    json.dump(dict(n=N, n_ok=int(ok.sum()), missing=[int(x) for x in dva.ann_id.values[~ok]][:50],
                   box_fill_median={lv: float(np.nanmedian(fill[(dva.level == lv).values])) for lv in ("object", "part")},
                   rule="interior frac>=0.9, boundary 0.1<frac<0.9, background frac<=0.1, over cells with C0 coverage>0"),
              open(os.path.join(a.out, "mask_done.json"), "w"), indent=1)
    print("masks ok", ok.sum(), "/", N)


if __name__ == "__main__":
    main()
