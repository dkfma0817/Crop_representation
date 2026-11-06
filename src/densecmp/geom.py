"""Letterbox / ROI geometry for an arbitrary canvas side S and patch P (grid G = S / P).

Same rules as the DINOv2 336/14 pipeline (dinov2_oracle_region_decoder data.preprocessing.Letterbox,
paco_full_vs_crop_ft common.box_cell_weights, paco_region_reader common_rr.roi_geo, src.readout.common.crop_canvas_geo),
only S / P / normalisation constants are parameters. With S=336, P=14 and ImageNet mean/std every function here
reproduces the original bit-exactly (checked by src.densecmp.verify geometry).

  Letterbox(w, h, S): s = S / max(w, h); nw = min(S, max(1, round(w s))); content pasted centred at
      (ox, oy) = ((S - nw) // 2, (S - nh) // 2); per-axis scale sx = nw / w (includes rounding).
  canvas tensor: PIL BICUBIC resize to (nw, nh), (x/255 - mean) / std, padding = 0 in normalised space (= mean colour).
  full-grid ROI geometry (R2 'full' mode): cov = |box ∩ cell| / P^2 (cells of the G x G grid), geo = [rel_x, rel_y, cov,
      log(bw/P), log(bh/P)] with box in canvas px, box4 = [X0, Y0, bw, bh] / S.
  crop-grid geometry (C0 mode): box = crop content rect on the crop canvas, cov = letterbox content coverage.
"""
import math

import numpy as np
import torch
from PIL import Image

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
OPENAI_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))   # EVA-CLIP / open_clip constants


class Letterbox:
    def __init__(self, w, h, S):
        self.w, self.h, self.S = int(w), int(h), int(S)
        s = S / max(self.w, self.h)
        self.nw = min(S, max(1, int(round(self.w * s))))
        self.nh = min(S, max(1, int(round(self.h * s))))
        self.ox, self.oy = (S - self.nw) // 2, (S - self.nh) // 2
        self.sx, self.sy = self.nw / self.w, self.nh / self.h

    def to_canvas(self, x, y):
        return self.ox + np.asarray(x, np.float64) * self.sx, self.oy + np.asarray(y, np.float64) * self.sy

    def coverage(self, P):
        G = self.S // P
        e = np.arange(G + 1) * P
        cx = np.clip(np.minimum(e[1:], self.ox + self.nw) - np.maximum(e[:-1], self.ox), 0, None) / P
        cy = np.clip(np.minimum(e[1:], self.oy + self.nh) - np.maximum(e[:-1], self.oy), 0, None) / P
        return (cy[:, None] * cx[None, :]).reshape(-1).astype(np.float32)


def to_tensor_canvas(img: Image.Image, lb: Letterbox, norm=OPENAI_CLIP) -> torch.Tensor:
    img = img.convert("RGB")
    assert img.size == (lb.w, lb.h), (img.size, lb.w, lb.h)
    mean = torch.tensor(norm[0])[:, None, None]
    std = torch.tensor(norm[1])[:, None, None]
    r = np.asarray(img.resize((lb.nw, lb.nh), resample=Image.BICUBIC), dtype=np.float32)
    x = torch.zeros(3, lb.S, lb.S)
    x[:, lb.oy:lb.oy + lb.nh, lb.ox:lb.ox + lb.nw] = (torch.from_numpy(r).permute(2, 0, 1) / 255.0 - mean) / std
    return x


def box_cell_weights(lb: Letterbox, ltrb, P):
    """(G*G,) area of (GT box on the canvas) ∩ (P x P cell) in cell units, row-major."""
    G = lb.S // P
    l, t, r, b = ltrb
    X0, Y0 = lb.to_canvas(l, t)
    X1, Y1 = lb.to_canvas(r, b)
    e = np.arange(G + 1) * P
    cx = np.clip(np.minimum(e[1:], X1) - np.maximum(e[:-1], X0), 0, None) / P
    cy = np.clip(np.minimum(e[1:], Y1) - np.maximum(e[:-1], Y0), 0, None) / P
    return (cy[:, None] * cx[None, :]).reshape(-1).astype(np.float32)


def roi_geo(lb: Letterbox, ltrb, P):
    """Full-grid R2 geometry (= common_rr.roi_geo) -> geo (G*G, 5), box4 (4,), cov (G*G,)."""
    S, G = lb.S, lb.S // P
    l, t, r, b = ltrb
    X0, Y0 = lb.to_canvas(l, t)
    X1, Y1 = lb.to_canvas(r, b)
    bw, bh = X1 - X0, Y1 - Y0
    ci, cj = np.divmod(np.arange(G * G), G)
    cx, cy = (cj + 0.5) * P, (ci + 0.5) * P
    cov = box_cell_weights(lb, ltrb, P)
    geo = np.stack([(cx - X0) / bw, (cy - Y0) / bh, cov, np.full(G * G, math.log(bw / P)), np.full(G * G, math.log(bh / P))], 1)
    box4 = np.array([X0 / S, Y0 / S, bw / S, bh / S], np.float32)
    return geo.astype(np.float32), box4, cov


def crop_canvas_geo(cl: Letterbox, P):
    """C0 crop-grid geometry (= src.readout.common.crop_canvas_geo): box = crop content rect on the crop canvas."""
    S, G = cl.S, cl.S // P
    X0, Y0, X1, Y1 = cl.ox, cl.oy, cl.ox + cl.nw, cl.oy + cl.nh
    bw, bh = X1 - X0, Y1 - Y0
    ci, cj = np.divmod(np.arange(G * G), G)
    cx, cy = (cj + 0.5) * P, (ci + 0.5) * P
    cov = cl.coverage(P)
    geo = np.stack([(cx - X0) / bw, (cy - Y0) / bh, cov, np.full(G * G, math.log(bw / P)), np.full(G * G, math.log(bh / P))], 1)
    return geo.astype(np.float32), np.array([X0 / S, Y0 / S, bw / S, bh / S], np.float32), cov
