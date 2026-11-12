"""D1 C1 Crop-from-384 input. Everything else is src.arch_ablate.bbdata (B0 'bil', C0 'c0').

C1 path, per image:
  original RGB -> (flip, same as C0) -> PIL BICUBIC resize to (full.nw, full.nh): the uint8 image LB.canvas normalises for the Full384 student, content only, so padding cannot be read
  -> GT box mapped to that image with the mapping of the Full384 ROI read (x * full.nw / W, y * full.nh / H; float, never rounded)
  -> PIL BICUBIC resize of the box to the C0 crop size (crop.nw, crop.nh). `box=` is fractional inside the enclosing integer patch, so filter support stops at the patch: only pixels overlapping the box are read
  -> paste at (crop.ox, crop.oy) and apply the DINOv3 normalisation once. Padding stays 0 in normalised space, as for C0 and Full384.
C1 and C0 share the same crop LB, hence the same canvas layout, coverage, geometry and R2 input.
"""
import math

import numpy as np
import torch
from PIL import Image

from src.arch_ablate.bbdata import BBImageDS as Base, bil_sample, crop_geo  # noqa: F401
from src.readout.common import C
from src.viz.c0_pca import LB, MEAN_T, STD_T


def full_rgb(img, full):
    """The exact uint8 Full384 content LB.canvas normalises (no padding)."""
    return img.convert("RGB").resize((full.nw, full.nh), resample=Image.BICUBIC)


def box_384(full, W, H, ltrb):
    l, t, r, b = ltrb
    return l * full.nw / W, t * full.nh / H, r * full.nw / W, b * full.nh / H


def c1_rgb(rgb384, full, W, H, ltrb, crop):
    """uint8 PIL of size (crop.nw, crop.nh), read from the Full384 content only."""
    x0, y0, x1, y1 = box_384(full, W, H, ltrb)
    ix0, iy0, ix1, iy1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
    assert 0 <= ix0 < ix1 <= full.nw and 0 <= iy0 < iy1 <= full.nh, (ltrb, W, H, full.nw, full.nh)
    return rgb384.crop((ix0, iy0, ix1, iy1)).resize((crop.nw, crop.nh), resample=Image.BICUBIC, box=(x0 - ix0, y0 - iy0, x1 - ix0, y1 - iy0))


def to_canvas(pil, lb):
    """LB.canvas without its resize: paste uint8 content at (ox, oy), normalise once, padding 0."""
    x = torch.zeros(3, lb.S, lb.S)
    x[:, lb.oy:lb.oy + lb.nh, lb.ox:lb.ox + lb.nw] = (torch.from_numpy(np.asarray(pil, dtype=np.float32)).permute(2, 0, 1) / 255.0 - MEAN_T) / STD_T
    return x


class BBImageDS(Base):
    """Base dataset plus mode 'c1' (same outputs as 'c0'; only the crop pixels differ)."""

    def __init__(self, df, Y, flip, mode, S, P, full_size=None):
        self.c1 = mode == "c1"
        super().__init__(df, Y, flip, "c0" if self.c1 else mode, S, P, full_size)

    def __getitem__(self, idx):
        if not self.c1:
            return super().__getitem__(idx)
        xs, per, rows_all = [], {k: [] for k in ("cov", "geo", "box4")}, []
        for gi in idx:
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[rows[0]]), self.p[rows[0]]
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            if f:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            full = LB(W, H, self.S, self.P)
            rgb384 = full_rgb(img, full)
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                crop = LB(bx[2] - bx[0], bx[3] - bx[1], self.S, self.P)
                xs.append(to_canvas(c1_rgb(rgb384, full, W, H, bx, crop), crop))
                geo, box4, cov = crop_geo(crop, self.S, self.P)
                per["geo"].append(geo)
                per["box4"].append(box4)
                per["cov"].append(cov)
            rows_all.append(rows)
        rows_all = np.concatenate(rows_all)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items()}
        out.update(side=self.S, x=torch.stack(xs), img_index=torch.arange(len(xs), dtype=torch.long), rows=torch.from_numpy(rows_all), lv=torch.from_numpy(self.lv[rows_all]), Y=self.Y[rows_all],
                   group=torch.as_tensor(np.concatenate([np.full(len(self.groups[gi]), k) for k, gi in enumerate(idx)]), dtype=torch.long))
        return out
