"""PACO image dataset for EVA-family encoders = src.readout.common.PacoImageDS with parametric canvas / patch.

mode 'full': whole image -> S x S letterbox (OpenAI-CLIP mean/std), per-ROI full-grid geometry (geom.roi_geo, patch 16)
mode 'c0'  : one crop per ROI = original pixels inside the manifest integer box (l, t, r, b) (= roi_geometry()['ltrb']),
             -> S x S letterbox, crop-grid geometry (geom.crop_canvas_geo)
Image grouping, flip rule (whole image mirrored, box mirrored in original px), row order: unchanged (common.C).
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.densecmp import geom
from src.readout.common import C


class EvaPacoDS(Dataset):
    def __init__(self, df, Y, flip, mode, side, patch=16, norm=geom.OPENAI_CLIP):
        assert mode in ("full", "c0") and side % patch == 0
        self.groups = C.image_groups(df)
        self.p, self.box = df.path.values, df[["l", "t", "r", "b"]].values.astype(int)
        self.WH = df[["W", "H"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.Y, self.flip, self.mode, self.side, self.patch, self.norm = torch.as_tensor(Y), flip, mode, side, patch, norm
        self.n_roi = np.array([len(g) for g in self.groups])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        xs, per, img_index, rows_all = [], {k: [] for k in ("cov", "geo", "box4", "box_ltrb", "WH")}, [], []
        for gi in idx:
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[rows[0]]), self.p[rows[0]]
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            if f:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.mode == "full":
                lb = geom.Letterbox(W, H, self.side)
                xs.append(geom.to_tensor_canvas(img, lb, self.norm))
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                if self.mode == "full":
                    g, b4, cov = geom.roi_geo(lb, bx, self.patch)
                else:
                    l, t, rr, bb = map(int, bx)
                    assert 0 <= l < rr <= W and 0 <= t < bb <= H, (bx, W, H)
                    cl = geom.Letterbox(rr - l, bb - t, self.side)
                    xs.append(geom.to_tensor_canvas(img.crop((l, t, rr, bb)), cl, self.norm))
                    g, b4, cov = geom.crop_canvas_geo(cl, self.patch)
                per["geo"].append(g)
                per["box4"].append(b4)
                per["cov"].append(cov)
                per["box_ltrb"].append(np.array(bx, np.float32))
                per["WH"].append(np.array([W, H], np.float32))
                img_index.append(len(xs) - 1)
            rows_all.append(rows)
        rows_all = np.concatenate(rows_all)
        out = {k: torch.from_numpy(np.stack(v)) for k, v in per.items()}
        out.update(side=self.side, x=torch.stack(xs), img_index=torch.as_tensor(img_index, dtype=torch.long), rows=torch.from_numpy(rows_all),
                   lv=torch.from_numpy(self.lv[rows_all]), Y=self.Y[rows_all])
        return out
