"""Step 2 checks (CPU): (a) the VAW geometry unit check on random boxes, (b) visual samples of the Full / Tight canvases
with the GT box mapped to the canvas and the R2 cells (cov > 0) shaded, padding visible, (c) feature-cache size estimate.

  python -m src.cocolvis.geocheck --out output/2026-11-07/cocolvis_headroom/geocheck
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.cocolvis.common import RUN
from src.vaw.data import check, gt_box, r2_geo, source_rect

TOK = 768 * 2          # bytes per token (bf16)


def draw(canvas, box4, cov, S=384, P=16):
    im = canvas.convert("RGB")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ov.ctx = ImageDraw.Draw(ov)
    for c in np.nonzero(cov > 0)[0]:
        i, j = divmod(int(c), 24)
        d.rectangle([j * P, i * P, (j + 1) * P - 1, (i + 1) * P - 1], fill=(0, 255, 0, int(40 + 120 * min(1, cov[c]))))
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    X0, Y0, w, h = (v * S for v in box4)
    ImageDraw.Draw(im).rectangle([X0, Y0, X0 + w, Y0 + h], outline=(255, 0, 0), width=2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    check()
    from src.viz.c0_pca import LB
    res = {}
    rng = np.random.default_rng(0)
    for ds in ("coco", "lvis"):
        V = pd.read_parquet(os.path.join(RUN, "data", f"instances_{ds}_val.parquet"))
        # samples: 2 S, 2 M, 2 L
        pick = np.concatenate([rng.choice(np.nonzero(V["size"].values == s)[0], 2, replace=False) for s in range(3)])
        tiles = []
        for r in pick:
            x = V.iloc[r]
            img = Image.open(x.path).convert("RGB")
            W, H = img.size
            box = gt_box([x.x, x.y, x.w, x.h], W, H)
            row = []
            for mode in ("full", "tight"):
                rect = source_rect(mode, box, W, H)
                geo, box4, cov, lb = r2_geo(box, rect)
                cv = lb.canvas(img if mode == "full" else img.crop(rect))
                cv = cv if isinstance(cv, Image.Image) else Image.fromarray(np.uint8(np.clip(cv.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1) * 255))
                row.append(draw(cv, box4, cov))
                res.setdefault(f"{ds}_samples", []).append(dict(ann_id=int(x.ann_id), size="SML"[int(x["size"])], mode=mode, box=[round(v, 2) for v in box],
                                                               rect=list(rect), cells=int((cov > 0).sum()), cov_sum=float(cov.sum()),
                                                               content=[lb.ox, lb.oy, lb.nw, lb.nh]))
            t = Image.new("RGB", (780, 410), "white")
            t.paste(row[0], (0, 22)), t.paste(row[1], (394, 22))
            ImageDraw.Draw(t).text((4, 4), f"{ds} ann {int(x.ann_id)} size {'SML'[int(x['size'])]} box {[round(v) for v in box]}  | left Full 384, right C0 Tight 384 (green = R2 cells cov>0, red = mapped GT box)", fill="black")
            tiles.append(t)
        M = Image.new("RGB", (780 * 2, 410 * 3), "white")
        for k, t in enumerate(tiles):
            M.paste(t, ((k % 2) * 780, (k // 2) * 410))
        M.save(os.path.join(a.out, f"samples_{ds}.jpg"), quality=88)
        # cache estimate (bf16, 576 tokens per canvas; ROI-only = cells with cov > 0)
        for sp in ("train", "val"):
            D = pd.read_parquet(os.path.join(RUN, "data", f"instances_{ds}_{sp}.parquet"))
            sub = D.sample(min(len(D), 20000), random_state=0)
            fc, tc = [], []
            for x in sub.itertuples():
                box = gt_box([x.x, x.y, x.w, x.h], x.W, x.H)
                fc.append(int((r2_geo(box, (0, 0, x.W, x.H))[2] > 0).sum()))
                tc.append(int((r2_geo(box, source_rect("tight", box, x.W, x.H))[2] > 0).sum()))
            n_img, n_roi = D.image_id.nunique(), len(D)
            res[f"cache_{ds}_{sp}"] = dict(
                images=int(n_img), rois=int(n_roi), mean_cells_full_roi=float(np.mean(fc)), mean_cells_tight_roi=float(np.mean(tc)),
                full_image_tokens_GB=n_img * 576 * TOK / 1e9, full_roi_only_GB=n_roi * np.mean(fc) * TOK / 1e9,
                tight_all_tokens_GB=n_roi * 576 * TOK / 1e9, tight_roi_only_GB=n_roi * np.mean(tc) * TOK / 1e9)
    json.dump(res, open(os.path.join(a.out, "geocheck.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("cache")}, indent=1))


if __name__ == "__main__":
    main()
