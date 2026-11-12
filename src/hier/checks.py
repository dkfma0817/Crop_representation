"""CPU input audit for H0/H1: geometry equality, pixel differences, native-size accounting, and the visual comparison figure."""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import COCO_TRAIN, STREAM  # noqa: E402
from src.hier.data import CANVAS, SCALE, detail_uint8, rect  # noqa: E402
from src.msd.pretrain import eval_exclusions  # noqa: E402
from src.viz.c0_pca import LB  # noqa: E402

BANDS = [("max side = 640 (native, no resampling of the full image)", lambda s: s == 640), ("max side > 640 (downscaled to 640)", lambda s: s > 640), ("max side < 640 (H1 is upsampled; no new information)", lambda s: s < 640)]


def stats(path):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    full = LB(W, H, 384, 16)
    ox, oy, nw, nh = rect(full)
    assert abs(ox - full.ox * SCALE) <= .5 and abs(nw - full.nw * SCALE) <= .5 and abs(oy - full.oy * SCALE) <= .5 and abs(nh - full.nh * SCALE) <= .5
    a = np.asarray(detail_uint8(img, full, "H0"), dtype=np.float32)
    b = np.asarray(detail_uint8(img, full, "H1"), dtype=np.float32)
    assert a.shape == b.shape
    d = a - b
    mse = float((d ** 2).mean())
    return dict(W=W, H=H, side=max(W, H), identity_resize=(nw, nh) == (W, H), mae=float(np.abs(d).mean()), psnr=float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse), scale=max(nw / W, nh / H))


def figure(paths, out):
    n = len(paths)
    fig, ax = plt.subplots(n, 4, figsize=(14, 3.5 * n), squeeze=False)
    for r, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        full = LB(*img.size, 384, 16)
        a, b = np.asarray(detail_uint8(img, full, "H0")), np.asarray(detail_uint8(img, full, "H1"))
        h, w = a.shape[:2]
        size = min(160, h, w)
        y0, x0 = (h - size) // 2, (w - size) // 2
        diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(2)
        panels = [(a, f"H0: original → 384 → 640 ({w}×{h})"), (b, f"H1: original {img.size[0]}×{img.size[1]} → 640"), (None, "|H0 − H1| × 8 (full)"), (None, "central 160 px: H0 | H1")]
        ax[r, 0].imshow(a)
        ax[r, 1].imshow(b)
        ax[r, 2].imshow(np.clip(diff * 8, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax[r, 3].imshow(np.concatenate([a[y0:y0 + size, x0:x0 + size], np.full((size, 4, 3), 255, dtype=np.uint8), b[y0:y0 + size, x0:x0 + size]], 1))
        for k, (_, title) in enumerate(panels):
            ax[r, k].set_title(title, fontsize=9)
            ax[r, k].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    run = Path(a.run)
    ex = set(eval_exclusions()["raev2_stream"])
    ids = np.unique(np.load(STREAM)[:12000].reshape(-1))
    ids = [int(i) for i in ids if int(i) not in ex]
    rng = np.random.default_rng(0)
    coco = [os.path.join(COCO_TRAIN, f"{ids[i]:012d}.jpg") for i in rng.choice(len(ids), a.n, replace=False)]
    M = pd.read_parquet("output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet")
    paths = M.drop_duplicates("image_id").path.values
    paco = list(paths[rng.choice(len(paths), min(a.n, len(paths)), replace=False)])
    report = {}
    for name, files in (("coco_train_stream", coco), ("paco_val", paco)):
        rows = pd.DataFrame([stats(p) for p in files])
        report[name] = dict(n=len(rows), identity_resize_fraction=float(rows.identity_resize.mean()), mae_mean=float(rows.mae.mean()), mae_median=float(rows.mae.median()), psnr_median=float(rows.psnr[np.isfinite(rows.psnr)].median()),
                            bands={label: dict(fraction=float(rows.side.map(test).mean()), mae_mean=float(rows.mae[rows.side.map(test)].mean()) if rows.side.map(test).any() else None) for label, test in BANDS},
                            scale_stats=dict(min=float(rows.scale.min()), median=float(rows.scale.median()), max=float(rows.scale.max())), identical_rect_tolerance_px=.5, canvas=CANVAS)
    (run / "input_audit.json").write_text(json.dumps(report, indent=1))
    (run / "figures").mkdir(exist_ok=True)
    figure(coco[:3] + paco[:3], run / "figures/detail_inputs.png")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
