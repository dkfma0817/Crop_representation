"""Reconstruction snapshots on FIXED COCO dev ROIs, so runs and steps can be compared by eye.

Rows: GT crop | teacher (cached GT-crop latent -> decoder) | bilinear (full map -> crop grid) | model at each step.
Columns: N dev ROIs spread over sizes (n_in on the RAE grid), the same ROIs for every run (first dev images, deterministic).
Padding of the crop letterbox is shown dimmed.

  during training : src/engine/train.py saves <out>/viz/step_<step>.png at every dev evaluation (rank 0)
  after training  : python -m src.viz.progress --run_dir output/2026-11-04/B      -> <run_dir>/viz/progress.png
                    (rows: init, then every ckpt_<step>.pt in the run folder)
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import src.data.dataset as DR
import src.rae.common as RC

N_IMG_SLOTS = 8          # first 8 dev images (32 ROIs) -> pick N_COLS of them by size
N_COLS = 8


def fixed_batch(dev):
    """-> (batch on device, column indices into the batch ROIs). Deterministic: first dev images, ROIs at n_in quantiles."""
    ds = DR.CachedSet("dev", rgb=True)
    b = DR.to_dev(ds[list(range(N_IMG_SLOTS))], dev)
    nin = b["geom"]["n_in"].cpu().numpy()
    order = np.argsort(nin, kind="stable")
    cols = order[np.linspace(0, len(order) - 1, N_COLS).round().astype(int)]
    return b, torch.as_tensor(np.sort(cols), device=dev)


@torch.no_grad()
def decode_rows(models, rae, b, P):
    """models: list of (label, CoordMLP or None) -> dict label -> (R,3,256,256) in [0,1]; adds gt / teacher / bilinear."""
    g = b["geom"]
    out = {"GT crop": b["rgb"], "teacher": None, "bilinear": RC.bilinear(b["full"], g)}
    Z = {"teacher": b["teacher"], "bilinear": out["bilinear"]}
    for label, m in models:
        was = m.training
        m.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Z[label] = m(RC.gather_roi(b["full"], g), None, g, P).float()
        m.train(was)
    for label, z in Z.items():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[label] = rae.decode(z, chunk=32).float().clamp(0, 1)
    return out


def save_labeled_grid(path, rows, mask, col_titles, tile=256, label_w=150):
    """rows: dict label -> (n,3,H,W) in [0,1]; mask (n,1,H,W). Writes a PNG with row labels and column titles."""
    labels = list(rows)
    n = mask.shape[0]
    head = 28
    W, H = label_w + n * tile, head + len(labels) * tile
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    mm = mask.detach().float().cpu()
    for j, t in enumerate(col_titles):
        draw.text((label_w + j * tile + 6, 5), t, fill=(40, 40, 40), font=font)
    for i, lab in enumerate(labels):
        r = rows[lab].detach().float().cpu()
        r = r * mm + (1 - mm) * (0.5 + 0.3 * r)
        y0 = head + i * tile
        draw.text((8, y0 + tile // 2 - 10), lab, fill=(20, 20, 20), font=font)
        for j in range(n):
            x = (r[j].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(x).resize((tile, tile)), (label_w + j * tile, y0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


class Snapshot:
    """Used by the training loop: fixed dev ROIs prepared once, one PNG per call."""

    def __init__(self, rae, P, dev):
        self.rae, self.P = rae, P
        self.b, self.cols = fixed_batch(dev)
        self.titles = [f"n_in {int(v)}" for v in self.b["geom"]["n_in"][self.cols].tolist()]

    def save(self, m, step, out_dir):
        rows = decode_rows([(f"step {step}", m)], self.rae, self.b, self.P)
        rows = {k: v[self.cols] for k, v in rows.items()}
        save_labeled_grid(os.path.join(out_dir, "viz", f"step_{step:05d}.png"), rows, self.b["mask"][self.cols], self.titles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="folder with ckpt_<step>.pt (e.g. output/2026-11-04/B)")
    ap.add_argument("--no_init", action="store_true", help="skip the step-0 (init) row")
    a = ap.parse_args()
    import src.engine.train as TR
    from src.config import INIT
    dev = "cuda"
    RC.set_fp32()
    rae = RC.FrozenRAE(dev)
    del rae.enc
    P = torch.as_tensor(RC.query_coords(), device=dev)
    b, cols = fixed_batch(dev)
    step_of = lambda p: int(re.findall(r"ckpt_(\d+)", p)[0])
    ck = sorted(glob.glob(os.path.join(a.run_dir, "ckpt_*.pt")), key=step_of)
    models = ([] if a.no_init else [("step 0 (init)", TR.load_model(INIT, dev))]) + \
             [("step %d" % step_of(p), TR.load_model(p, dev)) for p in ck]
    rows = decode_rows(models, rae, b, P)
    rows = {k: v[cols] for k, v in rows.items()}
    titles = [f"n_in {int(v)}" for v in b["geom"]["n_in"][cols].tolist()]
    path = os.path.join(a.run_dir, "viz", "progress.png")
    save_labeled_grid(path, rows, b["mask"][cols], titles)
    print("saved", path, "rows:", list(rows))


if __name__ == "__main__":
    main()
