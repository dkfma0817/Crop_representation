"""Pre-training check that the Native 768 teacher input is aligned with the 384 student input. CPU only; uses real COCO stream batches.

For every image: (a) the content rectangle of the native canvas is exactly 2 x the 384 rectangle; (b) 48 x 48 cell means of the native canvas agree with those of the 384 canvas (MAD, Pearson), vs a horizontally flipped
negative control; (c) the native canvas differs from the 384 canvas upscaled to 768 by real extra detail (within-cell variance ratio). Figures: 384 RGB / upscaled 768 / native 768 / |native - upscaled| / valid mask.

  python -m src.g1native.geometry_check --out <run>/geometry
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.config import COCO_TRAIN, STREAM  # noqa: E402
from src.g1native.train import SHARED, GridDS  # noqa: E402
from src.hcf.distill import Slots, cell_w  # noqa: E402
from src.msd.pretrain import eval_exclusions  # noqa: E402
from src.viz.c0_pca import LB, MEAN_T, STD_T  # noqa: E402


def denorm(x):
    return (x * STD_T + MEAN_T).clamp(0, 1)


def cells(x, px):
    return F.avg_pool2d(denorm(x), px)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:12000]
    ds = GridDS(SHARED)
    rows, figs = [], 0
    for s in range(a.batches):
        item = next(iter(Slots(stream, 0, 1, ex, s)))
        b = ds[item]
        up = F.interpolate(b["full"], size=(768, 768), mode="bilinear", align_corners=False)
        valid = (cell_w(b["content"], 48, 8) > 0).reshape(-1, 48, 48)
        for k, (_, iid) in enumerate(item[1]):
            with Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")) as im:
                W, H = im.size
            lb = LB(W, H, 384, 16)
            x384, nat, u = b["full"][k], b["full768"][k], up[k]
            rect = dict(ox=2 * lb.ox, oy=2 * lb.oy, x1=2 * (lb.ox + lb.nw), y1=2 * (lb.oy + lb.nh))
            inside = torch.zeros(768, 768, dtype=torch.bool)
            inside[rect["oy"]:rect["y1"], rect["ox"]:rect["x1"]] = True
            content_outside = float((nat.abs().sum(0)[~inside] > 0).float().mean()) if (~inside).any() else 0.
            v = valid[k]
            c384, cnat, cflip = cells(x384, 8), cells(nat, 16), cells(nat.flip(-1), 16)
            full_cells = (cell_w(torch.tensor([[lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh]], dtype=torch.float32), 48, 8).reshape(48, 48) >= 1 - 1e-6)   # cells entirely inside the content
            cu = cells(u, 16)
            within_nat = float(((denorm(nat) - F.interpolate(cnat[None], size=(768, 768), mode="nearest")[0]) ** 2).mean(0)[inside].mean())
            within_up = float(((denorm(u) - F.interpolate(cu[None], size=(768, 768), mode="nearest")[0]) ** 2).mean(0)[inside].mean())
            rows.append(dict(iid=iid, W=W, H=H, content_outside=content_outside, mad=float((c384 - cnat).abs().mean(0)[full_cells].mean()), mad_flip=float((c384 - cflip).abs().mean(0)[full_cells].mean()), pearson=pearson(c384[:, full_cells], cnat[:, full_cells]), pearson_flip=pearson(c384[:, full_cells], cflip[:, full_cells]), native_vs_up_rmse=float(((denorm(nat) - denorm(u)) ** 2).mean(0)[inside].mean().sqrt()), within_cell_var_native=within_nat, within_cell_var_upscaled=within_up, valid_fraction=float(v.float().mean()), valid_outside_content=float((v & ~full_cells & (cell_w(torch.tensor([[lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh]], dtype=torch.float32), 48, 8).reshape(48, 48) <= 0)).sum())))
            if figs < 6 and (s % 5 == 0):
                figs += 1
                fig, ax = plt.subplots(1, 5, figsize=(17, 3.8), facecolor="white")
                panels = [(denorm(x384), f"Student 384 (id {iid})"), (denorm(u), "G1U: 384 upscaled to 768"), (denorm(nat), "G1N: original directly to 768"), (None, "|native - upscaled| x 6"), (None, "valid 48x48 cells")]
                for i, (p, t) in enumerate(panels):
                    if p is not None:
                        ax[i].imshow(p.permute(1, 2, 0).numpy())
                    elif i == 3:
                        ax[i].imshow(((denorm(nat) - denorm(u)).abs() * 6).clamp(0, 1).permute(1, 2, 0).numpy())
                    else:
                        ax[i].imshow(v.numpy(), cmap="Greys", vmin=0, vmax=1)
                    ax[i].set_title(t, fontsize=10)
                    ax[i].axis("off")
                fig.tight_layout()
                fig.savefig(out / "figures" / f"geometry_{figs - 1}.png", dpi=110, facecolor="white")
                plt.close(fig)
    arr = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "iid"}
    summary = dict(n_images=len(rows), content_outside_max=float(arr["content_outside"].max()), mad_mean=float(arr["mad"].mean()), mad_max=float(arr["mad"].max()), mad_flip_mean=float(arr["mad_flip"].mean()), pearson_mean=float(arr["pearson"].mean()), pearson_min=float(arr["pearson"].min()), pearson_flip_mean=float(arr["pearson_flip"].mean()), native_vs_up_rmse_mean=float(arr["native_vs_up_rmse"].mean()), within_cell_var_native_mean=float(arr["within_cell_var_native"].mean()), within_cell_var_upscaled_mean=float(arr["within_cell_var_upscaled"].mean()), within_cell_var_ratio_mean=float((arr["within_cell_var_native"] / arr["within_cell_var_upscaled"]).mean()), valid_fraction_mean=float(arr["valid_fraction"].mean()), valid_cells_outside_content=int(arr["valid_outside_content"].sum()), note="MAD on [0,1] RGB of 48x48 cell means; cells entirely inside the content. The training GridDS has no flip or crop augmentation; both inputs are made from one PIL image.")
    assert summary["content_outside_max"] == 0, summary
    assert summary["mad_mean"] < 0.05 and summary["mad_mean"] < summary["mad_flip_mean"] / 3, summary
    assert summary["pearson_mean"] > 0.98 and summary["pearson_mean"] > summary["pearson_flip_mean"] + .2, summary
    assert summary["valid_cells_outside_content"] == 0, summary
    assert summary["native_vs_up_rmse_mean"] > 1e-3, summary               # the native input must really differ from the upscaled one
    (out / "geometry_check.json").write_text(json.dumps(dict(summary=summary, status="passed", per_image=rows), indent=1))
    print(json.dumps(dict(status="passed", **summary), indent=1), flush=True)


if __name__ == "__main__":
    main()
