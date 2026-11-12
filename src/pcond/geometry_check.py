"""Sanity of the ROI geometry used for M_r (the 24x24 crop-canvas-aligned conditioning; identical for P1 and P2).

Crop384 patch centre -> undo letterbox + resize -> original pixel -> Full384 canvas -> feature-grid coordinate. Checks: (1) that chain, written out here independently, equals the training `aux_grid`;
(2) a linear ramp on the 24x24 feature grid read through grid_sample returns the expected index; (3) RGB at matched positions agrees (control: horizontally flipped grid); (4) padding bookkeeping; (5) figures.
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
from src.hcf.distill import Slots  # noqa: E402
from src.msd.pretrain import eval_exclusions  # noqa: E402
from src.pcond.train import CropObjectiveDS, selected_rois  # noqa: E402
from src.viz.c0_pca import LB, MEAN_T, STD_T  # noqa: E402

S, P, G = 384, 16, 24


def chain(W, H, box, i, j):
    """Patch (row i, col j) of the crop canvas -> every coordinate on the way, written independently of src.viz.c0_pca.geometry."""
    l, t, r, b = box
    crop, full = LB(r - l, b - t, S, P), LB(W, H, S, P)
    cx, cy = (j + .5) * P, (i + .5) * P
    xo, yo = l + (cx - crop.ox) * (r - l) / crop.nw, t + (cy - crop.oy) * (b - t) / crop.nh
    X, Y = full.ox + xo * full.nw / W, full.oy + yo * full.nh / H
    return dict(crop_canvas=[cx, cy], original=[xo, yo], full_canvas=[X, Y], feature_index_24=[X / P - .5, Y / P - .5], grid=[X / (S / 2) - 1, Y / (S / 2) - 1])


def to_image(x):
    return (x * STD_T + MEAN_T).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared", default="output/2026-11-08/hcf_pilot/shared")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--figures", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ds = CropObjectiveDS(a.shared)
    stream = np.load(STREAM)[:12000]
    ex = eval_exclusions()["raev2_stream"]
    trace, errors, ramp, aligned, control, stats = [], [], [], [], [], dict(rois=0, cells=0, valid=0, padding=0, partial=0, valid_centre_outside_roi=0, padding_with_rgb=0, valid_without_rgb=0)
    figures = 0
    for step in range(a.batches):
        item = next(iter(Slots(stream, 0, 1, ex, step)))
        b = ds[item]
        idx = 0
        for k, (column, iid) in enumerate(item[1]):
            boxes, selected = selected_rois(ds, item[0], column, iid)
            image = Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")).convert("RGB")
            W, H = image.size
            for j in selected:
                box = tuple(int(x) for x in boxes[j])
                grid, cov = b["aux_grid"][idx].reshape(G, G, 2), b["aux_cov"][idx].reshape(G, G)
                ours = torch.tensor([[chain(W, H, box, r, c)["grid"] for c in range(G)] for r in range(G)], dtype=torch.float32)
                errors.append(float((ours - grid).abs().max()))
                # Grid-sample convention: a ramp valued by the 24-grid index must return the analytic feature index.
                xs = torch.arange(G, dtype=torch.float32).repeat(G, 1)[None, None]
                got = F.grid_sample(xs, grid[None], mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
                want = (grid[..., 0] + 1) * G / 2 - .5
                ramp.append(float((got - want.clamp(0, G - 1)).abs().max()))
                crop = LB(box[2] - box[0], box[3] - box[1], S, P)
                mask = (b["pixel_valid"][idx, 0].unsqueeze(0).unsqueeze(0).float())
                frac = F.avg_pool2d(mask, P)[0, 0]
                stats["rois"] += 1
                stats["cells"] += G * G
                stats["valid"] += int((cov > 0).sum())
                stats["padding"] += int((cov == 0).sum())
                stats["partial"] += int(((cov > 0) & (cov < 1)).sum())
                stats["padding_with_rgb"] += int(((cov == 0) & (frac > 0)).sum())
                stats["valid_without_rgb"] += int(((cov > 0) & (frac == 0)).sum())
                cx, cy = (torch.arange(G) + .5) * P, (torch.arange(G) + .5) * P
                inside = torch.outer((cy >= crop.oy) & (cy <= crop.oy + crop.nh), (cx >= crop.ox) & (cx <= crop.ox + crop.nw))
                stats["valid_centre_outside_roi"] += int(((cov > 0) & ~inside).sum())
                # RGB agreement at the exact sampling positions, on cells fully inside the crop: crop-canvas patch mean vs full canvas bilinear at the patch centre.
                full = b["full"][b["aux_ri"][idx]][None]
                sampled = F.grid_sample(full, grid[None], mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0)
                flip = grid.clone()
                flip[..., 0] = grid[..., 0].flip(1)
                wrong = F.grid_sample(full, flip[None], mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0)
                patch = F.avg_pool2d(b["crops"][idx][None], P)[0].permute(1, 2, 0)
                full_cells = cov >= 1
                aligned.append(((sampled[full_cells] - patch[full_cells]).abs().mean().item(), float(np.corrcoef(sampled[full_cells].flatten().numpy(), patch[full_cells].flatten().numpy())[0, 1]) if int(full_cells.sum()) > 3 else float("nan")))
                control.append(((wrong[full_cells] - patch[full_cells]).abs().mean().item(), float(np.corrcoef(wrong[full_cells].flatten().numpy(), patch[full_cells].flatten().numpy())[0, 1]) if int(full_cells.sum()) > 3 else float("nan")))
                if figures < a.figures and (crop.ox > 0 or crop.oy > 0) and k % 3 == 0:
                    picks = [(r, c) for r, c in ((3, 3), (12, 12), (20, 6), (8, 19)) if float(cov[r, c]) > 0]
                    records = [dict(cell=[r, c], coverage=float(cov[r, c]), **chain(W, H, box, r, c)) for r, c in picks]
                    trace.append(dict(image_id=int(iid), image_size=[W, H], roi_ltrb=list(box), crop_resize=[crop.nw, crop.nh], crop_offset=[crop.ox, crop.oy], tracked=records))
                    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
                    ax[0].imshow(image)
                    ax[0].add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="red", lw=2))
                    ax[0].set_title(f"original {W}x{H}, ROI {box}", fontsize=9)
                    ax[1].imshow(to_image(b["crops"][idx]))
                    ax[1].set_title(f"crop384 canvas (resize {crop.nw}x{crop.nh}, offset {crop.ox},{crop.oy}); padded cells hatched", fontsize=9)
                    hatch = (cov == 0).numpy()
                    ax[1].imshow(np.ma.masked_where(~np.kron(hatch, np.ones((P, P), bool)), np.ones((S, S))), cmap="gray", alpha=.35, vmin=0, vmax=1)
                    ax[2].imshow(to_image(b["full"][b["aux_ri"][idx]]))
                    ax[2].set_title("Full384 canvas; sample positions", fontsize=9)
                    for n, rec in enumerate(records):
                        color = plt.cm.tab10(n)
                        ax[0].plot(*rec["original"], "o", color=color, ms=6)
                        ax[1].plot(*rec["crop_canvas"], "o", color=color, ms=6)
                        ax[2].plot(*rec["full_canvas"], "o", color=color, ms=6)
                    for x in ax:
                        x.axis("off")
                    fig.tight_layout()
                    fig.savefig(out / f"figures/geometry_{figures}.png", dpi=130, facecolor="white")
                    plt.close(fig)
                    figures += 1
                idx += 1
    assert max(errors) < 1e-3, max(errors)
    assert max(ramp) < 1e-3, max(ramp)
    assert stats["padding_with_rgb"] == 0 and stats["valid_without_rgb"] == 0
    aligned_np, control_np = np.array(aligned), np.array(control)
    report = dict(n_rois=stats["rois"], chain_vs_training_grid_max_abs_error=max(errors), ramp_vs_analytic_max_abs_error=max(ramp), rgb_mean_abs_diff=dict(aligned=float(np.nanmean(aligned_np[:, 0])), flipped_control=float(np.nanmean(control_np[:, 0]))), rgb_correlation=dict(aligned=float(np.nanmean(aligned_np[:, 1])), flipped_control=float(np.nanmean(control_np[:, 1]))), cells=stats, valid_fraction=stats["valid"] / stats["cells"],
                  fraction_valid_cell_centres_outside_roi=stats["valid_centre_outside_roi"] / max(1, stats["valid"]), note="Padding cells: coverage 0 -> masked in P1 (key_padding_mask), zero vector in P2. Partial-coverage cells keep the feature sampled at the cell centre (as in the baseline); their centre can lie up to 8 canvas px outside the ROI.", tracked_examples=trace)
    (out / "geometry_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "tracked_examples"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
