"""Architecture figure of P-C768 (PixelDiT Hybrid + Native Crop768 teacher) and M1 (A0 with raw-feature MSE crop loss), white background.

  python -m src.viz.pc768mse_arch --out results/figures/pc768_m1_architecture.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.viz.g1_arch import (FINE, FINE_EDGE, INK, INTER, INTER_EDGE, LOSS, LOSS_EDGE, MUTED, READ, READ_EDGE, STUDENT, STUDENT_EDGE, TEACHER, TEACHER_EDGE, arrow,  # noqa: E402
                             box)

HEAD, HEAD_EDGE = "#e9f3fb", "#2f6f9f"
CHANGED = "#d1342f"


def panel(ax, x, y, w, h, edge, face="#fcfcfd"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=face, ec=edge, lw=1.8, zorder=0))


def elbow(ax, pts, color, ls="-"):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=1.5, ls=ls, zorder=1, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color, ls=ls)


def changed(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x - .8, y - .8), w + 1.6, h + 1.6, boxstyle="round,pad=0,rounding_size=1.2", fc="none", ec=CHANGED, lw=2.2, ls="--", zorder=4))


def draw(out):
    fig, ax = plt.subplots(figsize=(22, 15), dpi=170, facecolor="white")
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 150)
    ax.axis("off")
    ax.text(110, 146.5, "P-C768 vs M1: two one-factor changes on the Coarse-Fine student", ha="center", fontsize=20, fontweight="bold", color=INK)
    ax.text(110, 142.3, "P-C768 changes only the crop teacher of the PixelDiT Hybrid (384 -> Native 768); M1 changes only the crop loss of A0 (1 - cos -> raw MSE)", ha="center", fontsize=12.5, color=MUTED)

    # ---------------- shared student ----------------
    panel(ax, 2, 50, 78, 88, STUDENT_EDGE, "#fbfdfc")
    ax.text(4, 134.6, "Shared student: A0 Coarse-Fine (trainable, EMA)", fontsize=13.5, fontweight="bold", color=STUDENT_EDGE)
    box(ax, 5, 122, 72, 9, "COCO image, 16 images / step\n-> 384 x 384 letterbox canvas (Full RGB only)", "white", MUTED, 10.5)
    box(ax, 5, 98, 29, 18, "Coarse\nDINOv3-B/16, 12 blocks\n24 x 24 x 768", STUDENT, STUDENT_EDGE, 10.5)
    box(ax, 48, 98, 29, 18, "Fine\nconv patch 8 + 4 window\nblocks, 48 x 48 x 384", FINE, FINE_EDGE, 10.5)
    box(ax, 35.5, 101, 11, 12, "C <-> F\ninteraction\nx 4 stages", INTER, INTER_EDGE, 9)
    arrow(ax, (34, 107), (35.5, 107), INTER_EDGE, style="<|-|>")
    arrow(ax, (46.5, 107), (48, 107), INTER_EDGE, style="<|-|>")
    arrow(ax, (19.5, 122), (19.5, 116), MUTED)
    arrow(ax, (62.5, 122), (62.5, 116), MUTED)
    ax.text(21, 93.5, "CLS, A (24 x 24)", fontsize=10.5, color=STUDENT_EDGE, fontweight="bold")
    ax.text(64, 93.5, "B (48 x 48)", fontsize=10.5, color=FINE_EDGE, fontweight="bold")

    box(ax, 5, 76, 72, 13, "ROI read (area-weighted pool inside the GT box)\nQ_r = A_r + s_d W_d(B_r)      768-d, NOT L2-normalised", READ, READ_EDGE, 10.5)
    arrow(ax, (19.5, 98), (19.5, 89), STUDENT_EDGE)
    arrow(ax, (62.5, 98), (62.5, 89), FINE_EDGE)

    box(ax, 7, 58, 70, 12, "Global read: CLS, valid-cell mean of A\nvs. frozen teacher on the same 384 image\nL_global = 1/2 [(1 - cos CLS) + (1 - cos mean A)], weight 0.5 in both", READ, READ_EDGE, 9.8)
    elbow(ax, [(5, 103), (3.5, 103), (3.5, 64), (7, 64)], STUDENT_EDGE)
    ax.text(41, 52.3, "start: common initial.pt  |  12k steps  |  seed 0", fontsize=9.8, color=MUTED, ha="center")

    # ---------------- P-C768 ----------------
    panel(ax, 86, 96, 132, 42, HEAD_EDGE)
    ax.text(88, 134.6, "P-C768  (GPU 0)  =  PixelDiT Hybrid, crop teacher -> Native Crop768", fontsize=13.5, fontweight="bold", color=HEAD_EDGE)
    box(ax, 89, 117, 24, 14, "ROI token map\nM_r = fuse(grid_sample\nA, B) on the 24 x 24\ncrop grid", HEAD, HEAD_EDGE, 9.2)
    box(ax, 116, 117, 15, 14, "8 x 8\ncoverage pool\n-> 64 memory\ntokens", HEAD, HEAD_EDGE, 9.2)
    box(ax, 134, 117, 38, 14, "CropPixelDiT head (d 384, dropped at probe)\n2 DiT blocks on 24 x 24 noisy patches\nself-attn, cross-attn -> memory, AdaLN(t)\n+ pixel block (4 x 4 px groups)", HEAD, HEAD_EDGE, 9.2)
    box(ax, 175, 117, 20, 14, "predict v = noise - x\nx_t = (1-t) x + t noise\nt ~ U[0.1, 0.9]\nRGB crop @ 384", "white", HEAD_EDGE, 9.2)
    box(ax, 198, 117, 18, 14, "L_flow\npixel MSE\nweight 1", LOSS, LOSS_EDGE, 10, "bold")
    for a_, b_ in (((113, 124), (116, 124)), ((131, 124), (134, 124)), ((172, 124), (175, 124)), ((195, 124), (198, 124))):
        arrow(ax, a_, b_, HEAD_EDGE)

    box(ax, 89, 100.5, 22, 12.5, "L_crop\n1 - cos\nweight 1", LOSS, LOSS_EDGE, 10.5, "bold")
    box(ax, 118, 100.5, 98, 12.5, "Teacher (frozen DINOv3-B/16): GT ROI crop on a 768 canvas\n48 x 48 final-norm tokens, coverage-weighted mean\n(Hybrid baseline: the same ROI on a 384 canvas, 24 x 24 tokens)", TEACHER, TEACHER_EDGE, 9.8)
    changed(ax, 118, 100.5, 98, 12.5)
    arrow(ax, (118, 106.75), (111, 106.75), TEACHER_EDGE)
    ax.text(88, 97.6, "L = L_crop768 + 1.0 L_flow + 0.5 L_global      flow head, 8x8 conditioning, flow target, optimizer, EMA = Hybrid", fontsize=10, color=INK)

    # ---------------- M1 ----------------
    panel(ax, 86, 50, 132, 42, LOSS_EDGE)
    ax.text(88, 88.6, "M1  (GPU 1)  =  A0 Crop Cosine, crop loss -> raw-feature MSE", fontsize=13.5, fontweight="bold", color=LOSS_EDGE)
    box(ax, 112, 70, 36, 14, "L_crop = lambda mean((Q - T)^2)\nlambda = 3.0, no normalisation\n(A0 baseline: 1 - cos)", LOSS, LOSS_EDGE, 10, "bold")
    changed(ax, 112, 70, 36, 14)
    box(ax, 154, 70, 62, 14, "Teacher (frozen DINOv3-B/16): GT ROI crop on a\n384 canvas (C0), 24 x 24 tokens, coverage-weighted mean\n= same T as A0 (unchanged)", TEACHER, TEACHER_EDGE, 9.8)
    arrow(ax, (154, 77), (148, 77), TEACHER_EDGE)
    box(ax, 89, 55.5, 127, 9, "lambda fixed before training: median ||grad L_cos|| / ||grad L_mse|| at initial weights, 8 batches, all Coarse-Fine params = 2.97 -> 3.0\n"
        "cos(grad L_mse, grad L_cos) = 0.96 (coarse 0.97)      no flow head, no pixel branch", "white", MUTED, 9.8)
    ax.text(88, 51.4, "L = 3.0 L_mse + 0.5 L_global", fontsize=10, color=INK)

    # student -> condition arrows
    arrow(ax, (77, 86), (89, 106.75), READ_EDGE)
    arrow(ax, (77, 79), (112, 77), READ_EDGE)
    arrow(ax, (77, 112), (89, 124), HEAD_EDGE, ls="--")
    ax.text(81.5, 114, "A, B", fontsize=9.5, color=HEAD_EDGE, ha="center")

    # legend
    ax.add_patch(FancyBboxPatch((150, 42.9), 6, 2.2, boxstyle="round,pad=0,rounding_size=.4", fc="none", ec=CHANGED, lw=2, ls="--"))
    ax.text(157.5, 44, "the only component changed vs. its baseline", fontsize=9.5, color=CHANGED, va="center")

    # ---------------- evaluation ----------------
    ax.text(2, 44, "Evaluation (same for both; EMA encoder frozen, PixelDiT head dropped)", fontsize=12.5, fontweight="bold", color=INK)
    names = ["Frozen EMA encoder", "ROIAlign onto the 24 x 24 crop grid", "Q = A + s_d W_d(B)", "R2 reader (10 epochs, seed 0)", "Epoch = max (Obj + Part) / 2 on official Val", "PACO official Test, GT box + GT category\noracle attribute AP"]
    widths = [24, 37, 25, 33, 42, 47]
    x = 2
    for i, (name, w) in enumerate(zip(names, widths)):
        box(ax, x, 31, w, 9, name, "white", MUTED, 10)
        if i < len(names) - 1:
            arrow(ax, (x + w, 35.5), (x + w + 2, 35.5), MUTED)
        x += w + 2

    # ---------------- results ----------------
    ax.text(2, 25, "Official Test Overall (seed 0, point estimates; bootstrap still running)", fontsize=12.5, fontweight="bold", color=INK)
    box(ax, 2, 8, 106, 13, "P-C768  41.41   vs  PixelDiT Hybrid 42.07  ->  -0.66  (pre-specified)\n"
        "            vs  E1a Crop768 Cosine 41.57  ->  -0.17  (exploratory)\n"
        "Val best epoch 8;  Val Overall 56.37 vs Hybrid 56.71", "white", HEAD_EDGE, 10.5)
    box(ax, 112, 8, 106, 13, "M1  41.25   vs  A0 Crop Cosine 41.61  ->  -0.36  (pre-specified)\n"
        "Obj Pattern -1.34, Obj Material +0.70\n"
        "Val best epoch 8;  Val Overall 56.74 vs A0 56.53", "white", LOSS_EDGE, 10.5)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.15)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    draw(ap.parse_args().out)
