"""Architecture figure of S3 token splitting x2, trained with L2 (S3-L2) or distribution KD (S3-Distill), white background.

  python -m src.viz.tsplit_arch --out results/figures/tsplit_s3_l2_vs_distill_architecture.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from src.viz.g1_arch import INK, LOSS, LOSS_EDGE, MUTED, READ, READ_EDGE, STUDENT, STUDENT_EDGE, TEACHER, TEACHER_EDGE, arrow, box  # noqa: E402

CHANGED = "#d1342f"
L2C, L2E = "#e9f3fb", "#2f6f9f"


def panel(ax, x, y, w, h, edge, face="#fcfcfd", ls="-", lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=face, ec=edge, lw=lw, ls=ls, zorder=0))


def elbow(ax, pts, color):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=1.5, zorder=1, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color)


def split_icon(ax, x, y):
    """One parent token -> 2 x 2 child tokens."""
    ax.add_patch(Rectangle((x, y + .4), 3.6, 3.6, fc=TEACHER, ec=INK, lw=1.1, zorder=3))
    arrow(ax, (x + 4.0, y + 2.2), (x + 5.6, y + 2.2), INK)
    for a in range(2):
        for b in range(2):
            ax.add_patch(Rectangle((x + 6 + 1.9 * b, y + 2.3 - 1.9 * a), 1.9, 1.9, fc=STUDENT, ec=STUDENT_EDGE, lw=1.1, zorder=3))


def draw(out):
    fig, ax = plt.subplots(figsize=(22, 14), dpi=170, facecolor="white")
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 140)
    ax.axis("off")
    ax.text(110, 136.5, "S3 Token Splitting x2 (16x16 -> 8x8 sub-patch tokens): S3-L2 vs S3-Distill", ha="center", fontsize=20, fontweight="bold", color=INK)
    ax.text(110, 132.2, "Frozen DINOv3-B/16 at 384; only the small S3 decoder is trained (COCO, 12k steps x 16 images). "
            "Both runs share the architecture, init, data and optimizer; only the training loss differs.", ha="center", fontsize=12.5, color=MUTED)

    # ---------------- input ----------------
    box(ax, 2, 84, 20, 12, "COCO image\n16 / step, hflip 0.5", "white", MUTED, 10.5)

    # ---------------- teacher row ----------------
    ax.text(26, 124, "Teacher (frozen, target only)", fontsize=12.5, fontweight="bold", color=TEACHER_EDGE)
    box(ax, 26, 110, 22, 11, "resize x2\n768 x 768 letterbox", "white", MUTED, 10)
    box(ax, 52, 110, 30, 11, "Frozen DINOv3-B/16\n(same weights)\n48 x 48 x 768 tokens", TEACHER, TEACHER_EDGE, 10)
    box(ax, 88, 110, 42, 11, "Child targets for each ROI\nparent (i, j)  <->  teacher cells (2i+a, 2j+b)", TEACHER, TEACHER_EDGE, 10)
    elbow(ax, [(12, 96), (12, 115.5), (26, 115.5)], MUTED)
    arrow(ax, (48, 115.5), (52, 115.5), MUTED)
    arrow(ax, (82, 115.5), (88, 115.5), TEACHER_EDGE)
    arrow(ax, (130, 115.5), (164, 115.5), TEACHER_EDGE)
    ax.text(147, 117, "target children", ha="center", fontsize=9.5, color=TEACHER_EDGE)

    # ---------------- student encoder ----------------
    ax.text(26, 102, "Student input", fontsize=12.5, fontweight="bold", color=INK)
    box(ax, 26, 84, 22, 12, "384 x 384\nletterbox", "white", MUTED, 10.5)
    arrow(ax, (22, 90), (26, 90), MUTED)
    box(ax, 52, 76, 30, 22, "Frozen DINOv3-B/16\n12 blocks, 384 input\n\noutputs (24 x 24 x 768 each)\n- patch-embed tokens\n  (before block 0)\n- final-norm tokens", TEACHER, TEACHER_EDGE, 10)
    arrow(ax, (48, 90), (52, 90), MUTED)
    box(ax, 52, 57, 30, 13, "ROI sampler: 8 ROIs / image\neach 1-6 x 1-6 parent tokens\n(= attention set of step 4)", "white", MUTED, 10)

    # ---------------- S3 decoder ----------------
    panel(ax, 87, 51, 74, 56, STUDENT_EDGE, "#fbfdfc")
    ax.text(89.5, 103.2, "S3 decoder (trainable, the only part that learns)", fontsize=12.5, fontweight="bold", color=STUDENT_EDGE)
    bx, bw = 90, 56
    rows = [
        (93, "1. Concat per parent token:\n[final 768, LN(patch-embed) 768] -> 1536"),
        (84, "2. Split Linear 1536 -> 4 x 768: one parent -> 2 x 2 children\n(init = copy of the parent, so step 0 = nearest upsampling)"),
        (75, "3. Linear 768 -> 384  +  fixed sin-cos position\n(parent row / col, sub row / col)"),
        (66, "4. 4 Transformer blocks (width 384, 6 heads)\nself-attention over ALL children inside the ROI"),
        (57, "5. child + zero-init Linear 384 -> 768 (residual)\n-> ROI child map, 2h x 2w x 768"),
    ]
    for i, (y, t) in enumerate(rows):
        box(ax, bx, y, bw, 7, t, STUDENT, STUDENT_EDGE, 9.8)
        if i:
            arrow(ax, (bx + bw / 2, y + 9), (bx + bw / 2, y + 7), STUDENT_EDGE)
    split_icon(ax, 148.5, 85.2)
    ax.text(154.5, 92.3, "x2 split", ha="center", fontsize=9, color=STUDENT_EDGE)
    arrow(ax, (82, 96.5), (90, 96.5), TEACHER_EDGE)
    elbow(ax, [(82, 63.5), (86, 63.5), (86, 69.5), (90, 69.5)], MUTED)
    arrow(ax, (146, 60.5), (164, 60.5), STUDENT_EDGE)
    ax.text(155, 62, "predicted\nchildren", ha="center", va="bottom", fontsize=9.5, color=STUDENT_EDGE)

    # ---------------- losses ----------------
    panel(ax, 164, 51, 54, 76, CHANGED, "#fffdfb", ls="--", lw=2.2)
    ax.text(166, 123.3, "Training loss (the ONLY difference)", fontsize=12.5, fontweight="bold", color=CHANGED)
    box(ax, 166.5, 94, 49, 26,
        "\n\n"
        "L2 distance between each predicted child\n"
        "and the teacher token at the same position\n"
        "(mean over valid children, then over ROIs)\n\n"
        "-> copies the teacher feature itself\n"
        "    (direction AND size)", L2C, L2E, 10)
    box(ax, 166.5, 54, 49, 37,
        "\n\n"
        "Bank KL: similarity of each child to a FIXED bank\n"
        "of 8,192 teacher patch tokens (temperature 0.07);\n"
        "student distribution should match the teacher's\n\n"
        "Local KL: inside the ROI, which children each child\n"
        "is similar to should match the teacher\n\n"
        "-> copies relations / distributions, not the feature;\n"
        "    scale-free, so student norm drifted 11.3 -> 19.6", LOSS, LOSS_EDGE, 9.6)
    ax.text(191, 116.5, "S3-L2", ha="center", fontsize=12.5, fontweight="bold", color=L2E, zorder=4)
    ax.text(191, 86.8, "S3-Distill  =  Bank KL  +  1.0 x Local KL", ha="center", fontsize=11.5, fontweight="bold", color=LOSS_EDGE, zorder=4)
    ax.text(191, 92.4, "or", ha="center", va="center", fontsize=11, fontweight="bold", color=CHANGED)

    # ---------------- evaluation ----------------
    ax.text(2, 45, "Evaluation (identical for both; encoder + S3 frozen)", fontsize=12.5, fontweight="bold", color=INK)
    names = ["Frozen DINOv3 + trained S3", "ROI child map read on\nthe 24 x 24 crop grid", "R2 attention reader\n10 epochs, seed 0",
             "epoch = max (Obj + Part) / 2\non official Val", "PACO official Test\nGT box + GT category oracle attribute AP"]
    widths = [33, 36, 32, 40, 58]
    x = 2
    for i, (name, w) in enumerate(zip(names, widths)):
        box(ax, x, 31, w, 10, name, READ if i == 2 else "white", READ_EDGE if i == 2 else MUTED, 10)
        if i < len(names) - 1:
            arrow(ax, (x + w, 36), (x + w + 2, 36), MUTED)
        x += w + 2

    # ---------------- results ----------------
    ax.text(2, 25, "Official Test Overall AP (seed 0; paired bootstrap B = 1000)", fontsize=12.5, fontweight="bold", color=INK)
    res = [("F384 (no decoder)", "39.28", "white", MUTED), ("S3-L2", "39.05", L2C, L2E), ("S3-Distill", "38.70", LOSS, LOSS_EDGE),
           ("F768 (768 input, upper ref.)", "43.66", TEACHER, TEACHER_EDGE)]
    x = 2
    for name, v, fc, ec in res:
        box(ax, x, 7, 34, 13, f"{name}\n{v}", fc, ec, 12)
        x += 36
    box(ax, 148, 7, 70, 13, "S3-Distill - S3-L2: -0.35 (n.s.)     S3-L2 - F384: -0.23 (n.s.)\n"
        "F768 - S3-L2: +4.61*     F768 - S3-Distill: +4.96*\n"
        "Distill was better on Val (55.65 vs 55.19) but not on Test", "white", MUTED, 10.5)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.15)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    draw(ap.parse_args().out)
