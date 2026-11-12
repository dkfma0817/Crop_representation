"""Architecture figure of the G1 spatial distillation family (G1U, G1R, G1-Native), white background.

  python -m src.viz.g1_arch --out results/figures/g1_spatial_distill_architecture.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED = "#1f2933", "#5b6773"
STUDENT, STUDENT_EDGE = "#e3f1ec", "#2b7a66"
FINE, FINE_EDGE = "#e6edf8", "#3a64a8"
INTER, INTER_EDGE = "#f1e9f7", "#7a4fa3"
TEACHER, TEACHER_EDGE = "#eef0f2", "#6b7782"
LOSS, LOSS_EDGE = "#fdeee0", "#c9691d"
READ, READ_EDGE = "#fff8dc", "#a98a1b"


def box(ax, x, y, w, h, text, face, edge, size=11, weight="normal", color=INK, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=face, ec=edge, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight, color=color, zorder=3, linespacing=1.3)


def arrow(ax, a, b, color=INK, style="-|>", lw=1.5, rad=0.0, ls="-", z=1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, lw=lw, color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def draw(out):
    fig, ax = plt.subplots(figsize=(21, 12.4), dpi=170, facecolor="white")
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 124)
    ax.axis("off")
    ax.text(105, 120.5, "G1 Spatial Distillation: P384 Coarse-Fine Student with a 768-Input Teacher", ha="center", fontsize=19, fontweight="bold", color=INK)
    ax.text(105, 116.2, "G1U, G1R and G1-Native share the student and the losses; only the Grid teacher input and the Grid weights differ", ha="center", fontsize=12.5, color=MUTED)

    # input
    box(ax, 1, 66, 17.5, 12, "COCO image\n16 images / step", "white", MUTED, 10)
    box(ax, 1, 46, 17.5, 12, "384 x 384\nletterbox canvas", "white", MUTED, 10)
    arrow(ax, (9.75, 66), (9.75, 58))

    # student panel
    ax.add_patch(FancyBboxPatch((21, 33), 87, 76, boxstyle="round,pad=0,rounding_size=1.2", fc="#fbfdfc", ec=STUDENT_EDGE, lw=1.8, zorder=0))
    ax.text(23, 105.3, "Student (trainable, EMA 0.999)", fontsize=13.5, fontweight="bold", color=STUDENT_EDGE)

    # coarse row
    ax.text(23, 100.7, "Coarse: DINOv3-B/16, 24 x 24 x 768", fontsize=11, color=STUDENT_EDGE)
    xs = [25, 42, 59, 76]
    for x, name in zip(xs, ("Blocks 1-3", "Blocks 4-6", "Blocks 7-9", "Blocks 10-12")):
        box(ax, x, 89, 14, 9, name, STUDENT, STUDENT_EDGE, 10.5)
    for x in xs[:-1]:
        arrow(ax, (x + 14, 93.5), (x + 17, 93.5), STUDENT_EDGE)
    box(ax, 93, 89, 12, 9, "Final\nLayerNorm", STUDENT, STUDENT_EDGE, 9.5)
    arrow(ax, (90, 93.5), (93, 93.5), STUDENT_EDGE)

    # interaction
    for x in xs:
        box(ax, x + 1, 69, 12, 9, "Interaction\nC <-> F", INTER, INTER_EDGE, 9.5)
        arrow(ax, (x + 7, 89), (x + 7, 78), INTER_EDGE, style="<|-|>")
        arrow(ax, (x + 7, 69), (x + 7, 58), INTER_EDGE, style="<|-|>")

    # fine row
    box(ax, 25, 36, 14, 10, "Conv patch P=8\n+ 2D sincos", FINE, FINE_EDGE, 9.3)
    for k, x in enumerate(xs):
        box(ax, x, 49, 14, 9, f"FineBlock {k + 1}\n(shift {0 if k % 2 == 0 else 4})", FINE, FINE_EDGE, 10)
    for x in xs[:-1]:
        arrow(ax, (x + 14, 53.5), (x + 17, 53.5), FINE_EDGE)
    arrow(ax, (32, 46), (32, 49), FINE_EDGE)
    box(ax, 93, 49, 12, 9, "Fine\nLayerNorm", FINE, FINE_EDGE, 9.5)
    arrow(ax, (90, 53.5), (93, 53.5), FINE_EDGE)
    ax.text(42, 44.3, "Fine: RGB patch 8, 48 x 48 x 384, window attention (size 8, 6 heads)", fontsize=10.2, color=FINE_EDGE, va="center")
    ax.text(42, 38.2, "Interaction per stage: C += s L(AvgPool2x2 F),  F += s L(Broadcast2x2 C)", fontsize=9.6, color=INTER_EDGE, va="center")
    arrow(ax, (18.5, 52), (21, 52), MUTED)
    arrow(ax, (21, 52), (25, 93.5), MUTED, rad=-0.15)
    arrow(ax, (21, 52), (25, 41), MUTED, rad=0.12)

    # outputs
    ax.text(108.6, 99.4, "CLS, A", fontsize=11, color=STUDENT_EDGE, fontweight="bold")
    ax.text(109, 45.5, "B (Fine)", fontsize=11, color=FINE_EDGE, fontweight="bold")

    # read paths
    box(ax, 118, 89, 31, 13, "Global read\nCLS,  valid-cell mean of A", READ, READ_EDGE, 10.5)
    box(ax, 118, 66, 31, 15, "ROI read (area-weighted pool)\nA_r on 24 x 24,  B_r on 48 x 48\nQ_r = A_r + s_d  W_d(B_r)", READ, READ_EDGE, 10)
    box(ax, 118, 44, 31, 15, "Grid read (48 x 48)\nA48 = bilinear(A, 24 -> 48)\nF = A48 + s_d  W_d(B)", READ, READ_EDGE, 10)
    arrow(ax, (105, 95.5), (118, 95.5), STUDENT_EDGE)
    arrow(ax, (105, 91), (118, 75), STUDENT_EDGE)
    arrow(ax, (105, 90), (118, 54), STUDENT_EDGE)
    arrow(ax, (105, 55), (118, 70), FINE_EDGE)
    arrow(ax, (105, 52), (118, 50), FINE_EDGE)

    # losses
    box(ax, 152, 89, 14, 13, "L_global\nweight 0.5", LOSS, LOSS_EDGE, 11, "bold")
    box(ax, 152, 66, 14, 15, "L_crop\n1 - cos\nweight 1", LOSS, LOSS_EDGE, 11, "bold")
    box(ax, 152, 44, 14, 15, "L_grid\n1 - cos (w_j)\nweight 0.61", LOSS, LOSS_EDGE, 10.5, "bold")
    for y in (95.5, 73.5, 51.5):
        arrow(ax, (149, y), (152, y), LOSS_EDGE)

    # teachers
    ax.add_patch(FancyBboxPatch((170, 33), 38, 76, boxstyle="round,pad=0,rounding_size=1.2", fc="#fafbfc", ec=TEACHER_EDGE, lw=1.8, zorder=0))
    ax.text(172, 105.3, "Teacher: frozen DINOv3-B/16", fontsize=11.5, fontweight="bold", color=TEACHER_EDGE)
    box(ax, 172, 89, 34, 13, "Same 384 canvas\nCLS, valid-cell mean of tokens", TEACHER, TEACHER_EDGE, 10.5)
    box(ax, 172, 66, 34, 15, "GT ROI crop on a 384 canvas\n(COCO object + nested boxes)\npooled crop tokens", TEACHER, TEACHER_EDGE, 10.5)
    box(ax, 172, 37, 34, 25, "768 input -> 48 x 48 x 768\n\nG1U / G1R: 384 canvas\nupscaled bilinearly to 768\n\nG1-Native: original image resized\ndirectly to 768 (384 layout x 2)", TEACHER, TEACHER_EDGE, 10)
    for y in (95.5, 73.5, 51.5):
        arrow(ax, (172, y), (166, y), TEACHER_EDGE)

    # legend strip for Grid weights and total loss
    box(ax, 21, 18.5, 87, 10.5, "Total loss   L = L_crop + 0.61 L_grid + 0.5 L_global\nL_global = 1/2 [ (1 - cos(CLS, CLS_T)) + (1 - cos(mean A, mean T)) ]", "white", LOSS_EDGE, 12)
    box(ax, 112, 18.5, 96, 10.5, "Grid weight w_j on each 48 x 48 cell\nG1U, G1-Native: 1 on valid cells      G1R: valid cell x max ROI overlap in that cell", "white", LOSS_EDGE, 12)

    # evaluation strip
    ax.text(21, 13.2, "Evaluation after 12k steps (EMA, encoder frozen)", fontsize=12.5, fontweight="bold", color=INK)
    names = ["Frozen EMA encoder", "ROIAlign onto the 24 x 24 crop grid", "Q = A + s_d W_d(B)", "R2 reader (10 epochs, seed 0)", "PACO official GT Box + GT Category\nOracle Attribute AP"]
    widths = [26, 40, 26, 40, 47]
    x = 21
    for i, (name, w) in enumerate(zip(names, widths)):
        box(ax, x, 3, w, 8, name, "white", MUTED, 10.5)
        if i < len(names) - 1:
            arrow(ax, (x + w, 7), (x + w + 2, 7), MUTED)
        x += w + 2
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.15)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    draw(ap.parse_args().out)
