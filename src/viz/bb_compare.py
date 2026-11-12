"""Figure: Bilinear (enc(x_full) -> bilinear -> classifier) vs C0 (enc(x_crop) -> classifier), DINOv2 vs DINOv3,
PACO val M0 (dev-selected epochs), from the JSON written by src.ivc.compare_bb.

Encoding: hue = backbone (DINOv2 blue, DINOv3 orange; reference palette slots 1-2), fill = condition (Bilinear = 45 deg
hatch on a light tint, C0 = solid). Row 1: absolute M0; row 2: C0 - Bilinear gap with paired image-bootstrap 95% CI.
Runs missing from the JSON (still training) are drawn as an empty slot with 'running'.

  python -m src.viz.bb_compare --json output/2026-11-06/ivc_v3/compare_all.json --out output/2026-11-06/ivc_v3/bb_compare.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLS = ["Overall", "Object", "Part", "Obj D5-16", "Part D1-4", "Obj D17+"]
SERIES = [("v2_bil", "DINOv2 Bilinear", BLUE, True), ("v2_c0", "DINOv2 C0", BLUE, False),
          ("v3_bil", "DINOv3 Bilinear", ORANGE, True), ("v3_c0", "DINOv3 C0", ORANGE, False)]


def tint(hexc, a=0.22):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(c * a + (1 - a))


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bar(ax, x, h, w, color, hatched, base=0.0):
    fc = tint(color) if hatched else color
    ax.bar(x, h, w, bottom=base, color=fc, edgecolor=color if hatched else SURF, linewidth=1.2 if hatched else 2,
           hatch="////" if hatched else None)


def pipeline(ax):
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)

    def box(x, y, w, t, fc="#f1f0ec"):
        ax.add_patch(FancyBboxPatch((x, y), w, 2.6, boxstyle="round,pad=0.25,rounding_size=0.6", fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + 1.3, t, ha="center", va="center", fontsize=9.5, color=INK)

    def arrow(x0, x1, y):
        ax.annotate("", (x1, y + 1.3), (x0, y + 1.3), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    for y, lab, steps, hatched in ((6.4, "Bilinear", ["x_full", "enc (frozen)", "bilinear to crop grid", "classifier (R0 / R2)"], True),
                                   (1.4, "C0", ["x_crop (GT box)", "enc (frozen)", "classifier (R0 / R2)"], False)):
        ax.add_patch(plt.Rectangle((0.5, y + 0.4), 2.2, 1.8, fc=tint(INK2, 0.25) if hatched else INK2, ec=INK2,
                                   hatch="////" if hatched else None, lw=0.8))
        ax.text(3.6, y + 1.3, lab, va="center", fontsize=10.5, color=INK, weight="bold")
        xs = [12, 27, 42, 60] if hatched else [12, 27, 42]
        ws = [11, 11, 14, 15] if hatched else [11, 11, 15]
        for k, (x, w, t) in enumerate(zip(xs, ws, steps)):
            box(x, y, w, t)
            if k:
                arrow(xs[k - 1] + ws[k - 1] + 0.6, x - 0.6, y)
    ax.text(99, 5, "enc = DINOv2-B/14 @336 or DINOv3-B/16 @384\n(24 x 24 tokens, frozen)\nonly the classifier is trained",
            ha="right", va="center", fontsize=8.5, color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = json.load(open(a.json))
    fig = plt.figure(figsize=(15, 13), facecolor=SURF)
    gs = fig.add_gridspec(5, 1, height_ratios=[0.8, 2.2, 1.6, 2.2, 1.6], hspace=0.55)
    pipeline(fig.add_subplot(gs[0]))
    x = np.arange(len(COLS))
    w = 0.19
    for row, rd in ((1, "r2"), (3, "r0")):
        ax = fig.add_subplot(gs[row])
        style(ax)
        vals_all = []
        for k, (key, lab, col, hat) in enumerate(SERIES):
            m = J["m0"].get(f"{key}/{rd}")
            xs = x + (k - 1.5) * (w + 0.012)
            if m is None:
                for xx in xs:
                    ax.text(xx, 31, "running", rotation=90, ha="center", va="bottom", fontsize=7.5, color=INK2)
                continue
            v = np.array([m[c] for c in COLS])
            vals_all.append(v)
            bar(ax, xs, v - 30, w, col, hat, base=30)
            for xx, vv in zip(xs, v):
                ax.text(xx, vv + 0.4, f"{vv:.1f}", ha="center", va="bottom", fontsize=7.2, color=INK2)
        ax.set_ylim(30, np.max(vals_all) + 4)
        ax.set_xticks(x, COLS, fontsize=10, color=INK)
        ax.set_ylabel("val M0", color=INK2, fontsize=10)
        ax.set_title(f"{'R2 (attention classifier)' if rd == 'r2' else 'R0 (mean + linear)'}: PACO val M0, dev-selected epoch",
                     loc="left", fontsize=11.5, color=INK, pad=8)
        if row == 1:
            hs = [plt.Rectangle((0, 0), 1, 1, fc=tint(c) if h else c, ec=c, hatch="////" if h else None) for _, _, c, h in SERIES]
            ax.legend(hs, [s[1] for s in SERIES], ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02),
                      fontsize=9.5, labelcolor=INK)
        # gap panel
        ax2 = fig.add_subplot(gs[row + 1])
        style(ax2)
        ax2.axhline(0, color=INK2, lw=0.8)
        for k, (bb, col) in enumerate((("v2", BLUE), ("v3", ORANGE))):
            c = J["contrasts"].get(f"{bb}_c0-{bb}_bil/{rd}")
            xs = x + (k - 0.5) * 0.3
            if c is None:
                for xx in xs:
                    ax2.text(xx, 0.3, "running", rotation=90, ha="center", va="bottom", fontsize=7.5, color=INK2)
                continue
            p = np.array([c[cc]["point"] for cc in COLS])
            lo, hi = np.array([c[cc]["lo"] for cc in COLS]), np.array([c[cc]["hi"] for cc in COLS])
            ax2.bar(xs, p, 0.28, color=col, edgecolor=SURF, linewidth=2)
            ax2.errorbar(xs, p, yerr=[p - lo, hi - p], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
            for xx, pp, hh, ll in zip(xs, p, hi, lo):
                ax2.text(xx, (hh if pp >= 0 else ll) + (0.3 if pp >= 0 else -0.3), f"{pp:+.1f}", ha="center",
                         va="bottom" if pp >= 0 else "top", fontsize=7.5, color=INK2)
        ax2.set_xticks(x, COLS, fontsize=10, color=INK)
        ax2.set_ylabel("C0 − Bilinear", color=INK2, fontsize=10)
        ax2.set_title("crop gain C0 − Bilinear (same backbone), paired image bootstrap 95% CI", loc="left", fontsize=10,
                      color=INK2, pad=6)
        lo_all, hi_all = ax2.get_ylim()
        ax2.set_ylim(lo_all - 1.5, hi_all + 1.5)
        hs = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (BLUE, ORANGE)]
        ax2.legend(hs, ["DINOv2", "DINOv3"], ncol=2, frameon=False, loc="upper right", fontsize=9, labelcolor=INK)
    fig.suptitle("Bilinear vs C0 crop upper bound: DINOv2 vs DINOv3 (PACO val, seed 0)", x=0.125, ha="left",
                 fontsize=14, color=INK, y=0.995)
    fig.savefig(a.out, dpi=110, bbox_inches="tight", facecolor=SURF)
    print(a.out)


if __name__ == "__main__":
    main()
