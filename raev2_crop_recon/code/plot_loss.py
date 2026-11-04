"""A/B/C loss curves from perception_cache/runs/{A,B,C}/train_log.jsonl -> figures/loss_curves.png.
Top row: train (50-step log, light) + COCO dev 2,000 ROI (every 1k, markers) for feature loss (MSE_w + (1 - cos)_w) and pixel loss
(L1 + 0.2 LPIPS). Bottom row: dev latent cos vs teacher and dev RGB L1 (vs GT crop).
  python code/plot_loss.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {"A": "A · feature", "B": "B · pixel", "C": "C · feature + pixel"}
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"


def load(r):
    L = [json.loads(x) for x in open(os.path.join(HERE, "perception_cache", "runs", r, "train_log.jsonl"))]
    tr = [(d["step"], d["train"]) for d in L if "train" in d]
    dv = [(d["step"], d["dev"]) for d in L if "dev" in d]
    return tr, dv


def style(a, title):
    a.set_title(title, fontsize=10, color=INK, loc="left")
    a.set_xlabel("step (k)", color=INK2, fontsize=9)
    a.grid(color=GRID, lw=0.8)
    a.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def main():
    data = {r: load(r) for r in RUNS}
    refs = json.load(open(os.path.join(HERE, "results", "dev_refs.json")))       # code/eval_refs.py
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
    panels = [(ax[0, 0], "feat", "feat_loss", "feature loss  MSE + (1 − cos) ↓", True),
              (ax[0, 1], "pix", "pix_loss", "pixel loss  L1 + 0.2·LPIPS ↓", True),
              (ax[1, 0], None, "cos", "dev latent cos vs teacher ↑", False),
              (ax[1, 1], None, "rgb_l1", "dev RGB L1 vs GT crop ↓", False)]
    for a, tk, dk, title, has_train in panels:
        for key, lab, ls, c in (("bil", "Full (Bilinear)", "--", INK2), ("teacher", "GT crop oracle", ":", INK)):
            v = refs[key][dk]
            a.axhline(v, color=c, ls=ls, lw=1.4, label=lab)
            a.annotate(f"{lab} {v:.3f}", (12, v), xytext=(-2, 3), textcoords="offset points", color=c, fontsize=7.5,
                       va="bottom", ha="right")
        for r in RUNS:
            tr, dv = data[r]
            if has_train and not (r == "A" and tk == "pix"):        # A never computes the pixel loss in training
                a.plot([s / 1000 for s, _ in tr], [t[tk] for _, t in tr], color=COL[r], lw=1, alpha=0.35)
            a.plot([s / 1000 for s, _ in dv], [d[dk] for _, d in dv], color=COL[r], lw=2, marker="o", ms=4, label=RUNS[r])
        ends = sorted((data[r][1][-1][1][dk], r) for r in RUNS)
        lo, hi = a.get_ylim()
        groups = [[ends[0]]]
        for v, r in ends[1:]:
            (groups[-1].append((v, r)) if v - groups[-1][-1][0] < 0.03 * (hi - lo) else groups.append([(v, r)]))
        for g in groups:
            a.annotate("·".join(sorted(r for _, r in g)), (12, sum(v for v, _ in g) / len(g)), xytext=(6, 0),
                       textcoords="offset points", color=INK, fontsize=9, va="center")
        style(a, title)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.text(0.01, 0.935, "light = train (50-step), markers = dev (COCO dev 2,000 ROI, every 1k) · A: pixel loss not trained, "
             "dev only · Full/oracle = no training, same dev set", ha="left", va="bottom", fontsize=7.5, color=INK2)
    fig.suptitle("RAEv2 crop recon · CoordMLP A / B / C · 12k steps, 64 ROI/step", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out = os.path.join(HERE, "figures", "loss_curves.png")
    fig.savefig(out, dpi=130)
    print("saved", out)
    for r in RUNS:
        d0, d1 = data[r][1][0][1], data[r][1][-1][1]
        print(r, {k: (round(d0[k], 4), round(d1[k], 4)) for k in ("feat_loss", "pix_loss", "cos", "rgb_l1")})


if __name__ == "__main__":
    main()
