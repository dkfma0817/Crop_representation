"""Merge shards and summarise the RAE diagnosis -> reconstruction_results.csv, latent_cycle_results.csv, tables.md, figures/.

Groups: all / object / part / D-bins (manifest F336 n_in: D1-4, D5-16, D17+) x level; RAE-grid n_in bins secondary.
Paired differences: image-cluster bootstrap of the mean per-ROI difference (1,000, seed 0), same ROIs in both conditions.
  python -m src.raediag.report --run output/2026-11-05/rae_diag
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

CONDS = ["A", "B", "B_in", "C", "D", "E", "F", "MB"]
LABEL = {"A": "A GT crop -> enc -> dec", "B": "B full -> enc -> dec -> crop", "B_in": "B_in full INPUT pixels -> crop (ref)",
         "C": "C full feat -> bilinear -> dec", "D": "D full feat -> bicubic -> dec", "E": "E diag: C0 latent low-passed -> dec",
         "F": "F diag: bilinear + C0 mean -> dec", "MB": "MB existing Pixel-only CoordMLP -> dec"}
RGB = ["l1", "psnr", "lpips", "ssim", "hp_corr", "de_mean", "hf_ratio"]
LAT = ["re_tok_cos", "re_pool_cos", "re_tok_mse", "re_pool_l2", "re_svar_ratio", "re_ss_diff"]
INP = ["in_tok_cos", "in_pool_cos", "in_tok_mse", "in_svar_ratio", "cyc_tok_cos", "in_norm", "in_z2", "in_z3", "in_z2_pad", "in_adj_cos"]
PAIRS = [("B", "A"), ("C", "A"), ("D", "C"), ("B", "C"), ("B", "D"), ("B_in", "B"), ("E", "C"), ("F", "C"), ("E", "A"), ("MB", "C")]


def dbin(n):
    return np.select([n <= 4, n <= 16], ["D1-4", "D5-16"], "D17+")


def boot(df, x, y, metric, nb=1000):
    a = df[df.cond == x].set_index("row")
    b = df[df.cond == y].set_index("row").loc[a.index]
    d = (a[metric] - b[metric]).values
    ok = np.isfinite(d)
    d, img = d[ok], a.image_id.values[ok]
    u, inv = np.unique(img, return_inverse=True)
    S, N = np.bincount(inv, d), np.bincount(inv)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(nb):
        c = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        bs.append((c * S).sum() / (c * N).sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = a.run
    FIG = os.path.join(R, "figures")
    os.makedirs(FIG, exist_ok=True)
    rec = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "reconstruction_results_shard*.csv")))])
    lat = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "latent_cycle_results_shard*.csv")))])
    for d in (rec, lat):
        assert d.groupby("cond").row.nunique().nunique() == 1, "conditions cover different ROIs"
        assert not d.duplicated(["row", "cond"]).any()
    rec = rec.sort_values(["cond", "row"]).reset_index(drop=True)
    lat = lat.sort_values(["cond", "row"]).reset_index(drop=True)
    rec.to_csv(os.path.join(R, "reconstruction_results.csv"), index=False)
    lat.to_csv(os.path.join(R, "latent_cycle_results.csv"), index=False)
    D = rec.merge(lat, on=["row", "ann_id", "image_id", "level", "n_in", "n_in_rae", "cond"])
    D["bin"] = dbin(D.n_in.values)
    D["bin_rae"] = dbin(D.n_in_rae.values)
    n = D[D.cond == "A"].shape[0]
    L = [f"# RAE diagnosis tables (PACO val, {n:,} ROIs, {D[D.cond == 'A'].image_id.nunique():,} images; auto-generated)", ""]

    def table(cols, title, groups):
        L.extend([f"### {title}", "", "| group | cond | n | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)])
        for gname, sel in groups:
            for c in CONDS:
                d = D[sel & (D.cond == c)]
                if not len(d) or (cols[0].startswith("in_") and c in ("B", "B_in")):
                    continue
                L.append(f"| {gname} | {c} | {len(d)} | " + " | ".join(f"{d[k].mean():.3f}" for k in cols) + " |")
        L.append("")
    allsel = pd.Series(True, index=D.index)
    groups = [("all", allsel), ("object", D.level == "object"), ("part", D.level == "part")]
    groups += [(f"{lv} {b}", (D.level == lv) & (D.bin == b)) for lv in ("object", "part") for b in ("D1-4", "D5-16", "D17+")]
    L += ["Conditions: " + "; ".join(f"**{c}** = {LABEL[c]}" for c in CONDS), ""]
    table(RGB, "RGB reconstruction vs the GT crop canvas (content pixels; clamped [0,1]; mean per ROI)", groups)
    table(LAT, "Re-encoding: reconstructed crop canvas -> encoder vs C0 latent T (content cells, coverage-weighted)", groups)
    table(INP, "Decoder-input latent vs T, cycle (re-encoded vs input) and input distribution", groups[:3] + groups[3:])
    # RAE-grid bins (secondary)
    g2 = [(f"RAE-grid {b}", D.bin_rae == b) for b in ("D1-4", "D5-16", "D17+")]
    table(["l1", "lpips", "ssim", "hp_corr", "re_tok_cos", "re_pool_cos"], "Secondary: bins by RAE 16x16 grid n_in", g2)
    # distributions
    L += ["### Per-ROI distributions (p10 / p50 / p90)", "", "| cond | L1 | LPIPS | SSIM | hp_corr | re token cos |", "|---|---|---|---|---|---|"]
    for c in CONDS:
        d = D[D.cond == c]
        q = lambda k: " / ".join(f"{v:.3f}" for v in np.nanpercentile(d[k], [10, 50, 90]))  # noqa: E731
        L.append(f"| {c} | {q('l1')} | {q('lpips')} | {q('ssim')} | {q('hp_corr')} | {q('re_tok_cos')} |")
    L.append("")
    # paired differences
    L += ["### Paired differences (mean per-ROI x - y [image-bootstrap 95% CI]); all ROIs and by size bin", ""]
    mets = ["l1", "lpips", "ssim", "hp_corr", "re_tok_cos", "re_pool_cos"]
    out = {}
    for gname, sel in [("all", allsel)] + [(b, D.bin == b) for b in ("D1-4", "D5-16", "D17+")]:
        L += [f"**{gname}**", "", "| x - y | " + " | ".join(mets) + " |", "|---|" + "---|" * len(mets)]
        for x, y in PAIRS:
            cells = []
            for mt in mets:
                v, lo, hi = boot(D[sel], x, y, mt)
                out[f"{gname}/{x}-{y}/{mt}"] = dict(point=v, lo=lo, hi=hi)
                cells.append(f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            L.append(f"| {x} - {y} | " + " | ".join(cells) + " |")
        L.append("")
    # RGB vs latent fidelity
    L += ["### RGB fidelity vs latent fidelity", "", "Across conditions (means, all ROIs):", "",
          "| cond | LPIPS | SSIM | re token cos | re pooled cos |", "|---|---|---|---|---|"]
    cm = D.groupby("cond")[["lpips", "ssim", "re_tok_cos", "re_pool_cos"]].mean().loc[CONDS]
    for c, r in cm.iterrows():
        L.append(f"| {c} | {r.lpips:.3f} | {r.ssim:.3f} | {r.re_tok_cos:.3f} | {r.re_pool_cos:.3f} |")
    L += ["", f"Spearman across the 8 condition means: LPIPS vs re token cos {spearmanr(cm.lpips, cm.re_tok_cos)[0]:+.2f}; "
          f"SSIM vs re token cos {spearmanr(cm.ssim, cm.re_tok_cos)[0]:+.2f}.", "",
          "Within condition, across ROIs (Spearman LPIPS vs re token cos):", ""]
    for c in CONDS:
        d = D[D.cond == c]
        L.append(f"- {c}: {spearmanr(d.lpips, d.re_tok_cos, nan_policy='omit')[0]:+.3f}")
    L.append("")
    open(os.path.join(R, "tables.md"), "w").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(R, "paired_diffs.json"), "w"), indent=1)
    figures(R, FIG, D)
    print("\n".join(L))


def figures(R, FIG, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # distributions by bin
    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    for j, b in enumerate(("D1-4", "D5-16", "D17+")):
        for i, (k, rng) in enumerate((("lpips", (0, 0.8)), ("re_tok_cos", (-0.1, 1.0)))):
            for c in CONDS:
                d = D[(D.cond == c) & (D.bin == b)][k].dropna()
                ax[i, j].hist(d, bins=50, range=rng, histtype="step", lw=1.6, label=c)
            ax[i, j].set_title(f"{k} per ROI, {b}")
    ax[0, 0].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dist_lpips_recos_by_bin.png"), dpi=100)
    plt.close(fig)
    # RGB vs latent fidelity scatter (condition x bin means)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for c in CONDS:
        g = D[D.cond == c].groupby("bin")[["lpips", "re_tok_cos", "ssim"]].mean()
        ax[0].plot(g.lpips, g.re_tok_cos, "o-", label=c)
        ax[1].plot(g.ssim, g.re_tok_cos, "o-", label=c)
        for b, r in g.iterrows():
            ax[0].annotate(b, (r.lpips, r.re_tok_cos), fontsize=6)
    ax[0].set_xlabel("LPIPS (lower = better RGB)")
    ax[1].set_xlabel("SSIM (higher = better RGB)")
    for x in ax:
        x.set_ylabel("re-encoded token cos to C0 latent")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rgb_vs_latent_fidelity.png"), dpi=100)
    plt.close(fig)
    # example grids per level x bin
    viz = {}
    for f in sorted(glob.glob(os.path.join(R, "viz_shard*.pt"))):
        viz.update(torch.load(f, weights_only=False))
    smp = json.load(open("configs/raediag/sample_rois.json"))["viz_ann_ids"]
    order = ["gt", "A", "B_in", "B", "C", "D", "E", "F", "MB"]
    meta = D[D.cond == "A"].set_index("ann_id")
    for k0 in range(0, len(smp), 3):
        ids = [i for i in smp[k0:k0 + 3] if i in viz]
        if not ids:
            continue
        lv, b = meta.loc[ids[0], "level"], meta.loc[ids[0], "bin"]
        fig, ax = plt.subplots(len(ids), len(order) + 1, figsize=(2.3 * (len(order) + 1), 2.5 * len(ids)))
        ax = np.atleast_2d(ax)
        for r, aid in enumerate(ids):
            v = viz[aid]
            m = v["mask"]
            ax[r, 0].imshow(v["full"].permute(1, 2, 0).numpy())
            ax[r, 0].set_title(f"full input  ann {aid}", fontsize=7)
            for c, o in enumerate(order):
                x = v[o].float()
                x = x * m + (1 - m) * (0.5 + 0.3 * x)
                ax[r, c + 1].imshow(x.permute(1, 2, 0).clamp(0, 1).numpy())
                if o != "gt":
                    row = D[(D.ann_id == aid) & (D.cond == o)].iloc[0]
                    ax[r, c + 1].set_title(f"{o}  LPIPS {row.lpips:.2f}\nre-cos {row.re_tok_cos:.2f}", fontsize=7)
                else:
                    ax[r, c + 1].set_title(f"GT crop  n_in {int(meta.loc[aid, 'n_in'])}", fontsize=7)
        for x in ax.flat:
            x.set_xticks([])
            x.set_yticks([])
        fig.suptitle(f"{lv} {b} (fixed seed-0 sample; padding dimmed)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"examples_{lv}_{b}.png"), dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    main()
