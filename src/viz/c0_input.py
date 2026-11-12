"""C0 encoder input visualisation: what DINOv2 actually sees for a GT-crop ROI vs for the full image (PACO val only).

Per ROI (fixed list configs/ivc/sample_rois.json 'viz_ann_ids', chosen with seed 0 before any feature was computed):
  1. original image + GT box (integer box l, t, r, b of the manifest)
  2. original pixels around the box (box x3 context), box drawn
  3. C0 encoder input = to_tensor_canvas(img.crop(ltrb), roi_geometry(...)['crop']) de-normalised: 336 letterbox,
     24 x 24 patch grid (14 px), content rect; padding = ImageNet mean colour (0 after normalisation)
  4. F336 encoder input = to_tensor_canvas(img, Letterbox(W, H)) de-normalised, zoomed around the ROI, 14 px grid,
     cells overlapping the box filled (= n_in tokens the full-image encoder spends on this ROI)
The tensors come from the same functions the readers were trained with (src.msd.common), so the picture is the input.

  python -m src.viz.c0_input --out output/2026-11-06/c0_input
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import src.readout.common as RC  # noqa: E402
from src.config import CACHE, ROOT  # noqa: E402
from src.msd.common import S, Letterbox, roi_geometry, to_tensor_canvas  # noqa: E402

P = 14
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm(x):
    return np.clip(x.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)


def grid(ax, n=S // P, color="w", alpha=0.25):
    for k in range(n + 1):
        ax.axhline(k * P - 0.5, color=color, lw=0.4, alpha=alpha)
        ax.axvline(k * P - 0.5, color=color, lw=0.4, alpha=alpha)


def box(ax, l, t, r, b, color="yellow", lw=1.5):
    ax.add_patch(mpatches.Rectangle((l - 0.5, t - 0.5), r - l, b - t, fill=False, ec=color, lw=lw))


def full_cells(lb, W, H, ltrb):
    """F336 cells (row, col) whose 14 px square overlaps the box mapped to the full canvas."""
    l, t, r, b = ltrb
    X0, X1 = lb.ox + l * lb.nw / W, lb.ox + r * lb.nw / W
    Y0, Y1 = lb.oy + t * lb.nh / H, lb.oy + b * lb.nh / H
    cols = [k for k in range(S // P) if k * P < X1 and (k + 1) * P > X0]
    rows = [k for k in range(S // P) if k * P < Y1 and (k + 1) * P > Y0]
    return rows, cols, (X0, Y0, X1, Y1)


def draw_row(axs, row, y, names):
    img = Image.open(row.path).convert("RGB")
    W, H = img.size
    ltrb = (int(row.l), int(row.t), int(row.r), int(row.b))
    g = roi_geometry(W, H, ltrb)
    lb_c, lb_f = g["crop"], Letterbox(W, H)
    crop_in = denorm(to_tensor_canvas(img.crop(g["ltrb"]), lb_c))
    full_in = denorm(to_tensor_canvas(img, lb_f))
    l, t, r, b = ltrb
    pos = [names[a] for a in np.where(y == 1)[0]]

    a = axs[0]
    a.imshow(img)
    box(a, l, t, r, b, lw=2)
    a.set_title(f"original {W}x{H}  ann {row.ann_id}\n{row.level}: {row.category}", fontsize=9)

    a = axs[1]
    cx, cy, half = (l + r) / 2, (t + b) / 2, 1.5 * max(r - l, b - t)
    a.imshow(img)
    box(a, l, t, r, b, lw=2)
    a.set_xlim(max(cx - half, 0), min(cx + half, W))
    a.set_ylim(min(cy + half, H), max(cy - half, 0))
    a.set_title(f"original pixels around box (x3)\nbox {r - l}x{b - t} px", fontsize=9)

    a = axs[2]
    a.imshow(crop_in)
    grid(a, color="w", alpha=0.3)
    box(a, lb_c.ox, lb_c.oy, lb_c.ox + lb_c.nw, lb_c.oy + lb_c.nh, color="cyan", lw=1.2)
    n_valid = int((lb_c.coverage() > 0).sum())
    a.set_title(f"C0 encoder input (336 letterbox)\ncontent {lb_c.nw}x{lb_c.nh}, x{lb_c.nw / (r - l):.1f} upscale, "
                f"{n_valid}/576 tokens", fontsize=9)

    a = axs[3]
    rows, cols, (X0, Y0, X1, Y1) = full_cells(lb_f, W, H, ltrb)
    a.imshow(full_in)
    for i in rows:
        for j in cols:
            a.add_patch(mpatches.Rectangle((j * P - 0.5, i * P - 0.5), P, P, fc="red", alpha=0.25, ec="none"))
    grid(a, color="w", alpha=0.6)
    a.add_patch(mpatches.Rectangle((X0 - 0.5, Y0 - 0.5), X1 - X0, Y1 - Y0, fill=False, ec="yellow", lw=1.5))
    ccx, ccy = (X0 + X1) / 2, (Y0 + Y1) / 2
    hh = max(3 * P, 1.5 * max(X1 - X0, Y1 - Y0))
    a.set_xlim(max(ccx - hh, 0), min(ccx + hh, S))
    a.set_ylim(min(ccy + hh, S), max(ccy - hh, 0))
    a.set_title(f"F336 full-image input (zoom)\nbox {X1 - X0:.1f}x{Y1 - Y0:.1f} px, n_in {len(rows) * len(cols)} tokens "
                f"(manifest {int(row.n_in)})", fontsize=9)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
    axs[0].set_ylabel(f"D{row['size']}  n_in {int(row.n_in)}\n" + ", ".join(pos[:5]) + ("…" if len(pos) > 5 else ""), fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/2026-11-06/c0_input")
    ap.add_argument("--by", choices=("level", "bin"), default="bin")
    ap.add_argument("--extra", type=int, default=1, help="extra random ROIs per (level, bin) in --by bin")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ids = json.load(open(os.path.join(ROOT, "configs", "ivc", "sample_rois.json")))["viz_ann_ids"]
    names = json.load(open(os.path.join(CACHE, "paco_lp", "attributes.json")))["names"]
    df, Y = RC.load_data(splits=("val",))["val"]
    pick = [int(np.where(df.ann_id.values == a)[0][0]) for a in ids]
    lv, sz = df.level.values, df["size"].values
    if args.by == "level":
        for lev in ("object", "part"):
            rr = [i for i in pick if lv[i] == lev]
            save(rr, df, Y, names, f"PACO val {lev} ROIs", os.path.join(args.out, f"c0_input_{lev}.png"))
        return
    # by D-bin: the 2 fixed ROIs per (level, bin) + args.extra more per (level, bin) drawn uniformly at random
    # (default_rng(1), fixed ROIs excluded, sorted row order) -> rows = object first, then part
    rng = np.random.default_rng(1)
    fixed = set(pick)
    rec = []
    for b in BINS:
        rr = []
        for lev in ("object", "part"):
            fx = [i for i in pick if lv[i] == lev and sz[i] == b]
            pool = np.array([i for i in np.where((lv == lev) & (sz == b))[0] if i not in fixed])
            ex = sorted(rng.choice(pool, args.extra, replace=False).tolist())
            rr += fx + ex
            rec += [dict(bin=b, level=lev, ann_id=int(df.ann_id.values[i]), source=s)
                    for i, s in [(i, "fixed") for i in fx] + [(i, "extra_rng1") for i in ex]]
        save(rr, df, Y, names, f"PACO val D{b} ROIs (object, then part)",
             os.path.join(args.out, f"c0_input_D{b}.png"))
    json.dump(rec, open(os.path.join(args.out, "rois_by_bin.json"), "w"), indent=1)
    bin_stats(df, args.out)


BINS = ("1-4", "5-16", "17+")


def save(rr, df, Y, names, title, path):
    fig, axs = plt.subplots(len(rr), 4, figsize=(16, 4.1 * len(rr)))
    for k, i in enumerate(rr):
        draw_row(axs[k], df.iloc[i], Y[i], names)
    fig.suptitle(f"{title}: original vs C0 crop input vs F336 full input "
                 f"(row label = size bin, n_in, positive attributes)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(path)


def bin_stats(df, out):
    """All val ROIs: C0 upscale (content long side / box long side), valid C0 tokens (cov > 0), padding share, F336 n_in."""
    rows = []
    for W, H, l, t, r, b in df[["W", "H", "l", "t", "r", "b"]].values.astype(int):
        lb = roi_geometry(W, H, (l, t, r, b))["crop"]
        nv = int((lb.coverage() > 0).sum())
        rows.append((max(lb.nw, lb.nh) / max(r - l, b - t), nv, max(r - l, b - t), max(r - l, b - t) / max(1, min(r - l, b - t))))
    a = np.array(rows)
    st = df[["level", "size", "n_in"]].copy()
    st["upscale"], st["c0_tokens"], st["long_px"], st["aspect"] = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    st["upscale_gt4"] = st.upscale > 4
    st["pad_share"] = 1 - st.c0_tokens / 576
    g = st.groupby(["size", "level"]).agg(n=("n_in", "size"), n_in_med=("n_in", "median"), long_px_med=("long_px", "median"),
                                          upscale_med=("upscale", "median"), upscale_gt4=("upscale_gt4", "mean"),
                                          c0_tokens_med=("c0_tokens", "median"), pad_share_mean=("pad_share", "mean"),
                                          aspect_med=("aspect", "median"))
    g = g.reindex([(b, lv) for b in BINS for lv in ("object", "part")])
    g.to_csv(os.path.join(out, "bin_stats.csv"))
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
