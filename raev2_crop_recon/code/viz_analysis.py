"""Analysis / visualisation only (no training): what do Teacher / Bilinear / A / B / C latents reconstruct through the frozen RAEv2 decoder?

Reuses: rae_common (official encoder/decoder, 256 letterbox, roi_geometry, stats), 12k checkpoints runs/{A,B,C}/ckpt_12000.pt,
pixel.rgb_metrics (= recon_rae.py metrics). Every method gets the SAME ROI, the SAME GT crop canvas (256 letterbox of the original crop,
BICUBIC, ImageNet-mean padding) and the same content mask. Decoder BF16 autocast, output clamped to [0,1] for display/metrics (as recon_rae).

ROI selection (fixed rules, decided before looking at any reconstruction; rng = numpy default_rng(0) per group):
  G1 small objects  : hval (rae_crop_pilot head_val, COCO GT boxes outside all training pools), F336 n_in 5-16; COCO categories in the order
                      clock, cell phone, laptop, bicycle, tv, keyboard, dog, book -> one random ROI per available category, first 4.
  G2 small parts    : hval has NO part annotations -> PACO test-primary (images outside every decoder pool; the manifest used by
                      recon_rae.py 'pacot'), level part, F336 n_in 1-4; object categories in the order clock, cellular_telephone,
                      laptop_computer, bicycle, dog, shoe, television_set, mug -> one random part ROI per object, first 4.
  G3 pattern/detail : PACO test-primary objects, F336 n_in 5-16, positive pattern label in the order text, striped, logo, woven, floral,
                      checkered -> one random ROI per label (no repeated ROI), first 4. (hval has no attribute labels.)
High-pass (Figure 2) = exactly the hp_corr filter of rcp.hp_corr: luminance Y = .299 R + .587 G + .114 B, HP = Y - box7(Y)
  (avg_pool 7x7, stride 1, pad 3, count_include_pad=False), shown on the content region eroded by 3 px (box7 fully inside content),
  ONE fixed symmetric scale for every method: +-s, s = 99th percentile of |HP(GT)| over the content pixels of all selected ROIs.
Error map (Figure 3) = mean over RGB of |pred - GT| per pixel, one fixed scale [0, 0.5] (inferno) for every method; padding = light grey
  (not part of any metric). Content-only version zooms each tile to its content rectangle.
  python code/viz_analysis.py -> figures/*.png, results/visualization_selection.json, results/visualization_tiles.csv
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rae_common as RC  # noqa: E402
import pixel as PX  # noqa: E402
import train_rae as TR  # noqa: E402

FIG = os.path.join(RC.HERE, "figures")
PACO = "/workspace/Areum/research/2026-10-24/dinov2_oracle_region_decoder/perception_cache/paco_lp"
COCO_NAMES = {85: "clock", 77: "cell phone", 73: "laptop", 2: "bicycle", 72: "tv", 76: "keyboard", 18: "dog", 84: "book"}
METHODS = [("teacher", "Teacher"), ("bil", "Bilinear"), ("A", "A Feature-only"), ("B", "B Pixel-only"), ("C", "C Feature+Pixel")]


def select():
    sel = []
    hv = pd.read_parquet(os.path.join(RC.CACHE, "feat", "hval_rois.parquet"))
    d = hv[(hv.n_in >= 5) & (hv.n_in <= 16)]
    rng = np.random.default_rng(0)
    for cid in (85, 77, 73, 2, 72, 76, 18, 84):
        c = d[d.cat_id == cid]
        if len(c) and len([s for s in sel if s["group"] == "G1"]) < 4:
            r = c.iloc[int(rng.choice(len(c)))]
            sel.append(dict(group="G1", label="small object", source="hval", roi_id=str(r.roi_id), image_id=int(r.image_id),
                            path=os.path.join(RC.COCO_TRAIN, f"{int(r.image_id):012d}.jpg"), ltrb=[int(r.l), int(r.t), int(r.r), int(r.b)],
                            category=COCO_NAMES[cid], n_in_f336=int(r.n_in), attrs=""))
    names = json.load(open(os.path.join(PACO, "attributes.json")))["names"]
    df = pd.read_parquet(os.path.join(PACO, "manifest_test.parquet"))
    Y = np.load(os.path.join(PACO, "labels_test.npy"))
    keep = ((df.bw > 0) & (df.bh > 0)).values
    df, Y = df[keep].reset_index(drop=True), Y[keep]
    prim = ~df.in_decoder_pool.values

    def attrs(i):
        return ",".join(names[j] for j in np.flatnonzero(Y[i] == 1))

    def paco_rec(group, label, i):
        r = df.iloc[i]
        return dict(group=group, label=label, source="paco_test_primary", roi_id=f"paco_test_row{i}", image_id=int(r.image_id), path=r.path,
                    ltrb=[int(r.l), int(r.t), int(r.r), int(r.b)], category=r.category, n_in_f336=int(r.n_in), attrs=attrs(i))
    rng = np.random.default_rng(0)
    parts = np.flatnonzero(prim & (df.level.values == "part") & (df.n_in.values <= 4))
    pobj = df.category.values[parts]
    for ob in ("clock", "cellular_telephone", "laptop_computer", "bicycle", "dog", "shoe", "television_set", "mug"):
        c = parts[np.array([x.split(":")[0] == ob for x in pobj])]
        if len(c) and len([s for s in sel if s["group"] == "G2"]) < 4:
            sel.append(paco_rec("G2", "small part", int(c[rng.choice(len(c))])))
    rng = np.random.default_rng(0)
    objs = np.flatnonzero(prim & (df.level.values == "object") & (df.n_in.values >= 5) & (df.n_in.values <= 16))
    used = set()
    for pat in ("text", "striped", "logo", "woven", "floral", "checkered"):
        j = names.index(pat)
        c = [i for i in objs[Y[objs, j] == 1] if i not in used]
        if c and len([s for s in sel if s["group"] == "G3"]) < 4:
            i = int(c[rng.choice(len(c))])
            used.add(i)
            rec = paco_rec("G3", f"pattern: {pat}", i)
            sel.append(rec)
    return sel


@torch.no_grad()
def reconstruct(sel, dev):
    rae = RC.FrozenRAE(dev)
    lp = PX.load_lpips(dev)
    P = torch.as_tensor(RC.query_coords(), device=dev)
    decs = {k: TR.load_model(os.path.join(RC.CACHE, "runs", k, "ckpt_12000.pt"), dev).eval() for k in ("A", "B", "C")}
    tiles = []
    for s in sel:
        img = Image.open(s["path"]).convert("RGB")
        g = RC.roi_geometry(img.size[0], img.size[1], tuple(s["ltrb"]))
        full = rae.encode(RC.u8_canvas(img, g["full"])[None])
        crop_u8 = RC.u8_canvas(img.crop(g["ltrb"]), g["crop"])
        T = rae.encode(crop_u8[None])
        b = RC.to_device(RC.collate_geoms([{k: g[k] for k in RC.GEOM_KEYS}], [0]), dev)
        reps = {"teacher": T, "bil": RC.bilinear(full, b)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for k, m in decs.items():
                reps[k] = m(RC.gather_roi(full, b), None, b, P).float()
        gt = crop_u8.to(dev).float()[None] / 255
        mask = RC.content_mask([g["crop"].rect()], dev)
        out = dict(sel=s, gt=gt[0].cpu(), mask=mask[0, 0].cpu(), rect=g["crop"].rect(), n_in_rae=g["n_in"], rec={}, met={})
        for k, Z in reps.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = rae.decode(Z).float().clamp(0, 1)
            met = {kk: float(v[0]) for kk, v in PX.rgb_metrics(y, gt, mask, lp).items()}
            mse, cos = PX.feat_mse_cos(Z, T, b["valid_w"])
            met.update(cos=float(cos[0]), mse=float(mse[0]))
            out["rec"][k], out["met"][k] = y[0].cpu(), met
        tiles.append(out)
    return tiles


def lum(x):
    return 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]


def highpass(x, m):
    Yl = lum(x)[None, None]
    hp = (Yl - F.avg_pool2d(Yl, 7, 1, 3, count_include_pad=False))[0, 0]
    er = (F.avg_pool2d(m[None, None], 7, 1, 3) > 0.999)[0, 0]
    return hp, er


def show_rgb(ax, x, m):
    x = x.permute(1, 2, 0).numpy()
    mm = m.numpy()[..., None]
    ax.imshow(x * mm + (1 - mm) * (0.55 + 0.25 * x))


def row_label(t):
    s = t["sel"]
    return f"{s['group']} {s['label']}\n{s['category']}\nF336 n_in {s['n_in_f336']} / RAE {t['n_in_rae']}\n{s['roi_id']}"


def fig_grid(tiles, ncols):
    n = len(tiles)
    fig, axes = plt.subplots(n, ncols, figsize=(1.9 * ncols + 1.6, 1.95 * n))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def figure1(tiles, path):
    fig, axes = fig_grid(tiles, 6)
    for i, t in enumerate(tiles):
        show_rgb(axes[i, 0], t["gt"], t["mask"])
        axes[i, 0].set_ylabel(row_label(t), fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
        if i == 0:
            axes[i, 0].set_title("GT crop", fontsize=8)
        for j, (k, nm) in enumerate(METHODS, 1):
            show_rgb(axes[i, j], t["rec"][k], t["mask"])
            mt = t["met"][k]
            axes[i, j].set_title((f"{nm}\n" if i == 0 else "") + f"L1 {mt['l1']:.3f}  LPIPS {mt['lpips']:.2f}", fontsize=6.5)
    fig.suptitle("Figure 1. Reconstruction through the frozen RAEv2 decoder (12k ckpts, same ROI / GT crop per row; padding dimmed)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure2(tiles, path):
    vals = []
    for t in tiles:
        hp, er = highpass(t["gt"], t["mask"])
        vals.append(hp[er].abs().numpy())
    s = float(np.percentile(np.concatenate(vals), 99))
    fig, axes = fig_grid(tiles, 6)
    for i, t in enumerate(tiles):
        for j, (k, nm) in enumerate([("gt", "GT")] + METHODS):
            x = t["gt"] if k == "gt" else t["rec"][k]
            hp, er = highpass(x, t["mask"])
            img = np.where(er.numpy(), hp.numpy(), np.nan)
            cm = matplotlib.colormaps["gray"].copy()
            cm.set_bad((0.85, 0.80, 0.70))
            axes[i, j].imshow(img, cmap=cm, vmin=-s, vmax=s)
            ttl = (f"{nm} high-pass\n" if i == 0 else "")
            if k != "gt":
                ttl += f"hp_corr {t['met'][k]['hp_corr']:.2f}  hf {t['met'][k]['hf_ratio']:.2f}"
            axes[i, j].set_title(ttl, fontsize=6.5)
        axes[i, 0].set_ylabel(row_label(t), fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
    fig.suptitle(f"Figure 2. High-pass luminance Y - box7(Y) (count_include_pad=False), content eroded 3 px; one fixed scale +-{s:.3f} for all "
                 f"(99th pct |HP(GT)|); beige = padding/border", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return s


def figure3(tiles, path, zoom):
    fig, axes = fig_grid(tiles, 6)
    cm = matplotlib.colormaps["inferno"].copy()
    cm.set_bad((0.82, 0.82, 0.82))
    for i, t in enumerate(tiles):
        show_rgb(axes[i, 0], t["gt"], t["mask"])
        axes[i, 0].set_ylabel(row_label(t), fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
        if i == 0:
            axes[i, 0].set_title("GT crop", fontsize=8)
        m = t["mask"].numpy() > 0
        ox, oy, nw, nh = t["rect"]
        for j, (k, nm) in enumerate(METHODS, 1):
            e = (t["rec"][k] - t["gt"]).abs().mean(0).numpy()
            im = axes[i, j].imshow(np.where(m, e, np.nan), cmap=cm, vmin=0, vmax=0.5)
            axes[i, j].set_title((f"{nm} vs GT\n" if i == 0 else "") + f"content |err| {e[m].mean():.3f}", fontsize=6.5)
        if zoom:
            for j in range(6):
                axes[i, j].set_xlim(ox - 0.5, ox + nw - 0.5)
                axes[i, j].set_ylim(oy + nh - 0.5, oy - 0.5)
    cax = fig.add_axes([0.93, 0.3, 0.008, 0.4])
    fig.colorbar(im, cax=cax).set_label("mean |pred - GT| over RGB", fontsize=7)
    fig.suptitle("Figure 3. Pixel-wise absolute error vs GT crop, one scale [0, 0.5] for all methods" +
                 (" — zoomed to the crop content" if zoom else "; grey = letterbox padding (excluded from metrics)"), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.92, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    RC.set_fp32()
    sel = select()
    assert [s["group"] for s in sel].count("G1") == 4 and [s["group"] for s in sel].count("G2") == 4 and [s["group"] for s in sel].count("G3") == 4, sel
    tiles = reconstruct(sel, "cuda")
    figure1(tiles, os.path.join(FIG, "reconstruction_comparison.png"))
    s = figure2(tiles, os.path.join(FIG, "high_frequency_comparison.png"))
    figure3(tiles, os.path.join(FIG, "reconstruction_error_maps.png"), zoom=False)
    figure3(tiles, os.path.join(FIG, "reconstruction_error_maps_content.png"), zoom=True)
    rows = [dict(**{k: v for k, v in t["sel"].items() if k != "path"}, n_in_rae=t["n_in_rae"], method=k, **t["met"][k]) for t in tiles for k, _ in METHODS]
    pd.DataFrame(rows).to_csv(os.path.join(RC.RES, "visualization_tiles.csv"), index=False)
    json.dump(dict(rules=__doc__.split("ROI selection")[1].split("High-pass")[0].strip(), highpass_scale=s, rois=sel),
              open(os.path.join(RC.RES, "visualization_selection.json"), "w"), indent=1)
    print(pd.DataFrame(rows).groupby(["group", "method"])[["l1", "lpips", "de_mean", "hp_corr", "cos"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
