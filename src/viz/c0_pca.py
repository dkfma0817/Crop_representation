"""PCA-RGB of frozen C0 crop tokens (PACO val only), next to Bilinear (F336 tokens read on the C0 crop grid).

Features = exactly the C0 / Bilinear reader inputs of the IVC probes: timm vit_base_patch14_dinov2.lvd142m (RC.load_vit,
frozen), 336 letterbox, final-LN patch tokens 24x24x768 (RC.encode_layers(..., [12]), bf16 autocast -> fp32).
  C0       : img.crop(ltrb) -> roi_geometry()['crop'] letterbox -> encoder
  Bilinear : full image -> Letterbox(W, H) -> encoder -> grid_sample(bil_grid, bilinear, border, align_corners=False)
Valid cells = C0 letterbox coverage > 0; padding cells drawn grey and never used in a fit.

Two projections (top-3 principal components -> RGB):
  per-ROI : PCA fitted on THIS ROI's valid C0 tokens; Bilinear projected on the same basis/centre. Colour range = 2-98th
            percentile of the C0 projections of this ROI (shared by both panels). Shows the ROI's own spatial structure.
  global  : PCA fitted once on valid C0 tokens of 2,000 val ROIs (rng 0, <= 64 random valid tokens per ROI); colour range
            = 2-98th percentile of the fit sample. Same colours mean the same direction for every ROI and both conditions.
Signs of principal components are arbitrary. ROIs = rois_by_bin.json of src.viz.c0_input (fixed + rng-1 extras).

Backbones (--backbone): dinov2 = timm vit_base_patch14_dinov2.lvd142m @ 336 (the IVC reader features, RC.load_vit);
dinov3 = timm vit_base_patch16_dinov3.lvd1689m @ 384 (patch 16 -> the same 24 x 24 = 576 token grid; 1 CLS + 4 register
prefix tokens dropped). Geometry is recomputed here for any (S, P) with the formulas of data.preprocessing (Letterbox,
to_tensor_canvas, roi_geometry bil_grid); at (336, 14) it is asserted equal to the original functions for every drawn ROI.

  CUDA_VISIBLE_DEVICES=0 python -m src.viz.c0_pca --backbone dinov2 --out output/2026-11-06/c0_pca
  CUDA_VISIBLE_DEVICES=0 python -m src.viz.c0_pca --backbone dinov3 --out output/2026-11-06/c0_pca_dinov3
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import src.readout.common as RC  # noqa: E402
from src.msd.common import Letterbox, roi_geometry, to_tensor_canvas  # noqa: E402
from src.viz.c0_input import BINS, box, denorm  # noqa: E402

G = 24
GREY = np.array([0.6, 0.6, 0.6])


BACKBONES = {"dinov2": dict(S=336, P=14, name="DINOv2-B/14 lvd142m @336"),
             "dinov3": dict(S=384, P=16, name="DINOv3-B/16 lvd1689m @384")}
MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_T = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class LB:
    """data.preprocessing.Letterbox for an arbitrary canvas S and patch P (same rounding / centring)."""

    def __init__(self, w, h, S, P):
        self.w, self.h, self.S, self.P = int(w), int(h), S, P
        s = S / max(self.w, self.h)
        self.nw = min(S, max(1, int(round(self.w * s))))
        self.nh = min(S, max(1, int(round(self.h * s))))
        self.ox, self.oy = (S - self.nw) // 2, (S - self.nh) // 2

    def coverage(self):
        e = np.arange(G + 1) * self.P
        cx = np.clip(np.minimum(e[1:], self.ox + self.nw) - np.maximum(e[:-1], self.ox), 0, None) / self.P
        cy = np.clip(np.minimum(e[1:], self.oy + self.nh) - np.maximum(e[:-1], self.oy), 0, None) / self.P
        return (cy[:, None] * cx[None, :]).reshape(-1).astype(np.float32)

    def canvas(self, img):
        r = np.asarray(img.convert("RGB").resize((self.nw, self.nh), resample=Image.BICUBIC), dtype=np.float32)
        x = torch.zeros(3, self.S, self.S)
        x[:, self.oy:self.oy + self.nh, self.ox:self.ox + self.nw] = (torch.from_numpy(r).permute(2, 0, 1) / 255.0 - MEAN_T) / STD_T
        return x


def geometry(W, H, ltrb, S, P):
    """-> crop LB, full LB, bil_grid (576, 2): C0 cell centre -> original px -> full canvas -> grid_sample coordinate."""
    l, t, r, b = ltrb
    crop, full = LB(r - l, b - t, S, P), LB(W, H, S, P)
    i, j = np.divmod(np.arange(G * G), G)
    xo = l + ((j + 0.5) * P - crop.ox) * (r - l) / crop.nw
    yo = t + ((i + 0.5) * P - crop.oy) * (b - t) / crop.nh
    X, Y = full.ox + xo * full.nw / W, full.oy + yo * full.nh / H
    return crop, full, np.stack([X / (S / 2) - 1, Y / (S / 2) - 1], 1).astype(np.float32)


def load_roi(row, S, P, check=False):
    img = Image.open(row.path).convert("RGB")
    W, H = img.size
    ltrb = (int(row.l), int(row.t), int(row.r), int(row.b))
    crop, full, grid = geometry(W, H, ltrb, S, P)
    xc, xf = crop.canvas(img.crop(ltrb)), full.canvas(img)
    if check:                                          # (336, 14): must equal the functions the readers were trained with
        g = roi_geometry(W, H, ltrb)
        assert tuple(g["ltrb"]) == ltrb
        assert np.array_equal(g["crop"].coverage(), crop.coverage())
        assert np.abs(g["bil_grid"] - grid).max() < 1e-5, np.abs(g["bil_grid"] - grid).max()
        assert torch.equal(to_tensor_canvas(img.crop(ltrb), g["crop"]), xc)
        assert torch.equal(to_tensor_canvas(img, Letterbox(W, H)), xf)
    return img, crop, grid, xc, xf


def load_encoder(backbone, dev):
    import timm
    if backbone == "dinov2":
        vit = RC.load_vit()
    else:
        vit = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, img_size=384, dynamic_img_size=True)
        for p in vit.parameters():
            p.requires_grad_(False)
        vit.eval()
    return vit.to(dev)


@torch.no_grad()
def encode(vit, x, dev):
    """final-norm patch tokens (B, 576, D) fp32 (= RC.encode_layers(..., [12]) for dinov2), bf16 autocast."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = vit.forward_features(x.to(dev))
    return t[:, vit.num_prefix_tokens:].float()


@torch.no_grad()
def bilinear(full_tok, bil_grid):
    """full_tok (576, D), bil_grid (576, 2) -> (576, D), = src.ivc.common.sample_roi(mode='bilinear') for one ROI."""
    m = full_tok.T.reshape(1, -1, G, G)
    g = torch.as_tensor(bil_grid, dtype=torch.float32, device=m.device).view(1, G, G, 2)
    return F.grid_sample(m, g, mode="bilinear", padding_mode="border", align_corners=False).flatten(2)[0].T


def pca(X, k=3):
    mu = X.mean(0)
    U, S, V = torch.linalg.svd(X - mu, full_matrices=False)
    ev = S ** 2
    return mu, V[:k].T, (ev[:k] / ev.sum()).cpu().numpy()


def to_rgb(P, lo, hi, valid):
    rgb = np.clip((P - lo) / (hi - lo + 1e-8), 0, 1)
    rgb[~valid] = GREY
    return rgb.reshape(G, G, 3)


@torch.no_grad()
def global_basis(vit, df, dev, S, P, n_roi=2000, per_roi=64, bs=64):
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(df), n_roi, replace=False))
    toks = []
    for s in range(0, n_roi, bs):
        xs, vs = [], []
        for i in idx[s:s + bs]:
            _, crop, _, xc, _ = load_roi(df.iloc[i], S, P)
            xs.append(xc)
            vs.append(crop.coverage() > 0)
        T = encode(vit, torch.stack(xs), dev)
        for t, v in zip(T, vs):
            vi = np.where(v)[0]
            pick = rng.choice(vi, min(per_roi, len(vi)), replace=False)
            toks.append(t[torch.as_tensor(pick, device=dev)])
    X = torch.cat(toks)
    mu, B, evr = pca(X)
    P = ((X - mu) @ B).cpu().numpy()
    lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
    return dict(mu=mu, B=B, evr=evr, lo=lo, hi=hi, n_tok=len(X), roi_idx=idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/2026-11-06/c0_pca")
    ap.add_argument("--rois", default="output/2026-11-06/c0_input/rois_by_bin.json")
    ap.add_argument("--backbone", choices=tuple(BACKBONES), default="dinov2")
    args = ap.parse_args()
    S, P, bname = BACKBONES[args.backbone]["S"], BACKBONES[args.backbone]["P"], BACKBONES[args.backbone]["name"]
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    vit = load_encoder(args.backbone, dev)
    df, _ = RC.load_data(splits=("val",))["val"]
    rec = json.load(open(args.rois))

    gb = global_basis(vit, df, dev, S, P)
    np.savez(os.path.join(args.out, "global_basis.npz"), mu=gb["mu"].cpu().numpy(), B=gb["B"].cpu().numpy(), evr=gb["evr"],
             lo=gb["lo"], hi=gb["hi"], roi_idx=gb["roi_idx"])
    print(f"global PCA: {gb['n_tok']} tokens, top-3 EVR {gb['evr'].round(3)} (sum {gb['evr'].sum():.3f})")

    stats = []
    for b in BINS:
        rows = [r for r in rec if r["bin"] == b]
        fig, axs = plt.subplots(len(rows), 6, figsize=(24, 4.1 * len(rows)))
        for k, r in enumerate(rows):
            i = int(np.where(df.ann_id.values == r["ann_id"])[0][0])
            row = df.iloc[i]
            img, crop, grid, xc, xf = load_roi(row, S, P, check=(S, P) == (336, 14))
            T = encode(vit, torch.stack([xc, xf]), dev)
            c0, bil = T[0], bilinear(T[1], grid)
            valid = crop.coverage() > 0
            vt = torch.as_tensor(valid, device=dev)

            mu, B, evr = pca(c0[vt])
            Pc, Pb = ((c0 - mu) @ B).cpu().numpy(), ((bil - mu) @ B).cpu().numpy()
            lo, hi = np.percentile(Pc[valid], 2, 0), np.percentile(Pc[valid], 98, 0)
            Gc, Gb = ((c0 - gb["mu"]) @ gb["B"]).cpu().numpy(), ((bil - gb["mu"]) @ gb["B"]).cpu().numpy()
            # share of Bilinear variance (valid cells) captured by the C0 per-ROI top-3 axes, and its spread vs C0
            var_c = ((c0[vt] - c0[vt].mean(0)) ** 2).sum(1).mean().item()
            var_b = ((bil[vt] - bil[vt].mean(0)) ** 2).sum(1).mean().item()
            cos = F.cosine_similarity(c0[vt], bil[vt], dim=1).mean().item()
            stats.append(dict(bin=b, level=row.level, ann_id=int(row.ann_id), category=row.category, n_in=int(row.n_in),
                              c0_tokens=int(valid.sum()), c0_evr3=evr.round(3).tolist(), spatial_var_c0=round(var_c, 2),
                              spatial_var_bil=round(var_b, 2), var_ratio_bil_c0=round(var_b / var_c, 3), token_cos=round(cos, 3)))

            l, t, rr, bb = int(row.l), int(row.t), int(row.r), int(row.b)
            a = axs[k, 0]
            cx, cy, half = (l + rr) / 2, (t + bb) / 2, 1.5 * max(rr - l, bb - t)
            a.imshow(img)
            box(a, l, t, rr, bb, lw=2)
            a.set_xlim(max(cx - half, 0), min(cx + half, img.size[0]))
            a.set_ylim(min(cy + half, img.size[1]), max(cy - half, 0))
            a.set_title(f"ann {row.ann_id}  {row.level}: {row.category}\nbox {rr - l}x{bb - t} px, F336 n_in {int(row.n_in)}",
                        fontsize=9)
            a.set_ylabel(f"D{b}  {row.level}", fontsize=10)
            axs[k, 1].imshow(denorm(xc))
            axs[k, 1].set_title(f"C0 input ({S})\n{int(valid.sum())}/576 valid tokens", fontsize=9)
            axs[k, 2].imshow(to_rgb(Pc, lo, hi, valid), interpolation="nearest")
            axs[k, 2].set_title(f"C0 PCA (per-ROI)\nEVR top3 {evr[0]:.2f}/{evr[1]:.2f}/{evr[2]:.2f}", fontsize=9)
            axs[k, 3].imshow(to_rgb(Pb, lo, hi, valid), interpolation="nearest")
            axs[k, 3].set_title(f"Bilinear on C0 per-ROI basis\nspatial var / C0 = {var_b / var_c:.2f}, cos {cos:.2f}",
                                fontsize=9)
            axs[k, 4].imshow(to_rgb(Gc, gb["lo"], gb["hi"], valid), interpolation="nearest")
            axs[k, 4].set_title("C0 PCA (global basis)", fontsize=9)
            axs[k, 5].imshow(to_rgb(Gb, gb["lo"], gb["hi"], valid), interpolation="nearest")
            axs[k, 5].set_title("Bilinear (global basis)", fontsize=9)
            for a in axs[k]:
                a.set_xticks([])
                a.set_yticks([])
        fig.suptitle(f"PACO val D{b}: frozen {bname} final-norm tokens, top-3 PCA -> RGB (grey = letterbox padding). "
                     f"Global basis: 2,000 val ROIs, EVR {gb['evr'].sum():.2f}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        p = os.path.join(args.out, f"c0_pca_D{b}.png")
        fig.savefig(p, dpi=80)
        plt.close(fig)
        print(p)
    json.dump(stats, open(os.path.join(args.out, "roi_stats.json"), "w"), indent=1)
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()
