"""Per-ROI figures from MEASURED features (no conceptual maps): re-encodes the listed val ROIs with the exact feature
code of src.ivc.features and draws, per ROI:
  row 1: original + GT box | C0 input crop | PCA-RGB of Bilinear / Bicubic / C0 tokens (one PCA fitted jointly on the
         three conditions' valid tokens of this ROI, same projection for all three; padding cells grey)
  row 2: token cosine to C0, Bilinear | Bicubic (fixed scale [-0.2, 1]) | per-token MSE to C0, Bilinear | Bicubic (shared
         scale per ROI: 0 .. 99th pct of both) | C0 letterbox coverage
  row 3: self-similarity (cosine) on the fully covered rectangle: Bicubic | C0 (scale [-1, 1]) | Bicubic - C0 ([-1, 1])
Padding cells (coverage 0) are drawn grey and excluded from every number.

  CUDA_VISIBLE_DEVICES=2 python -m src.ivc.viz --ann_ids 1,2,3 --tags a,b,c --out <dir>
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import src.msd.common as MS
import src.readout.common as RC
from src.ivc.common import EXCL, G, full_rect, sample_roi

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


@torch.no_grad()
def extract(dva, Yva, rows, dev):
    vit = RC.load_vit().to(dev)
    ds = MS.MSDImageDS(dva, Yva, flip=False, want_crops=True)
    groups = RC.C.image_groups(dva)
    img_of = np.empty(len(dva), int)
    for gi, g in enumerate(groups):
        img_of[g] = gi
    out = {}
    for r in rows:
        b = MS.to_dev(ds[[int(img_of[r])]], dev)
        k = int(np.flatnonzero(b["rows"].cpu().numpy() == r)[0])
        full = RC.encode_layers(vit, b["full"], (12,))[12]
        c0 = RC.encode_layers(vit, b["crops"][k:k + 1], (12,))[12][0]
        geom = {"img_index": b["geom"]["img_index"][k:k + 1], "bil_grid": b["geom"]["bil_grid"][k:k + 1]}
        out[r] = dict(bil=sample_roi(full, geom, "bilinear")[0].cpu(), bic=sample_roi(full, geom, "bicubic")[0].cpu(), c0=c0.cpu(),
                      cov=b["cov_c"][k].float().cpu(), crop=b["crops"][k].cpu(), full=b["full"][0].cpu())
    return out


def pca_rgb(feats, valid):
    X = torch.cat([f[valid] for f in feats])
    mu = X.mean(0)
    U, S_, V = torch.pca_lowrank(X - mu, q=3, center=False)
    proj = [((f - mu) @ V[:, :3]) for f in feats]
    allp = torch.cat([p[valid] for p in proj])
    lo, hi = torch.quantile(allp, 0.01, dim=0), torch.quantile(allp, 0.99, dim=0)
    ims = []
    for p in proj:
        im = ((p - lo) / (hi - lo + 1e-9)).clamp(0, 1)
        im[~valid] = 0.5
        ims.append(im.view(G, G, 3).numpy())
    return ims


def draw(d, info, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    cov = d["cov"]
    valid = cov > 0
    grey = np.where(valid.view(G, G).numpy(), np.nan, 1.0)
    fig, ax = plt.subplots(3, 5, figsize=(21, 13))
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    img = Image.open(info.path).convert("RGB")
    l, t, r, b = (int(info[c]) for c in ("l", "t", "r", "b"))
    ax[0, 0].imshow(img)
    ax[0, 0].add_patch(Rectangle((l, t), r - l, b - t, fill=False, ec="yellow", lw=2))
    ax[0, 0].set_title(f"original + GT ROI\n{info.level} {info.category} n_in={info.n_in}", fontsize=10)
    ax[0, 1].imshow((d["crop"] * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy())
    ax[0, 1].set_title("C0 input crop (336 letterbox)", fontsize=10)
    ims = pca_rgb([d["bil"], d["bic"], d["c0"]], valid)
    for k, (nm, im) in enumerate(zip(("Bilinear", "Bicubic", "C0"), ims)):
        ax[0, 2 + k].imshow(im)
        ax[0, 2 + k].set_title(f"{nm} tokens, joint PCA-RGB (grey = padding)", fontsize=10)
    cs = {nm: F.cosine_similarity(d[nm], d["c0"], dim=-1) for nm in ("bil", "bic")}
    se = {nm: (d[nm] - d["c0"]).pow(2).mean(-1) for nm in ("bil", "bic")}
    vmax = float(torch.quantile(torch.cat([se["bil"][valid], se["bic"][valid]]), 0.99))
    w = cov / cov.sum()
    for k, nm in enumerate(("bil", "bic")):
        im = ax[1, k].imshow(np.where(valid.view(G, G), cs[nm].view(G, G).numpy(), np.nan), vmin=-0.2, vmax=1, cmap="viridis")
        ax[1, k].imshow(grey, cmap="Greys", vmin=0, vmax=2)
        ax[1, k].set_title(f"token cos({'Bilinear' if nm == 'bil' else 'Bicubic'}, C0)  mean_w={float((w * cs[nm]).sum()):.3f}", fontsize=10)
        plt.colorbar(im, ax=ax[1, k], fraction=0.046)
        im = ax[1, 2 + k].imshow(np.where(valid.view(G, G), se[nm].view(G, G).numpy(), np.nan), vmin=0, vmax=vmax, cmap="magma")
        ax[1, 2 + k].imshow(grey, cmap="Greys", vmin=0, vmax=2)
        ax[1, 2 + k].set_title(f"{'Bilinear' if nm == 'bil' else 'Bicubic'} - C0 per-token MSE  mean_w={float((w * se[nm]).sum()):.3f}", fontsize=10)
        plt.colorbar(im, ax=ax[1, 2 + k], fraction=0.046)
    im = ax[1, 4].imshow(cov.view(G, G).numpy(), vmin=0, vmax=1)
    ax[1, 4].set_title("C0 coverage (valid mask)", fontsize=10)
    plt.colorbar(im, ax=ax[1, 4], fraction=0.046)
    rr = full_rect(cov)
    if rr is not None:
        i0, i1, j0, j1 = rr
        def ss(X):
            Y = F.normalize(X.view(G, G, -1)[i0:i1, j0:j1].reshape(-1, X.shape[-1]), dim=-1)
            return (Y @ Y.T).numpy()
        Sa, Sc = ss(d["bic"]), ss(d["c0"])
        for k, (M, nm, cm, lim) in enumerate(((Sa, "Bicubic self-similarity", "RdBu_r", 1), (Sc, "C0 self-similarity", "RdBu_r", 1),
                                              (Sa - Sc, "Bicubic - C0 self-similarity", "PuOr_r", 1))):
            im = ax[2, k].imshow(M, vmin=-lim, vmax=lim, cmap=cm)
            ax[2, k].set_title(f"{nm}\n({i1 - i0}x{j1 - j0} fully covered cells, row-major)", fontsize=10)
            plt.colorbar(im, ax=ax[2, k], fraction=0.046)
        off = ~np.eye(len(Sa), dtype=bool)
        ax[2, 3].hist([Sa[off], Sc[off]], bins=40, range=(-0.5, 1), label=["Bicubic", "C0"], color=["tab:orange", "tab:blue"])
        ax[2, 3].legend()
        ax[2, 3].set_xticks(np.linspace(-0.5, 1, 7))
        ax[2, 3].set_title("off-diagonal self-similarity distribution", fontsize=10)
    xf = (d["full"] * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    ax[2, 4].imshow(xf)
    ax[2, 4].set_title("F336 full-image input", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_ids", required=True)
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dva, Yva = RC.load_data(EXCL, splits=("val",))["val"]
    ids = [int(x) for x in a.ann_ids.split(",")]
    tags = a.tags.split(",") if a.tags else [""] * len(ids)
    rows = [int(np.flatnonzero(dva.ann_id.values == x)[0]) for x in ids]
    D = extract(dva, Yva, rows, torch.device("cuda"))
    for r, tg, x in zip(rows, tags, ids):
        info = dva.iloc[r]
        draw(D[r], info, os.path.join(a.out, f"{tg + '_' if tg else ''}ann{x}.png"), f"{tg}  ann {x} ({info.level}, {info.category}, n_in {info.n_in})")
    print("figures", len(rows))


if __name__ == "__main__":
    main()
