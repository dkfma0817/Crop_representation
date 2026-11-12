"""Picture-style architecture figure of P1 (Small ViT + late split x2), with real tokens of one PACO Val ROI.

All feature maps are real: frozen DINOv3-B/16 tokens of the 384 canvas (24 x 24) and the native 768 canvas (48 x 48), and the trained
P1 output for the ROI parents (output/2026-11-12/p1_smallvit_latesplit_s0/train/model.pt). Colours = one shared PCA basis fitted on
the L2-normalised 768 teacher tokens. Patch CE bars use the real FA prefill queue (4,096 teacher tokens), SIGReg panel the real P1 Z.
Runs on CPU (no GPU needed).

  python -m src.viz.fte_p1_arch --out results/figures/fte_p1_architecture.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

IMG = "/home/irteam/datasets/coco/train2017/000000345562.jpg"   # PACO val image 345562, ann 693939 (clock)
BOX = [276.19, 50.21, 106.45, 97.49]
CKPT = "output/2026-11-12/p1_smallvit_latesplit_s0/train/model.pt"
QUEUE = "output/2026-11-11/fine_aware_shared_s0/queue_prefill.pt"
TAU = 0.07

INK, MUTED = "#1f2933", "#5b6773"
FROZEN, FROZEN_EDGE = "#e4e7eb", "#7b8794"
TRAIN, TRAIN_EDGE = "#cdebdc", "#2b7a66"
ROI = "#e0332b"
TEA = "#8a96a3"


# ----------------------------------------------------------------------------------------------------------------------- compute
@torch.no_grad()
def compute(cache):
    from src.fte.models import P1
    from src.g1native.native import native_canvas
    from src.hcf.model import load_vit
    from src.viz.c0_pca import LB, MEAN_T, STD_T

    torch.manual_seed(0)
    img = Image.open(IMG).convert("RGB")
    W, H = img.size
    lb = LB(W, H, 384, 16)
    x384, x768 = lb.canvas(img), native_canvas(img, lb)
    vit = load_vit("cpu").eval()
    f384 = vit.forward_features(x384[None])[0, 5:]                          # (576, 768)
    t768 = vit.forward_features(x768[None])[0, 5:]                          # (2304, 768)

    s = lb.nw / W
    x0, y0 = BOX[0] * s + lb.ox, BOX[1] * s + lb.oy
    x1, y1 = (BOX[0] + BOX[2]) * s + lb.ox, (BOX[1] + BOX[3]) * s + lb.oy
    px0, px1, py0, py1 = int(x0 // 16), int((x1 - 1e-6) // 16), int(y0 // 16), int((y1 - 1e-6) // 16)
    rect = torch.tensor([[py0, px0, py1 - py0 + 1, px1 - px0 + 1]])

    m = P1(vit)
    m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
    m.eval()
    Y, Z, pid, pok = m(f384[None], torch.tensor([0]), rect)                 # (1, K, 4, 768)
    h, w = int(rect[0, 2]), int(rect[0, 3])
    Yc = Y[0].view(h, w, 2, 2, 768).permute(0, 2, 1, 3, 4).reshape(2 * h, 2 * w, 768)
    T48 = t768.view(48, 48, 768)
    Tc = T48[2 * py0:2 * py0 + 2 * h, 2 * px0:2 * px0 + 2 * w]

    # SIGReg latent: Z for all parents of the image (6 x 6 tiles), raw
    tiles = torch.tensor([[i, j, 6, 6] for i in range(0, 24, 6) for j in range(0, 24, 6)])
    _, Zall, _, _ = m(f384[None], torch.zeros(len(tiles), dtype=torch.long), tiles)
    Zall = Zall.reshape(-1, 768)
    dirs = F.normalize(torch.randn(768, 2), dim=0)
    zproj = (Zall @ dirs).numpy()

    # shared PCA on normalised teacher tokens
    valid = np.zeros((48, 48), bool)
    valid[2 * lb.oy // 16:(2 * (lb.oy + lb.nh) + 15) // 16, 2 * lb.ox // 16:(2 * (lb.ox + lb.nw) + 15) // 16] = True
    Tn = F.normalize(t768, dim=-1)
    mu = Tn[torch.from_numpy(valid.reshape(-1))].mean(0)
    _, _, V = torch.linalg.svd(Tn[torch.from_numpy(valid.reshape(-1))] - mu, full_matrices=False)
    B = V[:3].T
    proj = lambda X: ((F.normalize(X, dim=-1) - mu) @ B).numpy()  # noqa: E731
    pt = proj(t768)
    lo, hi = np.percentile(pt[valid.reshape(-1)], 2, 0), np.percentile(pt[valid.reshape(-1)], 98, 0)
    rgb = lambda P: np.clip((P - lo) / (hi - lo), 0, 1)  # noqa: E731

    # patch CE for one child: distributions over the real queue
    Q = F.normalize(torch.load(QUEUE, weights_only=False)["features"].float(), dim=-1)
    ci = (h, w)                                                              # centre child of the ROI
    ts, ss = F.normalize(Tc[ci], dim=-1), F.normalize(Yc[ci], dim=-1)
    pT, pS = torch.softmax(ts @ Q.T / TAU, -1), torch.softmax(ss @ Q.T / TAU, -1)
    top = torch.argsort(pT, descending=True)[:20]
    qcol = rgb(proj(Q[top]))

    canvas = lambda x: ((x * STD_T + MEAN_T).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # noqa: E731
    np.savez_compressed(cache, img=np.asarray(img), c384=canvas(x384), c768=canvas(x768), rect=rect[0].numpy(), box=np.array([x0, y0, x1, y1]),
                        f384=rgb(proj(f384)).reshape(24, 24, 3), t768=rgb(pt).reshape(48, 48, 3), yc=rgb(proj(Yc.reshape(-1, 768))).reshape(2 * h, 2 * w, 3),
                        tc=rgb(proj(Tc.reshape(-1, 768))).reshape(2 * h, 2 * w, 3), valid48=valid, child=np.array(ci),
                        pT=pT[top].numpy(), pS=pS[top].numpy(), qcol=qcol, zproj=zproj,
                        cos=F.cosine_similarity(Yc, Tc, dim=-1).numpy())


# ----------------------------------------------------------------------------------------------------------------------- draw
def arrow(ax, a, b, color=INK, lw=1.8, rad=0.0, ls="-", style="-|>", ms=16):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=5, shrinkA=0, shrinkB=0))


def grid(ax, x, y, size, n, color="white", lw=0.5, alpha=0.6, sub=None):
    """n x n grid lines over the square [x, x + size] x [y, y + size]; sub = (i0, j0, h, w) restricts to a cell rectangle."""
    c = size / n
    i0, j0, hh, ww = sub if sub else (0, 0, n, n)
    for k in range(ww + 1):
        ax.plot([x + (j0 + k) * c] * 2, [y + size - i0 * c, y + size - (i0 + hh) * c], color=color, lw=lw, alpha=alpha, zorder=4)
    for k in range(hh + 1):
        ax.plot([x + j0 * c, x + (j0 + ww) * c], [y + size - (i0 + k) * c] * 2, color=color, lw=lw, alpha=alpha, zorder=4)


def cellrect(ax, x, y, size, n, i0, j0, hh, ww, color=ROI, lw=2.2, ls="-"):
    c = size / n
    ax.add_patch(Rectangle((x + j0 * c, y + size - (i0 + hh) * c), ww * c, hh * c, fill=False, ec=color, lw=lw, ls=ls, zorder=6))
    return x + j0 * c, y + size - (i0 + hh) * c, ww * c, hh * c


def tokens(ax, arr, x, y, cell, gap=0.0, ec=None, lw=0.6):
    n_i, n_j = arr.shape[:2]
    for i in range(n_i):
        for j in range(n_j):
            ax.add_patch(Rectangle((x + j * (cell + gap), y + (n_i - 1 - i) * (cell + gap)), cell, cell, fc=arr[i, j], ec=ec or "none", lw=lw, zorder=3))


def stack(ax, x, y, w, h, n, face, edge, label, sub, tcol):
    """n slanted slabs = transformer blocks."""
    d = h / (n + 3)
    for k in range(n):
        yy = y + k * d
        ax.add_patch(Polygon([(x, yy), (x + w * .72, yy), (x + w, yy + 3 * d), (x + w * .28, yy + 3 * d)], closed=True, fc=face, ec=edge, lw=1.1, zorder=3 + k * .01))
    ax.text(x + w / 2, y + h + 1.2, label, ha="center", va="bottom", fontsize=12, fontweight="bold", color=tcol)
    ax.text(x + w / 2, y - 1.3, sub, ha="center", va="top", fontsize=10, color=MUTED, linespacing=1.25)


def caption(ax, x, y, t, size=10.5, color=MUTED, weight="normal", ha="center"):
    ax.text(x, y, t, ha=ha, va="top", fontsize=size, color=color, fontweight=weight, linespacing=1.25)


def draw(cache, out):
    d = np.load(cache)
    py0, px0, h, w = (int(v) for v in d["rect"])
    fig, ax = plt.subplots(figsize=(22, 12.2), dpi=160, facecolor="white")
    ax.set_xlim(0, 204)
    ax.set_ylim(0, 113)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(2, 110, "P1 · Small ViT + Late Split (×2)", fontsize=21, fontweight="bold", color=INK, va="top")
    ax.text(2, 105.3, "Predict 768-level child tokens of one ROI from the 384 parent tokens. Colours = real tokens (shared PCA). "
            "Grey = frozen, green = trained.", fontsize=12, color=MUTED, va="top")

    # ---------- original image + zoom-ins ----------
    img = d["img"]
    H0, W0 = img.shape[:2]
    iw = 30
    ih = iw * H0 / W0
    ix, iy = 2, 43
    ax.imshow(img, extent=(ix, ix + iw, iy, iy + ih), zorder=2)
    bx = [BOX[0] / W0 * iw + ix, iy + ih - (BOX[1] + BOX[3]) / H0 * ih, BOX[2] / W0 * iw, BOX[3] / H0 * ih]
    ax.add_patch(Rectangle(bx[:2], bx[2], bx[3], fill=False, ec=ROI, lw=2, zorder=6))
    caption(ax, ix + iw / 2, iy - 1, "PACO image + GT box\n(clock)")

    # ---------- canvases ----------
    S, cx = 32, 40
    ys, yt = 62, 9
    ax.imshow(d["c384"], extent=(cx, cx + S, ys, ys + S), zorder=2)
    grid(ax, cx, ys, S, 24, alpha=.35, lw=.45)
    grid(ax, cx, ys, S, 24, color=ROI, alpha=1, lw=.9, sub=(py0, px0, h, w))
    r384 = cellrect(ax, cx, ys, S, 24, py0, px0, h, w)
    caption(ax, cx + S / 2, ys - 1, "384 × 384 input\n24 × 24 patches of 16 px")
    ax.imshow(d["c768"], extent=(cx, cx + S, yt, yt + S), zorder=2)
    grid(ax, cx, yt, S, 48, alpha=.25, lw=.35)
    grid(ax, cx, yt, S, 48, color=ROI, alpha=1, lw=.7, sub=(2 * py0, 2 * px0, 2 * h, 2 * w))
    r768 = cellrect(ax, cx, yt, S, 48, 2 * py0, 2 * px0, 2 * h, 2 * w)
    caption(ax, cx + S / 2, yt - 1, "768 × 768 input (same image, ×2 resize)\n48 × 48 patches")
    arrow(ax, (ix + iw + .5, iy + ih * .75), (cx - .5, ys + S / 2), MUTED, rad=-.15)
    arrow(ax, (ix + iw + .5, iy + ih * .25), (cx - .5, yt + S / 2), MUTED, rad=.15)

    # zoom insets of the ROI (callouts, not part of the flow)
    zs = 15
    for (rx, ry, rw, rh), zy, n, txt, src in [(r384, 86.5, 1, "ROI: 5 × 5 parents", "c384"), (r768, 2, 2, "ROI: 10 × 10 children", "c768")]:
        zx = 9.5
        sc = d[src].shape[0] / S
        canvas_y = ys if src == "c384" else yt
        r0 = int(round((canvas_y + S - (ry + rh)) * sc))
        c0 = int(round((rx - cx) * sc))
        crop = d[src][r0:r0 + int(round(rh * sc)), c0:c0 + int(round(rw * sc))]
        ax.imshow(crop, extent=(zx, zx + zs, zy, zy + zs), zorder=2, interpolation="lanczos")
        grid(ax, zx, zy, zs, n * h, color="white", alpha=.9, lw=1.0 if n == 1 else .7)
        ax.add_patch(Rectangle((zx, zy), zs, zs, fill=False, ec=ROI, lw=2, zorder=6))
        caption(ax, zx + zs / 2, zy - .8 if n == 1 else zy + zs + 2.6, txt, size=10.5, color=ROI)
        for (ax0, ay0), (bx0, by0) in [((zx + zs, zy + zs), (rx, ry + rh)), ((zx + zs, zy), (rx, ry))]:
            ax.plot([ax0, bx0], [ay0, by0], color=ROI, lw=.8, ls=(0, (3, 3)), zorder=1, alpha=.7)

    # ---------- frozen encoders ----------
    ex = 77
    stack(ax, ex, ys + 6, 12, 20, 12, FROZEN, FROZEN_EDGE, "DINOv3-B/16", "12 blocks · frozen", INK)
    stack(ax, ex, yt + 6, 12, 20, 12, FROZEN, FROZEN_EDGE, "DINOv3-B/16", "same weights · frozen", INK)
    arrow(ax, (cx + S + .5, ys + S / 2), (ex - .5, ys + S / 2), INK)
    arrow(ax, (cx + S + .5, yt + S / 2), (ex - .5, yt + S / 2), INK)

    # ---------- token maps ----------
    fx, fs = 95, 24
    fy, ty = ys + 4, yt + 4
    ax.imshow(d["f384"], extent=(fx, fx + fs, fy, fy + fs), interpolation="nearest", zorder=2)
    grid(ax, fx, fy, fs, 24, color="white", alpha=.35, lw=.4)
    f_r = cellrect(ax, fx, fy, fs, 24, py0, px0, h, w)
    caption(ax, fx + fs / 2, fy - 1, "24 × 24 tokens (768-d)")
    t48 = d["t768"].copy()
    ax.imshow(t48, extent=(fx, fx + fs, ty, ty + fs), interpolation="nearest", zorder=2)
    t_r = cellrect(ax, fx, ty, fs, 48, 2 * py0, 2 * px0, 2 * h, 2 * w)
    caption(ax, fx + fs / 2, ty - 1, "48 × 48 tokens (768-d)")
    arrow(ax, (ex + 12.5, ys + S / 2), (fx - .5, ys + S / 2), INK)
    arrow(ax, (ex + 12.5, yt + S / 2), (fx - .5, yt + S / 2), INK)

    # ---------- P1: parents -> small ViT -> split ----------
    par = d["f384"][py0:py0 + h, px0:px0 + w]
    kx, cell, gap = 125, 1.9, .45
    kw = w * cell + (w - 1) * gap
    ky = ys + S / 2 - kw / 2
    tokens(ax, par, kx, ky, cell, gap, ec=INK, lw=.5)
    caption(ax, kx + kw / 2, ky - 1, "25 ROI parent\ntokens (no grad)")
    arrow(ax, (f_r[0] + f_r[2], f_r[1] + f_r[3] / 2), (kx - .5, ys + S / 2), ROI, rad=-.2, lw=1.5)

    vx = 140
    stack(ax, vx, ys + 6, 13, 20, 6, TRAIN, TRAIN_EDGE, "Small ViT", "6 blocks · 384-d\n+ 2D position", TRAIN_EDGE)
    arrow(ax, (kx + kw + .5, ys + S / 2), (vx - .5, ys + S / 2), INK)

    ox, osz = 163, 15
    oy = ys + S / 2 - osz / 2
    # split icon above the arrow
    sx, sy = 155.2, ys + S / 2 + 2.6
    ax.add_patch(Rectangle((sx, sy), 2.2, 2.2, fc=TRAIN, ec=TRAIN_EDGE, lw=1, zorder=4))
    arrow(ax, (sx + 2.5, sy + 1.1), (sx + 3.7, sy + 1.1), TRAIN_EDGE, lw=1.2, ms=9)
    for a in range(2):
        for b in range(2):
            ax.add_patch(Rectangle((sx + 4 + 1.1 * b, sy + 1.1 - 1.1 * a), 1.1, 1.1, fc=TRAIN, ec=TRAIN_EDGE, lw=.8, zorder=4))
    ax.text(sx + 2.6, sy + 3.0, "split", ha="center", fontsize=9.5, color=TRAIN_EDGE)
    arrow(ax, (vx + 13.5, ys + S / 2), (ox - .5, ys + S / 2), TRAIN_EDGE)
    ax.imshow(d["yc"], extent=(ox, ox + osz, oy, oy + osz), interpolation="nearest", zorder=2)
    grid(ax, ox, oy, osz, 2 * h, color="white", alpha=.6, lw=.5)
    grid(ax, ox, oy, osz, h, color=INK, alpha=.9, lw=1.0)
    ax.add_patch(Rectangle((ox, oy), osz, osz, fill=False, ec=TRAIN_EDGE, lw=2.2, zorder=6))
    caption(ax, ox + osz / 2, oy - 1, "P1 output\n10 × 10 children")

    # teacher target
    tgy = yt + S / 2 - osz / 2
    ax.imshow(d["tc"], extent=(ox, ox + osz, tgy, tgy + osz), interpolation="nearest", zorder=2)
    grid(ax, ox, tgy, osz, 2 * h, color="white", alpha=.6, lw=.5)
    grid(ax, ox, tgy, osz, h, color=INK, alpha=.9, lw=1.0)
    ax.add_patch(Rectangle((ox, tgy), osz, osz, fill=False, ec=TEA, lw=2.2, zorder=6))
    caption(ax, ox + osz / 2, tgy - 1, "Teacher target\n10 × 10 children")
    arrow(ax, (t_r[0] + t_r[2], t_r[1] + t_r[3] / 2), (ox - .5, yt + S / 2), ROI, rad=.08, lw=1.5)

    # highlight the child used in the CE panel
    ci, cj = (int(v) for v in d["child"])
    for yy in (oy, tgy):
        c = osz / (2 * h)
        ax.add_patch(Rectangle((ox + cj * c, yy + osz - (ci + 1) * c), c, c, fill=False, ec="#ffd400", lw=2.2, zorder=7))

    # ---------- losses ----------
    lx = 183
    ax.text(lx, 61, "Patch CE", fontsize=12.5, fontweight="bold", color=INK, va="bottom")
    # queue strip
    qc = d["qcol"]
    n = len(qc)
    bw = 19 / n
    for k in range(n):
        ax.add_patch(Rectangle((lx + k * bw, 56.6), bw * .9, bw * .9, fc=qc[k], ec="none", zorder=3))
    caption(ax, lx + 9.5, 56.2, "queue of 4,096 teacher tokens", size=9)
    # bars
    pT, pS = d["pT"], d["pS"]
    top = max(pT.max(), pS.max())
    by = 39.5
    for k in range(n):
        ax.add_patch(Rectangle((lx + k * bw, by), bw * .42, 9 * pT[k] / top, fc=TEA, ec="none", zorder=3))
        ax.add_patch(Rectangle((lx + k * bw + bw * .45, by), bw * .42, 9 * pS[k] / top, fc=TRAIN_EDGE, ec="none", zorder=3))
    ax.plot([lx, lx + 19], [by, by], color=INK, lw=.8)
    ax.add_patch(Rectangle((lx + 3, by + 11.6), 1.1, 1.1, fc=TEA))
    ax.text(lx + 4.6, by + 12.15, "teacher", fontsize=9, va="center", color=MUTED)
    ax.add_patch(Rectangle((lx + 11, by + 11.6), 1.1, 1.1, fc=TRAIN_EDGE))
    ax.text(lx + 12.6, by + 12.15, "P1", fontsize=9, va="center", color=MUTED)
    caption(ax, lx + 9.5, by - .8, "similarity to the queue\n(top-20 teacher tokens):\nmatch the distributions", size=9.5)
    arrow(ax, (ox + osz / 2, oy - 4.7), (lx + 3, 64.6), TRAIN_EDGE, rad=.25, lw=1.4)
    arrow(ax, (ox + osz + .5, tgy + osz / 2), (lx - .5, by + 4), TEA, rad=-.35, lw=1.4)

    # SIGReg
    zy = 74
    ax.text(lx, zy + 15.5, "+ SIGReg on Z", fontsize=12.5, fontweight="bold", color=INK, va="bottom")
    zp = d["zproj"][:, 0]
    zp = (zp - zp.mean()) / zp.std()
    hist, edges = np.histogram(zp, bins=24, range=(-3.5, 3.5), density=True)
    for k in range(len(hist)):
        ax.add_patch(Rectangle((lx + (edges[k] + 3.5) / 7 * 19, zy), 19 / 24 * .9, hist[k] * 30, fc=TRAIN, ec=TRAIN_EDGE, lw=.5, zorder=3))
    xs = np.linspace(-3.5, 3.5, 200)
    ax.plot(lx + (xs + 3.5) / 7 * 19, zy + 30 * np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.3, ls="--", zorder=4)
    ax.plot([lx, lx + 19], [zy, zy], color=INK, lw=.8)
    caption(ax, lx + 9.5, zy - .8, "random 1-D projections of Z\n→ isotropic Gaussian", size=9.5)
    arrow(ax, (ox + osz + .5, oy + osz / 2), (lx - .5, zy + 6), TRAIN_EDGE, rad=.2, lw=1.4)

    # ---------- result ----------
    ax.text(lx - 1, 9, "Official Test AP\nP1  34.04\nF384  39.28\nF768  43.66", fontsize=11, color=INK, va="top", linespacing=1.45,
            bbox=dict(boxstyle="round,pad=.5", fc="#f7f8fa", ec=FROZEN_EDGE))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=.15)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="results/figures/fte_p1_arch_tokens.npz")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    if a.recompute or not Path(a.cache).exists():
        compute(a.cache)
    draw(a.cache, a.out)
