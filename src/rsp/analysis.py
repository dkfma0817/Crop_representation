"""Held-out feature analysis of RSP-Masked / noPix vs the dense teacher (COCO dev images disjoint from the training stream; analysis only).

Images: src.tsplit.train.dev_set() (held-out COCO, PACO duplicates removed), no flip. ROIs per image (rng (1e6, id), = S3 held-out):
8 'train' rectangles (1..6 parents) + 2 'large' (7..12, clipped). Every valid fine (4 px) cell of each rectangle is predicted.
Compared at the same fine positions (all in DINOv3 final-token space):
  RSP-Masked, RSP-noPix (de-normalised predictions) | F384-bil (bilinear read of the 24 x 24 map) | F384-NN (parent copy)
  and the teacher's relation to F768 (native 768 canvas, bilinear read of the 48 x 48 map) for context.
Metrics per ROI over valid fine cells, then averaged over ROIs: cos, L1 in the normalised space (the training loss), MSE raw, norm ratio;
within-parent residual (x - mean of the same parent's valid fine cells): R^2 vs the teacher residual, residual cosine, parent-mean MSE.

  python -m src.rsp.analysis --pix <model.pt> --nopix <model.pt> --out <json> [--n_images 300]
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import COCO_TRAIN
from src.rsp.common import GF, PREFIX, PT, R_FP, fine_rect_index
from src.rsp.data import evidence, frects_of, rect_box
from src.rsp.encoder import load_rsp
from src.rsp.teacher import DenseTeacher
from src.tsplit.features import frozen_vit, teacher as t768
from src.tsplit.train import MAXS, dev_set, sample_rects
from src.g1native.native import native_canvas
from src.viz.c0_pca import LB


def load(iid):
    with Image.open(os.path.join(COCO_TRAIN, f"{iid:012d}.jpg")) as im:
        img = im.convert("RGB")
    lb = LB(img.size[0], img.size[1], 384, 16)
    return img, lb


def bil(fm, cen, side):
    """fm (B, side*side, D) map, cen (R, N, 2) (y, x) canvas px, per-ROI image index applied by caller -> (R, N, D)."""
    B = fm.shape[0]
    m = fm.transpose(1, 2).reshape(B, -1, side, side)
    g = (cen.flip(-1) / 384 * 2 - 1)[:, :, None]
    return F.grid_sample(m, g, mode="bilinear", padding_mode="border", align_corners=False)[..., 0].transpose(1, 2)


def roi_metrics(Y, T, v, par, mu, sd):
    """Y, T (N, D) float, v (N,) bool, par (N,) parent id -> dict."""
    Y, T, pr = Y[v], T[v], par[v]
    u, inv = torch.unique(pr, return_inverse=True)
    cnt = torch.zeros(len(u), device=Y.device).index_add_(0, inv, torch.ones(len(inv), device=Y.device))
    def pmean(X):
        return (torch.zeros(len(u), X.shape[1], device=X.device).index_add_(0, inv, X) / cnt[:, None])
    mY, mT = pmean(Y), pmean(T)
    rY, rT = Y - mY[inv], T - mT[inv]
    nz = rT.norm(dim=-1) > 1e-6
    return dict(cos=float(F.cosine_similarity(Y, T, dim=-1).mean()), l1_norm=float(((Y - mu) / sd - (T - mu) / sd).abs().mean()),
                mse_raw=float((Y - T).pow(2).mean()), norm_ratio=float(Y.norm(dim=-1).mean() / T.norm(dim=-1).mean()),
                parent_mean_mse=float((mY - mT).pow(2).mean()), res_se=float((rY - rT).pow(2).sum()), res_energy=float(rT.pow(2).sum()),
                res_cos=float(F.cosine_similarity(rY[nz], rT[nz], dim=-1).mean()) if nz.any() else float("nan"),
                res_norm_ratio=float(rY.norm(dim=-1).sum() / rT.norm(dim=-1).sum().clamp_min(1e-12)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pix", required=True)
    ap.add_argument("--nopix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_images", type=int, default=300)
    a = ap.parse_args()
    dev = torch.device("cuda")
    vit = frozen_vit(dev)
    teach = DenseTeacher(vit).to(dev)
    models = {}
    for name, path in (("RSP-Masked", a.pix), ("RSP-Masked-noPix", a.nopix)):
        m, mu, sd, ck = load_rsp(path, dev)
        models[name] = m
    ids = dev_set()[:a.n_images]
    acc = {}
    for k0 in range(0, len(ids), 8):
        chunk = ids[k0:k0 + 8]
        xs, x768, content, rects, rimg, boxes, evs, bucket = [], [], [], [], [], [], [], []
        for bi, iid in enumerate(chunk):
            img, lb = load(iid)
            xs.append(lb.canvas(img))
            x768.append(native_canvas(img, lb))
            content.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
            rng = np.random.default_rng([10 ** 6, iid])
            for r in sample_rects(rng, lb, 8) + sample_rects(rng, lb, 2, 7, 12):
                bx = rect_box(r, lb)
                rects.append(r)
                rimg.append(bi)
                boxes.append(bx)
                evs.append(evidence(img, lb, bx))
                s = max(r[2], r[3])
                bucket.append("large_7to12" if s > MAXS else ("side_1to2" if s <= 2 else "side_3to4" if s <= 4 else "side_5to6"))
        x = torch.stack(xs).to(dev)
        content = torch.tensor(content, dtype=torch.float32, device=dev)
        rects = torch.tensor(rects, device=dev)
        rimg = torch.tensor(rimg, device=dev)
        box = torch.tensor(boxes, dtype=torch.float32, device=dev)
        ev = torch.stack(evs).to(dev)
        fr = frects_of(rects)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res, final = teach.targets(x, fr, rimg, content)
            T768 = t768(vit, torch.stack(x768).to(dev))
        final = final.float()
        idx, uy, ux, ok = fine_rect_index(fr)
        cen = torch.stack([(uy.float() + .5) * PT, (ux.float() + .5) * PT], -1)
        par = (uy // R_FP) * 24 + ux // R_FP
        preds = {}
        for name, m in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cache = m.context(final)
                l1, z = m.level1(cache, rimg, box, ev if m.pix else None)
                out = []
                for s0 in range(0, len(rects), 16):
                    sl = slice(s0, s0 + 16)
                    out.append(m.queries(cache, [(kk[sl], vv[sl]) for kk, vv in l1], rimg[sl], z[sl], cen[sl]).float())
            preds[name] = torch.cat(out) * sd + mu
        fin = final[:, PREFIX:]
        preds["F384-bil"] = bil(fin[rimg], cen, 24)
        preds["F384-NN"] = fin[rimg[:, None], par]
        f768 = bil(T768[rimg], cen, 48)
        for r in range(len(rects)):
            T, v, ix = res[r]
            n = len(T)
            for name, Y in list(preds.items()) + [("F768-bil (vs teacher)", f768)]:
                mt = roi_metrics(Y[r, :n].float(), T, v, par[r, :n], mu, sd)
                for bk in (bucket[r], "all", "train_1to6" if bucket[r] != "large_7to12" else None):
                    if bk is None:
                        continue
                    d = acc.setdefault(name, {}).setdefault(bk, {})
                    for kk, vv in mt.items():
                        d.setdefault(kk, []).append(vv)
            for name, Y in (("F384-bil", preds["F384-bil"]),):            # how far F384 is from F768 at the same points (context)
                mt = roi_metrics(Y[r, :n].float(), f768[r, :n].float(), v, par[r, :n], mu, sd)
                d = acc.setdefault("F384-bil vs F768-bil", {}).setdefault("all", {})
                for kk, vv in mt.items():
                    d.setdefault(kk, []).append(vv)
        print(f"{k0 + len(chunk)}/{len(ids)} images", flush=True)
    out = {}
    for name, bks in acc.items():
        out[name] = {}
        for bk, d in bks.items():
            r = {k: float(np.nanmean(v)) for k, v in d.items() if k not in ("res_se", "res_energy")}
            r["within_parent_residual_r2"] = 1 - float(np.sum(d["res_se"]) / max(np.sum(d["res_energy"]), 1e-12))
            r["n_roi"] = len(d["cos"])
            out[name][bk] = r
    out["_meta"] = dict(n_images=len(ids), rois_per_image="8 x (1..6) + 2 x (7..12) parents", targets="dense teacher p_T 4 px, all valid fine cells",
                        note="RSP rows/cols: prediction vs dense teacher; 'F768-bil (vs teacher)': F768 map read at the fine centres vs the dense teacher")
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({n: {b: {k: round(v, 4) for k, v in r.items() if k in ("cos", "l1_norm", "within_parent_residual_r2", "norm_ratio")} for b, r in d.items() if b in ("all", "train_1to6", "large_7to12")} for n, d in out.items() if n != "_meta"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
