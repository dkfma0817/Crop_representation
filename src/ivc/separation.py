"""Object-vs-background token separation inside the ROI box, C0 crop vs Bilinear (PACO VAL only, no training).

Question: are C0 crop tokens of the PACO instance / part mask separated from the in-box background tokens, how much
better than Bilinear (full-image tokens read on the same crop grid), by size bin -- and (DINOv2 only, where the IVC R2
readers exist) does the separation track the per-ROI C0 gain?

Features (frozen, final-norm patch tokens, 24 x 24 crop grid, bf16 autocast -> fp32; geometry of src.viz.c0_pca):
  dinov2 = vit_base_patch14_dinov2.lvd142m @ 336 (exactly the IVC reader inputs), dinov3 = vit_base_patch16_dinov3.lvd1689m @ 384.
Mask fraction per cell = src.ivc.masks rule on the backbone's C0 canvas (PIL BOX resize of the box-cropped mask, mask
area / content area per cell); at (336, 14) asserted equal to output/2026-11-05/ivc/features/mask_frac.npz.
Regions (the IVC rule, fixed before results): over cells with C0 coverage > 0, fg = frac >= 0.9, bg = frac <= 0.1.
ROI metrics need >= 2 fg and >= 2 bg cells (else NaN), for cond in {c0, bil}:
  cent_dist  = 1 - cos(mean fg token, mean bg token)
  fisher     = ||mu_fg - mu_bg||^2 / mean within-region squared distance to own centroid (scale free)
  km_bacc    = 2-means (cosine, deterministic farthest-point init, 30 iters) on fg U bg tokens; balanced accuracy of the
               better cluster->region matching vs the mask (0.5 = chance, 1 = clusters = mask)
  fgpool_cos = cos(mean fg token, coverage-weighted mean of all valid tokens) (= how much the R0 pooled vector is the object)
Score link (dinov2): per-ROI mean over M0-unit positive labels of rank margin (src.ivc.analyze.rank_margin) of C0 R2
minus Bilinear R2 (IVC dev-selected epochs); Spearman and partial Spearman (rank n_in + level regressed out) with image
bootstrap 95% CI (300, seed 0).

  CUDA_VISIBLE_DEVICES=0 python -m src.ivc.separation --backbone dinov2 --out output/2026-11-06/separation/dinov2
  CUDA_VISIBLE_DEVICES=1 python -m src.ivc.separation --backbone dinov3 --out output/2026-11-06/separation/dinov3
"""
import argparse
import json
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import src.readout.common as RC
from src.config import PACO_DIR
from src.ivc.common import EXCL
from src.viz.c0_pca import BACKBONES, encode, geometry, load_encoder

G = 24
IVC = "output/2026-11-05/ivc"
BINS = (("D1-4", 1, 4), ("D5-16", 5, 16), ("D17+", 17, 10 ** 9))
METRICS = ("cent_dist", "fisher", "km_bacc", "fgpool_cos")
_M = {}


# ------------------------------------------------------------------------------------------------ masks
def _mask_job(r):
    ann = _M["anns"].get(int(_M["ann_id"][r]))
    if ann is None or not ann.get("segmentation"):
        return r, None
    from pycocotools import mask as mu
    W, H = int(_M["W"][r]), int(_M["H"][r])
    l, t, rr, bb = (int(x) for x in _M["box"][r])
    S, P = _M["S"], _M["P"]
    seg = ann["segmentation"]
    rle = mu.merge(mu.frPyObjects(seg, H, W)) if isinstance(seg, list) else (mu.frPyObjects(seg, H, W) if isinstance(seg["counts"], list) else seg)
    m = mu.decode(rle).astype(np.uint8)
    if m.ndim == 3:
        m = m.max(2)
    crop, _, _ = geometry(W, H, (l, t, rr, bb), S, P)
    mc = Image.fromarray(m[t:bb, l:rr] * 255).resize((crop.nw, crop.nh), resample=Image.BOX)
    canvas, content = np.zeros((S, S)), np.zeros((S, S))
    canvas[crop.oy:crop.oy + crop.nh, crop.ox:crop.ox + crop.nw] = np.asarray(mc, np.float64) / 255.0
    content[crop.oy:crop.oy + crop.nh, crop.ox:crop.ox + crop.nw] = 1.0
    ms = canvas.reshape(G, P, G, P).sum((1, 3)).reshape(-1)
    cs = content.reshape(G, P, G, P).sum((1, 3)).reshape(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return r, np.where(cs > 0, ms / np.maximum(cs, 1e-9), np.nan).astype(np.float32)


def mask_frac(dva, S, P, workers=48):
    js = json.load(open(os.path.join(PACO_DIR, "paco_lvis_v1_val.json")))
    _M.update(anns={a["id"]: a for a in js["annotations"]}, ann_id=dva.ann_id.values, W=dva.W.values, H=dva.H.values,
              box=dva[["l", "t", "r", "b"]].values, S=S, P=P)
    out = np.full((len(dva), G * G), np.nan, np.float32)
    with get_context("fork").Pool(workers) as pool:
        for r, f in pool.imap_unordered(_mask_job, range(len(dva)), chunksize=32):
            if f is not None:
                out[r] = f
    return out


# ------------------------------------------------------------------------------------------------ features
class ROIDS(Dataset):
    def __init__(self, df, S, P):
        self.df, self.S, self.P = df, S, P

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.path).convert("RGB")
        W, H = img.size
        ltrb = (int(row.l), int(row.t), int(row.r), int(row.b))
        crop, full, grid = geometry(W, H, ltrb, self.S, self.P)
        return crop.canvas(img.crop(ltrb)), full.canvas(img), torch.from_numpy(grid), torch.from_numpy(crop.coverage()), i


def bil_batch(full_tok, grid):
    m = full_tok.transpose(1, 2).reshape(full_tok.shape[0], -1, G, G)
    return F.grid_sample(m, grid.view(-1, G, G, 2), mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)


def two_means_bacc(X, fg):
    """X (n, D) tokens of fg U bg cells, fg (n,) bool -> balanced accuracy of deterministic cosine 2-means vs the mask."""
    Xn = F.normalize(X, dim=1)
    mu = Xn.mean(0, keepdim=True)
    c1 = Xn[(Xn @ F.normalize(mu, dim=1).T).argmin()]                 # farthest from the mean direction
    c2 = Xn[(Xn @ c1).argmin()]                                        # farthest from c1
    C = torch.stack([c1, c2])
    for _ in range(30):
        a = (Xn @ C.T).argmax(1)
        newC = torch.stack([F.normalize(Xn[a == k].mean(0), dim=0) if (a == k).any() else C[k] for k in range(2)])
        if torch.allclose(newC, C):
            break
        C = newC
    a = (Xn @ C.T).argmax(1).cpu().numpy() == 0
    f = fg.cpu().numpy()

    def bacc(p):
        return 0.5 * (p[f].mean() + (~p[~f]).mean())
    return float(max(bacc(a), bacc(~a)))


def roi_metrics(T, cov, frac):
    """T (576, D) one condition, cov (576,), frac (576,) numpy -> dict of METRICS (NaN if < 2 fg or < 2 bg cells)."""
    valid = cov > 0
    fg, bg = valid & (frac >= 0.9), valid & (frac <= 0.1)
    out = dict(n_fg=int(fg.sum()), n_bg=int(bg.sum()))
    if fg.sum() < 2 or bg.sum() < 2:
        return {**out, **{m: np.nan for m in METRICS}}
    dev = T.device
    tf, tb = T[torch.as_tensor(fg, device=dev)], T[torch.as_tensor(bg, device=dev)]
    mf, mb = tf.mean(0), tb.mean(0)
    w = torch.as_tensor(cov[valid], device=dev)
    pooled = (T[torch.as_tensor(valid, device=dev)] * w[:, None]).sum(0) / w.sum()
    within = torch.cat([((tf - mf) ** 2).sum(1), ((tb - mb) ** 2).sum(1)]).mean()
    X = torch.cat([tf, tb])
    lab = torch.cat([torch.ones(len(tf), dtype=torch.bool), torch.zeros(len(tb), dtype=torch.bool)])
    return {**out, "cent_dist": float(1 - F.cosine_similarity(mf, mb, dim=0)), "fisher": float(((mf - mb) ** 2).sum() / within),
            "km_bacc": two_means_bacc(X, lab), "fgpool_cos": float(F.cosine_similarity(mf, pooled, dim=0))}


# ------------------------------------------------------------------------------------------------ summaries
def dbin_of(n_in):
    return np.select([n_in <= 4, n_in <= 16], ["D1-4", "D5-16"], "D17+")


def summary_table(R):
    rows = []
    for (b, lv), g in R.groupby(["dbin", "level"]):
        d = dict(dbin=b, level=lv, n=len(g), n_defined=int(np.isfinite(g.c0_km_bacc).sum()))
        for m in METRICS:
            c, a = g[f"c0_{m}"], g[f"bil_{m}"]
            d[f"c0_{m}"], d[f"bil_{m}"], d[f"diff_{m}"] = float(np.nanmedian(c)), float(np.nanmedian(a)), float(np.nanmedian(c - a))
        rows.append(d)
    order = {(b, lv): k for k, (b, lv) in enumerate([(b, lv) for b, _, _ in BINS for lv in ("object", "part")])}
    return pd.DataFrame(rows).sort_values(by=["dbin", "level"], key=lambda s: s.map(lambda x: x)).assign(
        _o=lambda d: [order[(b, lv)] for b, lv in zip(d.dbin, d.level)]).sort_values("_o").drop(columns="_o")


def score_link(R, dva, Yva, out):
    """dinov2: per-ROI C0 - Bilinear R2 rank margin (positives) vs separation metrics."""
    import src.ivc.analyze as AN
    from src.readout.evaluate import Boot
    data = RC.load_data(EXCL)
    B = Boot(dva, Yva, RC.ytr_by_level(data))
    units = list(B.keys_o)
    lvi = (dva.level.values == "part").astype(int)
    ok_unit = np.zeros((2, 59), bool)
    for lv, at in units:
        ok_unit[int(lv == "part"), at] = True
    okp = (Yva == 1) & ok_unit[lvi]
    rm = {c: AN.rank_margin(AN.load_probe(IVC, c, "r2")["Sv"], Yva, lvi, units) for c in ("bil", "c0")}
    n = okp.sum(1)
    d_pos = np.where(n > 0, np.where(okp, rm["c0"] - rm["bil"], 0).sum(1) / np.maximum(n, 1), np.nan)
    R["d_rank_c0_bil"] = d_pos
    W = AN.boot_weights(dva, nb=300, seed=0)
    ctrl_all = np.column_stack([np.log(dva.n_in.values.astype(np.float64)), lvi.astype(np.float64)])
    res = {}
    subsets = {"all": np.ones(len(R), bool), **{b: (R.dbin == b).values for b, _, _ in BINS},
               "object": (R.level == "object").values, "part": (R.level == "part").values,
               "Obj D5-16": ((R.dbin == "D5-16") & (R.level == "object")).values}
    feats = [f"c0_{m}" for m in METRICS] + [f"diff_{m}" for m in METRICS]
    for m in METRICS:
        R[f"diff_{m}"] = R[f"c0_{m}"] - R[f"bil_{m}"]
    for nm, sm in subsets.items():
        for f in feats:
            x, y = R[f].values, d_pos
            ok = sm & np.isfinite(x) & np.isfinite(y)
            ctrl = ctrl_all[ok]
            pt = AN.spearman(x[ok], y[ok], ctrl)
            bs = [AN.spearman(x[ok], y[ok], ctrl, w=W[k][ok]) for k in range(W.shape[0])]
            res[f"{nm}|{f}"] = dict(n=int(ok.sum()), rho_raw=AN.spearman(x[ok], y[ok]), partial=pt,
                                    lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)))
    json.dump(res, open(os.path.join(out, "score_link.json"), "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=tuple(BACKBONES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S, P = BACKBONES[a.backbone]["S"], BACKBONES[a.backbone]["P"]
    dva, Yva = RC.load_data(EXCL, splits=("val",))["val"]
    if a.limit:
        dva, Yva = dva.iloc[:a.limit].reset_index(drop=True), Yva[:a.limit]
    N = len(dva)

    frac = mask_frac(dva, S, P)
    if (S, P) == (336, 14):
        ref = np.load(os.path.join(IVC, "features", "mask_frac.npz"))
        assert np.array_equal(ref["ann_id"][:N], dva.ann_id.values)
        rf = ref["frac"][:N]
        assert np.array_equal(np.isnan(rf), np.isnan(frac)) and np.nanmax(np.abs(rf - frac)) < 1e-5, "mask_frac mismatch"
        print("mask_frac == IVC mask_frac.npz")
    print(f"masks: {np.isfinite(frac).any(1).sum()} / {N} ROIs")

    dev = "cuda"
    vit = load_encoder(a.backbone, dev)
    dl = DataLoader(ROIDS(dva, S, P), batch_size=a.bs, num_workers=a.workers, shuffle=False)
    rec = [None] * N
    for k, (xc, xf, grid, cov, idx) in enumerate(dl):
        Tc = encode(vit, xc, dev)
        Tb = bil_batch(encode(vit, xf, dev), grid.to(dev))
        for j, i in enumerate(idx.tolist()):
            cv = cov[j].numpy()
            mc, mbl = roi_metrics(Tc[j], cv, frac[i]), roi_metrics(Tb[j], cv, frac[i])
            rec[i] = dict(n_fg=mc["n_fg"], n_bg=mc["n_bg"], **{f"c0_{m}": mc[m] for m in METRICS}, **{f"bil_{m}": mbl[m] for m in METRICS})
        if k % 20 == 0:
            print(f"batch {k}/{len(dl)}", flush=True)
    R = pd.DataFrame(rec)
    R.insert(0, "ann_id", dva.ann_id.values)
    R.insert(1, "image_id", dva.image_id.values)
    R.insert(2, "level", dva.level.values)
    R.insert(3, "n_in", dva.n_in.values)
    R.insert(4, "dbin", dbin_of(dva.n_in.values))
    tab = summary_table(R)
    tab.to_csv(os.path.join(a.out, "summary_by_bin.csv"), index=False)
    print(tab.round(3).to_string(index=False))
    if a.backbone == "dinov2" and not a.limit:
        res = score_link(R, dva, Yva, a.out)
        for k, v in res.items():
            print(f"{k:32s} n={v['n']:5d} raw {v['rho_raw']:+.3f} partial {v['partial']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    R.to_parquet(os.path.join(a.out, "roi_separation.parquet"))
    json.dump(dict(backbone=a.backbone, S=S, P=P, N=N, rule="fg frac>=0.9, bg frac<=0.1 over cov>0 cells; >=2 each"),
              open(os.path.join(a.out, "done.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
