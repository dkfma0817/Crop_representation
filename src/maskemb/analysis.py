"""Extra analyses without training (self analysis, not PACO metrics).
A (CPU) token allocation on the official val ROIs: Full = #24x24 cells with cov > 0 (+ sum of cov = effective cells),
  C0 = #valid (non-padding) crop cells; by level and original-box-area bin (COCO small / medium / large).
B (GPU, one device) feature comparison on N val images (rng 0) / their ROIs, same precision as the probes:
  token level  : cos(CLIP, MaskEmbed) per token, linear CKA over tokens, token-norm stats, high-norm outliers
                 (norm > 3 x image median), spatial locality = mean cos of 4-neighbour tokens minus mean cos of random
                 same-image pairs (after removing the image-mean token)
  ROI level    : cov-weighted ROI pooled features of CLIP-Full / MaskEmbed-Full vs the CLIP-C0 pooled crop feature:
                 linear CKA across ROIs and ridge R^2 (fit on half the images, test on the other half) -> how much of
                 the crop representation is linearly recoverable from each full-image feature (space-invariant)
  python -m src.maskemb.analysis --n_img 400 --prec tf32
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.maskemb import common as MC
from src.maskemb.probe import encoder_tokens, set_precision

DATA = "output/2026-11-07/paco_oracle_attr/data"


def size_bin(a):
    return np.where(a < 32 ** 2, "small", np.where(a < 96 ** 2, "medium", "large"))


def alloc(M):
    W, H = M.W.values.astype(float), M.H.values.astype(float)
    l, t, r, b = (M[c].values.astype(float) for c in ("l", "t", "r", "b"))
    s = MC.S / np.maximum(W, H)
    nw, nh = np.minimum(MC.S, np.maximum(1, np.round(W * s))), np.minimum(MC.S, np.maximum(1, np.round(H * s)))
    ox, oy = (MC.S - nw) // 2, (MC.S - nh) // 2
    X0, X1, Y0, Y1 = ox + l * nw / W, ox + r * nw / W, oy + t * nh / H, oy + b * nh / H
    e = np.arange(MC.G + 1) * MC.P
    cx = np.clip(np.minimum(e[1:][None], X1[:, None]) - np.maximum(e[:-1][None], X0[:, None]), 0, None) / MC.P
    cy = np.clip(np.minimum(e[1:][None], Y1[:, None]) - np.maximum(e[:-1][None], Y0[:, None]), 0, None) / MC.P
    n_full = (cx > 0).sum(1) * (cy > 0).sum(1)
    eff_full = cx.sum(1) * cy.sum(1)
    cw, ch = r - l, b - t
    cs = MC.S / np.maximum(cw, ch)
    cnw, cnh = np.minimum(MC.S, np.maximum(1, np.round(cw * cs))), np.minimum(MC.S, np.maximum(1, np.round(ch * cs)))
    cox, coy = (MC.S - cnw) // 2, (MC.S - cnh) // 2
    ccx = np.clip(np.minimum(e[1:][None], (cox + cnw)[:, None]) - np.maximum(e[:-1][None], cox[:, None]), 0, None)
    ccy = np.clip(np.minimum(e[1:][None], (coy + cnh)[:, None]) - np.maximum(e[:-1][None], coy[:, None]), 0, None)
    n_c0 = (ccx > 0).sum(1) * (ccy > 0).sum(1)
    D = pd.DataFrame(dict(level=M.level.values, bin=size_bin(M.bw.values * M.bh.values), n_full=n_full, eff_full=eff_full, n_c0=n_c0,
                          has_attr=M.has_attr.values if "has_attr" in M else True))
    out = {}
    for (lv, bn), g in list(D.groupby(["level", "bin"])) + [(("all", "all"), D)]:
        q = lambda v: {f"p{p}": float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)}  # noqa: E731
        out[f"{lv}/{bn}"] = dict(n=len(g), full_cells=dict(mean=float(g.n_full.mean()), **q(g.n_full)), full_eff_cells_mean=float(g.eff_full.mean()),
                                 c0_cells=dict(mean=float(g.n_c0.mean()), **q(g.n_c0)), ratio_c0_over_full_median=float(np.median(g.n_c0 / g.n_full)),
                                 frac_full_le_4_cells=float((g.n_full <= 4).mean()))
    return out


def cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def ridge_r2(X, Y, tr, te, lam=1.0):
    X, Y = X.double(), Y.double()
    mx, my = X[tr].mean(0), Y[tr].mean(0)
    Xt, Yt = X[tr] - mx, Y[tr] - my
    A = Xt.T @ Xt + lam * len(tr) * torch.eye(X.shape[1], dtype=X.dtype, device=X.device) * (Xt.pow(2).sum() / (len(tr) * X.shape[1]))
    Wt = torch.linalg.solve(A, Xt.T @ Yt)
    Pr = (X[te] - mx) @ Wt + my
    return float(1 - (Y[te] - Pr).pow(2).sum() / (Y[te] - Y[te].mean(0)).pow(2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_img", type=int, default=400)
    ap.add_argument("--prec", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    M = pd.read_parquet(os.path.join(DATA, "manifest_val.parquet"))
    Y = np.load(os.path.join(DATA, "labels_val.npy"))
    M = M.assign(has_attr=(Y == 1).any(1))
    res = dict(token_allocation_all_anns=alloc(M), token_allocation_attr_anns=alloc(M[M.has_attr].reset_index(drop=True)))
    CP = MC.set_clip_norm()
    from src.g1x.data import PacoDS
    from src.ivc.bbdata import BBImageDS
    dev = torch.device(a.device)
    torch.set_num_threads(48)
    set_precision(a.prec)
    clip, _ = MC.build_encoder("clip", dev)
    me, _ = MC.build_encoder("maskembed", dev)
    rng = np.random.default_rng(0)
    imgs = rng.choice(M.image_id.unique(), a.n_img, replace=False)
    sub = M[M.image_id.isin(imgs)].reset_index(drop=True)
    Z = np.zeros((len(sub), 59), np.int8)
    fd, cd = PacoDS(sub, Z, False, "full", MC.S, MC.P), BBImageDS(sub, Z, False, "c0", MC.S, MC.P)
    cosv, Tc, Tm, nrm_c, nrm_m, out_c, out_m, loc_c, loc_m = [], [], [], [], [], [], [], [], []
    Pc, Pm, Pk, img_of = [], [], [], []
    g = torch.Generator(device="cpu").manual_seed(0)
    nb = torch.tensor([(i, j) for i in range(MC.G) for j in range(MC.G - 1)])
    with torch.no_grad():
        for gi in range(len(fd)):
            bf, bc = fd[[gi]], cd[[gi]]
            x = bf["x"].to(dev)
            tc, tm = encoder_tokens(clip, x, a.prec)[0], encoder_tokens(me, x, a.prec)[0]
            valid = torch.as_tensor(CP.LB(int(sub.W.iloc[int(bf["rows"][0])]), int(sub.H.iloc[int(bf["rows"][0])]), MC.S, MC.P).coverage() > 0, device=dev)
            vc, vm = tc[valid], tm[valid]
            cosv.append(torch.nn.functional.cosine_similarity(vc, vm, dim=-1).cpu())
            sel = torch.randperm(len(vc), generator=g)[:64]
            Tc.append(vc[sel.to(dev)].cpu())
            Tm.append(vm[sel.to(dev)].cpu())
            for t, v, nr, oo, lo in ((tc, vc, nrm_c, out_c, loc_c), (tm, vm, nrm_m, out_m, loc_m)):
                n = v.norm(dim=-1)
                nr.append(n.cpu())
                oo.append(float((n > 3 * n.median()).float().mean()))
                c = t - v.mean(0)
                c = torch.nn.functional.normalize(c.view(MC.G, MC.G, -1), dim=-1)
                vm2 = valid.view(MC.G, MC.G)
                h = (c[:, :-1] * c[:, 1:]).sum(-1)[vm2[:, :-1] & vm2[:, 1:]]
                vv = (c[:-1] * c[1:]).sum(-1)[vm2[:-1] & vm2[1:]]
                cv = c[vm2]
                p = torch.randint(0, len(cv), (2, 2000), generator=g).to(dev)
                lo.append(float(torch.cat([h, vv]).mean() - (cv[p[0]] * cv[p[1]]).sum(-1).mean()))
            cov = bf["cov"].to(dev)
            Pc.append(((cov[..., None] * tc[None]).sum(1) / cov.sum(1, keepdim=True)).cpu())
            Pm.append(((cov[..., None] * tm[None]).sum(1) / cov.sum(1, keepdim=True)).cpu())
            k0 = encoder_tokens(clip, bc["x"].to(dev), a.prec)
            ccov = bc["cov"].to(dev)
            Pk.append(((ccov[..., None] * k0).sum(1) / ccov.sum(1, keepdim=True)).cpu())
            img_of += [gi] * len(cov)
    cosv = torch.cat(cosv)
    Tc, Tm = torch.cat(Tc).double(), torch.cat(Tm).double()
    Pc, Pm, Pk = torch.cat(Pc).double(), torch.cat(Pm).double(), torch.cat(Pk).double()
    img_of = np.array(img_of)
    tr, te = np.nonzero(img_of % 2 == 0)[0], np.nonzero(img_of % 2 == 1)[0]
    q = lambda v: {f"p{p}": float(np.percentile(v.numpy(), p)) for p in (1, 5, 50, 95)}  # noqa: E731
    res["features"] = dict(
        n_images=int(a.n_img), n_rois=int(len(Pc)), precision=a.prec,
        token_cos_clip_vs_maskembed=dict(mean=float(cosv.mean()), **q(cosv)),
        token_linear_cka_clip_vs_maskembed=cka(Tc, Tm),
        token_norm=dict(clip=dict(mean=float(torch.cat(nrm_c).mean()), **q(torch.cat(nrm_c))),
                        maskembed=dict(mean=float(torch.cat(nrm_m).mean()), **q(torch.cat(nrm_m)))),
        high_norm_outlier_frac=dict(clip=float(np.mean(out_c)), maskembed=float(np.mean(out_m))),
        spatial_locality_neighbour_minus_random_cos=dict(clip=float(np.mean(loc_c)), maskembed=float(np.mean(loc_m))),
        roi_pooled_vs_clip_c0=dict(
            cka=dict(clip_full=cka(Pc, Pk), maskembed_full=cka(Pm, Pk), clip_full_vs_maskembed_full=cka(Pc, Pm)),
            ridge_r2_test=dict(clip_full=ridge_r2(Pc, Pk, tr, te), maskembed_full=ridge_r2(Pm, Pk, tr, te),
                               clip_c0_self_upper=1.0),
            cos_clip_full_vs_clip_c0=float(torch.nn.functional.cosine_similarity(Pc, Pk, dim=-1).mean()),
            note="CKA / ridge are invariant to the feature space (MaskEmbed has its own final norm); higher = the full-image "
                 "ROI feature carries more of the CLIP crop representation linearly"))
    json.dump(res, open(MC.run_path("analysis.json"), "w"), indent=1)
    print(json.dumps(res["features"], indent=1))


if __name__ == "__main__":
    main()
