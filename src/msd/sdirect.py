"""S-series direct checks (PACO val, the same 700 images / ROIs as src.msd.direct; report only).

Per ROI on valid teacher cells, in the normalised block-12 space (x - mu12) / sd12:
  L12, cos; spatial variance ratio var_pred / var_teacher; effective rank exp(H(singular values of the centred cell x
  channel matrix)) and its ratio to the teacher; neighbour difference ratio (mean |x_i - x_j| over 4-neighbour valid cells,
  pred / teacher); centred cosine = cos(x_p - mean_p, t_p - mean_t) (does the per-position pattern match, not only the mean);
  patch-affinity gap (mean |A_pred - A_teacher|).
Stages: A0 / A1 = output head applied to each block's state (upsampled to 24 x 24); B0 / B1 = the working map W after each
stage in the decoder's hidden space (variance and effective rank relative to the initial map W0).
Strata: object n_in 1-16, object 17+, part, patterned (non-plain pattern positive), multi-material.
Cache check: checksum of the frozen encoder maps before / after every decoder forward.

  CUDA_VISIBLE_DEVICES=2 python -m src.msd.sdirect --run output/2026-11-05/sdec --h1 output/2026-11-05/msd/H1/best_dev.pt
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import src.msd.common as MS
import src.readout.common as RC
from src.msd.direct import roi_metrics
from models.roi_reader import bilinear  # noqa: E402

EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"


def erank(X, m):
    """X (R,576,C), m (R,576) bool -> effective rank of the centred valid-cell matrix per ROI."""
    out = []
    for x, mm in zip(X, m):
        v = x[mm]
        v = v - v.mean(0, keepdim=True)
        s = torch.linalg.svdvals(v.float())
        p = s / s.sum().clamp(min=1e-12)
        out.append(torch.exp(-(p * (p + 1e-12).log()).sum()))
    return torch.stack(out)


def neigh_diff(X, m):
    G = 24
    Xg, mg = X.view(len(X), G, G, -1), m.view(len(X), G, G)
    dh = (Xg[:, :, 1:] - Xg[:, :, :-1]).norm(dim=-1)
    dv = (Xg[:, 1:] - Xg[:, :-1]).norm(dim=-1)
    mh, mv = (mg[:, :, 1:] & mg[:, :, :-1]).float(), (mg[:, 1:] & mg[:, :-1]).float()
    return ((dh * mh).sum((1, 2)) + (dv * mv).sum((1, 2))) / (mh.sum((1, 2)) + mv.sum((1, 2))).clamp(min=1)


def centred_cos(P, T, m):
    mf = m.float()[..., None]
    pc = P - (P * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True)
    tc = T - (T * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True)
    c = F.cosine_similarity(pc, tc, dim=-1)
    return (c * m).sum(1) / m.sum(1)


def svar(X, m):
    mf = m.float()[..., None]
    mu = (X * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True)
    return (((X - mu) ** 2) * mf).sum(1).mean(-1) / m.sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--h1", default="output/2026-11-05/msd/H1/best_dev.pt")
    ap.add_argument("--stats", default="output/2026-11-05/msd/teacher_stats.npz")
    ap.add_argument("--n_img", type=int, default=700)
    a = ap.parse_args()
    dev = torch.device("cuda")
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    nin = np.load(RC.NIN_SG.format("val"))
    norm = MS.Norm(a.stats, dev)
    ds = MS.MSDImageDS(dva, Yva, flip=False, want_crops=True)
    sel = np.sort(np.random.default_rng(0).choice(len(ds), min(a.n_img, len(ds)), replace=False))
    chunks = [list(sel[i:i + 8]) for i in range(0, len(sel), 8)]
    rows = np.concatenate([ds.groups[i] for i in sel])
    from pom_common import NAMES, T_OF  # noqa: E402
    mat = np.array([T_OF[k] == "material" for k in range(len(NAMES))])
    pat = np.array([T_OF[k] == "pattern_marking" for k in range(len(NAMES))]) & (np.arange(len(NAMES)) != NAMES.index("plain"))
    Ys, lv = Yva[rows], dva.level.values[rows]
    strata = {"all": np.ones(len(rows), bool), "object n_in 1-16": (lv == "object") & (nin[rows] <= 16),
              "object 17+": (lv == "object") & (nin[rows] > 16), "part": lv == "part",
              "patterned (non-plain)": (Ys[:, pat] == 1).any(1), "multi-material": (Ys[:, mat] == 1).sum(1) >= 2}
    enc = MS.Student(adapters=False).to(dev).eval()
    conds = {"bil": None, "H1": a.h1}
    for c in ("A0", "A1", "B0", "B1"):
        f = os.path.join(a.run, c, "best_dev.pt")
        if os.path.exists(f):
            conds[c] = f
    res, cache_ok = {}, True
    for c, path in conds.items():
        dec = None if path is None else MS.load_decoder(torch.load(path, map_location="cpu"), dev)
        acc, stage_acc = {}, {}
        with torch.no_grad():
            for ch in chunks:
                b = MS.to_dev(ds[ch], dev)
                g = b["geom"]
                feats, _ = enc(b["full"])
                T = MS.teacher_feats(enc.vit, b["crops"])
                m = g["valid_w"] > 0
                w = g["valid_w"].float()
                before = {d: float(v.double().sum()) for d, v in feats.items()}
                stages = None
                if dec is None:
                    Z = bilinear(feats[12], g)
                elif hasattr(dec, "arch") and dec.arch in ("flat", "hier", "oneway", "twoway"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        Z, stages = dec(feats, g, return_stages=True)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        Z = dec(feats, g)
                after = {d: float(v.double().sum()) for d, v in feats.items()}
                cache_ok &= before == after
                Z = Z.float()
                Zn, Tn = norm.n(Z, 12), norm.n(T[12], 12)
                rec = {}
                l = (Zn - Tn).pow(2).mean(-1) + (1 - F.cosine_similarity(Zn, Tn, dim=-1))
                rec["L12"] = (l * w).sum(1) / w.sum(1)
                sp = roi_metrics(Zn, Tn, w)
                rec["var_ratio"], rec["affinity_gap"], rec["cos12_norm"] = sp["var_ratio"], sp["affinity_gap"], sp["cos"]
                er_p, er_t = erank(Zn, m), erank(Tn, m)
                rec["erank_pred"], rec["erank_teacher"], rec["erank_ratio"] = er_p, er_t, er_p / er_t
                rec["neigh_diff_ratio"] = neigh_diff(Zn, m) / neigh_diff(Tn, m).clamp(min=1e-6)
                rec["centred_cos"] = centred_cos(Zn, Tn, m)
                for k, v in rec.items():
                    acc.setdefault(k, []).append(v.cpu())
                if stages:
                    for i, S in enumerate(stages):
                        S = S.float()
                        if S.shape[-1] == 768:          # A: stage outputs in block-12 space
                            Sn = norm.n(S, 12)
                            sv = svar(Sn, m) / svar(Tn, m).clamp(min=1e-8)
                            se = erank(Sn, m) / er_t
                        else:                            # B: working map in hidden space, relative to W0
                            sv, se = svar(S, m), erank(S, m)
                        stage_acc.setdefault(f"stage{i + 1}_var", []).append(sv.cpu())
                        stage_acc.setdefault(f"stage{i + 1}_erank", []).append(se.cpu())
        arr = {k: torch.cat(v).numpy() for k, v in acc.items()}
        np.savez(os.path.join(a.run, f"sdirect_{c}.npz"), rows=rows, **arr)
        res[c] = {sn: {k: float(np.nanmean(v[mk])) for k, v in arr.items()} | {"n_rois": int(mk.sum())} for sn, mk in strata.items()}
        if stage_acc:
            st = {k: float(torch.cat(v).mean()) for k, v in stage_acc.items()}
            res[c]["stages"] = st
        print(c, json.dumps(res[c]["all"]), json.dumps(res[c].get("stages", {})), flush=True)
    res["cache_unchanged_all_forwards"] = bool(cache_ok)
    json.dump(res, open(os.path.join(a.run, "sdirect.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
