"""Direct capability checks of the pre-trained student encoder + ROI decoder (PACO VAL images, report only; no selection).

For 700 fixed val images (rng 0, every ROI of each image), per ROI on valid teacher cells (crop content coverage > 0):
  1. feature matching vs the C0 teacher at blocks 4 / 8 / 12: L_d (normalised MSE + 1 - cos, the training loss) and cos
     (block 12: raw Z vs raw teacher; blocks 4 / 8: aux-head prediction in normalised space - aux heads are training-only)
  2. spatial: mean pairwise token cosine inside the ROI (pred vs teacher) and spatial variance ratio
     var_pred / var_teacher (mean over channels of the variance across ROI cells, normalised block-12 space)
  3. patch-to-patch affinity gap: mean |A_pred - A_teacher| over valid cell pairs (A = cosine-similarity matrix)
  4. strata: object n_in 1-16 (SigLIP2 n_in, existing bins), object 17+, part, ROIs with a material / pattern positive
  5. global change vs the frozen original encoder (all 2,051 val images): cos of CLS, of mean patch token, mean per-patch cos
Reference row "bil": no decoder, bilinear read of the ORIGINAL block d maps at the crop grid (d = 4, 8, 12).

  CUDA_VISIBLE_DEVICES=2 python -m src.msd.direct --run output/2026-11-05/msd
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.msd.common as MS
import src.readout.common as RC
from models.roi_reader import bilinear  # noqa: E402

TYPES = RC.ET.__dict__.get("TYPES", None)
EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"


def roi_metrics(P, T, w):
    """P, T (R,576,D) in the same space, w (R,576) -> dict of (R,) tensors."""
    m = (w > 0).float()
    cos = F.cosine_similarity(P, T, dim=-1)
    out = {"cos": (cos * m).sum(1) / m.sum(1)}
    Pn, Tn = F.normalize(P, dim=-1), F.normalize(T, dim=-1)
    Ap, At = Pn @ Pn.transpose(1, 2), Tn @ Tn.transpose(1, 2)
    mm = m[:, :, None] * m[:, None, :]
    eye = torch.eye(P.shape[1], device=P.device)[None]
    off = mm * (1 - eye)
    out["pair_cos_pred"] = (Ap * off).sum((1, 2)) / off.sum((1, 2)).clamp(min=1)
    out["pair_cos_teacher"] = (At * off).sum((1, 2)) / off.sum((1, 2)).clamp(min=1)
    out["affinity_gap"] = ((Ap - At).abs() * off).sum((1, 2)) / off.sum((1, 2)).clamp(min=1)

    def svar(X):
        mu = (X * m[..., None]).sum(1, keepdim=True) / m.sum(1)[:, None, None]
        return (((X - mu) ** 2) * m[..., None]).sum(1).mean(-1) / m.sum(1)
    out["var_ratio"] = svar(P) / svar(T).clamp(min=1e-8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n_img", type=int, default=700)
    a = ap.parse_args()
    dev = torch.device("cuda")
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    nin = np.load(RC.NIN_SG.format("val"))
    names = RC.ET.__dict__  # noqa: F841
    norm = MS.Norm(os.path.join(a.run, "teacher_stats.npz"), dev)
    ds = MS.MSDImageDS(dva, Yva, flip=False, want_crops=True)
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(ds), min(a.n_img, len(ds)), replace=False))
    chunks = [list(sel[i:i + 8]) for i in range(0, len(sel), 8)]
    rows = np.concatenate([ds.groups[i] for i in sel])
    import pandas as pd
    attr = RC.ET.EM.__dict__.get("T_OF")
    from pom_common import NAMES, T_OF  # noqa: E402
    mat = np.array([T_OF[k] == "material" for k in range(len(NAMES))])
    pat = np.array([T_OF[k] == "pattern_marking" for k in range(len(NAMES))])
    Ys = Yva[rows]
    lv = dva.level.values[rows]
    strata = {"all": np.ones(len(rows), bool), "object n_in 1-16": (lv == "object") & (nin[rows] <= 16),
              "object 17+": (lv == "object") & (nin[rows] > 16), "part": lv == "part",
              # almost every ROI has some material / pattern positive -> use the informative cases instead
              "material-positive ROI": (Ys[:, mat] == 1).sum(1) >= 2,
              "pattern-positive ROI": (Ys[:, pat & (np.arange(len(NAMES)) != NAMES.index("plain"))] == 1).any(1)}
    vit_orig = MS.Student(adapters=False).to(dev).eval()
    conds = ["bil"] + [c for c in ("H0", "H1", "H2", "H3") if os.path.exists(os.path.join(a.run, c, "best_dev.pt"))]
    res = {}
    for c in conds:
        if c == "bil":
            student, dec = vit_orig, None
        else:
            ck = torch.load(os.path.join(a.run, c, "best_dev.pt"), map_location="cpu")
            student = MS.Student(adapters=ck["adapters"] is not None).to(dev).eval()
            if ck["adapters"] is not None:
                student.adapters.load_state_dict(ck["adapters"])
            dec = MS.load_decoder(ck, dev)
        acc = {}
        with torch.no_grad():
            for ch in chunks:
                b = MS.to_dev(ds[ch], dev)
                g = b["geom"]
                feats, _ = student(b["full"])
                T = MS.teacher_feats(vit_orig.vit, b["crops"])
                w = g["valid_w"].float()
                if dec is None:
                    pred = {d: bilinear(feats[d], g) for d in MS.DEPTHS}
                    predn = {d: norm.n(pred[d], d) for d in MS.DEPTHS}
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        Z = dec(feats, g)
                    Z = Z.float()
                    Zn = norm.n(Z, 12)
                    aux = {d: v.float() for d, v in dec.aux_pred(Zn).items()}
                    pred = {12: Z}
                    predn = {12: Zn, 4: aux[4], 8: aux[8]}
                rec = {}
                for d in MS.DEPTHS:
                    tn = norm.n(T[d], d)
                    l = (predn[d] - tn).pow(2).mean(-1) + (1 - F.cosine_similarity(predn[d], tn, dim=-1))
                    rec[f"L{d}"] = (l * w).sum(1) / w.sum(1)
                    rec[f"cos{d}_norm"] = ((F.cosine_similarity(predn[d], tn, dim=-1)) * (w > 0)).sum(1) / (w > 0).sum(1)
                sp = roi_metrics(norm.n(pred[12], 12), norm.n(T[12], 12), w)
                rec.update({f"{k}_b12": v for k, v in sp.items()})
                rec["cos12_raw"] = roi_metrics(pred[12], T[12], w)["cos"]
                for k, v in rec.items():
                    acc.setdefault(k, []).append(v.cpu())
        arr = {k: torch.cat(v).numpy() for k, v in acc.items()}
        np.savez(os.path.join(a.run, f"direct_{c}.npz"), rows=rows, **arr)
        res[c] = {sn: {k: float(np.nanmean(v[m])) for k, v in arr.items()} | {"n_rois": int(m.sum())} for sn, m in strata.items()}
        # global representation change vs the original encoder (all val images)
        if c in ("H2", "H3"):
            dl = DataLoader(MS.MSDImageDS(dva, Yva, flip=False, want_crops=False), sampler=[list(range(i, min(len(ds), i + 24))) for i in range(0, len(ds), 24)],
                            batch_size=1, num_workers=8, collate_fn=MS.collate_one)
            gl = {"cls_cos": [], "gap_cos": [], "patch_cos": []}
            with torch.no_grad():
                for b in dl:
                    x = b["full"].to(dev)
                    f1, c1 = student(x)
                    f0, c0 = student(x, use_adapters=False)
                    gl["cls_cos"].append(F.cosine_similarity(c1, c0, dim=-1).cpu())
                    gl["gap_cos"].append(F.cosine_similarity(f1[12].mean(1), f0[12].mean(1), dim=-1).cpu())
                    gl["patch_cos"].append(F.cosine_similarity(f1[12], f0[12], dim=-1).mean(1).cpu())
            res[c]["global"] = {k: float(torch.cat(v).mean()) for k, v in gl.items()} | {"n_images": len(ds)}
        elif c != "bil":
            res[c]["global"] = {"cls_cos": 1.0, "gap_cos": 1.0, "patch_cos": 1.0, "note": "encoder frozen (no adapters): identical by construction"}
        print(c, json.dumps(res[c]["all"]), flush=True)
    json.dump(res, open(os.path.join(a.run, "direct.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
