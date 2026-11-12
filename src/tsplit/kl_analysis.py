"""Held-out feature comparison S3-L2 vs S3-Distill (vs B0), analysis only. Same 468 COCO dev images and fixed ROIs as src.tsplit.train.Held.

Two views per model and ROI bucket:
  raw  : the decoder output as it goes to R2 (MSE, cosine, within-parent residual MSE / R^2, parent-mean MSE, mean L2 norms)
  unit : every child feature (student and teacher) L2-normalised first -> the same metrics on directions only (the KL losses are
         scale-invariant, so this separates a scale change from a structural change)

  python -m src.tsplit.kl_analysis --l2 <S3-L2 S3.pt> --kl <S3-Distill S3.pt> --out <json>
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.tsplit.features import up2
from src.tsplit.model import TokenSplit, rect_index
from src.tsplit.train import Held, dev_set
from src.tsplit.features import frozen_vit


def metrics(y, t, w, H, W):
    n = len(y)
    mse = ((y - t).pow(2).mean(-1) * w).sum(1) / w.sum(1)
    cos = (F.cosine_similarity(y, t, dim=-1) * w).sum(1) / w.sum(1)
    yy, tt, oo = y.view(n, H, 2, W, 2, -1), t.view(n, H, 2, W, 2, -1), w.view(n, H, 2, W, 2)
    full = (oo.sum((2, 4)) == 4).float()
    ry, rt = yy - yy.mean((2, 4), keepdim=True), tt - tt.mean((2, 4), keepdim=True)
    rs = ((ry - rt).pow(2).mean(-1).sum((2, 4)) / 4 * full).sum((1, 2))
    re = (rt.pow(2).mean(-1).sum((2, 4)) / 4 * full).sum((1, 2))
    pm = ((yy.mean((2, 4)) - tt.mean((2, 4))).pow(2).mean(-1) * full).sum((1, 2))
    ny = (y.norm(dim=-1) * w).sum(1) / w.sum(1)
    nt = (t.norm(dim=-1) * w).sum(1) / w.sum(1)
    return dict(mse=mse, cos=cos, res_se=rs, res_en=re, pm_se=pm, parents=full.sum((1, 2)), norm_s=ny, norm_t=nt)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", required=True)
    ap.add_argument("--kl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    vit = frozen_vit(dev)
    held = Held(vit, dev, dev_set())
    models = {}
    for name, p in (("S3-L2", a.l2), ("S3-Distill", a.kl)):
        m = TokenSplit("S3").to(dev).eval()
        m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["model"])
        models[name] = m
    acc = {}
    for part in held.parts:
        T, fin, ear = part["T"].float(), part["fin"].float(), part["ear"].float()
        shapes = {}
        for i, (H, W) in enumerate(part["rects"][:, 2:].tolist()):
            shapes.setdefault((H, W), []).append(i)
        cmaps = {"B0": up2(fin)}
        for k, m in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cmaps[k] = m.child_map(fin, ear).float()
        for (H, W), rows in shapes.items():
            sel = torch.tensor(rows, device=dev)
            ri = part["rimg"][sel]
            idx, cy, cx, ok = rect_index(part["rects"][sel])
            key_ok = ok & part["v48"][ri[:, None], idx]
            t = T[ri[:, None], idx]
            w = key_ok.float()
            for k in ("B0",) + tuple(models):
                if k == "B0":
                    y = cmaps[k][ri[:, None], idx]
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = models[k].refine(cmaps[k], ri, idx, cy, cx, key_ok).float()
                for view, (yv, tv) in (("raw", (y, t)), ("unit", (F.normalize(y, dim=-1), F.normalize(t, dim=-1)))):
                    r = {kk: v.tolist() for kk, v in metrics(yv, tv, w, H, W).items()}
                    for j, i in enumerate(rows):
                        for bk in (part["bucket"][i], "all") + (("1x1",) if (H, W) == (1, 1) else ()):
                            d = acc.setdefault((k, view, bk), {})
                            d["n"] = d.get("n", 0) + 1
                            for kk, v in r.items():
                                d[kk] = d.get(kk, 0.0) + v[j]
    out = {}
    for (k, view, bk), d in acc.items():
        n, P = d["n"], max(d["parents"], 1)
        out.setdefault(k, {}).setdefault(view, {})[bk] = dict(n_roi=n, mse=d["mse"] / n, cos=d["cos"] / n, within_parent_residual_mse=d["res_se"] / P, within_parent_r2=1 - d["res_se"] / max(d["res_en"], 1e-12),
                                                             parent_mean_mse=d["pm_se"] / P, norm_student=d["norm_s"] / n, norm_teacher=d["norm_t"] / n)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k in out:
        for view in ("raw", "unit"):
            v = out[k][view]["all"]
            print(k, view, {kk: round(vv, 5) for kk, vv in v.items()}, flush=True)


if __name__ == "__main__":
    main()
