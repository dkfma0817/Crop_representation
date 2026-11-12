"""Held-out feature analysis of FA-A / FA-B (468 COCO dev images, the fixed ROIs of src.tsplit.train.dev_batch: 8 x 1..6, 2 x 7..12,
1 whole image; no flip). Analysis only. Per ROI shape group (no padding), every child of the rectangle is decoded in one pass.
Metrics over valid children (letterbox-invalid excluded), parent statistics over valid children:
  child MSE, cosine to teacher, norm ratio ||Y|| / ||T||, parent-mean MSE, within-parent residual MSE and R^2 (1 - SSE / teacher residual
  energy), residual norm ratio, residual cosine (r_S vs r_T), correction sizes ||DeltaMean||, ||DeltaRes||, and residual positional
  diversity = mean pairwise cosine between the 4 residual vectors of a parent (student vs teacher; parents with 4 valid children).

  python -m src.finedec.analysis --a <A model.pt> --b <B model.pt> --out <json>
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.finedec.feats import student_pre
from src.finedec.losses import parent_stats
from src.finedec.model import FineAware
from src.tsplit.features import frozen_vit, teacher, valid48
from src.tsplit.model import rect_index
from src.tsplit.train import dev_batch, dev_set


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    dev = torch.device("cuda")
    vit = frozen_vit(dev)
    models = {}
    for k, p in (("FA-A", x.a), ("FA-B", x.b)):
        m = FineAware(vit).to(dev).eval()
        m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["model"])
        models[k] = m
    ids = dev_set()
    acc = {}
    for i in range(0, len(ids), 50):
        b = dev_batch(ids[i:i + 50])
        xx, x768 = b["x"].to(dev), b["x768"].to(dev)
        final, xpre = student_pre(vit, xx)
        T = teacher(vit, x768)
        v48 = valid48(b["content"].to(dev))
        rects, rimg = b["rects"].to(dev), b["rimg"].to(dev)
        shapes = {}
        for r, (H, W) in enumerate(rects[:, 2:].tolist()):
            shapes.setdefault((H, W), []).append(r)
        side = {}
        for k, m in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                side[k] = m.image_side(xx, final)
        for (H, W), rows in shapes.items():
            sel = torch.tensor(rows, device=dev)
            rs, ri = rects[sel], rimg[sel]
            idx, cy, cx, ok = rect_index(rs)
            v = ok & v48[ri[:, None], idx]
            vf = v.float()
            Tt = T[ri[:, None], idx]
            lp = (cy // 2 - rs[:, 0:1]) * rs[:, 3:4] + (cx // 2 - rs[:, 1:2])
            mut, _, _ = parent_stats(Tt, lp, v)
            rT = Tt - mut
            n = len(rows)
            full4 = vf.view(n, H, 2, W, 2).sum((2, 4)) == 4
            for k, m in models.items():
                fine, kvs = side[k]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Y, Z, dm, dr = m.decode(fine, kvs, xpre, ri, idx, cy, cx, ok, v, rs[:, 0], rs[:, 1], rs[:, 3])
                Y = Y.float()
                mus, _, _ = parent_stats(Y, lp, v)
                rS = Y - mus
                per = dict(child_mse=((Y - Tt).pow(2).mean(-1) * vf).sum(1), cos=(F.cosine_similarity(Y, Tt, dim=-1) * vf).sum(1),
                           normY=(Y.norm(dim=-1) * vf).sum(1), normT=(Tt.norm(dim=-1) * vf).sum(1), pm_mse=((mus - mut).pow(2).mean(-1) * vf).sum(1),
                           res_se=((rS - rT).pow(2).sum(-1) * vf).sum(1), res_en=(rT.pow(2).sum(-1) * vf).sum(1), res_mse=((rS - rT).pow(2).mean(-1) * vf).sum(1),
                           rnS=(rS.norm(dim=-1) * vf).sum(1), rnT=(rT.norm(dim=-1) * vf).sum(1),
                           rcos=(F.cosine_similarity(rS, rT, dim=-1) * vf).sum(1), dmn=(dm.float().norm(dim=-1) * vf).sum(1), drn=(dr.float().norm(dim=-1) * vf).sum(1), cnt=vf.sum(1))
                # positional diversity: pairwise cosine of the 4 residuals of a parent
                for who, rr in (("S", rS), ("T", rT)):
                    q = F.normalize(rr.view(n, H, 2, W, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(n, H, W, 4, -1), dim=-1)
                    gram = q @ q.transpose(-1, -2)
                    pc = (gram.sum((-1, -2)) - 4) / 12
                    per[f"pair_cos_{who}"] = (pc * full4).sum((1, 2))
                per["n_full4"] = full4.float().sum((1, 2))
                per = {kk: vv.tolist() for kk, vv in per.items()}
                for j, r in enumerate(rows):
                    for bk in (b["bucket"][r], "all"):
                        a = acc.setdefault((k, bk), {})
                        a["n_roi"] = a.get("n_roi", 0) + 1
                        for kk, vv in per.items():
                            a[kk] = a.get(kk, 0.0) + vv[j]
    out = {}
    for (k, bk), a in acc.items():
        c = a["cnt"]
        out.setdefault(k, {})[bk] = dict(n_roi=a["n_roi"], child_mse=a["child_mse"] / c, cos=a["cos"] / c, norm_ratio=a["normY"] / a["normT"], parent_mean_mse=a["pm_mse"] / c,
                                         residual_mse=a["res_mse"] / c, residual_r2=1 - a["res_se"] / a["res_en"], residual_norm_ratio=a["rnS"] / a["rnT"], residual_cos=a["rcos"] / c,
                                         delta_mean_norm=a["dmn"] / c, delta_res_norm=a["drn"] / c, residual_pair_cos_student=a["pair_cos_S"] / max(a["n_full4"], 1),
                                         residual_pair_cos_teacher=a["pair_cos_T"] / max(a["n_full4"], 1))
    Path(x.out).write_text(json.dumps(out, indent=1))
    for k in out:
        print(k, {kk: round(vv, 5) for kk, vv in out[k]["all"].items()}, flush=True)


if __name__ == "__main__":
    main()
