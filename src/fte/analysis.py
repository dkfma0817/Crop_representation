"""Held-out COCO feature analysis of P1 / P2 (468 dev images, fixed ROIs of src.tsplit.train.dev_batch; analysis only).
Per model, over valid children: teacher cosine, norm ratio, parent-mean MSE, within-parent residual MSE and R^2, residual cosine, child
diversity = mean pairwise cosine of the 4 children of a parent (student vs teacher, full post-norm features), per child slot (k = 2a + b)
residual norm ratio and residual cosine. P2: relative weight change of the copied blocks 9-12 vs the pretrained ones.

  python -m src.fte.analysis --p1 <P1 model.pt> --p2 <P2 model.pt> --out <json>
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.fte.common import child_ids, rect_parents
from src.fte.models import P1, P2
from src.fte.train import student_h8
from src.tsplit.features import frozen_vit, teacher, valid48
from src.tsplit.train import dev_batch, dev_set


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    vit = frozen_vit(dev)
    models = {}
    for k, cls, p in (("P1", P1, a.p1), ("P2", P2, a.p2)):
        m = cls(vit).to(dev).eval()
        m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["model"])
        models[k] = m
    out = dict(P2_block_change={})
    for k, (b0, b1) in enumerate(zip(vit.blocks[8:], models["P2"].blocks)):
        num = sum(float((p1 - p0).float().pow(2).sum()) for p0, p1 in zip(b0.parameters(), b1.parameters()))
        den = sum(float(p0.float().pow(2).sum()) for p0 in b0.parameters())
        out["P2_block_change"][f"block{9 + k}"] = (num / den) ** 0.5
    acc = {}
    ids = dev_set()
    for i in range(0, len(ids), 25):
        b = dev_batch(ids[i:i + 25])
        x, x768 = b["x"].to(dev), b["x768"].to(dev)
        final, h8 = student_h8(vit, x)
        T = teacher(vit, x768)
        v48 = valid48(b["content"].to(dev))
        rects, rimg = b["rects"].to(dev), b["rimg"].to(dev)
        for r0 in range(0, len(rects), 64):
            rs, ri = rects[r0:r0 + 64], rimg[r0:r0 + 64]
            buckets = b["bucket"][r0:r0 + 64]
            for name, m in models.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Y, Z, pid, pok = m(final if name == "P1" else h8, ri, rs)
                cid, _, _ = child_ids(pid)
                v = pok[..., None] & v48[ri[:, None, None], cid]
                Tt = T[ri[:, None, None], cid].float()
                Y = Y.float()
                vf = v.float()
                mS = (Y * vf[..., None]).sum(2, keepdim=True) / vf.sum(2, keepdim=True)[..., None].clamp_min(1)
                mT = (Tt * vf[..., None]).sum(2, keepdim=True) / vf.sum(2, keepdim=True)[..., None].clamp_min(1)
                rS, rT = Y - mS, Tt - mT
                full = v.all(-1)

                def pc(X):
                    q = F.normalize(X, dim=-1)
                    return (q @ q.transpose(-1, -2)).sum((-1, -2)).sub(4).div(12)
                per = dict(cos=(F.cosine_similarity(Y, Tt, dim=-1) * vf).sum((1, 2)), nY=(Y.norm(dim=-1) * vf).sum((1, 2)), nT=(Tt.norm(dim=-1) * vf).sum((1, 2)),
                           pm=(((mS - mT)[:, :, 0].pow(2).mean(-1)) * vf.sum(-1)).sum(1), rse=((rS - rT).pow(2).sum(-1) * vf).sum((1, 2)), ren=(rT.pow(2).sum(-1) * vf).sum((1, 2)),
                           rmse=((rS - rT).pow(2).mean(-1) * vf).sum((1, 2)), rcos=(F.cosine_similarity(rS, rT, dim=-1) * vf).sum((1, 2)), cnt=vf.sum((1, 2)),
                           divS=(pc(Y) * full).sum(1), divT=(pc(Tt) * full).sum(1), nfull=full.float().sum(1))
                for s in range(4):
                    per[f"slot{s}_rn_S"] = (rS[:, :, s].norm(dim=-1) * vf[:, :, s]).sum(1)
                    per[f"slot{s}_rn_T"] = (rT[:, :, s].norm(dim=-1) * vf[:, :, s]).sum(1)
                    per[f"slot{s}_rcos"] = (F.cosine_similarity(rS[:, :, s], rT[:, :, s], dim=-1) * vf[:, :, s]).sum(1)
                    per[f"slot{s}_cnt"] = vf[:, :, s].sum(1)
                per = {kk: vv.tolist() for kk, vv in per.items()}
                for j, bk in enumerate(buckets):
                    for key in (bk, "all"):
                        d = acc.setdefault((name, key), {})
                        d["n_roi"] = d.get("n_roi", 0) + 1
                        for kk, vv in per.items():
                            d[kk] = d.get(kk, 0.0) + vv[j]
    for (name, key), d in acc.items():
        c = d["cnt"]
        out.setdefault(name, {})[key] = dict(n_roi=d["n_roi"], cos=d["cos"] / c, norm_ratio=d["nY"] / d["nT"], parent_mean_mse=d["pm"] / c, residual_mse=d["rmse"] / c,
                                            residual_r2=1 - d["rse"] / d["ren"], residual_cos=d["rcos"] / c, child_pair_cos_student=d["divS"] / max(d["nfull"], 1),
                                            child_pair_cos_teacher=d["divT"] / max(d["nfull"], 1),
                                            per_slot={s: dict(residual_norm_ratio=d[f"slot{s}_rn_S"] / max(d[f"slot{s}_rn_T"], 1e-9), residual_cos=d[f"slot{s}_rcos"] / max(d[f"slot{s}_cnt"], 1)) for s in range(4)})
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(P2_block_change=out["P2_block_change"], **{k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in out[k]["all"].items() if kk != "per_slot"} for k in ("P1", "P2")}), indent=1), flush=True)


if __name__ == "__main__":
    main()
