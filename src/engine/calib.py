"""Pre-registered loss-weight calibration at the shared init (configs/protocol.md), on FIXED train batches = the first 8 rows
of the training stream (8 x 64 ROIs), micro-batched exactly as in training:
  r = median_b ||dLPIPS/dZ|| / ||dL1/dZ||          -> beta = 1 if r <= 1 else largest of {0.1, 0.2, 0.5, 1} with beta * r <= 1
  q = median_b ||d(L1 + beta LPIPS)/dZ|| / ||dL_feat/dZ||
                                                   -> lam = largest of {0.01 .. 100} with lam * q <= 1 (pixel term on equal footing)
Z = CoordMLP output (raw latent), losses are batch means over the 64 ROIs (as in training). Param-level norms are recorded too
(at init only the zero-init last layer receives gradient). Weights are then fixed for A/B/C and never changed.
  python -m src.engine.calib --micro M  -> results/calibration.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

import src.rae.common as RC  # noqa: E402
import src.losses.pixel as PX  # noqa: E402
import src.data.dataset as DR  # noqa: E402
import src.engine.train as TR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--micro", type=int, required=True)
ap.add_argument("--nb", type=int, default=8)
a = ap.parse_args()
dev = "cuda"
RC.set_fp32()
rae = RC.FrozenRAE(dev)
del rae.enc
lp = PX.load_lpips(dev)
m = TR.load_model(TR.INIT, dev)
m.train()
ps = [p for p in m.parameters()]
P = torch.as_tensor(RC.query_coords(), device=dev)
ds = DR.CachedSet("stream", rgb=True)
rows = []
for batch in DR.loader(ds, DR.stream_slots(a.nb), 4):
    b = DR.to_dev(batch, dev)
    R = b["teacher"].shape[0]
    gz = {k: 0.0 for k in ("feat", "l1", "lpips")}
    gp = {k: [torch.zeros_like(p) for p in ps] for k in gz}
    gzs = {k: [] for k in gz}
    for i in range(0, R, a.micro):
        ix = torch.arange(i, min(R, i + a.micro), device=dev)
        g = TR.sub(b["geom"], ix)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Z = m(RC.gather_roi(b["full"], g), None, g, P).float()
        mse, cos = PX.feat_mse_cos(Z, b["teacher"][ix], g["valid_w"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = rae.decode(Z).float()
        l1, lpv = PX.pix_terms(y, b["rgb"][ix], b["mask"][ix], lp)
        terms = dict(feat=(mse + 1 - cos).sum() / R, l1=l1.sum() / R, lpips=lpv.sum() / R)
        for k, L in terms.items():
            out = torch.autograd.grad(L, [Z] + ps, retain_graph=True, allow_unused=True)
            gzs[k].append(out[0])
            for acc, x in zip(gp[k], out[1:]):
                if x is not None:
                    acc += x
    rec = {}
    for k in gz:
        rec[f"gz_{k}"] = float(torch.cat(gzs[k]).norm())
        rec[f"gp_{k}"] = float(torch.sqrt(sum((x ** 2).sum() for x in gp[k])))
    Gf, G1, Gl = (torch.cat(gzs[k]).flatten() for k in ("feat", "l1", "lpips"))
    rec["cos_feat_l1"] = float(torch.nn.functional.cosine_similarity(Gf, G1, dim=0))
    rec["cos_feat_lpips"] = float(torch.nn.functional.cosine_similarity(Gf, Gl, dim=0))
    rec["_G"] = (Gf, G1, Gl)
    rows.append(rec)
    print({k: v for k, v in rec.items() if k != "_G"}, flush=True)
r = float(np.median([x["gz_lpips"] / x["gz_l1"] for x in rows]))
beta = 1.0 if r <= 1 else max(bb for bb in (0.1, 0.2, 0.5, 1.0) if bb * r <= 1)
q = float(np.median([float((x["_G"][1] + beta * x["_G"][2]).norm()) / x["gz_feat"] for x in rows]))
grid = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
lam = max(l for l in grid if l * q <= 1)
qp = float(np.median([float(np.hypot(x["gp_l1"], beta * x["gp_lpips"])) / x["gp_feat"] for x in rows]))
cos_fp = float(np.median([float(torch.nn.functional.cosine_similarity(x["_G"][0], x["_G"][1] + beta * x["_G"][2], dim=0)) for x in rows]))
out = dict(rule="beta: r<=1 -> 1 else largest {0.1,0.2,0.5,1} with beta*r<=1; lam: largest grid value with lam*q<=1",
           init=TR.INIT, batches="first %d stream rows (64 ROIs each), micro %d" % (a.nb, a.micro),
           per_batch=[{k: v for k, v in x.items() if k != "_G"} for x in rows],
           ratio_lpips_over_l1=r, beta=beta, ratio_pix_over_feat_Z=q, lam=lam, lam_times_q=lam * q,
           param_level_ratio_pix_over_feat_approx=qp, cos_gradZ_feat_pix_median=cos_fp)
json.dump(out, open(os.path.join(RC.RES, "calibration.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_batch"}, indent=1))
