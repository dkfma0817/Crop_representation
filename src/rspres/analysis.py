"""Held-out COCO feature diagnosis for RSP-Residual vs RSP-Masked vs Step-0 (= F384-bil) against the p_T = 4 dense teacher.

Same images / ROIs / metrics as src.rsp.analysis (dev_set()[:300], rng (1e6, id): 8 rects 1..6 + 2 rects 7..12 parents, every valid
fine cell; roi_metrics). Adds for RSP-Residual: |Delta| / |F384_bil| per ROI (raw space, mean over valid cells, ratio of sums).

  python -m src.rspres.analysis --res <model.pt> --rsp <rsp pix model.pt> --out <json>
"""
import argparse
import json

import numpy as np
import torch

from src.rsp.analysis import load, roi_metrics
from src.rsp.common import PREFIX, PT, R_FP, fine_rect_index
from src.rsp.data import evidence, frects_of, rect_box
from src.rsp.encoder import load_rsp
from src.rsp.teacher import DenseTeacher
from src.rspres.encoder import load_res
from src.rspres.model import f384_bil
from src.tsplit.features import frozen_vit
from src.tsplit.train import MAXS, dev_set, sample_rects


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--rsp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_images", type=int, default=300)
    a = ap.parse_args()
    dev = torch.device("cuda")
    vit = frozen_vit(dev)
    teach = DenseTeacher(vit).to(dev)
    res, _ = load_res(a.res, dev)
    rsp, mu, sd, _ = load_rsp(a.rsp, dev)
    assert torch.equal(mu, res.mu) and torch.equal(sd, res.sd)
    ids = dev_set()[:a.n_images]
    acc = {}

    def add(name, bk, mt):
        d = acc.setdefault(name, {}).setdefault(bk, {})
        for kk, vv in mt.items():
            d.setdefault(kk, []).append(vv)
    for k0 in range(0, len(ids), 8):
        chunk = ids[k0:k0 + 8]
        xs, content, rects, rimg, boxes, evs, bucket = [], [], [], [], [], [], []
        for bi, iid in enumerate(chunk):
            img, lb = load(iid)
            xs.append(lb.canvas(img))
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
            tg, final = teach.targets(x, fr, rimg, content)
        final = final.float()
        idx, uy, ux, ok = fine_rect_index(fr)
        cen = torch.stack([(uy.float() + .5) * PT, (ux.float() + .5) * PT], -1)
        par = (uy // R_FP) * 24 + ux // R_FP
        preds, delta = {}, None
        for name, m in (("RSP-Residual", res), ("RSP-Masked", rsp)):
            outs, dl = [], []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cache = m.context(final)
                l1, z = m.level1(cache, rimg, box, ev)
                for s0 in range(0, len(rects), 16):
                    sl = slice(s0, s0 + 16)
                    outs.append(m.queries(cache, [(kk[sl], vv[sl]) for kk, vv in l1], rimg[sl], z[sl], cen[sl]).float())
            o = torch.cat(outs)
            if name == "RSP-Residual":
                base = f384_bil(final, rimg, cen)
                delta = o * sd
                preds[name] = base + delta
            else:
                preds[name] = o * sd + mu
        preds["Step-0 (= F384-bil)"] = f384_bil(final, rimg, cen)
        for r in range(len(rects)):
            T, v, _ = tg[r]
            n = len(T)
            bks = [bucket[r], "all"] + (["train_1to6"] if bucket[r] != "large_7to12" else [])
            for name, Y in preds.items():
                mt = roi_metrics(Y[r, :n].float(), T, v, par[r, :n], mu, sd)
                for bk in bks:
                    add(name, bk, mt)
            dv, bv = delta[r, :n][v], preds["Step-0 (= F384-bil)"][r, :n][v]
            dm = dict(delta_norm=float(dv.norm(dim=-1).sum()), bil_norm=float(bv.norm(dim=-1).sum()),
                      delta_cos_to_teacher_residual=float(torch.nn.functional.cosine_similarity(dv, T[v] - bv, dim=-1).mean()))
            for bk in bks:
                add("_delta", bk, dm)
        print(f"{k0 + len(chunk)}/{len(ids)} images", flush=True)
    out = {}
    for name, bks in acc.items():
        out[name] = {}
        for bk, d in bks.items():
            if name == "_delta":
                out[name][bk] = dict(delta_over_bil_norm=float(np.sum(d["delta_norm"]) / np.sum(d["bil_norm"])),
                                     delta_cos_to_teacher_minus_bil=float(np.mean(d["delta_cos_to_teacher_residual"])), n_roi=len(d["delta_norm"]))
                continue
            r = {k: float(np.nanmean(v)) for k, v in d.items() if k not in ("res_se", "res_energy")}
            r["within_parent_residual_r2"] = 1 - float(np.sum(d["res_se"]) / max(np.sum(d["res_energy"]), 1e-12))
            r["n_roi"] = len(d["cos"])
            out[name][bk] = r
    out["_meta"] = dict(n_images=len(ids), rois_per_image="8 x (1..6) + 2 x (7..12) parents", targets="dense teacher p_T 4 px, all valid fine cells",
                        note="parent_mean_mse = MSE between per-parent means of prediction and teacher (parent-mean preservation); "
                             "_delta: |Delta| / |F384_bil| (sums over valid cells) and cos(Delta, T - F384_bil)")
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({n: {b: {k: round(v, 4) for k, v in r.items() if k in ("cos", "l1_norm", "within_parent_residual_r2", "norm_ratio", "parent_mean_mse", "delta_over_bil_norm")}
                          for b, r in d.items() if b in ("all", "train_1to6", "large_7to12")} for n, d in out.items() if n != "_meta"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
