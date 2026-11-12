"""COCO / LVIS headroom report (CPU): Full vs C0 Tight for each dataset, on the saved selected-epoch val predictions.

Classification (primary): top-1, top-5, mAcc (+ S/M/L, + LVIS r/c/f) from all-class scores (src.cocolvis.metrics).
GT-box oracle AP (secondary): official cocoapi / lvis-api on detections built by src.cocolvis.apeval (RegionCLIP GT rule).
Bootstrap: image-level, B = 1000, seed 0; the SAME multinomial image weights for Full and Crop (paired); all ROIs of an
image move together. AP resamples use the verified weighted re-accumulation of the official per-image matching.
Size bins: classification uses the GT BOX area (original px); AP is given with the official GT 'area' (segmentation) bins
and with GT-box-area bins. Single probe seed 0: CIs cover val-image sampling only, NOT training-seed variance.

  python -m src.cocolvis.report --run output/2026-11-07/cocolvis_headroom
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.cocolvis import apeval as AE
from src.cocolvis.metrics import macc, per_instance, summary

B, SEED = 1000, 0
_G = {}


def _boot_ap(b):
    W = _G["W"][b]
    out = {}
    for key, F in _G["F"].items():
        out[key] = F.summary(W)
    return out


def ci(v):
    v = np.asarray(v, float)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def cls_boot(c1, c5, y, size, freq, img_idx, W, K):
    """-> dict metric -> (B,) weighted values."""
    out = {k: [] for k in ["top1", "top5", "mAcc"] + [f"{m}_{s}" for m in ("top1", "mAcc") for s in "SML"]
           + ([f"{m}_{f}" for m in ("top1", "mAcc") for f in "rcf"] if freq is not None else [])}
    groups = [("S", size == 0), ("M", size == 1), ("L", size == 2)]
    if freq is not None:
        groups += [(f, freq == f) for f in "rcf"]
    for b in range(len(W)):
        w = W[b][img_idx]
        sw = w.sum()
        out["top1"].append((w * c1).sum() / sw)
        out["top5"].append((w * c5).sum() / sw)
        out["mAcc"].append(macc(c1, y, w, K))
        for g, m in groups:
            ww = w[m]
            out[f"top1_{g}"].append((ww * c1[m]).sum() / ww.sum() if ww.sum() > 0 else np.nan)
            out[f"mAcc_{g}"].append(macc(c1[m], y[m], ww, K))
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    res = dict(note="val = checkpoint selection AND reported numbers (user protocol); single seed 0; CI = image bootstrap only",
               datasets={})
    for ds in ("coco", "lvis"):
        conds = [f"{ds}_full", f"{ds}_tight"]
        if not all(os.path.exists(os.path.join(a.run, c, "final_scored.json")) for c in conds):
            print("skip", ds, "(not finished)")
            continue
        V = pd.read_parquet(os.path.join(a.run, "data", f"instances_{ds}_val.parquet"))
        cc = pd.read_csv(os.path.join(a.run, "data", f"class_counts_{ds}.csv"))
        K = len(cc)
        y, size = V.label.values, V["size"].values
        freq = cc.frequency.values[y] if ds == "lvis" else None
        J = {"seg": AE.gt_dict(ds, "seg"), "box": AE.gt_dict(ds, "box")}
        img_ids = sorted(im["id"] for im in J["seg"]["images"])
        pos = {i: k for k, i in enumerate(img_ids)}
        img_idx = np.array([pos[i] for i in V.image_id])
        rng = np.random.default_rng(SEED)
        W = rng.multinomial(len(img_ids), np.full(len(img_ids), 1.0 / len(img_ids)), size=a.B).astype(np.float64)
        D = dict(n_val_rois=len(V), n_val_images_with_roi=int(V.image_id.nunique()), n_eval_images=len(img_ids),
                 classes_evaluable=int(len(np.unique(y))), points={}, ap={}, ap_box_bins={}, selection={}, n_dets={})
        cb, Fs = {}, {}
        for c in conds:
            sel = json.load(open(os.path.join(a.run, c, "selection.json")))
            curves = [json.loads(l) for l in open(os.path.join(a.run, c, "curves.jsonl"))]
            D["selection"][c] = dict(selected_epoch=sel["selected_epoch"], val_top1_by_epoch=sel["all"],
                                     last_epoch_metrics=curves[-1]["val"],
                                     train_s=sum(r["epoch_train_s"] for r in curves), val_s=sum(r["val_s"] for r in curves))
            top5 = np.load(os.path.join(a.run, c, "val_top5.npy")).astype(np.int64)
            c1, c5 = per_instance(top5, y)
            D["points"][c] = summary(c1, c5, y, size, freq, K)
            cb[c] = cls_boot(c1.astype(float), c5.astype(float), y, size, freq, img_idx, W, K)
            prob = np.load(os.path.join(a.run, c, "val_prob.npy"), mmap_mode="r")
            assert np.allclose(prob[:1000].argmax(1), top5[:1000, 0])
            dets = AE.make_dets(ds, np.asarray(prob), V, cc.cat_id.values)
            D["n_dets"][c] = len(dets)
            del prob
            for area in ("seg", "box"):
                st, E = AE.run_official(ds, dets, J[area])
                F = AE.FastDetAP(E, ds)
                F.verify(st)
                assert F.I == len(img_ids)
                (D["ap"] if area == "seg" else D["ap_box_bins"])[c] = st
                Fs[(c, area)] = F
                del E
        # AP bootstrap (parallel over resamples; fork shares the FastDetAP objects)
        _G["W"], _G["F"] = W, {f"{c}|{area}": F for (c, area), F in Fs.items()}
        with Pool(a.procs) as p:
            BA = p.map(_boot_ap, range(a.B), chunksize=4)
        x, z = conds[1], conds[0]                                         # tight - full
        con = {}
        for k in cb[x]:
            lo, hi = ci(cb[x][k] - cb[z][k])
            con[k] = dict(full=D["points"][z][k], crop=D["points"][x][k], delta=D["points"][x][k] - D["points"][z][k], lo=lo, hi=hi)
        for area, src in (("seg", "ap"), ("box", "ap_box_bins")):
            for k in D[src][z]:
                if k == "AP75" or k not in BA[0][f"{x}|{area}"]:
                    continue
                d = [r[f"{x}|{area}"][k] - r[f"{z}|{area}"][k] for r in BA]
                lo, hi = ci(d)
                name = k if area == "seg" else f"{k}(boxbin)"
                if area == "box" and k not in ("APs", "APm", "APl"):
                    continue
                con[name] = dict(full=D[src][z][k], crop=D[src][x][k], delta=D[src][x][k] - D[src][z][k], lo=lo, hi=hi)
        D["crop_minus_full"] = con
        res["datasets"][ds] = D
        print(ds, json.dumps({k: round(v["delta"] * 100, 2) for k, v in con.items()}), flush=True)
    json.dump(res, open(os.path.join(a.run, "results.json"), "w"), indent=1, default=float)
    write_md(res, a.run)


def f(v):
    return f"{100 * v:.2f}"


def dci(c):
    s = "*" if c["lo"] > 0 or c["hi"] < 0 else ""
    return f"{100 * c['delta']:+.2f} [{100 * c['lo']:+.2f}, {100 * c['hi']:+.2f}]{s}"


def write_md(res, run):
    L = ["# COCO / LVIS GT-box headroom results (auto-generated by src.cocolvis.report)", "",
         "Frozen DINOv3-B/16, R2 + linear head, seed 0. Val selects the epoch AND gives the numbers. `*` = 95% image-level "
         "paired bootstrap CI excludes 0 (B = 1000). CIs do NOT include training-seed variance. AP = GT-box ORACLE AP "
         "(GT boxes as the only candidates), not detection AP.", ""]
    for ds, D in res["datasets"].items():
        C = D["crop_minus_full"]
        L += [f"## {ds.upper()}: {D['n_val_rois']:,} val GT regions / {D['n_val_images_with_roi']:,} images, "
              f"{D['classes_evaluable']} evaluable classes", "",
              "| metric | Full | Crop | Crop − Full [95% CI] |", "|---|---:|---:|---|"]
        for k, c in C.items():
            L.append(f"| {k} | {f(c['full'])} | {f(c['crop'])} | {dci(c)} |")
        L += ["", "| condition | selected epoch | val top-1 by epoch | train s | val s | detections |", "|---|---:|---|---:|---:|---:|"]
        for c, s in D["selection"].items():
            L.append(f"| {c} | {s['selected_epoch']} | " + ", ".join(f"{e}:{100 * v:.2f}" for e, v in s["val_top1_by_epoch"].items())
                     + f" | {s['train_s']:.0f} | {s['val_s']:.0f} | {D['n_dets'][c]:,} |")
        p = D["points"][f"{ds}_full"]
        L += ["", "ROIs / classes per bin: " + ", ".join(f"{g}: {p[f'n_{g}']:,} ROIs / {p[f'n_classes_{g}']} classes"
                                                      for g in (["S", "M", "L"] + (["r", "c", "f"] if ds == "lvis" else []))), ""]
    open(os.path.join(run, "results.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
