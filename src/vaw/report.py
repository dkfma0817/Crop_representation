"""VAW report: official metrics on the saved TEST probabilities (each condition's test was scored once), subsets, paired
image-level bootstrap. CPU only.

Subsets (same rows for every condition):
  all        : official VAW test (31,819 instances)
  leakfree   : test images whose VG coco_id is NOT a G1 distillation image (and not a confirmed dHash duplicate of one)
  nodup      : test images without a cross-split duplicate (same photo also in VAW train/val; coco_id / dHash)  [sensitivity]
  size S/M/L : COCO pixel-area bins of the clipped GT box in ORIGINAL VG pixels (< 32^2, 32^2-96^2, >= 96^2)  [own analysis]
Metrics = the official evaluator (eval/evaluator.py, unmodified): mAP (per-class AP mean), mR@15 (per-class top-15 recall),
mA (per-class balanced accuracy @0.5), F1@15 (overall top-15 F1), head/medium/tail and parent-type mAP. On a subset,
attributes with no labeled positive or no labeled negative in that subset are passed as the evaluator's `exclude_atts`
(GT-only rule, identical for every condition; the count is reported). On 'all' nothing is excluded (= official number).
Bootstrap: images resampled with replacement (B = 1000, seed 0), the SAME resample for both conditions of a contrast;
AP via a weighted re-implementation checked to equal the official per-class AP at weight 1; per resample, attributes
with 0 weighted positives or negatives are dropped for both conditions. CI = 2.5 / 97.5 percentiles of the difference.
Seed 0 single probe run: the CI covers test-image sampling only, NOT probe-training seed variance.

  python -m src.vaw.report --run output/2026-11-07/vaw_g1
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.vaw.common import attr_index, image_sizes, label_matrix, load_split
from src.vaw.data import gt_box
from src.vaw.probe import official_evaluator

CONDS = {"frozen_full": "Frozen DINOv3 · Full 384", "frozen_tight": "Frozen · C0 Tight Crop 384",
         "frozen_expand": "Frozen · C0 Expanded Crop 384", "g1pool_full": "G1-pool · Full 384",
         "g1cross_full": "G1-cross (Encoder Only) · Full 384"}
CONTRASTS = [("frozen_tight", "frozen_full"), ("frozen_expand", "frozen_full"), ("frozen_expand", "frozen_tight"),
             ("g1pool_full", "frozen_full"), ("g1cross_full", "frozen_full"), ("g1cross_full", "g1pool_full")]
B, SEED = 1000, 0


def official(prob, Y, i2a, exclude=()):
    ev = official_evaluator()
    ev.exclude_atts = list(exclude)
    so, pc = ev.evaluate(prob, Y)
    ev2 = official_evaluator()
    ev2.exclude_atts = list(exclude)
    sok, pck = ev2.evaluate(prob, Y, threshold_type="topk")
    groups = ["all", "head", "medium", "tail"] + list(ev.attribute_parent_type.keys())
    out = dict(mAP=pc["all"]["ap"], mR15=pck["all"]["recall"], mA=pc["all"]["bacc"], F1_15=sok["all"]["f1"])
    out.update({f"mAP_{g}": pc[g]["ap"] for g in groups[1:]})
    per_class = np.array([ev.get_score_class(j).ap if i2a[j] not in set(exclude) else np.nan for j in range(len(i2a))])
    return {k: float(v) * 100 for k, v in out.items()}, per_class


class FastAP:
    """weighted per-class AP (sklearn average_precision_score rule) for many resamples of one score matrix."""

    def __init__(self, prob, Y, rows_img, cls_ok):
        segs, self.y, self.img, self.last = [], [], [], []
        self.cls = []
        for j in np.nonzero(cls_ok)[0]:
            r = np.nonzero(Y[:, j] < 2)[0]
            s = prob[r, j]
            o = np.argsort(-s, kind="mergesort")
            r, s = r[o], s[o]
            last = np.r_[s[1:] != s[:-1], True]               # last element of each run of tied scores
            self.y.append((Y[r, j] == 1).astype(np.float64))
            self.img.append(rows_img[r])
            self.last.append(last)
            segs.append(len(r))
            self.cls.append(j)
        self.seg = np.repeat(np.arange(len(segs)), segs)
        self.start = np.r_[0, np.cumsum(segs)[:-1]]
        self.y, self.img, self.last = np.concatenate(self.y), np.concatenate(self.img), np.concatenate(self.last)
        self.n_cls = len(segs)

    def ap(self, wimg):
        """wimg: per-image weights -> per-class AP (nan where no pos or no neg)."""
        w = wimg[self.img]
        wp, wn = w * self.y, w * (1 - self.y)
        ctp, cfp = np.cumsum(wp), np.cumsum(wn)
        base_tp = np.r_[0, ctp][self.start][self.seg]
        base_fp = np.r_[0, cfp][self.start][self.seg]
        tp, fp = ctp - base_tp, cfp - base_fp
        P = np.bincount(self.seg, wp, self.n_cls)
        N = np.bincount(self.seg, wn, self.n_cls)
        L = self.last
        tpL, fpL, sL = tp[L], fp[L], self.seg[L]
        prev = np.r_[0.0, tpL[:-1]]
        first = np.r_[True, sL[1:] != sL[:-1]]
        prev[first] = 0.0
        ps = tpL + fpL
        prec = np.divide(tpL, ps, out=np.zeros_like(tpL), where=ps > 0)
        ap = np.bincount(sL, (tpL - prev) * prec, self.n_cls) / np.where(P > 0, P, 1)
        ap[(P <= 0) | (N <= 0)] = np.nan
        return ap


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--run", required=True)
    ap_.add_argument("--B", type=int, default=B)
    a = ap_.parse_args()
    a2i, i2a = attr_index()
    X = load_split("test")
    Y = label_matrix(X, a2i)
    sizes = image_sizes()
    leak = pd.read_parquet(os.path.join(a.run, "leak", "instances.parquet"))
    lt = leak[leak.split == "test"].reset_index(drop=True)
    assert np.array_equal(lt.instance_id.values, np.array([int(x["instance_id"]) for x in X]))
    sub = json.load(open(os.path.join(a.run, "leak", "subsets.json")))      # written by src.vaw.subsets
    img = np.array([int(x["image_id"]) for x in X])
    area = np.array([(lambda b: (b[2] - b[0]) * (b[3] - b[1]))(gt_box(x["instance_bbox"], *sizes[int(x["image_id"])])) for x in X])
    masks = {"all": np.ones(len(X), bool),
             "leakfree": ~np.isin(img, sub["g1_exposed_test_images"]),
             "nodup": ~np.isin(img, sub["cross_split_dup_test_images"]),
             "size_S": area < 32 ** 2, "size_M": (area >= 32 ** 2) & (area < 96 ** 2), "size_L": area >= 96 ** 2}
    conds = [c for c in CONDS if os.path.exists(os.path.join(a.run, c, "test_prob.npy"))]
    prob = {c: np.load(os.path.join(a.run, c, "test_prob.npy")) for c in conds}
    sel = {c: json.load(open(os.path.join(a.run, c, "selection.json"))) for c in conds}
    res = dict(conditions={c: CONDS[c] for c in conds}, selection=sel, subsets={}, contrasts={})
    _, img_idx = np.unique(img, return_inverse=True)
    rng = np.random.default_rng(SEED)
    for sname, m in masks.items():
        Ys = Y[m]
        excl = [i2a[j] for j in range(620) if not ((Ys[:, j] == 1).any() and (Ys[:, j] == 0).any())]
        ok = np.array([i2a[j] not in set(excl) for j in range(620)])
        uimg, rimg = np.unique(img_idx[m], return_inverse=True)
        pts, fast = {}, {}
        for c in conds:
            pts[c], pcap = official(prob[c][m], Ys, i2a, excl)
            fast[c] = FastAP(prob[c][m], Ys, rimg, ok)
            chk = fast[c].ap(np.ones(len(uimg)))
            assert np.nanmax(np.abs(chk - pcap[fast[c].cls])) < 1e-9, (sname, c, "fast AP != official AP")
        res["subsets"][sname] = dict(n_instances=int(m.sum()), n_images=int(len(uimg)), n_attr_excluded=len(excl), points=pts)
        W = rng.multinomial(len(uimg), np.full(len(uimg), 1.0 / len(uimg)), size=a.B).astype(np.float64)
        hmt = {g: np.array([official_evaluator().get_attr_head_tail(i2a[j]) == g for j in fast[conds[0]].cls]) for g in ("head", "medium", "tail")}
        for x, y in CONTRASTS:
            if x not in conds or y not in conds:
                continue
            d = {k: [] for k in ("all", "head", "medium", "tail")}
            for b in range(a.B):
                ax, ay = fast[x].ap(W[b]), fast[y].ap(W[b])
                v = ~(np.isnan(ax) | np.isnan(ay))
                d["all"].append(100 * (ax[v].mean() - ay[v].mean()))
                for g, gm in hmt.items():
                    vv = v & gm
                    d[g].append(100 * (ax[vv].mean() - ay[vv].mean()) if vv.any() else np.nan)
            pk = {"all": "mAP", "head": "mAP_head", "medium": "mAP_medium", "tail": "mAP_tail"}
            res["contrasts"].setdefault(f"{x} - {y}", {})[sname] = {
                g: dict(delta=pts[x][pk[g]] - pts[y][pk[g]], lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)))
                for g, v in d.items()}
        print(sname, json.dumps({c: round(pts[c]["mAP"], 2) for c in conds}), flush=True)
    json.dump(res, open(os.path.join(a.run, "results.json"), "w"), indent=1)
    write_md(res, a.run)


def f2(v):
    return f"{v:.2f}"


def ci(c):
    sig = "*" if c["lo"] > 0 or c["hi"] < 0 else ""
    return f"{c['delta']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]{sig}"


def write_md(res, run):
    S, K = res["subsets"], res["contrasts"]
    conds = list(res["conditions"])
    L = ["# VAW results (auto-generated by src.vaw.report)", "",
         "Test scored once per condition with the val-selected epoch. `*` = 95% image-level paired bootstrap CI excludes 0 "
         "(B = 1000). Single probe seed 0: CIs do NOT include training-seed variance.", ""]
    L += ["## Selected epochs (official VAW val mAP)", "", "| condition | epoch | val mAP |", "|---|---:|---:|"]
    for c in conds:
        L.append(f"| {res['conditions'][c]} | {res['selection'][c]['selected_epoch']} | {100 * res['selection'][c]['val_mAP']:.2f} |")
    types = [k for k in S["all"]["points"][conds[0]] if k.startswith("mAP_") and k not in ("mAP_head", "mAP_medium", "mAP_tail")]
    for sname, s in S.items():
        L += ["", f"## Test subset `{sname}`: {s['n_images']:,} images / {s['n_instances']:,} instances, "
                  f"{s['n_attr_excluded']} attributes excluded (no pos or no neg)", "",
              "| condition | mAP | mR@15 | mA | F1@15 | Head | Medium | Tail | " + " | ".join(t[4:] for t in types) + " |",
              "|---|" + "---:|" * (7 + len(types))]
        for c in conds:
            p = s["points"][c]
            L.append(f"| {res['conditions'][c]} | " + " | ".join(f2(p[k]) for k in ["mAP", "mR15", "mA", "F1_15", "mAP_head", "mAP_medium", "mAP_tail"] + types) + " |")
    L += ["", "## Paired contrasts (Δ, 95% CI)", "", "| contrast | subset | ΔmAP | ΔHead | ΔMedium | ΔTail |", "|---|---|---|---|---|---|"]
    for k, v in K.items():
        for sname, c in v.items():
            L.append(f"| {k} | {sname} | {ci(c['all'])} | {ci(c['head'])} | {ci(c['medium'])} | {ci(c['tail'])} |")
    open(os.path.join(run, "results.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
