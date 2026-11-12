"""GT-box ORACLE detection AP (secondary metric) with the OFFICIAL cocoapi / lvis-api, plus a weighted re-accumulation of
the official per-image matching results for the image-level paired bootstrap.

GT-box oracle AP != detection AP: the candidate regions are exactly the GT boxes (perfect localisation, no missed
objects, one candidate per object), so AP here measures how well the class scores RANK the GT regions per category
across the val set. It is not comparable with any detector AP.

Detections (RegionCLIP GT-box zero-shot evaluation rule, cfg CROP_REGION_TYPE=GT / NO_BOX_DELTA / MULTIPLY_RPN_SCORE False,
detectron2 fast_rcnn_inference_single_image; identical for Full and Crop):
  every GT box (clipped to the image, original pixels) x every class with softmax score > thr
  (COCO 0.001 = RegionCLIP COCO config, LVIS 0.0001 = RegionCLIP LVIS GT script) -> class-wise NMS IoU 0.5 inside the image
  -> top max_det per image by score (COCO 100, LVIS 300). The GT category is never used to build or filter detections.
  Differences from RegionCLIP: softmax over the K classes only (our trained linear head has no background column),
  COCO uses all 80 classes (RegionCLIP COCO zero-shot GT eval uses the 65-class OVD split), trained head (not text).
Size bins: the official evaluators bin by the GT 'area' field (= SEGMENTATION area); variant `box` replaces the GT
'area' by the GT box area w*h (the bin rule used for all classification metrics). Both are reported.
"""
import copy
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from torchvision.ops import batched_nms

from src.cocolvis.common import ANN, LVIS_API

sys.path.insert(0, LVIS_API)
THR = {"coco": 0.001, "lvis": 0.0001}
MAXDET = {"coco": 100, "lvis": 300}
NMS = 0.5


def _dets_one(args):
    img, boxes, prob, cat_ids, thr, maxdet = args
    r, c = np.nonzero(prob > thr)
    if len(r) == 0:
        return []
    s = prob[r, c]
    b = torch.as_tensor(boxes[r], dtype=torch.float32)
    keep = batched_nms(b, torch.as_tensor(s, dtype=torch.float32), torch.as_tensor(c), NMS).numpy()[:maxdet]
    out = []
    for k in keep:
        x0, y0, x1, y1 = boxes[r[k]]
        out.append(dict(image_id=int(img), category_id=int(cat_ids[c[k]]), bbox=[float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        score=float(s[k])))
    return out


def make_dets(ds, prob, V, cat_ids, procs=32):
    """prob (N, K) softmax rows aligned with V (instances_{ds}_val.parquet). Boxes clipped to the image."""
    x0, y0 = V.x.clip(lower=0).values, V.y.clip(lower=0).values
    x1, y1 = np.minimum(V.x + V.w, V.W).values, np.minimum(V.y + V.h, V.H).values
    B = np.stack([x0, y0, x1, y1], 1)
    jobs = []
    for img, idx in V.groupby("image_id", sort=True).indices.items():
        jobs.append((img, B[idx], prob[idx], cat_ids, THR[ds], MAXDET[ds]))
    with Pool(procs) as p:
        R = p.map(_dets_one, jobs, chunksize=64)
    return [d for r in R for d in r]


def gt_dict(ds, area="seg"):
    J = json.load(open(ANN[(ds, "val")]))
    if area == "box":
        for a in J["annotations"]:
            a["area"] = float(a["bbox"][2] * a["bbox"][3])
    return J


def run_official(ds, dets, J, img_ids=None):
    """-> (stats dict, evaluator object). J = GT dict (official, or with box areas). img_ids: smoke tests only."""
    if ds == "coco":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        gt = COCO()
        gt.dataset = copy.deepcopy(J)
        gt.createIndex()
        dt = gt.loadRes(copy.deepcopy(dets)) if dets else COCO()
        E = COCOeval(gt, dt, "bbox")
        if img_ids is not None:
            E.params.imgIds = sorted(img_ids)
        E.evaluate()
        E.accumulate()
        E.summarize()
        st = dict(zip(["AP", "AP50", "AP75", "APs", "APm", "APl"], [float(v) for v in E.stats[:6]]))
        return st, E
    from lvis import LVIS, LVISEval, LVISResults
    gt = LVIS.__new__(LVIS)
    import logging
    gt.logger = logging.getLogger("lvis")
    gt.dataset = copy.deepcopy(J)
    gt._create_index()
    dt = LVISResults(gt, copy.deepcopy(dets), max_dets=MAXDET[ds])
    E = LVISEval(gt, dt, "bbox")
    if img_ids is not None:
        E.params.img_ids = sorted(img_ids)
    E.run()
    E.print_results()
    st = {k: float(E.results[k]) for k in ("AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf")}
    return st, E


class FastDetAP:
    """Weighted re-accumulation of an official evaluator's per-(category, area, image) matching (no re-matching: matching
    is within an image, so an image-level bootstrap only re-weights images). With all weights 1 it must reproduce the
    official per-category precision exactly (checked in `verify`). Areas: 0 all, 1 small, 2 medium, 3 large."""

    def __init__(self, E, kind):
        self.kind = kind
        if kind == "coco":
            p = E.params
            K, A, I = len(p.catIds), len(p.areaRng), len(p.imgIds)
            ev, md = E.evalImgs, p.maxDets[-1]
            self.rec = p.recThrs
            self.T = len(p.iouThrs)
            get = lambda e: (e["dtScores"][:md], e["dtMatches"][:, :md], e["dtIgnore"][:, :md], e["gtIgnore"])  # noqa: E731
        else:
            p = E.params
            K, A, I = len(p.cat_ids), len(p.area_rng), len(p.img_ids)
            ev = E.eval_imgs
            self.rec = p.rec_thrs
            self.T = len(p.iou_thrs)
            get = lambda e: (e["dt_scores"], e["dt_matches"], e["dt_ignore"], e["gt_ignore"])  # noqa: E731
            self.freq_groups = E.freq_groups
        self.K, self.A, self.I = K, A, I
        self.seg = {}
        for a in range(A):
            S, TP, FP, IMG, seg_id, GIMG, gseg, valid = [], [], [], [], [], [], [], []
            n = 0
            for k in range(K):
                Es = [(i, ev[k * A * I + a * I + i]) for i in range(I)]
                Es = [(i, e) for i, e in Es if e is not None]
                if not Es:
                    continue
                parts = [get(e) for _, e in Es]
                sc = np.concatenate([np.asarray(q[0], dtype=np.float64) for q in parts]) if parts else np.zeros(0)
                im = np.concatenate([np.full(len(q[0]), i) for (i, _), q in zip(Es, parts)])
                dtm = np.concatenate([np.asarray(q[1]).reshape(self.T, -1) for q in parts], axis=1)
                dti = np.concatenate([np.asarray(q[2]).reshape(self.T, -1) for q in parts], axis=1)
                gti = np.concatenate([np.asarray(q[3]).reshape(-1) for q in parts])
                gim = np.concatenate([np.full(len(np.asarray(q[3]).reshape(-1)), i) for (i, _), q in zip(Es, parts)])
                if np.count_nonzero(gti == 0) == 0:
                    continue
                o = np.argsort(-sc, kind="mergesort")
                S.append(sc[o]), IMG.append(im[o])
                TP.append(np.logical_and(dtm[:, o] != 0, ~(dti[:, o] != 0)))
                FP.append(np.logical_and(dtm[:, o] == 0, ~(dti[:, o] != 0)))
                seg_id.append(np.full(len(o), n))
                GIMG.append(gim[gti == 0]), gseg.append(np.full(int((gti == 0).sum()), n))
                valid.append(k)
                n += 1
            self.seg[a] = dict(img=np.concatenate(IMG).astype(np.int64), tp=np.concatenate(TP, 1), fp=np.concatenate(FP, 1),
                               seg=np.concatenate(seg_id).astype(np.int64), gimg=np.concatenate(GIMG).astype(np.int64),
                               gseg=np.concatenate(gseg).astype(np.int64), cats=np.array(valid), n=n)

    def per_cat(self, w, a):
        """w: per-image weights (len I) -> (T, n_valid_cats) AP (mean over the 101 recall points), nan where npig == 0."""
        s = self.seg[a]
        n, seg = s["n"], s["seg"]
        npig = np.bincount(s["gseg"], w[s["gimg"]], n)
        wd = w[s["img"]]
        out = np.full((self.T, n), np.nan)
        start = np.r_[0, np.cumsum(np.bincount(seg, minlength=n))[:-1]]
        cnt = np.bincount(seg, minlength=n)
        R = len(self.rec)
        for t in range(self.T):
            ctp, cfp = np.cumsum(s["tp"][t] * wd), np.cumsum(s["fp"][t] * wd)
            btp = np.r_[0.0, ctp][start][seg]
            bfp = np.r_[0.0, cfp][start][seg]
            tp, fp = ctp - btp, cfp - bfp
            rc = tp / np.where(npig[seg] > 0, npig[seg], 1)
            pr = tp / (fp + tp + np.spacing(1))
            # precision envelope (max to the right) inside every segment: reverse cummax of pr - 2*seg (pr in [0, 1], so
            # values of a later segment, seen first in the reversed scan, can never exceed those of an earlier one)
            off = 2.0 * seg
            env = np.maximum.accumulate((pr - off)[::-1])[::-1] + off
            q = np.zeros((n, R))
            for j in range(n):                      # exact per-segment searchsorted (official: side='left', miss -> 0)
                if cnt[j] == 0:
                    continue
                lo = start[j]
                idx = np.searchsorted(rc[lo:lo + cnt[j]], self.rec, side="left")
                ok = idx < cnt[j]
                q[j, ok] = env[lo + idx[ok]]
            ap = q.mean(1)
            ap[npig <= 0] = np.nan
            out[t] = ap
        return out

    def summary(self, w):
        """-> dict AP, AP50, APs, APm, APl (+ APr, APc, APf for LVIS): official definitions (mean over valid categories)."""
        res = {}
        for a, name in enumerate(["AP", "APs", "APm", "APl"]):
            pc = self.per_cat(w, a)
            res[name] = float(np.nanmean(pc))
            if a == 0:
                res["AP50"] = float(np.nanmean(pc[0]))
                if self.kind == "lvis":
                    cats = self.seg[0]["cats"]
                    for g, nm in zip(range(3), ("APr", "APc", "APf")):
                        m = np.isin(cats, self.freq_groups[g])
                        res[nm] = float(np.nanmean(pc[:, m])) if m.any() else float("nan")
        return res

    def verify(self, official):
        r = self.summary(np.ones(self.I))
        for k, v in r.items():
            assert abs(v - official[k]) < 1e-9, (k, v, official[k])
        return r
