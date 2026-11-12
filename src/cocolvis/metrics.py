"""GT-region classification metrics (independent of AP).

top1 / top5 : instance-level accuracy (a GT region is correct if its label is the arg-max / in the top-5 of ALL K scores)
mAcc        : mean over classes with >= 1 evaluated region of the per-class top-1 accuracy (CLIPSelf `_macc` rule, top-1)
subsets     : the same metrics restricted to S / M / L regions (original GT box area) and, for LVIS, r / c / f classes.
"""
import numpy as np


def ranks_topk(scores, k=5):
    """scores (N, K) array -> (N, k) indices of the k largest (descending)."""
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(scores, idx, 1), axis=1, kind="stable")
    return np.take_along_axis(idx, o, 1)


def per_instance(top5, y):
    return top5[:, 0] == y, (top5 == y[:, None]).any(1)


def macc(c1, y, w=None, ncls=None):
    """per-class mean of top-1 correctness; w = per-instance weights (bootstrap); classes with 0 weight are skipped."""
    w = np.ones(len(y)) if w is None else w
    n = np.bincount(y, w, ncls)
    k = np.bincount(y, w * c1, ncls)
    v = n > 0
    return float((k[v] / n[v]).mean()) if v.any() else float("nan")


def summary(c1, c5, y, size, freq=None, ncls=None):
    out = dict(top1=float(c1.mean()), top5=float(c5.mean()), mAcc=macc(c1, y, ncls=ncls), n=int(len(y)),
               n_classes=int(len(np.unique(y))))
    for i, s in enumerate("SML"):
        m = size == i
        out[f"top1_{s}"] = float(c1[m].mean()) if m.any() else float("nan")
        out[f"mAcc_{s}"] = macc(c1[m], y[m], ncls=ncls) if m.any() else float("nan")
        out[f"n_{s}"], out[f"n_classes_{s}"] = int(m.sum()), int(len(np.unique(y[m])))
    if freq is not None:
        for f in "rcf":
            m = freq == f
            out[f"top1_{f}"] = float(c1[m].mean()) if m.any() else float("nan")
            out[f"mAcc_{f}"] = macc(c1[m], y[m], ncls=ncls) if m.any() else float("nan")
            out[f"n_{f}"], out[f"n_classes_{f}"] = int(m.sum()), int(len(np.unique(y[m])))
    return out
