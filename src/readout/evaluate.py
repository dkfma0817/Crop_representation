"""CPU evaluation shared by D0 and the report: paired image-bootstrap M2 on PACO VAL (never test).

M2 = PACO category-macro AP exactly as eval_tables.evaluate (units = eval_mlr.build_units 'cat', COCO 101-point AP,
attr_means -> hier_groups); bootstrap row 0 = identity weights = the point estimate; rows 1..B = image bootstrap
(pom_common.image_bootstrap_weights, seed 0, same draws for every condition -> paired differences).
Groups: overall, object, part, color, pattern_marking, material, transparency (+ object/<type>, part/<type>).
Object 5-16 is NOT a group here: its official M2 on val is NaN (too few eligible units) and is never imputed.
"""
import os
import sys

import numpy as np

from src.readout.common import ET, LEVELS

EM = ET.EM
sys.path.insert(0, os.path.dirname(ET.__file__))
from aggregate import attr_means, flat_groups, hier_groups  # noqa: E402
from pom_common import ap_unit, image_bootstrap_weights  # noqa: E402

GROUPS = ["overall", "object", "part", "color", "pattern_marking", "material", "transparency"]


class Boot:
    def __init__(self, df, Y, Ytr_by, nb=1000, seed=0, mask=None):
        self.df, self.Y = df, Y
        m = np.ones(len(df), bool) if mask is None else mask
        U = EM.build_units(df, Y, Ytr_by, m)
        self.U, self.Uo = U["cat"], U["orig"]
        W = image_bootstrap_weights(df.image_id.values, nb, seed).astype(np.float32)
        self.W = np.concatenate([np.ones((1, len(df)), np.float32), W])
        self.keys = [k for k, _, _ in self.U]
        self.keys_o = [k for k, _, _ in self.Uo]

    def m0(self, S):
        """M0 = paco_lp pooled (level, attr) AP (sklearn-style), flat group means -> {group: (B+1,)}; [0] = ET.evaluate M0."""
        if not self.Uo:
            return None
        o = np.stack([ap_unit(S[r, a], self.Y[r, a] == 1, self.W[:, r], ("sk",))["sk"] for _, r, a in self.Uo], 1)
        return {g: 100 * v for g, v in flat_groups(o, self.keys_o).items()}

    def m2(self, S):
        """-> {group: (B+1,) array}; [0] = point estimate (equal to ET.evaluate M2)."""
        c = np.stack([ap_unit(S[r, a], self.Y[r, a] == 1, self.W[:, r], ("coco",))["coco"] for _, r, a in self.U], 1)
        a_co, la = attr_means(c, self.keys)
        return {g: 100 * v for g, v in hier_groups(a_co, la).items()}


def summarize(x, y=None):
    """x (B+1,) [minus y] -> dict(point, lo, hi) with percentile CI over rows 1..B."""
    d = x if y is None else x - y
    b = d[1:][np.isfinite(d[1:])]
    if not len(b) or not np.isfinite(d[0]):
        return dict(point=float(d[0]), lo=float("nan"), hi=float("nan"))
    return dict(point=float(d[0]), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))
