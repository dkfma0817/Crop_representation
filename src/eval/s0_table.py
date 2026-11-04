"""Simplified pilot table (user 2026-11-04 ~18:30 KST): every condition with R2 AttnReader and R0 Linear, reader seed 0 only (corrected ~18:35 KST), 12k decoders.
PACO M2 point estimates (eval_rae.point_metrics = eval_vr / eval_fu metric code) on test-primary (+ val-primary for reference),
best epoch = best val pooled mAP (existing rule). Bins: F336 D-bins primary, SigLIP2 5-16 and RAE-grid bins auxiliary.
Scale reference for noise: A's R2 seeds 0/1/2 spread (computed here from the existing A runs).
  python -m src.eval.s0_table -> results/s0_table.md / .csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]
import src.eval.paco as E  # noqa: E402

G = E.G
G["df"], G["Y"] = {}, {}
for sp in ("train", "val", "test"):
    G["df"][sp], G["Y"][sp] = E.ER.load_split(sp)
G["Ytr_by"] = {lv: G["Y"]["train"][(G["df"]["train"].level == lv).values] for lv in E.ER.LEVELS}
CONDS = [("full", "Full (direct GT-ROI read-out)"), ("bil", "Bilinear"), ("crop", "GT crop oracle"),
         ("A_12000", "A Feature-only"), ("B_12000", "B Pixel-only"), ("C_12000", "C Feature+Pixel")]
TAG = os.environ.get("S0_TAG", "")
if TAG == "cl1":                                   # user request 2026-11-04 ~20:20 KST: C without LPIPS (beta 0)
    CONDS.append(("C_L1_12000", "C_L1 Feature+L1 (no LPIPS)"))
COLS = [("overall", "overall"), ("object", "object"), ("part", "part"), ("D5-16 object", "Obj D5-16"), ("D17+ object", "Obj D17+"),
        ("D1-4 part", "Part D1-4"), ("color", "color"), ("material", "material"), ("pattern_marking", "pattern"),
        ("5-16 object", "Obj 5-16 (SigLIP2)"), ("R5-16 object", "Obj R5-16 (RAE)"), ("R1-4 part", "Part R1-4 (RAE)")]
rows = []
READERS = (("r2", "r2_lr0.0003_s0"), ("r0", "r0_lr0.0001_s0"))
for kind, rn in READERS:
    for c, lab in CONDS:
        d = os.path.join(E.PROBE, c, rn)
        if not os.path.exists(os.path.join(d, "best.pt")):
            print("missing", c, kind)
            continue
        ep, v, n = E.ER.best_of(d)
        for split, sub in (("test", "testp"), ("val", "valp")):
            f = os.path.join(d, f"scores_{split}_ep{ep}.npy")
            if not os.path.exists(f):
                continue
            m = E.point_metrics([np.load(f).astype(np.float32)], split)
            rows.append(dict(cond=c, reader=kind, label=lab, subset=sub, best_epoch=ep, val_pooled=v, epochs_logged=n, **{k: m[k] for k, _ in COLS}))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(E.RES, f"s0_table{'_' + TAG if TAG else ''}.csv"), index=False)
L = ["# RAEv2 pilot — reader seed 0 (R2 AttnReader and R0 Linear), 12k decoders, PACO M2 (auto: src/eval/s0_table.py)\n"]
for kind in ("r2", "r0"):
    for sub, title in (("testp", "test-primary"), ("valp", "val-primary (reference, noisy)")):
        T = D[(D.subset == sub) & (D.reader == kind)]
        if not len(T):
            continue
        L += [f"\n## {kind.upper()} seed 0 — {title}\n", "| condition | best ep | " + " | ".join(n for _, n in COLS) + " |", "|---" * (len(COLS) + 2) + "|"]
        for r in T.itertuples():
            L.append(f"| {r.label} | {r.best_epoch} | " + " | ".join(f"{D.loc[r.Index, k]:.2f}" for k, _ in COLS) + " |")
        if sub == "testp":
            g = T.set_index("cond")
            L += [f"\n### {kind.upper()} differences (seed 0, test-primary, point estimates)\n", "| comparison | " + " | ".join(n for _, n in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
            for a, b in (("C_12000", "A_12000"), ("B_12000", "A_12000"), ("C_12000", "B_12000"), ("A_12000", "full"), ("B_12000", "full"), ("C_12000", "full"),
                         ("A_12000", "bil"), ("B_12000", "bil"), ("C_12000", "bil"), ("crop", "full"), ("bil", "full"),
                         ("C_L1_12000", "C_12000"), ("C_L1_12000", "A_12000"), ("C_L1_12000", "full")):
                if a in g.index and b in g.index:
                    L.append(f"| {a.rsplit('_', 1)[0]} − {b.rsplit('_', 1)[0]} | " + " | ".join(f"{g.loc[a, k] - g.loc[b, k]:+.2f}" for k, _ in COLS) + " |")
# seed spread of A (noise scale, existing seed 1/2 runs kept for this only)
for kind, pat in (("r2", "r2_lr0.0003_s{}"), ("r0", "r0_lr0.0001_s{}")):
    sp = []
    for s_ in (0, 1, 2):
        d = os.path.join(E.PROBE, "A_12000", pat.format(s_))
        ep = E.ER.best_of(d)[0]
        sp.append(E.point_metrics([np.load(os.path.join(d, f"scores_test_ep{ep}.npy")).astype(np.float32)], "test"))
    L += [f"\n### Noise scale: A {kind.upper()} reader seeds 0/1/2 (test-primary), max − min\n", "| " + " | ".join(n for _, n in COLS) + " |", "|---" * len(COLS) + "|",
          "| " + " | ".join(f"{max(x[k] for x in sp) - min(x[k] for x in sp):.2f}" for k, _ in COLS) + " |"]
open(os.path.join(E.RES, f"s0_table{'_' + TAG if TAG else ''}.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
