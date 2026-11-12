"""PACO VAL comparison of IVC-protocol probes across backbones (CPU). Same rules as src.ivc.analyze:
epoch per (run, reader) = max PACO-dev M0 Overall over epochs 1..10; M0 = src.readout.evaluate.Boot (paco_lp pooled AP);
D-bins = manifest n_in (F336 24 x 24 grid) D1-4 / D5-16 / D17+; paired image bootstrap B = 1000, seed 0, same draws for
every run (all runs share the val manifest rows, asserted by score shape).

  python -m src.ivc.compare_bb --runs v2_bil=output/2026-11-05/ivc/probe_bil v3_bil=output/2026-11-06/ivc_v3/probe_bil \
      --contrasts v3_bil-v2_bil --out output/2026-11-06/ivc_v3/compare_bil.md
"""
import argparse
import json
import os

import numpy as np

import src.readout.common as RC
from src.ivc.common import DBINS, EXCL
from src.readout.evaluate import Boot, summarize

COLS = [("Overall", "all", "overall"), ("Object", "all", "object"), ("Part", "all", "part"),
        ("Obj D5-16", "D5-16", "object"), ("Part D1-4", "D1-4", "part"), ("Obj D17+", "D17+", "object"),
        ("Color", "all", "color"), ("Material", "all", "material"), ("Pattern", "all", "pattern_marking")]


def jl(p):
    return [json.loads(l) for l in open(p)]


def pick(d):
    dv = [x for x in jl(os.path.join(d, "curves_dev.jsonl")) if x["epoch"] >= 1]
    b = max(dv, key=lambda x: x["dev"]["M0"]["overall"])
    return b["epoch"], b["dev"]["M0"]["overall"], np.load(os.path.join(d, f"scores_val_ep{b['epoch']}.npy")).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=probe_dir")
    ap.add_argument("--contrasts", nargs="*", default=[], help="a-b")
    ap.add_argument("--readers", nargs="+", default=["r0", "r2"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    Ytr_by = RC.ytr_by_level(data)
    BB = {"all": Boot(dva, Yva, Ytr_by), **{nm: Boot(dva, Yva, Ytr_by, mask=(dva.n_in.values >= lo) & (dva.n_in.values <= hi))
                                            for nm, lo, hi in DBINS}}
    runs = dict(r.split("=", 1) for r in a.runs)
    M, L, OUT = {}, [], {"runs": runs, "epochs": {}, "m0": {}, "contrasts": {}}
    hdr = "| run | reader | ep (dev M0) | " + " | ".join(c for c, _, _ in COLS) + " |"
    L += ["**PACO val M0 (dev-selected epoch)**", "", hdr, "|" + "---|" * (len(COLS) + 3)]
    for rd in a.readers:
        for nm, d in runs.items():
            ep, dm0, S = pick(os.path.join(d, rd))
            assert S.shape == (len(dva), 59), (nm, S.shape)
            M[(nm, rd)] = {b: BB[b].m0(S) for b in BB}
            pts = [float(M[(nm, rd)][b][g][0]) for _, b, g in COLS]
            OUT["epochs"][f"{nm}/{rd}"] = dict(epoch=ep, dev_M0=dm0)
            OUT["m0"][f"{nm}/{rd}"] = dict(zip([c for c, _, _ in COLS], pts))
            L.append(f"| {nm} | {rd.upper()} | {ep} ({dm0:.2f}) | " + " | ".join(f"{v:.2f}" for v in pts) + " |")
    if a.contrasts:
        L += ["", "**Paired image bootstrap 95% CI (M0, B = 1000, seed 0)**", "",
              "| contrast | reader | " + " | ".join(c for c, _, _ in COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
        for rd in a.readers:
            for c in a.contrasts:
                x, y = c.split("-")
                cells, rec = [], {}
                for col, b, g in COLS:
                    s = summarize(M[(x, rd)][b][g], M[(y, rd)][b][g])
                    rec[col] = s
                    cells.append(f"{s['point']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]")
                OUT["contrasts"][f"{c}/{rd}"] = rec
                L.append(f"| {x} − {y} | {rd.upper()} | " + " | ".join(cells) + " |")
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(OUT, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
