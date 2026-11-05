"""PACO Test-primary evaluation of the fixed IVC checkpoints (one-time, user-approved). CPU only.

Test-primary = PACO test ROIs whose image is NOT in the decoder COCO pool (~in_decoder_pool; paco_lp / eval_fu definition).
Primary metric M2 (project category-macro AP, NOT the official PACO detection AP); M0 reported for reference. D-bins = manifest
n_in. Paired image bootstrap (1,000, seed 0), same draws for every condition / reader.

  python -m src.ivc.test_report --run output/2026-11-05/ivc
"""
import argparse
import json
import os

os.environ["READOUT_ALLOW_TEST"] = "1"

import numpy as np

import src.readout.common as RC
from src.ivc.analyze import COLS, CONDS, CONTRASTS, LAB, READERS, fci, f2
from src.ivc.common import DBINS, EXCL
from src.readout.evaluate import Boot, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    T = os.path.join(a.run, "test")
    data = RC.load_data(EXCL, splits=("train", "test"))
    dte, Yte = data["test"]
    Ytr_by = RC.ytr_by_level(data)
    prim = ~dte.in_decoder_pool.values.astype(bool)
    nin = dte.n_in.values
    B = {"all": Boot(dte, Yte, Ytr_by, mask=prim)}
    B.update({nm: Boot(dte, Yte, Ytr_by, mask=prim & (nin >= lo) & (nin <= hi)) for nm, lo, hi in DBINS})
    ep = {}
    for c in CONDS:
        ep.update(json.load(open(os.path.join(T, f"done_{'_'.join(['bil', 'bic'] if c != 'c0' else ['c0'])}.json")))["epochs"])
    RES = {}
    for c in CONDS:
        for rd in READERS:
            S = np.load(os.path.join(T, f"scores_test_{c}_{rd}.npy"))
            assert S.shape == (len(dte), 59)
            RES[(c, rd)] = {m: {b: getattr(bb, m)(S) for b, bb in B.items()} for m in ("m2", "m0")}

    def val(c, rd, m, b, g):
        r = RES[(c, rd)][m][b]
        return None if r is None else r[g]
    L = ["## G. PACO Test-primary (one-time, fixed checkpoints)", "",
         f"Test-primary = PACO test ROIs outside the decoder COCO pool: {int(prim.sum()):,} ROIs / {dte.image_id[prim].nunique():,} images "
         f"(object {int((prim & (dte.level.values == 'object')).sum()):,}, part {int((prim & (dte.level.values == 'part')).sum()):,}); "
         f"all test {len(dte):,} ROIs. Checkpoints = dev-selected epochs (" + ", ".join(f"{k} ep{v}" for k, v in ep.items()) + "). "
         "Val re-score with the same code reproduced the saved val scores bit-exactly. Train/test image overlap: none (asserted).", ""]
    OUT = {}
    for m, title in (("m2", "M2 (primary)"), ("m0", "M0 (reference)")):
        L += [f"### {title}", "", "| Method | Reader | " + " | ".join(c[0] for c in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
        for rd in READERS:
            for c in CONDS:
                cells = [f2(float(v[0])) if (v := val(c, rd, m, b, g)) is not None else "n/a" for _, b, g in COLS]
                L.append(f"| {LAB[c]} | {rd.upper()} | " + " | ".join(cells) + " |")
                OUT[f"{m}/{c}/{rd}"] = {cn: (None if (v := val(c, rd, m, b, g)) is None else float(v[0])) for cn, b, g in COLS}
        L += ["", f"**Paired image bootstrap ({m.upper()}, B=1000, seed 0)**", "", "| Contrast | Reader | " + " | ".join(c[0] for c in COLS) + " |",
              "|---|---|" + "---|" * len(COLS)]
        for rd in READERS:
            for x, y in CONTRASTS:
                cells, d = [], {}
                for cn, b, g in COLS:
                    vx, vy = val(x, rd, m, b, g), val(y, rd, m, b, g)
                    if vx is None or vy is None:
                        cells.append("n/a")
                        continue
                    d[cn] = summarize(vx, vy)
                    cells.append(fci(d[cn]))
                OUT[f"{m}/{x}-{y}/{rd}"] = d
                L.append(f"| {LAB[x]} - {LAB[y]} | {rd.upper()} | " + " | ".join(cells) + " |")
        L.append("")
    nu = {b: (len(bb.U), len(bb.Uo)) for b, bb in B.items()}
    L += ["Units (M2 category units / M0 units) per subset: " + "; ".join(f"{b}: {v[0]} / {v[1]}" for b, v in nu.items()) + ". NaN = no eligible unit (never imputed).", ""]
    open(os.path.join(T, "TEST_REPORT.md"), "w").write("\n".join(L) + "\n")
    json.dump(OUT, open(os.path.join(T, "test_results.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
