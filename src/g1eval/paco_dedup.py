"""PACO val leakage check for the G1 encoders + R2 re-evaluation on the de-duplicated val set (CPU, no re-training).

Training images of BOTH G1 runs (identical): the existing COCO 12k stream rows 0..11999 (16 train2017 ids each) minus the
2 images excluded at training time (config.json 'excluded.stream') -> 50,489 images.
Duplicate search vs PACO val (2,051 images):
  exact  : COCO image_id, file byte md5, decoded-pixel md5, dHash (data_check image_keys.parquet)
  near   : dHash Hamming <= 4 and grayscale 32x32 thumbnail correlation (incl. horizontal flip) over ALL val x train
           pairs; every candidate with corr >= 0.85 (67) or Hamming <= 4 (11) was inspected visually; confirmed
           same-photo / same-session pairs are listed in NEAR_DUP (images removed WHOLE, with all their ROIs).
Scores: the saved per-ROI R2 val scores of the PACO probes (dev-selected epoch, = src.ivc.compare_bb.pick); AP recomputed
from scratch on the filtered rows with the unchanged rule (src.readout.evaluate.Boot: paco_lp pooled AP for M0,
COCO-style for M2, unit inclusion from the full PACO train labels, D-bins on manifest n_in), image bootstrap B = 1000.

  python -m src.g1eval.paco_dedup --out output/2026-11-07/g1_leakcheck_ade20k/paco
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import src.readout.common as RC
from src.config import STREAM
from src.ivc.common import DBINS, EXCL
from src.ivc.compare_bb import pick
from src.msd.pretrain import KEYS
from src.readout.evaluate import Boot, summarize

G1 = "output/2026-11-06/g1_cross_distill"
RUNS = {"Frozen": "probe_frozen_full", "G1-pool": "probe_G1pool_full", "G1-cross Encoder Only": "probe_G1cross_full"}
# visually confirmed (val_id, train_id, relation); evidence montages in <out>/evidence
NEAR_DUP = [(110596, 200149, "same photo, colour-processed variant (thumb corr 1.000, dHash 1)"),
            (110170, 393664, "same scene, adjacent burst frame (corr 0.972)"),
            (137586, 361382, "same location / photo session, different rider pose (corr 0.905)"),
            (3286, 68084, "same photo with an added frame / caption overlay (corr 0.886)")]
COLS = [("M0 Overall", "m0", "all", "overall"), ("Obj D5-16", "m0", "D5-16", "object"), ("Part D1-4", "m0", "D1-4", "part"),
        ("M2 Overall", "m2", "all", "overall"), ("Object", "m0", "all", "object"), ("Part", "m0", "all", "part")]


def train_ids():
    cfg = [json.load(open(os.path.join(G1, r, "config.json"))) for r in ("G1pool", "G1cross")]
    assert cfg[0]["steps"] == cfg[1]["steps"] and cfg[0]["excluded"]["stream"] == cfg[1]["excluded"]["stream"]
    st = np.load(STREAM)[: cfg[0]["steps"]]
    return set(int(i) for i in st.ravel()) - set(cfg[0]["excluded"]["stream"]), cfg[0]


def evaluate(dva, Yva, Ytr_by, scores, keep):
    d, Y = dva[keep].reset_index(drop=True), Yva[keep]
    BB = {"all": Boot(d, Y, Ytr_by), **{nm: Boot(d, Y, Ytr_by, mask=(d.n_in.values >= lo) & (d.n_in.values <= hi)) for nm, lo, hi in DBINS}}
    M = {m: {b: dict(m0=BB[b].m0(S[keep]), m2=BB[b].m2(S[keep])) for b in BB} for m, S in scores.items()}
    pts = {m: {c: float(M[m][b][k][g][0]) for c, k, b, g in COLS} for m in scores}
    con = {}
    for x, y in (("G1-pool", "Frozen"), ("G1-cross Encoder Only", "Frozen"), ("G1-cross Encoder Only", "G1-pool")):
        con[f"{x} - {y}"] = {c: summarize(M[x][b][k][g], M[y][b][k][g]) for c, k, b, g in COLS}
    return dict(n_images=int(d.image_id.nunique()), n_rois=int(len(d)), points=pts, contrasts=con)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr, cfg = train_ids()
    keys = pd.read_parquet(KEYS)
    s, v = keys[(keys.source == "raev2_stream") & keys.image_id.isin(tr)], keys[keys.source == "paco_val"]
    exact = {k: sorted(int(i) for i in v.image_id[v[k].isin(set(s[k]))]) for k in ("image_id", "md5", "pix_md5", "dhash")}
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    Ytr_by = RC.ytr_by_level(data)
    assert set(dva.image_id.unique()) == set(v.image_id), "manifest val ids != data_check paco_val ids"
    assert all(t in tr for _, t, _ in NEAR_DUP) and all(int(x) in set(dva.image_id) for x, _, _ in NEAR_DUP)
    drop_exact = set().union(*map(set, exact.values()))
    drop = drop_exact | {x for x, _, _ in NEAR_DUP}
    leak = dict(train_images=len(tr), train_rule=f"stream rows 0..{cfg['steps'] - 1} minus {cfg['excluded']['stream']} (identical for G1-pool / G1-cross)",
                paco_val_images=int(dva.image_id.nunique()), paco_val_rois=int(len(dva)), exact_overlap=exact,
                near_duplicates=[dict(val_id=x, train_id=t, relation=r, val_rois=int((dva.image_id == x).sum())) for x, t, r in NEAR_DUP],
                removed_images=sorted(drop), removed_rois=int(dva.image_id.isin(drop).sum()))
    leak["removed_image_pct"] = 100 * len(drop) / leak["paco_val_images"]
    leak["removed_roi_pct"] = 100 * leak["removed_rois"] / leak["paco_val_rois"]
    scores, epochs = {}, {}
    for m, d in RUNS.items():
        ep, dm0, S = pick(os.path.join(G1, d, "r2"))
        assert S.shape == (len(dva), 59)
        scores[m], epochs[m] = S, dict(epoch=ep, dev_M0=dm0)
    full = evaluate(dva, Yva, Ytr_by, scores, np.ones(len(dva), bool))
    dedup = evaluate(dva, Yva, Ytr_by, scores, ~dva.image_id.isin(drop).values)
    out = dict(leak=leak, epochs=epochs, full=full, dedup=dedup)
    json.dump(out, open(os.path.join(a.out, "paco_dedup.json"), "w"), indent=1)
    f = lambda s: f"{s['point']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]{'*' if s['lo'] > 0 or s['hi'] < 0 else ''}"  # noqa: E731
    L = ["## Leakage", "", f"- G1 training images: {leak['train_images']:,} ({leak['train_rule']})",
         f"- PACO val: {leak['paco_val_images']:,} images / {leak['paco_val_rois']:,} ROIs",
         f"- exact overlap (id / md5 / pixel md5 / dHash): " + ", ".join(f"{k} {len(v_)}" for k, v_ in exact.items()),
         f"- near duplicates (visually confirmed): {len(NEAR_DUP)} images / {sum(n['val_rois'] for n in leak['near_duplicates'])} ROIs",
         f"- removed: {len(drop)} images ({leak['removed_image_pct']:.2f}%) / {leak['removed_rois']} ROIs ({leak['removed_roi_pct']:.2f}%)", "",
         "## Table A (R2, dev-selected epoch; AP recomputed on the filtered rows)", "",
         "| Method | set | Val Images | Val ROIs | M0 Overall | Obj D5–16 | Part D1–4 | M2 Overall |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for nm, r in (("full", full), ("dedup", dedup)):
        for m in RUNS:
            p = r["points"][m]
            L.append(f"| {m} | {nm} | {r['n_images']:,} | {r['n_rois']:,} | {p['M0 Overall']:.2f} | {p['Obj D5-16']:.2f} | {p['Part D1-4']:.2f} | {p['M2 Overall']:.2f} |")
    L += ["", "## Paired image bootstrap 95% CI (B = 1000, seed 0)", "", "| set | contrast | M0 Overall | Obj D5–16 | Part D1–4 | M2 Overall |", "|---|---|---|---|---|---|"]
    for nm, r in (("full", full), ("dedup", dedup)):
        for k, c in r["contrasts"].items():
            L.append(f"| {nm} | {k.replace(' - ', ' − ')} | {f(c['M0 Overall'])} | {f(c['Obj D5-16'])} | {f(c['Part D1-4'])} | {f(c['M2 Overall'])} |")
    open(os.path.join(a.out, "paco_dedup.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
