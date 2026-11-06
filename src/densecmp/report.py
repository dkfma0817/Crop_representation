"""Tables for the EVA-CLIP / CLIPSelf / DeCLIP PACO comparison (CPU). Writes final_tables.md, results.csv, results.json.

Test-primary = PACO test ROIs not in the decoder COCO pool (src.ivc.test_report definition). M2 primary (project
category-macro AP, src.readout.evaluate.Boot = eval_tables.evaluate), M0 reference. Size bins = manifest n_in on the DINOv2
24x24 grid (src.ivc.common.DBINS; not redefined for the EVA grid). Paired image bootstrap B=1000 seed 0, same draws for every
model. Also: PACO val at the dev-selected epoch; exposure sensitivity = test-primary images from COCO val2017 only.

  python -m src.densecmp.report --run output/2026-11-06/densecmp
"""
import argparse
import csv
import json
import os

os.environ["READOUT_ALLOW_TEST"] = "1"

import numpy as np

import src.readout.common as RC
from src.ivc.common import DBINS, EXCL
from src.readout.evaluate import Boot, summarize

MODELS = [  # key, probe dir, label, group
    ("eva1024", "eva_full1024", "EVA-CLIP", "primary"),
    ("clipself1024", "clipself_full1024", "CLIPSelf", "primary"),
    ("declip1024", "declip_full1024", "DeCLIP", "primary"),
    ("evac0", "eva_c0224", "EVA-CLIP GT crop @224 (oracle)", "oracle"),
    ("eva560", "eva_full560", "EVA-CLIP @560", "secondary"),
    ("clipself560", "clipself_full560", "CLIPSelf @560", "secondary"),
    ("declip560", "declip_full560", "DeCLIP @560", "secondary"),
]
CONTRASTS = [("clipself1024", "eva1024"), ("declip1024", "eva1024"), ("declip1024", "clipself1024"),
             ("evac0", "eva1024"), ("evac0", "clipself1024"), ("evac0", "declip1024"),
             ("clipself560", "eva560"), ("declip560", "eva560"), ("declip560", "clipself560"),
             ("evac0", "eva560"), ("evac0", "clipself560"), ("evac0", "declip560")]
COLS = [("Overall", "all", "overall"), ("Obj 5-16", "D5-16", "object"), ("Part 1-4", "D1-4", "part"), ("Color", "all", "color"),
        ("Material", "all", "material"), ("Pattern", "all", "pattern_marking"), ("Object", "all", "object"), ("Part", "all", "part"),
        ("Obj 17+", "D17+", "object"), ("Part 5-16", "D5-16", "part"), ("Part 17+", "D17+", "part")]
DINO_REF = "experiments/ivc/TEST_REPORT.md"   # reference only (different encoder / reader input), never mixed


def f2(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}"


def fci(d):
    if d is None or not np.isfinite(d["point"]):
        return "n/a"
    s = "*" if (d["lo"] > 0 or d["hi"] < 0) else ""
    return f"{d['point']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]{s}"


def jl(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    P = os.path.join(a.run, "probes")
    data = RC.load_data(EXCL, splits=("train", "val", "test"))
    Ytr_by = RC.ytr_by_level(data)
    dte, Yte = data["test"]
    dva, Yva = data["val"]
    prim = ~dte.in_decoder_pool.values.astype(bool)
    v17 = ~dte.path.str.contains("train2017").values
    nin, ninv = dte.n_in.values, dva.n_in.values
    have = [m for m in MODELS if os.path.isfile(os.path.join(P, m[1], "scores_test.npy"))]
    print("models with test scores:", [m[0] for m in have])
    subsets = {"test": dict(all=Boot(dte, Yte, Ytr_by, mask=prim), **{nm: Boot(dte, Yte, Ytr_by, mask=prim & (nin >= lo) & (nin <= hi)) for nm, lo, hi in DBINS}),
               "test_val2017": dict(all=Boot(dte, Yte, Ytr_by, mask=prim & v17),
                                    **{nm: Boot(dte, Yte, Ytr_by, mask=prim & v17 & (nin >= lo) & (nin <= hi)) for nm, lo, hi in DBINS}),
               "val": dict(all=Boot(dva, Yva, Ytr_by), **{nm: Boot(dva, Yva, Ytr_by, mask=(ninv >= lo) & (ninv <= hi)) for nm, lo, hi in DBINS})}
    RES, META = {}, {}
    for key, d, lab, grp in have:
        td = json.load(open(os.path.join(P, d, "test_done.json")))
        rc = json.load(open(os.path.join(P, d, "val_recheck.json")))
        cfg = json.load(open(os.path.join(P, d, "config.json")))
        fit = json.load(open(os.path.join(P, d, "train_fit.json")))
        dv = jl(os.path.join(P, d, "r2", "curves_dev.jsonl"))
        cv = jl(os.path.join(P, d, "r2", "curves.jsonl"))
        ep = td["epoch"]
        St = np.load(os.path.join(P, d, "scores_test.npy"))
        Sv = np.load(os.path.join(P, d, "r2", f"scores_val_ep{ep}.npy"))
        assert St.shape == (len(dte), 59) and Sv.shape == (len(dva), 59)
        RES[key] = {sp: {m: {b: getattr(bb, m)(St if sp != "val" else Sv) for b, bb in subsets[sp].items()} for m in ("m2", "m0")} for sp in subsets}
        META[key] = dict(label=lab, group=grp, probe=d, epoch=ep, dev_M0=td["dev_M0"], val_recheck_max_abs=rc["max_abs_diff"],
                         token_grid=cfg["token_grid"], side=cfg["args"]["side"], mode=cfg["args"]["mode"],
                         val_best_epoch_by_val_M2=max((x for x in cv if x["epoch"] >= 1), key=lambda x: x["val"]["M2"]["overall"])["epoch"],
                         dev_M0_curve=[round(x["dev"]["M0"]["overall"], 2) for x in dv if x["epoch"] >= 1],
                         val_M2_curve=[round(x["val"]["M2"]["overall"], 2) for x in cv if x["epoch"] >= 1],
                         train_subset_M2=fit["train_subset_M2"]["overall"], time_s=json.load(open(os.path.join(P, d, "done.json")))["time_s"])

    def val(k, sp, m, b, g):
        r = RES[k][sp][m][b]
        return None if r is None or g not in r else r[g]

    L, rows = [], []
    nu = {sp: {b: (len(bb.U), len(bb.Uo)) for b, bb in s.items()} for sp, s in subsets.items()}
    for sp, title in (("test", "PACO test-primary"), ("test_val2017", "PACO test-primary, COCO val2017 images only (exposure sensitivity)"),
                      ("val", "PACO val (dev-selected epoch)")):
        for m in ("m2", "m0"):
            L += [f"## {title} — {m.upper()}{' (primary metric)' if m == 'm2' else ' (reference)'}", "",
                  "| Model | grid | ep | " + " | ".join(c[0] for c in COLS) + " |", "|---|---|---|" + "---|" * len(COLS)]
            for key, _, lab, grp in have:
                cells = []
                for cn, b, g in COLS:
                    v = val(key, sp, m, b, g)
                    x = None if v is None else float(v[0])
                    cells.append(f2(x))
                    rows.append(dict(split=sp, metric=m.upper(), kind="score", model=key, label=lab, group=grp, column=cn, value=x, lo="", hi=""))
                g_ = META[key]["token_grid"]
                L.append(f"| {lab} | {g_[0]}x{g_[1]} | {META[key]['epoch']} | " + " | ".join(cells) + " |")
            L += ["", f"Paired image bootstrap ({m.upper()}, B=1000, seed 0; * = 95% CI excludes 0)", "",
                  "| Contrast | " + " | ".join(c[0] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
            for x, y in CONTRASTS:
                if x not in RES or y not in RES:
                    continue
                cells = []
                for cn, b, g in COLS:
                    vx, vy = val(x, sp, m, b, g), val(y, sp, m, b, g)
                    d = None if vx is None or vy is None else summarize(vx, vy)
                    cells.append(fci(d))
                    rows.append(dict(split=sp, metric=m.upper(), kind="diff", model=f"{x}-{y}", label=f"{META[x]['label']} - {META[y]['label']}",
                                     group="contrast", column=cn, value=None if d is None else d["point"], lo="" if d is None else d["lo"], hi="" if d is None else d["hi"]))
                L.append(f"| {META[x]['label']} − {META[y]['label']} | " + " | ".join(cells) + " |")
            L.append("")
        L += [f"Units (M2 category units / M0 units): " + "; ".join(f"{b}: {v[0]} / {v[1]}" for b, v in nu[sp].items()) + ". n/a = no eligible unit (never imputed).", ""]
    L += ["## Checkpoint selection / sanity", "", "| Model | probe | epoch (dev M0) | val-best epoch (val M2, reference only) | val re-score max |Δ| | train-subset M2 | time (min) | dev M0 ep1..10 |",
          "|---|---|---|---|---|---|---|---|"]
    for key, d, lab, _ in have:
        mm = META[key]
        L.append(f"| {lab} | {d} | {mm['epoch']} ({mm['dev_M0']:.2f}) | {mm['val_best_epoch_by_val_M2']} | {mm['val_recheck_max_abs']:.4f} | "
                 f"{mm['train_subset_M2']:.2f} | {mm['time_s'] / 60:.0f} | {mm['dev_M0_curve']} |")
    L += ["", f"Test-primary: {int(prim.sum()):,} ROIs / {dte.image_id[prim].nunique():,} images (object {int((prim & (dte.level.values == 'object')).sum()):,}, "
          f"part {int((prim & (dte.level.values == 'part')).sum()):,}); COCO-val2017 subset {int((prim & v17).sum()):,} ROIs / {dte.image_id[prim & v17].nunique():,} images.", ""]
    os.makedirs(a.run, exist_ok=True)
    open(os.path.join(a.run, "final_tables.md"), "w").write("\n".join(L) + "\n")
    with open(os.path.join(a.run, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(meta=META, units=nu), open(os.path.join(a.run, "results.json"), "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
