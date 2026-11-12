"""Full-224 vs Full-1024 (and EVA crop oracle @224) on PACO VAL only (CPU; no test is read).

Every model: R2 scores on PACO val at its dev-selected epoch (max PACO-dev M0 Overall over epochs 1..10, = test_eval rule),
taken from the probe's saved scores_val_ep<k>.npy. 1024 probes / oracle = output/2026-11-06/densecmp/probes (same code,
same split, same reader/protocol; only --side differs, checked by the config diff below). M2 primary, M0 reference,
size bins = manifest n_in (src.ivc.common.DBINS), paired image bootstrap B=1000 seed 0 on val (same draws for all models).

  python -m src.densecmp.report_val --run output/2026-11-06/densecmp224 --ref output/2026-11-06/densecmp
"""
import argparse
import csv
import json
import os

import numpy as np

import src.readout.common as RC
from src.densecmp.report import f2, fci, jl
from src.densecmp.test_eval import select_epoch
from src.ivc.common import DBINS, EXCL
from src.readout.evaluate import Boot, summarize

COLS = [("Overall", "all", "overall"), ("Object", "all", "object"), ("Part", "all", "part"), ("Obj D5-16", "D5-16", "object"),
        ("Part D1-4", "D1-4", "part"), ("Color", "all", "color"), ("Material", "all", "material"), ("Pattern", "all", "pattern_marking"),
        ("Obj D17+", "D17+", "object"), ("Part D5-16", "D5-16", "part"), ("Part D17+", "D17+", "part")]
LAB = {"eva": "EVA-CLIP", "clipself": "CLIPSelf", "declip": "DeCLIP"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    M = {}
    for e in ("eva", "clipself", "declip"):
        M[f"{e}224"] = (os.path.join(a.run, "probes", f"{e}_full224"), f"{LAB[e]} Full 224")
        M[f"{e}1024"] = (os.path.join(a.ref, "probes", f"{e}_full1024"), f"{LAB[e]} Full 1024")
    M["oracle"] = (os.path.join(a.ref, "probes", "eva_c0224"), "EVA-CLIP GT crop 224 (oracle)")
    data = RC.load_data(EXCL, splits=("train", "val"))
    Ytr_by = RC.ytr_by_level(data)
    dva, Yva = data["val"]
    nin = dva.n_in.values
    B = dict(all=Boot(dva, Yva, Ytr_by), **{nm: Boot(dva, Yva, Ytr_by, mask=(nin >= lo) & (nin <= hi)) for nm, lo, hi in DBINS})
    RES, META = {}, {}
    for k, (d, lab) in M.items():
        assert os.path.isfile(os.path.join(d, "done.json")), d
        ep, dm0 = select_epoch(d)
        S = np.load(os.path.join(d, "r2", f"scores_val_ep{ep}.npy"))
        assert S.shape == (len(dva), 59)
        RES[k] = {m: {b: getattr(bb, m)(S) for b, bb in B.items()} for m in ("m2", "m0")}
        cfg = json.load(open(os.path.join(d, "config.json")))
        META[k] = dict(label=lab, probe=d, epoch=ep, dev_M0=dm0, token_grid=cfg["token_grid"], side=cfg["args"]["side"], mode=cfg["args"]["mode"],
                       frozen=json.load(open(os.path.join(d, "frozen_check.json")))["unchanged"],
                       time_min=round(json.load(open(os.path.join(d, "done.json")))["time_s"] / 60),
                       dev_M0_curve=[round(x["dev"]["M0"]["overall"], 2) for x in jl(os.path.join(d, "r2", "curves_dev.jsonl")) if x["epoch"] >= 1])

    def val(k, m, b, g):
        r = RES[k][m][b]
        return None if r is None or g not in r else r[g]

    def diff(x, y, m, b, g):
        vx, vy = val(x, m, b, g), val(y, m, b, g)
        return None if vx is None or vy is None else summarize(vx, vy)

    # config diff: 224 vs 1024 of the same encoder (only args.side / out / token_grid may differ)
    cdiff = {}
    ign = {"elapsed", "time"}
    for e in ("eva", "clipself", "declip"):
        c2 = json.load(open(os.path.join(M[f"{e}224"][0], "config.json")))
        c1 = json.load(open(os.path.join(M[f"{e}1024"][0], "config.json")))
        keys = sorted(set(c1) | set(c2))
        dd = [k for k in keys if k != "args" and c1.get(k) != c2.get(k) and k not in ign]
        da = [k for k in sorted(set(c1["args"]) | set(c2["args"])) if c1["args"].get(k) != c2["args"].get(k)]
        cdiff[e] = dict(top_level=dd, args=da)
    L, rows = [], []
    L += ["## Summary (PACO val, R2, M2; dev-selected epoch)", "",
          "| Method | Full 1024 Val M2 | Full 224 Val M2 | Δ (224 − 1024) [95% CI] | Obj D5–16 (224) | Part D1–4 (224) |", "|---|---|---|---|---|---|"]
    for e in ("eva", "clipself", "declip"):
        a1, a2 = val(f"{e}1024", "m2", "all", "overall"), val(f"{e}224", "m2", "all", "overall")
        L.append(f"| {LAB[e]} | {f2(float(a1[0]))} | {f2(float(a2[0]))} | {fci(diff(f'{e}224', f'{e}1024', 'm2', 'all', 'overall'))} | "
                 f"{f2(float(val(f'{e}224', 'm2', 'D5-16', 'object')[0]))} | {f2(float(val(f'{e}224', 'm2', 'D1-4', 'part')[0]))} |")
    o = "oracle"
    L.append(f"| GT Crop 224 (EVA-CLIP, reused) | – (crop input) | {f2(float(val(o, 'm2', 'all', 'overall')[0]))} (crop 224) | – | "
             f"{f2(float(val(o, 'm2', 'D5-16', 'object')[0]))} | {f2(float(val(o, 'm2', 'D1-4', 'part')[0]))} |")
    L.append("")
    order = ["eva224", "clipself224", "declip224", "eva1024", "clipself1024", "declip1024", "oracle"]
    contrasts = [("clipself224", "eva224"), ("declip224", "eva224"), ("declip224", "clipself224"),
                 ("eva224", "eva1024"), ("clipself224", "clipself1024"), ("declip224", "declip1024"),
                 ("oracle", "eva224"), ("oracle", "clipself224"), ("oracle", "declip224"),
                 ("clipself1024", "eva1024"), ("declip1024", "eva1024")]
    for m in ("m2", "m0"):
        L += [f"## PACO val — {m.upper()}{' (primary)' if m == 'm2' else ' (reference)'}", "",
              "| Model | grid | ep (dev M0) | " + " | ".join(c[0] for c in COLS) + " |", "|---|---|---|" + "---|" * len(COLS)]
        for k in order:
            cells = []
            for cn, b, g in COLS:
                v = val(k, m, b, g)
                x = None if v is None else float(v[0])
                cells.append(f2(x))
                rows.append(dict(split="val", metric=m.upper(), kind="score", model=k, label=META[k]["label"], column=cn, value=x, lo="", hi=""))
            gr = META[k]["token_grid"]
            L.append(f"| {META[k]['label']} | {gr[0]}x{gr[1]} | {META[k]['epoch']} ({META[k]['dev_M0']:.2f}) | " + " | ".join(cells) + " |")
        L += ["", f"Paired image bootstrap on val ({m.upper()}, B=1000, seed 0; * = 95% CI excludes 0)", "",
              "| Contrast | " + " | ".join(c[0] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
        for x, y in contrasts:
            cells = []
            for cn, b, g in COLS:
                dd = diff(x, y, m, b, g)
                cells.append(fci(dd))
                rows.append(dict(split="val", metric=m.upper(), kind="diff", model=f"{x}-{y}", label=f"{META[x]['label']} - {META[y]['label']}",
                                 column=cn, value=None if dd is None else dd["point"], lo="" if dd is None else dd["lo"], hi="" if dd is None else dd["hi"]))
            L.append(f"| {META[x]['label']} − {META[y]['label']} | " + " | ".join(cells) + " |")
        L.append("")
    nu = {b: (len(bb.U), len(bb.Uo)) for b, bb in B.items()}
    L += ["Units on val (M2 category / M0): " + "; ".join(f"{b}: {v[0]} / {v[1]}" for b, v in nu.items()) + ". n/a = no eligible unit.", "",
          "## Probes", "", "| Model | probe dir | epoch (dev M0) | encoder unchanged | time (min) | dev M0 ep1..10 |", "|---|---|---|---|---|---|"]
    for k in order:
        mm = META[k]
        L.append(f"| {mm['label']} | `{mm['probe']}` | {mm['epoch']} ({mm['dev_M0']:.2f}) | {mm['frozen']} | {mm['time_min']} | {mm['dev_M0_curve']} |")
    L += ["", "Config diff 224 vs 1024 (same encoder; keys whose values differ): " + json.dumps(cdiff), ""]
    open(os.path.join(a.run, "final_tables.md"), "w").write("\n".join(L) + "\n")
    with open(os.path.join(a.run, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(meta=META, units=nu, config_diff=cdiff), open(os.path.join(a.run, "results.json"), "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
