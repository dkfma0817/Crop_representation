"""Official-protocol Dev AP, step 4 (CPU, env paco_eval): table + paired image bootstrap on the dev images + M0 comparison.
Per model: the probe's ALREADY SELECTED epoch (training-time rule: max dev M0) -> Dev M0, Official-protocol Dev Obj / Part
AP (+ types), official TEST Obj / Part AP (existing results, not re-run). Contrasts on dev: paired bootstrap B = 1000 (seed
0) over the 1,000 dev images, the official per-image matching re-accumulated (src.pacoattr.boot.Pack, identical rule).
Per-epoch table (reference only, no re-selection): dev M0 vs Dev AP per epoch and which epoch each rule would pick.

  PYTHONPATH=<paco>:<lvis-api>:. python -m src.paco_devap.compare --models clip_full=<test eval dir> ... --out <md>
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

import src.pacoattr.boot as BT
from paco.evaluation.paco_eval_api import PACO

RUN = "output/2026-11-07/paco_devap"
TYPES = ("color", "pattern_marking", "material", "transparency")
KEYS = ["obj", "part"] + [f"{lv}_{t}" for lv in ("obj", "part") for t in TYPES]
RULE = "selection score = (Dev Obj AP + Dev Part AP) / 2, max over epochs 1..N, ties -> earlier epoch"


def score(r):
    return (r["obj"] + r["part"]) / 2


def _one(b):
    w = BT.G["W"][b]
    return {c: BT.hier(pk.precision(w)) for c, pk in BT.G["packs"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=<official test eval dir or ->")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(RUN, "results_dev.md"))
    a = ap.parse_args()
    models = dict(m.split("=", 1) for m in a.models)
    rows, per_ep = {}, {}
    for n, tdir in models.items():
        d = os.path.join(RUN, "preds", n)
        eps = sorted(int(e[2:]) for e in os.listdir(d) if e.startswith("ep"))
        recs = {}
        for e in eps:
            f = os.path.join(d, f"ep{e}", "official_dev.json")
            if not os.path.exists(f):
                continue
            o, m = json.load(open(f)), json.load(open(os.path.join(d, f"ep{e}", "meta.json")))
            recs[e] = dict(obj=o["obj"], part=o["part"], **{f"obj_{t}": o["obj_types"][t] for t in TYPES}, **{f"part_{t}": o["part_types"][t] for t in TYPES},
                           m0=m["m0_repro"]["m0_training_curve"], m0_re=m["m0_repro"]["m0_recomputed"], sel=m["is_selected"],
                           pairs=o["evaluable_pairs"], attrs=o["evaluable_attrs_per_type"], dropped_lab=o["dropped_with_known_label"])
        per_ep[n] = recs
        sel = [e for e, r in recs.items() if r["sel"]][0]
        t = None
        if tdir != "-":
            h = json.load(open(os.path.join(tdir, "official_results.json")))["hierarchical"]
            t = dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"])
        rows[n] = dict(epoch=sel, **recs[sel], test=t)
    # paired bootstrap on the selected epochs
    BT.G["gt"] = PACO(os.path.join(RUN, "data", "paco_dev_gt.json"))
    BT.G["packs"] = {n: BT.Pack(os.path.join(RUN, "preds", n, f"ep{r['epoch']}", "boot_pack.pkl")) for n, r in rows.items()}
    pk0 = next(iter(BT.G["packs"].values()))
    for n, pk in BT.G["packs"].items():
        assert pk.img_ids == pk0.img_ids, "models evaluated on different dev images"
        assert np.array_equal(pk.precision(np.ones(pk.I)), pk.official), n
    rng = np.random.default_rng(0)
    BT.G["W"] = rng.multinomial(pk0.I, np.full(pk0.I, 1.0 / pk0.I), size=a.B).astype(float)
    with Pool(32) as p:
        R = p.map(_one, range(a.B), chunksize=4)
    names = list(rows)
    con = {}
    for i, x in enumerate(names):
        for y in names[:i]:
            for xx, yy in ((x, y),):
                d = {}
                for k in KEYS:
                    s = np.array([r[xx][k] - r[yy][k] for r in R])
                    d[k] = dict(delta=rows[xx][k] - rows[yy][k], lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)))
                con[f"{xx}-{yy}"] = d
    out = dict(protocol="Official-protocol Dev AP: official PACO attribute evaluator (unmodified) on the PACO train-internal 1,000-image dev "
                        "(GT box + category oracle, labelled-first submission order); not an official split",
               rule_for_future_selection=RULE, rows=rows, per_epoch=per_ep, contrasts=con, B=a.B)
    json.dump(out, open(a.out.replace(".md", ".json"), "w"), indent=1, default=float)
    f = lambda d: f"{d['delta']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]" + ("*" if d["lo"] > 0 or d["hi"] < 0 else "")  # noqa: E731
    L = ["# Official-protocol Dev AP (PACO train-internal dev, 1,000 images, 13,931 GT; NOT an official split)", "",
         "| Method | R2 epoch | Dev M0 | Dev Obj AP | Dev Part AP | Test Obj AP | Test Part AP |", "|---|---:|---:|---:|---:|---:|---:|"]
    for n, r in rows.items():
        t = r["test"] or {}
        L.append(f"| {n} | {r['epoch']} | {r['m0']:.2f} | {r['obj']:.2f} | {r['part']:.2f} | {t.get('obj', float('nan')):.2f} | {t.get('part', float('nan')):.2f} |")
    L += ["", "| Dev type AP | " + " | ".join(f"Obj {t}" for t in TYPES) + " | " + " | ".join(f"Part {t}" for t in TYPES) + " |", "|---|" + "---:|" * 8]
    for n, r in rows.items():
        L.append(f"| {n} | " + " | ".join(f"{r[f'{lv}_{t}']:.2f}" for lv in ("obj", "part") for t in TYPES) + " |")
    r0 = next(iter(rows.values()))
    L += ["", f"Evaluable pairs (>= 1 positive dev GT): object {r0['pairs']['obj']}, part {r0['pairs']['part']}; attributes per type: {r0['attrs']}; "
              f"labelled GT dropped by max_dets: {r0['dropped_lab']}", "",
          f"## Dev contrasts (paired image bootstrap over the 1,000 dev images, B = {a.B}, * = CI excludes 0) and Test direction", "",
          "| Contrast | Dev M0 Δ | Dev Obj Δ | Dev Part Δ | Test Obj Δ | Test Part Δ |", "|---|---:|---|---|---:|---:|"]
    for k, d in con.items():
        x, y = k.split("-", 1)
        tx, ty = rows[x]["test"], rows[y]["test"]
        tt = (f"{tx['obj'] - ty['obj']:+.2f} | {tx['part'] - ty['part']:+.2f}") if tx and ty else "– | –"
        L.append(f"| {x} − {y} | {rows[x]['m0'] - rows[y]['m0']:+.2f} | {f(d['obj'])} | {f(d['part'])} | {tt} |")
    L += ["", "## Per epoch (reference only; the probes keep their dev-M0-selected epoch)", "",
          "| Model | epoch: Dev M0 / Dev Obj / Dev Part | pick by M0 | pick by Dev AP rule |", "|---|---|---:|---:|"]
    for n, recs in per_ep.items():
        pm = max(recs, key=lambda e: (recs[e]["m0"], -e))
        pa = max(recs, key=lambda e: (score(recs[e]), -e))
        L.append(f"| {n} | " + "; ".join(f"{e}: {r['m0']:.2f}/{r['obj']:.2f}/{r['part']:.2f}" for e, r in recs.items()) + f" | {pm} | {pa} |")
    L += ["", f"Future selection rule (fixed): {RULE}."]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
