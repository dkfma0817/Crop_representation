"""Report: PACO-LVIS GT-Box + Category Oracle Attribute AP (Official Evaluator). CPU; reads eval/ + boot/ outputs.

  python -m src.pacoattr.report --run output/2026-11-07/paco_oracle_attr
"""
import argparse
import json
import os

import numpy as np

CONDS = {"frozen_full": "Frozen Full", "g1pool_full": "G1-pool", "g1cross_full": "G1-cross Encoder Only",
         "g1cross_dec": "G1-cross + Decoder", "c0_crop": "C0 Crop Oracle"}
TYPES = ("color", "pattern_marking", "material", "transparency")
TNAME = {"color": "Color", "pattern_marking": "Pattern & Markings", "material": "Material", "transparency": "Transparency"}
CONTR = [("A", "g1pool_full", "frozen_full"), ("B", "g1cross_full", "frozen_full"), ("C", "g1cross_full", "g1pool_full"),
         ("D", "g1cross_dec", "g1cross_full"), ("E", "g1cross_dec", "c0_crop"), ("F", "g1cross_full", "c0_crop")]
OLD = {"frozen_full": "Frozen Full", "g1pool_full": "G1-pool", "g1cross_full": "G1-cross Encoder Only",
       "g1cross_dec": "G1-cross + Decoder", "c0_crop": "C0 Crop Oracle"}
KEYS = ["obj"] + [f"obj_{t}" for t in TYPES] + ["part"] + [f"part_{t}" for t in TYPES]


def load_eval(run, c, v):
    return json.load(open(os.path.join(run, "eval", c, v, "official_results.json")))


def flat(h):
    return dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"], **{f"obj_{t}": h["obj-attrs"][t] for t in TYPES},
                **{f"part_{t}": h["obj-part-attrs"][t] for t in TYPES})


def counts(ev, attr_type):
    """valid (category, attribute) pairs / attributes per level x type from the official per-pair AP (nan = no positive)."""
    out = {}
    for name, ap in ev["hierarchical"]["per-obj-part-attr-AP"].items():
        if ap != ap:
            continue
        *cat, attr = name.split(":")
        if attr in ("other(pattern_marking)", "other(material)"):
            continue
        lvl = "part" if len(cat) > 1 else "obj"
        t = attr_type[attr]
        d = out.setdefault((lvl, t), dict(pairs=0, attrs=set()))
        d["pairs"] += 1
        d["attrs"].add(attr)
    return {f"{l}_{t}": dict(pairs=v["pairs"], attrs=len(v["attrs"])) for (l, t), v in out.items()}


def ci(a, b):
    d = np.asarray(a) - np.asarray(b)
    lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
    return float(lo), float(hi)


def fmt(p, lo=None, hi=None):
    if lo is None:
        return f"{p:.2f}"
    s = "*" if lo > 0 or hi < 0 else ""
    return f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = a.run
    att = json.load(open(os.path.join(run, "audit.json")))["vocab"]
    names = dict(att["attributes"])
    attr_type = {names[i]: t for t, ix in att["attr_type_to_attr_idxs"].items() for i in ix}
    res = dict(points={}, contrasts={}, counts={}, bookkeeping={}, ranks={})
    for v in ("val", "val_minus_prior4", "val_dedup", "test", "test_dedup"):
        res["points"][v] = {c: flat(load_eval(run, c, v)["hierarchical"]) for c in CONDS}
        e0 = load_eval(run, "frozen_full", v)
        res["counts"][v] = counts(e0, attr_type)
        res["bookkeeping"][v] = {c: {k: load_eval(run, c, v)[k] for k in ("n_eval_images", "n_gt_anns", "n_dets_given", "n_dets_kept",
                                                                         "dets_dropped_by_max_dets", "images_with_dropped", "alignment_mismatch",
                                                                         "n_valid_pairs_area_all")} for c in CONDS}
    for bname in ("test_full", "test_dedup", "val_full"):
        B = json.load(open(os.path.join(run, "boot", f"{bname}.json")))
        v = {"test_full": "test", "test_dedup": "test_dedup", "val_full": "val"}[bname]
        for c in CONDS:                                 # weighted point (all w = 1 on the subset) == official point
            for k in KEYS:
                assert abs(B["points"][c][k] - res["points"][v][c][k]) < 1e-9, (bname, c, k)
        vp = np.array(B["valid_pairs"]["boot"])
        res.setdefault("valid_pairs_boot", {})[bname] = dict(point=B["valid_pairs"]["point"], min=int(vp.min()), p5=float(np.percentile(vp, 5)),
                                                              median=float(np.median(vp)), max=int(vp.max()))
        cons = [(f"{x} − Frozen", x, "frozen_full") for x in CONDS if x != "frozen_full"] + [(f"{L}: {CONDS[x]} − {CONDS[y]}", x, y) for L, x, y in CONTR]
        for name, x, y in cons:
            res["contrasts"].setdefault(bname, {})[name] = {k: dict(delta=res["points"][v][x][k] - res["points"][v][y][k],
                                                                    lo=ci(B["boot"][x][k], B["boot"][y][k])[0], hi=ci(B["boot"][x][k], B["boot"][y][k])[1])
                                                            for k in KEYS}
    old = json.load(open("output/2026-11-06/g1_cross_distill/results.json"))["points"]
    rk = lambda d: {c: int(r) for c, r in zip(d, (-np.array(list(d.values()))).argsort().argsort() + 1)}  # noqa: E731
    res["ranks"] = {"M0 Overall (probe val subset, own metric)": rk({c: old[OLD[c]]["M0 Overall"] for c in CONDS}),
                    "M2 Overall (probe val subset, own metric)": rk({c: old[OLD[c]]["M2 Overall"] for c in CONDS}),
                    "Official Obj AP (val)": rk({c: res["points"]["val"][c]["obj"] for c in CONDS}),
                    "Official Part AP (val)": rk({c: res["points"]["val"][c]["part"] for c in CONDS}),
                    "Official Obj AP (test)": rk({c: res["points"]["test"][c]["obj"] for c in CONDS}),
                    "Official Part AP (test)": rk({c: res["points"]["test"][c]["part"] for c in CONDS})}
    res["old_points"] = {c: {k: old[OLD[c]][k] for k in ("M0 Overall", "M2 Overall")} for c in CONDS}
    res["infer"] = {c: {sp: json.load(open(os.path.join(run, "preds", c, f"infer_{sp}.json"))) for sp in ("val", "test")} for c in CONDS}
    vb = os.path.join(run, "boot", "verify_val_frozen_full.json")
    res["boot_verify"] = json.load(open(vb)) if os.path.exists(vb) else None
    json.dump(res, open(os.path.join(run, "results.json"), "w"), indent=1, default=float)
    write_md(res, run)


def write_md(res, run):
    P, K = res["points"], res["contrasts"]
    L = ["# PACO-LVIS GT-Box + Category Oracle Attribute AP (Official Evaluator) — auto-generated by src.pacoattr.report", "",
         "Oracle input: GT box + GT category (score 1.0) per official GT annotation; attribute scores = the model. NOT the official "
         "PACO detection benchmark; not comparable with Model Zoo detection AP. `*` = paired image bootstrap 95% CI excludes 0 "
         "(B = 1000, seed 0; single probe seed 0 — CIs exclude training-seed variance).", ""]
    for v, title in (("test", "Official TEST (9,443 images)"), ("test_dedup", "TEST minus G1-exact / confirmed near-duplicate images"),
                     ("val", "Official VAL (2,410 images, Phase 1)"), ("val_minus_prior4", "VAL minus the 4 earlier near-dups"),
                     ("val_dedup", "VAL minus G1-exact / confirmed near-duplicates")):
        L += [f"## {title}", "", "| Method | Obj All | Obj Color | Obj Pattern | Obj Material | Obj Transp. | Part All | Part Color | Part Pattern | Part Material | Part Transp. |",
              "|---|" + "---:|" * 10]
        for c, nm in CONDS.items():
            L.append(f"| {nm} | " + " | ".join(fmt(P[v][c][k]) for k in KEYS) + " |")
        cnt = res["counts"][v]
        L += ["", "valid pairs / attributes: " + ", ".join(f"{k} {d['pairs']}/{d['attrs']}" for k, d in sorted(cnt.items())), ""]
    for b in ("test_full", "test_dedup", "val_full"):
        L += [f"## Contrasts ({b}; valid pairs per resample: {res['valid_pairs_boot'][b]})", "",
              "| contrast | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
        for name, d in K[b].items():
            L.append(f"| {name} | " + " | ".join(fmt(d[k]['delta'], d[k]['lo'], d[k]['hi']) for k in KEYS) + " |")
        L.append("")
    L += ["## Ranks", "", "| ranking | " + " | ".join(CONDS.values()) + " |", "|---|" + "---:|" * 5]
    for k, r in res["ranks"].items():
        L.append(f"| {k} | " + " | ".join(str(r[c]) for c in CONDS) + " |")
    open(os.path.join(run, "results.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
