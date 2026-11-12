"""Collect the official val results, the paired bootstrap, the self analyses and the bookkeeping into results.json /
results.md (README.md is written by hand from these). CPU only.
  python -m src.maskemb.report
"""
import json
import os

import numpy as np

from src.maskemb import common as MC

CONDS = [("clip_full", "Frozen CLIP Full"), ("clip_c0", "Frozen CLIP C0 Crop"), ("maskembed_full", "MaskEmbed Full")]
CONTRASTS = [("A", "clip_c0", "clip_full", "CLIP Crop − CLIP Full"), ("B", "maskembed_full", "clip_full", "MaskEmbed Full − CLIP Full"),
             ("C", "maskembed_full", "clip_c0", "MaskEmbed Full − CLIP Crop")]
TYPES = [("color", "Color"), ("pattern_marking", "Pattern & Markings"), ("material", "Material"), ("transparency", "Transparency")]
KEYS = ["obj", "part"] + [f"{lv}_{t}" for lv in ("obj", "part") for t, _ in TYPES]


def jl(p):
    return json.load(open(p))


def fmt_ci(d, lo, hi):
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


def main():
    global CONDS, CONTRASTS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    sp = ap.parse_args().split
    sfx = "" if sp == "val" else "_test"
    R = MC.RUN
    CONDS = [(c, n) for c, n in CONDS if os.path.exists(os.path.join(R, "eval", c, sp, "official_results.json"))]
    have = {c for c, _ in CONDS}
    CONTRASTS = [t for t in CONTRASTS if t[1] in have and t[2] in have]
    out = dict(conds=dict(CONDS), points={}, contrasts={}, recovery={}, selection={}, infer={}, official_meta={})
    for c, _ in CONDS:
        o = jl(os.path.join(R, "eval", c, sp, "official_results.json"))
        h = o["hierarchical"]
        out["points"][c] = dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"], **{f"obj_{t}": h["obj-attrs"][t] for t, _ in TYPES},
                                **{f"part_{t}": h["obj-part-attrs"][t] for t, _ in TYPES})
        out["official_meta"][c] = {k: o[k] for k in ("n_eval_images", "n_gt_anns", "n_dets_given", "n_dets_kept", "dets_dropped_by_max_dets",
                                                    "alignment_mismatch", "n_joint_pairs", "n_valid_pairs_area_all", "seconds")}
        out["selection"][c] = jl(os.path.join(R, "probes", c, "r2", "selected_epoch.json"))
        out["selection"][c]["dev_curve_M0"] = [round(json.loads(l)["dev"]["M0"]["overall"], 2) for l in open(os.path.join(R, "probes", c, "r2", "curves_dev.jsonl"))]
        out["infer"][c] = {k: v for k, v in jl(os.path.join(R, "preds", c, f"infer_{sp}.json")).items() if k not in ("encoder",)}
    B = jl(os.path.join(R, "boot", f"{sp}_full.json"))
    for c, _ in CONDS:
        for k in KEYS:
            assert abs(B["points"][c][k] - out["points"][c][k]) < 1e-6, (c, k)
    for tag, x, y, nm in CONTRASTS:
        d = {}
        for k in KEYS:
            s = np.array(B["boot"][x][k]) - np.array(B["boot"][y][k])
            d[k] = dict(delta=out["points"][x][k] - out["points"][y][k], lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)))
        out["contrasts"][tag] = dict(name=nm, **d)
    for k in (KEYS if {"clip_c0", "clip_full", "maskembed_full"} <= have else []):   # share of the crop headroom recovered by MaskEmbed (point estimate + bootstrap ratio CI)
        hr = np.array(B["boot"]["clip_c0"][k]) - np.array(B["boot"]["clip_full"][k])
        gm = np.array(B["boot"]["maskembed_full"][k]) - np.array(B["boot"]["clip_full"][k])
        H = out["points"]["clip_c0"][k] - out["points"]["clip_full"][k]
        Gm = out["points"]["maskembed_full"][k] - out["points"]["clip_full"][k]
        ok = hr > 0
        out["recovery"][k] = dict(headroom=H, gain=Gm, share=(Gm / H if H > 0 else None),
                                  share_ci=[float(np.percentile(gm[ok] / hr[ok], 2.5)), float(np.percentile(gm[ok] / hr[ok], 97.5))] if ok.mean() > 0.9 else None,
                                  frac_resamples_headroom_pos=float(ok.mean()))
    out["boot_meta"] = dict(B=B["B"], n_images=B["n_images"], valid_pairs_point=B["valid_pairs"]["point"],
                            valid_pairs_boot=[int(min(B["valid_pairs"]["boot"])), int(np.median(B["valid_pairs"]["boot"])), int(max(B["valid_pairs"]["boot"]))])
    fa = os.path.join(R, "eval_area", f"{sp}_box_area_bins.json")
    if os.path.exists(fa):
        out["area_bins"] = jl(fa)
        out["area_all_slice_minus_main"] = {}
        for c, _ in CONDS:      # test: <= 0.002 AP (box-area GT copy x the official max_dets cut in 11 images); val: exact
            dd = {lv: out["area_bins"][c]["all"][lv] - out["points"][c][lv] for lv in ("obj", "part")}
            out["area_all_slice_minus_main"][c] = dd
            assert all(abs(v) < 0.01 for v in dd.values()), ("area 'all' slice != main result", c, dd)
    if os.path.exists(os.path.join(R, "analysis.json")):
        out["analysis"] = jl(os.path.join(R, "analysis.json"))
    json.dump(out, open(os.path.join(R, f"results{sfx}.json"), "w"), indent=1, default=float)

    P = out["points"]
    L = [f"# results (PACO-LVIS official {sp.upper()}, GT box + category oracle, official attribute box-AP evaluator)", "",
         "## Table A — Object / Part attribute AP", "", "| Method | Object AP | Part AP | Color (Obj) | Pattern (Obj) | Material (Obj) |", "|---|---:|---:|---:|---:|---:|"]
    for c, nm in CONDS:
        L.append(f"| {nm} | {P[c]['obj']:.2f} | {P[c]['part']:.2f} | {P[c]['obj_color']:.2f} | {P[c]['obj_pattern_marking']:.2f} | {P[c]['obj_material']:.2f} |")
    for lv, title in (("obj", "Object"), ("part", "Object-Part")):
        L += ["", f"## {title} attribute AP by official type", "", "| Method | All | " + " | ".join(n for _, n in TYPES) + " |", "|---|---:|" + "---:|" * len(TYPES)]
        for c, nm in CONDS:
            L.append(f"| {nm} | {P[c][lv]:.2f} | " + " | ".join(f"{P[c][f'{lv}_{t}']:.2f}" for t, _ in TYPES) + " |")
    L += ["", f"## Contrasts (Δ AP [paired image bootstrap 95% CI], B = {B['B']}, * = CI excludes 0)", "",
          "| Contrast | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for tag, *_ in CONTRASTS:
        d = out["contrasts"][tag]
        L.append(f"| {tag}. {d['name']} | " + " | ".join(fmt_ci(d[k]["delta"], d[k]["lo"], d[k]["hi"]) for k in KEYS) + " |")
    if out["recovery"]:
        L += ["", "## Crop headroom recovered by MaskEmbed", "", "| metric | headroom (C0 − Full) | MaskEmbed gain | share | share 95% CI |", "|---|---:|---:|---:|---|"]
    for k in out["recovery"]:
        r = out["recovery"][k]
        L.append(f"| {k} | {r['headroom']:+.2f} | {r['gain']:+.2f} | {('%.0f%%' % (100 * r['share'])) if r['share'] is not None else '–'} | "
                 + (f"[{100 * r['share_ci'][0]:.0f}%, {100 * r['share_ci'][1]:.0f}%]" if r["share_ci"] else f"– (headroom>0 in {100 * r['frac_resamples_headroom_pos']:.0f}% of resamples)") + " |")
    if "area_bins" in out:
        L += ["", "## SELF analysis: AP by original box area (COCO bins; not the PACO main metric)", "",
              "| Method | Obj S | Obj M | Obj L | Part S | Part M | Part L |", "|---|---:|---:|---:|---:|---:|---:|"]
        for c, nm in CONDS:
            a = out["area_bins"][c]
            L.append(f"| {nm} | " + " | ".join(f"{a[b][lv]:.2f}" for lv in ("obj", "part") for b in ("small", "medium", "large")) + " |")
        a = out["area_bins"][CONDS[0][0]]
        L.append("| valid pairs | " + " | ".join(str(a[b]["n_valid_pairs"]) for b in ("small", "medium", "large")) + " | (pairs are shared obj/part) | | |")
    L += ["", "## Selection / inference", ""]
    for c, nm in CONDS:
        s, i = out["selection"][c], out["infer"][c]
        L.append(f"- {nm}: epoch {s['epoch']} (dev M0 {s['dev_M0']:.2f}; curve {s['dev_curve_M0']}); encoder calls/image {i['encoder_calls_per_image']:.2f}; "
                 f"enc ms/img {i['ms_encoder_per_image']:.1f}, enc ms/ROI {i['ms_encoder_per_roi']:.1f}, reader ms/ROI {i['ms_reader_per_roi']:.2f}; checks {i['checks']}")
    open(os.path.join(R, f"results{sfx}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
