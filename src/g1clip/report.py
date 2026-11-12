"""G1-CLIP report (CPU): official PACO TEST GT box + category oracle attribute AP only (obj / part / official types),
paired image bootstrap (src.pacoattr.boot output). Reference row = Frozen CLIP Full of maskembed_paco (same manifest,
evaluator, adapter, probe protocol; its eval dir is symlinked, read-only). No size bins / custom metrics.
  python -m src.g1clip.report
"""
import json
import os

import numpy as np

R = "output/2026-11-07/g1clip"
CONDS = [("clip_full", "Frozen CLIP Full (reference)"), ("g1pool_enc", "G1-pool Encoder Only"), ("g1cross_enc", "G1-cross Encoder Only"),
         ("g1cross_dec", "G1-cross + Decoder")]
CONTRASTS = [("g1pool_enc", "clip_full"), ("g1cross_enc", "clip_full"), ("g1cross_dec", "clip_full"), ("g1cross_enc", "g1pool_enc"),
             ("g1cross_dec", "g1cross_enc")]
TYPES = [("color", "Color"), ("pattern_marking", "Pattern & Markings"), ("material", "Material"), ("transparency", "Transparency")]
KEYS = ["obj", "part"] + [f"{lv}_{t}" for lv in ("obj", "part") for t, _ in TYPES]


def main():
    out = dict(points={}, contrasts={}, meta={}, selection={}, infer={})
    for c, _ in CONDS:
        o = json.load(open(os.path.join(R, "eval", c, "test", "official_results.json")))
        h = o["hierarchical"]
        out["points"][c] = dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"], **{f"obj_{t}": h["obj-attrs"][t] for t, _ in TYPES},
                                **{f"part_{t}": h["obj-part-attrs"][t] for t, _ in TYPES})
        out["meta"][c] = {k: o[k] for k in ("n_eval_images", "n_gt_anns", "n_dets_kept", "dets_dropped_by_max_dets", "alignment_mismatch", "n_valid_pairs_area_all")}
        if c != "clip_full":
            out["selection"][c] = json.load(open(os.path.join(R, "probes", c, "r2", "selected_epoch.json")))
            out["infer"][c] = {k: v for k, v in json.load(open(os.path.join(R, "preds", c, "infer_test.json"))).items() if k in ("checks", "encoder", "encoder_calls_per_image", "ms_encoder_per_image", "ms_reader_per_roi", "n_rows", "n_images")}
    assert len({json.dumps(m, sort_keys=True) for m in out["meta"].values()}) == 1, "conditions evaluated on different GT / detections"
    B = json.load(open(os.path.join(R, "boot", "test_full.json")))
    for c, _ in CONDS:
        for k in KEYS:
            assert abs(B["points"][c][k] - out["points"][c][k]) < 1e-6
    for x, y in CONTRASTS:
        d = {}
        for k in KEYS:
            s = np.array(B["boot"][x][k]) - np.array(B["boot"][y][k])
            d[k] = dict(delta=out["points"][x][k] - out["points"][y][k], lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)))
        out["contrasts"][f"{x}-{y}"] = d
    json.dump(out, open(os.path.join(R, "results.json"), "w"), indent=1, default=float)
    P, nm = out["points"], dict(CONDS)
    f = lambda d: f"{d['delta']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]" + ("*" if d["lo"] > 0 or d["hi"] < 0 else "")  # noqa: E731
    L = ["# G1 on CLIP ViT-L/14@336 — PACO official TEST, GT box + category oracle attribute AP", "",
         "| Method | Object AP | Part AP |", "|---|---:|---:|"]
    L += [f"| {nm[c]} | {P[c]['obj']:.2f} | {P[c]['part']:.2f} |" for c, _ in CONDS]
    for lv, title in (("obj", "Object"), ("part", "Object-Part")):
        L += ["", f"## {title} attribute AP by official type", "", "| Method | All | " + " | ".join(n for _, n in TYPES) + " |", "|---|---:|" + "---:|" * 4]
        L += [f"| {nm[c]} | {P[c][lv]:.2f} | " + " | ".join(f"{P[c][f'{lv}_{t}']:.2f}" for t, _ in TYPES) + " |" for c, _ in CONDS]
    L += ["", f"## Contrasts (Δ AP [paired image bootstrap 95% CI], B = {B['B']}, * = CI excludes 0)", "",
          "| Contrast | Object | Part | " + " | ".join(f"Obj {n}" for _, n in TYPES) + " | " + " | ".join(f"Part {n}" for _, n in TYPES) + " |",
          "|---|" + "---|" * 10]
    for k_, d in out["contrasts"].items():
        x, y = k_.split("-")
        L.append(f"| {nm[x]} − {nm[y]} | " + " | ".join(f(d[k]) for k in KEYS) + " |")
    L += ["", "## Bookkeeping", ""] + [f"- {nm[c]}: R2 epoch {s['epoch']} (dev M0 {s['dev_M0']:.2f}); checks {out['infer'][c]['checks']}" for c, s in out["selection"].items()]
    L.append(f"- evaluation scope (identical for all rows): {out['meta']['clip_full']}")
    open(os.path.join(R, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
