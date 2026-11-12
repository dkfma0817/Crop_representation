"""PACO official TEST table for the decoder study (user request 2026-11-08), in the layout of the user's sheet, + deltas vs
G1-cross Encoder Only with paired image bootstrap 95% CI. -> <run>/test/results_test.md / .json"""
import json
import os

import numpy as np

T = "output/2026-11-07/decoder_study/test"
ROWS = [("clip_full", "Original CLIP (ep6)"), ("maskembed_full", "MaskEmbed (ep6)"), ("c0", "CLIP-L GT Crop C0 (ep6, new)"),
        ("g1pool_enc", "G1-pool (ep6)"), ("g1cross_enc", "G1-cross Encoder Only (ep6)"), ("g1cross_dec", "G1-cross + Decoder (existing, ep9 = M0 pick)"),
        ("g1cross_dec_ep6", "G1-cross + Decoder (existing, ep6)"), ("b0", "B0 Bilinear Only (ep6)"), ("d1", "D1 Bilinear + Residual (ep6)"),
        ("l1", "L1 D1 + Covariance (ep6)"), ("l2", "L2 D1 + Relational (ep6)")]
COLS = ["obj", "obj_color", "obj_pattern_marking", "obj_material", "obj_transparency", "part", "part_color", "part_pattern_marking", "part_material", "part_transparency"]
HEAD = ["Obj AP", "Obj AP col", "Obj AP pat", "Obj AP mat", "Obj AP ref", "Part AP", "Part AP col", "Part AP pat", "Part AP mat", "Part AP ref"]


def main():
    b = json.load(open(os.path.join(T, "boot_test.json")))
    P = b["points"]
    L = ["# PACO-LVIS Official Test — GT Box + Category Oracle (decoder study rows added 2026-11-08)", "",
         "Official evaluator (src.pacoattr.official_eval, unmodified PACO/lvis-api), 9,443 test images. ref = official type `transparency`. "
         "All new rows: R2 epoch 6 fixed in advance (no test-based selection). \\* = paired image bootstrap (B=1000) 95% CI excludes 0.", "",
         "| Model | " + " | ".join(HEAD) + " |", "|---|" + "---:|" * len(HEAD)]
    for c, n in ROWS:
        if c not in P:
            L.append(f"| {n} | " + " | ".join(["-"] * len(COLS)) + " |")
            continue
        L.append(f"| {n} | " + " | ".join(f"{P[c][k]:.2f}" for k in COLS) + " |")
    L += ["", "## Δ vs G1-cross Encoder Only (paired bootstrap 95% CI)", "", "| Model | Obj AP | Part AP | Obj mat | Obj pat | Part pat | Part col |", "|---|---|---|---|---|---|---|"]
    out = {}
    for c, n in ROWS:
        if c not in P or c == "g1cross_enc":
            continue
        row = []
        for k in ("obj", "part", "obj_material", "obj_pattern_marking", "part_pattern_marking", "part_color"):
            d = np.array(b["boot"][c][k]) - np.array(b["boot"]["g1cross_enc"][k])
            lo, hi = np.percentile(d, [2.5, 97.5])
            pt = P[c][k] - P["g1cross_enc"][k]
            out.setdefault(c, {})[k] = dict(delta=pt, lo=lo, hi=hi)
            row.append(f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("\\*" if lo > 0 or hi < 0 else ""))
        L.append(f"| {n} | " + " | ".join(row) + " |")
    L += ["", "## Decoder-study contrasts", "", "| Contrast | Obj AP | Part AP | Obj mat | Part pat |", "|---|---|---|---|---|"]
    for n, a, c in (("D1 − B0", "d1", "b0"), ("L1 − D1", "l1", "d1"), ("L2 − D1", "l2", "d1"), ("D1 − existing decoder (ep6)", "d1", "g1cross_dec_ep6"),
                    ("B0 − Encoder Only", "b0", "g1cross_enc"), ("C0 crop − Encoder Only", "c0", "g1cross_enc")):
        if a not in P or c not in P:
            continue
        row = []
        for k in ("obj", "part", "obj_material", "part_pattern_marking"):
            d = np.array(b["boot"][a][k]) - np.array(b["boot"][c][k])
            lo, hi = np.percentile(d, [2.5, 97.5])
            row.append(f"{P[a][k] - P[c][k]:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("\\*" if lo > 0 or hi < 0 else ""))
        L.append(f"| {n} | " + " | ".join(row) + " |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(T, "results_test.md"), "w").write(txt)
    json.dump(dict(points=P, delta_vs_enc=out), open(os.path.join(T, "results_test.json"), "w"), indent=1, default=float)
    print(txt)


if __name__ == "__main__":
    main()
