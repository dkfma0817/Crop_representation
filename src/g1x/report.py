"""G1-pool / G1-cross pilot report (CPU): PACO val R2 (dev-selected epoch, = src.ivc.compare_bb rule) M0 / M2 with paired
image bootstrap 95% CI (B = 1000, seed 0), COCO-dev feature diagnostics, gradient-flow checks.

  python -m src.g1x.report --run output/2026-11-06/g1_cross_distill
"""
import argparse
import json
import os

import numpy as np

import src.readout.common as RC
from src.ivc.common import DBINS, EXCL
from src.ivc.compare_bb import pick
from src.readout.evaluate import Boot, summarize

C0_ORACLE = "output/2026-11-06/ivc_v3/probe_c0"
RUNS = [("Frozen Full", "probe_frozen_full"), ("G1-pool", "probe_G1pool_full"), ("G1-cross Encoder Only", "probe_G1cross_full"),
        ("G1-cross + Decoder", "probe_G1cross_dec"), ("C0 Crop Oracle", None)]
COLS = [("M0 Overall", "m0", "all", "overall"), ("M2 Overall", "m2", "all", "overall"), ("Object", "m0", "all", "object"),
        ("Part", "m0", "all", "part"), ("Obj D5-16", "m0", "D5-16", "object"), ("Part D1-4", "m0", "D1-4", "part"),
        ("Obj D17+", "m0", "D17+", "object"), ("Color", "m0", "all", "color"), ("Material", "m0", "all", "material"),
        ("Pattern", "m0", "all", "pattern_marking")]
CONTRASTS = [("G1-pool", "Frozen Full", "Q1"), ("G1-cross Encoder Only", "Frozen Full", "Q2"),
             ("G1-cross + Decoder", "G1-cross Encoder Only", "Q3"), ("G1-cross Encoder Only", "G1-pool", "Q4"),
             ("G1-cross + Decoder", "G1-pool", "Q4"), ("G1-cross + Decoder", "Frozen Full", "-"),
             ("C0 Crop Oracle", "G1-cross + Decoder", "gap"), ("C0 Crop Oracle", "Frozen Full", "gap")]


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reader", default="r2")
    a = ap.parse_args()
    R = a.run
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    Ytr_by = RC.ytr_by_level(data)
    BB = {"all": Boot(dva, Yva, Ytr_by), **{nm: Boot(dva, Yva, Ytr_by, mask=(dva.n_in.values >= lo) & (dva.n_in.values <= hi))
                                            for nm, lo, hi in DBINS}}
    M, OUT, L = {}, {"epochs": {}, "points": {}, "contrasts": {}, "features": {}, "grads": {}}, []
    for nm, d in RUNS:
        path = os.path.join(C0_ORACLE if d is None else os.path.join(R, d), a.reader)
        if not os.path.exists(os.path.join(path, "curves_dev.jsonl")):
            print(f"missing {path}")
            continue
        ep, dm0, S = pick(path)
        assert S.shape == (len(dva), 59), (nm, S.shape)
        M[nm] = {b: dict(m0=BB[b].m0(S), m2=BB[b].m2(S)) for b in BB}
        OUT["epochs"][nm] = dict(epoch=ep, dev_M0=dm0, dir=path)
        OUT["points"][nm] = {c: float(M[nm][b][m][g][0]) for c, m, b, g in COLS}
        print(nm, ep, round(OUT["points"][nm]["M0 Overall"], 2), flush=True)

    # ---------------- feature diagnostics (COCO dev, final EMA; frozen = step-0 init row of the G1-pool run)
    def last(run, model="ema"):
        rows = [r for r in jl(os.path.join(R, run, "diag.jsonl")) if r["model"] == model]
        return rows[-1] if rows else {}
    init = [r for r in jl(os.path.join(R, "G1pool", "diag.jsonl")) if r["model"] == "init"]
    feat = {"Frozen Full": init[0] if init else {}, "G1-pool": last("G1pool"), "G1-cross Encoder Only": last("G1cross"),
            "G1-cross + Decoder": last("G1cross")}
    OUT["features"] = feat
    OUT["features_student"] = {"G1-pool": last("G1pool", "student"), "G1-cross": last("G1cross", "student")}
    for run in ("G1pool", "G1cross"):
        OUT["grads"][run] = dict(check=jl(os.path.join(R, run, "grad_check.jsonl")),
                                 train=[r for r in jl(os.path.join(R, run, "train_log.jsonl")) if r["step"] in (100, 1000, 3000, 6000, 9000, 12000)])
    OUT["weights"] = json.load(open(os.path.join(R, "calib", "weights.json"))) if os.path.exists(os.path.join(R, "calib", "weights.json")) else {}

    f3 = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    L += ["## Main table (PACO val, R2, dev-selected epoch; features = COCO dev, final EMA)", "",
          "| Method | R2 M0 | R2 M2 | Obj D5–16 | Part D1–4 | Full CLS Cos | Crop ROI Cos |", "|---|---|---|---|---|---|---|"]
    for nm, _ in RUNS:
        p = OUT["points"].get(nm)
        f = feat.get(nm, {})
        roi = f.get("dec_pool_cos") if nm == "G1-cross + Decoder" else f.get("roi_pool_cos")
        if nm == "C0 Crop Oracle":
            cls, roi_s = "1 (crop teacher itself)", "1"
        else:
            cls, roi_s = f3(f.get("full_cls_cos")), f3(roi)
        if p is None:
            L.append(f"| {nm} | n/a | n/a | n/a | n/a | {cls} | {roi_s} |")
        else:
            L.append(f"| {nm} | {p['M0 Overall']:.2f} | {p['M2 Overall']:.2f} | {p['Obj D5-16']:.2f} | {p['Part D1-4']:.2f} | {cls} | {roi_s} |")
    L += ["", "Crop ROI Cos: encoder rows = cos(area-weighted pooled full tokens, pooled crop teacher); G1-cross + Decoder = "
          "cos(pooled decoder output, pooled crop teacher).", "",
          "## PACO val, all columns (M0 unless noted)", "",
          "| Method | ep (dev M0) | " + " | ".join(c for c, _, _, _ in COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for nm, _ in RUNS:
        if nm in OUT["points"]:
            e = OUT["epochs"][nm]
            L.append(f"| {nm} | {e['epoch']} ({e['dev_M0']:.2f}) | " + " | ".join(f"{OUT['points'][nm][c]:.2f}" for c, _, _, _ in COLS) + " |")
    L += ["", "## Paired image bootstrap 95% CI (B = 1000, seed 0)", "",
          "| Q | contrast | " + " | ".join(c for c, _, _, _ in COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for x, y, q in CONTRASTS:
        if x not in M or y not in M:
            continue
        cells, rec = [], {}
        for c, m, b, g in COLS:
            s = summarize(M[x][b][m][g], M[y][b][m][g])
            rec[c] = s
            star = "*" if (s["lo"] > 0 or s["hi"] < 0) else ""
            cells.append(f"{s['point']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]{star}")
        OUT["contrasts"][f"{x} - {y}"] = rec
        L.append(f"| {q} | {x} − {y} | " + " | ".join(cells) + " |")
    L += ["", "\\* = 95% CI excludes 0.", "", "## Feature diagnostics (COCO dev, 468 images / 1,872 ROIs; final EMA weights)", "",
          "| Method | ROI pooled cos (enc) | Dec token cos | Dec pooled cos | SV ratio | Relation err | Full CLS cos | "
          "Full mean-token cos | Full per-token cos | Inside-ROI token cos | Outside-ROI token cos | Outside-all-ROI token cos |",
          "|" + "---|" * 12]
    for nm in ("Frozen Full", "G1-pool", "G1-cross Encoder Only", "G1-cross + Decoder"):
        f = feat.get(nm, {})
        dec = nm == "G1-cross + Decoder"
        g = lambda k: f3(f.get(k))  # noqa: E731
        L.append(f"| {nm} | {g('roi_pool_cos')} | {g('dec_tok_cos') if dec else '–'} | {g('dec_pool_cos') if dec else '–'} | "
                 f"{g('sv_ratio') if dec else '–'} | {g('rel_err') if dec else '–'} | {g('full_cls_cos')} | {g('full_mean_cos')} | "
                 f"{g('full_tok_cos')} | {g('in_tok_cos')} | {g('out_tok_cos')} | {g('img_out_tok_cos')} |")
    L += ["", "Token cosines of the full image are student vs ORIGINAL DINOv3 on the same full image; inside / outside = full cells "
          "overlapping / not overlapping the ROI box (outside-all = cells outside every ROI of the image).", "",
          "## Gradient flow (local batch, mean losses; encoder / decoder gradient L2 norms)", "",
          "| run | step | enc L_roi | enc L_global | enc L_detail | dec L_detail |", "|---|---|---|---|---|---|"]
    for run in ("G1pool", "G1cross"):
        for r in OUT["grads"][run]["check"]:
            L.append(f"| {run} | {r['step']} | {r['enc_roi']:.4f} | {r['enc_global']:.4f} | {r.get('enc_detail', float('nan')):.4f} | "
                     f"{r.get('dec_detail', float('nan')):.4f} |")
    L += ["", "| run | step | L_roi | L_detail | L_global | total enc grad norm (pre-clip) | total dec grad norm |", "|---|---|---|---|---|---|---|"]
    for run in ("G1pool", "G1cross"):
        for r in OUT["grads"][run]["train"]:
            L.append(f"| {run} | {r['step']} | {r['l_roi']:.4f} | {r['l_detail']:.4f} | {r['l_global']:.6f} | {r['gnorm_enc']:.3f} | {r['gnorm_dec']:.3f} |")
    open(os.path.join(R, "results.md"), "w").write("\n".join(L) + "\n")
    json.dump(OUT, open(os.path.join(R, "results.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
