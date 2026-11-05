"""S-series report: per pair (A: S1 hierarchical vs flat, B: S3 two-way vs one-way) hypothesis / decoder / loss / direct
checks / PACO (frozen encoder + decoder, new R2, val at the dev-selected epoch) / cost / verdict.
References (same PACO probe protocol, earlier run output/2026-11-05/msd): H1, Bilinear, Full ML+R2, C0.

  python -m src.msd.sreport --run output/2026-11-05/sdec --ref output/2026-11-05/msd
"""
import argparse
import json
import os

import numpy as np

import src.readout.common as RC
from src.msd.report import EXCL, f, jl, p
from src.readout.evaluate import Boot, summarize

SEC = ["object", "part", "color", "material", "pattern_marking"]
PAIRS = {
    "A": dict(title="S1 Hierarchical vs Flat", ctrl="A0", exp="A1",
              hyp="Coarse-to-fine refinement (4x4 -> 8x8 -> 16x16 -> 24x24) builds ROI structure first and adds detail later, "
                  "giving more spatially varied, teacher-like ROI maps than the same blocks run flat at 24x24.",
              arch="A0: 4 blocks [SA, CA to ROI+ring cache tokens, FFN x4], d 384, all at 24x24. A1: the same 4 blocks at 4/8/16/24, "
                   "bilinear state upsampling + stage grid input between stages. Same params (10.71M w/o aux heads)."),
    "B": dict(title="S3 Two-way vs One-way", ctrl="B0", exp="B1",
              hyp="Letting the ROI working map be updated by the queries (two-way) keeps and sharpens spatial structure better than "
                   "updating only the queries over a fixed working map.",
              arch="16 box-conditioned queries, 24x24 working map W (from the fixed cache), 3 stages: q reads cache and W, FFN; "
                   "B0 adds SA(q) + FFN2(q) (control), B1 instead W <- CA(W, q) + FFN_w(W). Final readout W + CA(W, q) + FFN -> output head. "
                   "Same params (11.47M w/o aux heads)."),
}
LOSS = ("Same as H1: L12 + 0.5 L8 + 0.5 L4, L_d = coverage-weighted [mean_c (p - t)^2 + (1 - cos)] on per-depth teacher-normalised "
        "features, teacher = frozen DINOv2 on the original-pixel COCO GT crop (C0), p_4 / p_8 from aux heads (removed for R2). "
        "COCO 12k stream (64 ROIs/step), AdamW 1e-4 (0.9, 0.95), wd 0.05, warmup 600 + cosine, clip 1.0, seed 0, frozen encoder, "
        "checkpoint = best COCO-dev L12. Output Z = Bilinear(F12) + sd12 * Linear_out(h) with Linear_out zero-init (all four).")
REFS = {"H1": "msd/probe_H1", "bil": "msd/probe_bil", "mlr2": "msd/probe_mlr2", "c0": "msd/probe_c0"}
LAB = {"A0": "A0 flat", "A1": "A1 hierarchical", "B0": "B0 one-way", "B1": "B1 two-way", "H1": "H1 (CoordMLP, ref)",
       "bil": "Bilinear (ref)", "mlr2": "Full ML+R2 (ref)", "c0": "GT crop C0 (ref)"}


def load_probe(d, B, bins):
    dv = jl(os.path.join(d, "r2", "curves_dev.jsonl"))
    if not os.path.exists(os.path.join(d, "done.json")) or not dv:
        return None
    ep = max(dv[1:] or dv, key=lambda r: r["dev"]["M0"]["overall"])["epoch"]
    S = np.load(os.path.join(d, "r2", f"scores_val_ep{ep}.npy")).astype(np.float32)
    fz = os.path.join(d, "frozen_check.json")
    return dict(epoch=ep, m0=B.m0(S), m2=B.m2(S), bins={k: b.m0(S) for k, b in bins.items()},
                frozen=json.load(open(fz)) if os.path.exists(fz) else None, done=json.load(open(os.path.join(d, "done.json"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", default="output/2026-11-05/msd")
    a = ap.parse_args()
    R = a.run
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    Ytr_by = RC.ytr_by_level(data)
    B = Boot(dva, Yva, Ytr_by)
    nin = np.load(RC.NIN_SG.format("val"))
    bins = {nm: Boot(dva, Yva, Ytr_by, mask=(nin >= lo) & (nin <= hi)) for nm, lo, hi in (("1-4", 1, 4), ("5-16", 5, 16), ("17+", 17, 10 ** 9))}
    P = {c: load_probe(os.path.join(R, f"probe_{c}"), B, bins) for c in ("A0", "A1", "B0", "B1")}
    base = os.path.dirname(a.ref.rstrip("/"))
    P.update({k: load_probe(os.path.join(base, v), B, bins) for k, v in REFS.items()})
    pre = {c: dict(done=json.load(open(os.path.join(R, c, "done.json"))) if os.path.exists(os.path.join(R, c, "done.json")) else None,
                   cfg=json.load(open(os.path.join(R, c, "config.json"))) if os.path.exists(os.path.join(R, c, "config.json")) else None,
                   gc=json.load(open(os.path.join(R, c, "grad_check.json"))) if os.path.exists(os.path.join(R, c, "grad_check.json")) else None)
           for c in ("A0", "A1", "B0", "B1")}
    sd = json.load(open(os.path.join(R, "sdirect.json"))) if os.path.exists(os.path.join(R, "sdirect.json")) else {}
    L = ["# S-series spatial ROI decoder pilot (COCO pre-training -> frozen PACO R2 probe)", "",
         "PACO VAL only (test never evaluated), R2 epoch by PACO dev (1,000 train images held out), primary = val **M0 Overall**, "
         "paired image bootstrap 95% CI. References come from the earlier run with the identical probe protocol "
         f"(`{a.ref}`). Direct checks: 700 PACO val images (report only). Cache unchanged in every direct-check forward: "
         f"**{sd.get('cache_unchanged_all_forwards')}**.", "", f"**Common loss / recipe.** {LOSS}", ""]
    out = {}
    for key, spec in PAIRS.items():
        c0, c1 = spec["ctrl"], spec["exp"]
        L += [f"## Pair {key}: {spec['title']} ({c1} vs {c0})", "", f"**Hypothesis.** {spec['hyp']}", "", f"**Decoder.** {spec['arch']}", ""]
        L += ["**Training / checks**", "", "| cond | trainable (decoder incl. aux) | dev step | dev L12 | dev L4 | dev L8 | output shape | cache unchanged | trunk grad step1 / step2 | aux-only grad into trunk | weight change trunk | time | peak mem |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in (c0, c1):
            x = pre[c]
            if not x["done"]:
                L.append(f"| {c} | FAILED / missing |" + " |" * 12)
                continue
            s, g = x["done"]["selected"], x["gc"]
            tk = [k for k in g["step1_grad_norm"] if k.startswith("decoder_trunk")][0]
            L.append(f"| {c} | {x['cfg']['trainable_total']:,} | {s['step']} | {s['dev']['L12']:.4f} | {s['dev']['L4']:.4f} | {s['dev']['L8']:.4f} | {g['decoder_output_shape']} | "
                     f"{g['cache_unchanged']} | {g['step1_grad_norm'][tk]:.1e} / {g['step2_grad_norm'][tk]:.1e} | {g['aux_loss_only_grad_norm_into_decoder_body']:.3f} | "
                     f"{x['done']['weight_change_norm_total'][tk]:.2f} | {x['done']['time_s'] / 60:.0f} min | {x['done']['peak_mem_gb']:.1f} GB |")
        L += ["", "**Direct checks** (normalised block-12 space; teacher = C0 crop; ratios pred / teacher, 1 = teacher-like)", "",
              "| cond | L12 | var ratio | eff. rank (ratio) | neighbour-diff ratio | centred cos | affinity gap | L12 obj 1-16 | L12 part | L12 patterned | var ratio obj 1-16 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in ("bil", "H1", c0, c1):
            if c not in sd:
                continue
            al = sd[c]["all"]
            L.append(f"| {LAB[c]} | {al['L12']:.3f} | {al['var_ratio']:.3f} | {al['erank_pred']:.1f} ({al['erank_ratio']:.3f}) | {al['neigh_diff_ratio']:.3f} | "
                     f"{al['centred_cos']:.3f} | {al['affinity_gap']:.3f} | {sd[c]['object n_in 1-16']['L12']:.3f} | {sd[c]['part']['L12']:.3f} | "
                     f"{sd[c]['patterned (non-plain)']['L12']:.3f} | {sd[c]['object n_in 1-16']['var_ratio']:.3f} |")
        st_rows = [(c, sd[c]["stages"]) for c in (c0, c1) if c in sd and "stages" in sd[c]]
        if st_rows:
            L += ["", "**Stage-wise spatial representation** (" + ("A: output head on each block state, var / eff-rank ratio to the teacher" if key == "A"
                  else "B: working map W in hidden space after each stage (stage1 = W0); B0 keeps W fixed by design") + ")", "",
                  "| cond | " + " | ".join(k for k in st_rows[0][1]) + " |", "|---|" + "---|" * len(st_rows[0][1])]
            for c, st in st_rows:
                L.append(f"| {c} | " + " | ".join(f"{v:.3f}" for v in st.values()) + " |")
        L += ["", "**PACO** (val at the dev-selected epoch)", "",
              "| probe | epoch | **M0 Overall** | Object | Part | Color | Material | Pattern | M2 | M0 n_in 1-4 | 5-16 | 17+ | frozen check |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in (c0, c1, "H1", "bil", "mlr2", "c0"):
            x = P.get(c)
            if x is None:
                L.append(f"| {LAB[c]} | missing |" + " |" * 11)
                continue
            fz = x["frozen"]
            fzs = "n/a" if fz is None else ("ok" if fz["unchanged"] and not fz["grads_present"] and not fz["requires_grad_any"] else "FAILED")
            L.append(f"| {LAB[c]} | {x['epoch']} | **{x['m0']['overall'][0]:.2f}** | " + " | ".join(f"{x['m0'][g][0]:.2f}" for g in SEC) +
                     f" | {x['m2']['overall'][0]:.2f} | " + " | ".join(p(float(x['bins'][nm]['overall'][0])) for nm in ("1-4", "5-16", "17+")) + f" | {fzs} |")
        L += ["", "| contrast | M0 Overall | Object | Part | Color | Material | Pattern | M2 | M0 5-16 |", "|---|---|---|---|---|---|---|---|---|"]
        res = {}
        for nm, x, y in [(f"{c1} - {c0}", c1, c0), (f"{c1} - H1", c1, "H1"), (f"{c0} - H1", c0, "H1"), (f"{c1} - Bilinear", c1, "bil"),
                         (f"{c1} - Full ML+R2", c1, "mlr2"), (f"{c1} - C0", c1, "c0")]:
            if P.get(x) is None or P.get(y) is None:
                continue
            d = {g: summarize(P[x]["m0"][g], P[y]["m0"][g]) for g in ["overall"] + SEC}
            d2 = summarize(P[x]["m2"]["overall"], P[y]["m2"]["overall"])
            db = summarize(P[x]["bins"]["5-16"]["overall"], P[y]["bins"]["5-16"]["overall"])
            res[nm] = dict(m0=d, m2=d2, m0_5_16=db)
            L.append(f"| {nm} | {f(d['overall'])} | " + " | ".join(f(d[g]) for g in SEC) + f" | {f(d2)} | {f(db)} |")
        # verdict
        main_c = res.get(f"{c1} - {c0}")
        a0, a1 = sd.get(c0, {}).get("all"), sd.get(c1, {}).get("all")
        ability = None
        if a0 and a1:
            ability = (a1["var_ratio"] > a0["var_ratio"]) and (a1["erank_ratio"] > a0["erank_ratio"]) and (a1["affinity_gap"] < a0["affinity_gap"])
        paco = bool(main_c and main_c["m0"]["overall"]["lo"] > 0)
        L += ["", "**Cost**: " + "; ".join(f"{c}: pretrain {pre[c]['done']['time_s'] / 60:.0f} min, peak {pre[c]['done']['peak_mem_gb']:.1f} GB/GPU"
                                           + (f", probe {P[c]['done']['time_s'] / 60:.0f} min" if P.get(c) else "") for c in (c0, c1) if pre[c]["done"]), "",
              "**Verdict**", "",
              f"- Spatial ability shown in the direct check ({c1} vs {c0}: higher var ratio AND higher eff-rank ratio AND lower affinity gap): **{ability}**",
              f"- PACO val M0 Overall {c1} - {c0}: {f(main_c['m0']['overall']) if main_c else 'n/a'} -> improved (CI lo > 0): **{paco}**",
              f"- Versus H1 (same protocol): {f(res.get(f'{c1} - H1', {}).get('m0', {}).get('overall'))}; versus Full ML+R2: {f(res.get(f'{c1} - Full ML+R2', {}).get('m0', {}).get('overall'))}",
              f"- **Verdict: {'success' if paco else 'not a success'}**" + (" (direct-check gain without PACO gain is not a success)" if ability and not paco else ""), ""]
        out[key] = dict(contrasts=res, ability=ability, paco_improved=paco)
    open(os.path.join(R, "REPORT.md"), "w").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(R, "report.json"), "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
