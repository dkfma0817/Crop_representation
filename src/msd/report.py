"""MSD pilot report: hypothesis / trained parts / direct checks / PACO (frozen R2 probe) / global / cost / verdict.

PACO: R2 epoch per probe = max PACO-dev M0 Overall (dev = 1,000 train images held out, rng 0); numbers = PACO VAL at that
epoch. Primary = val M0 Overall; secondary = M0 object / part / color / material / pattern; M2 auxiliary. Paired image
bootstrap 95% CI (1,000, seed 0). Size bins (SigLIP2 n_in 1-4 / 5-16 / 17+) reported when the bin has M0 units.

  python -m src.msd.report --run output/2026-11-05/msd
"""
import argparse
import json
import os

import numpy as np

import src.readout.common as RC
from src.readout.evaluate import Boot, summarize

EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
PROBES = ["H0", "H1", "H2", "H3", "bil", "mlr2", "c0"]
LABEL = {"H0": "H0 frozen enc + dec, L12", "H1": "H1 frozen enc + dec, L4/8/12", "H2": "H2 adapters + dec, L12",
         "H3": "H3 adapters + dec, L4/8/12", "bil": "Bilinear (no decoder)", "mlr2": "Full Multi-layer + R2", "c0": "GT crop C0 (oracle)"}
SEC = ["object", "part", "color", "material", "pattern_marking"]
CONTRASTS = [("H1 - H0 (multi-depth supervision)", "H1", "H0"), ("H2 - H0 (shared-encoder adaptation + L_glob)", "H2", "H0"),
             ("H3 - H2 (multi-depth on top of adaptation)", "H3", "H2"), ("H3 - H0 (both)", "H3", "H0")]
REFS = [("bil", "Bilinear"), ("mlr2", "Full ML+R2"), ("c0", "C0")]


def f(d, nd=2):
    if d is None or not np.isfinite(d.get("point", np.nan)):
        return "n/a"
    if not np.isfinite(d.get("lo", np.nan)):
        return f"{d['point']:.{nd}f}"
    return f"{d['point']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def p(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = a.run
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    Ytr_by = RC.ytr_by_level(data)
    B = Boot(dva, Yva, Ytr_by)
    nin = np.load(RC.NIN_SG.format("val"))
    bins = {nm: Boot(dva, Yva, Ytr_by, mask=(nin >= lo) & (nin <= hi)) for nm, lo, hi in (("1-4", 1, 4), ("5-16", 5, 16), ("17+", 17, 10 ** 9))}
    P = {}
    for k in PROBES:
        d = os.path.join(R, f"probe_{k}", "r2")
        dv = jl(os.path.join(d, "curves_dev.jsonl"))
        if not os.path.exists(os.path.join(R, f"probe_{k}", "done.json")) or not dv:
            P[k] = None
            continue
        ep = max(dv[1:] or dv, key=lambda r: r["dev"]["M0"]["overall"])["epoch"]
        S = np.load(os.path.join(d, f"scores_val_ep{ep}.npy")).astype(np.float32)
        fz = os.path.join(R, f"probe_{k}", "frozen_check.json")
        P[k] = dict(epoch=ep, dev_M0=max(r["dev"]["M0"]["overall"] for r in (dv[1:] or dv)), m0=B.m0(S), m2=B.m2(S),
                    bins={nm: bb.m0(S) for nm, bb in bins.items()}, frozen=json.load(open(fz)) if os.path.exists(fz) else None,
                    curves=jl(os.path.join(d, "curves.jsonl")), done=json.load(open(os.path.join(R, f"probe_{k}", "done.json"))))
    pre = {}
    for c in ("H0", "H1", "H2", "H3"):
        dn, cf, gc = (os.path.join(R, c, x) for x in ("done.json", "config.json", "grad_check.json"))
        pre[c] = dict(done=json.load(open(dn)) if os.path.exists(dn) else None, cfg=json.load(open(cf)) if os.path.exists(cf) else None,
                      gc=json.load(open(gc)) if os.path.exists(gc) else None, log=jl(os.path.join(R, c, "train_log.jsonl")))
    direct = json.load(open(os.path.join(R, "direct.json"))) if os.path.exists(os.path.join(R, "direct.json")) else {}

    L = ["# MSD pilot: Multi-depth Crop-to-Shared Distillation (COCO pre-training -> frozen PACO R2 probe)", "",
         "Protocol: `configs/msd/protocol.md`. Decoder pre-training on COCO train (existing 12k stream, 64 ROIs/step, seed 0, no PACO labels); "
         "checkpoint = best COCO-dev L12. PACO: encoder + decoder frozen, new R2 on PACO train (minus a 1,000-image dev for epoch selection), "
         "**final numbers on PACO VAL**; test never evaluated. Primary = val **M0 Overall**; CIs = paired image bootstrap (1,000).", ""]
    # ---------------- pre-training
    L += ["## 1. COCO pre-training (what was trained)", "",
          "| cond | encoder | supervision | trainable params | selected step (dev) | dev L12 | dev L4 | dev L8 | dev cos12 (raw) | time |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in ("H0", "H1", "H2", "H3"):
        x = pre[c]
        if not x["done"]:
            L.append(f"| {c} | - | - | - | FAILED / missing | | | | | |")
            continue
        s = x["done"]["selected"]
        enc = "frozen" if c in ("H0", "H1") else "adapters 4/8/12 (backbone frozen)"
        sup = "L12" + (" + 0.5 L8 + 0.5 L4" if c in ("H1", "H3") else "") + (" + 1.0 L_glob" if c in ("H2", "H3") else "")
        L.append(f"| {c} | {enc} | {sup} | {x['cfg']['trainable_total']:,} | {s['step']} | {s['dev']['L12']:.4f} | {s['dev']['L4']:.4f} | "
                 f"{s['dev']['L8']:.4f} | {s['dev']['cos12_raw']:.4f} | {x['done']['time_s'] / 60:.0f} min |")
    L += ["", "Dev L4 / L8 of H0 / H2 come from aux heads that receive no supervision (weight 0) and are not meaningful.", ""]
    L += ["**Gradient / update checks** (`<cond>/grad_check.json`, after DDP all-reduce and clipping):", "",
          "| cond | encoder backbone grad | decoder body grad (step 1) | enc adapter fc2 grad step 1 / step 2 | enc adapter fc1 grad step 3-5 (max) | aux-loss-only grad into decoder body | total weight change: body / enc adapters |",
          "|---|---|---|---|---|---|---|"]
    for c in ("H0", "H1", "H2", "H3"):
        g = pre[c]["gc"]
        if not g:
            continue
        s1, s2 = g["step1_grad_norm"], g["step2_grad_norm"]
        fc2_1 = [v for k, v in s1.items() if "fc2" in k]
        fc2_2 = [v for k, v in s2.items() if "fc2" in k]
        fc1 = [g.get(f"step{i}_grad_norm", {}).get(k) for i in (3, 4, 5) for k in s1 if "fc1" in k]
        fc1 = [v for v in fc1 if v is not None]
        wc = pre[c]["done"]["weight_change_norm_total"] if pre[c]["done"] else {}
        wa = sum(v for k, v in wc.items() if k.startswith("encoder_adapter"))
        L.append(f"| {c} | {s1['encoder_backbone (frozen, = teacher weights)']} | {p(s1['decoder_body (CoordMLP)'], 3)} | "
                 f"{('%.1e / %.1e' % (max(fc2_1), max(fc2_2))) if fc2_1 else 'n/a (no adapters)'} | {('%.1e' % max(fc1)) if fc1 else 'n/a'} | "
                 f"{p(g['aux_loss_only_grad_norm_into_decoder_body'], 3)} | {p(wc.get('decoder_body (CoordMLP)'), 2)} / {p(wa, 3) if fc2_1 else 'n/a'} |")
    L.append("")
    # ---------------- direct
    L += ["## 2. Direct capability checks (PACO val, 700 images, report only)", ""]
    if direct:
        L += ["| cond | L4 | L8 | L12 | cos12 raw | pair-cos pred / teacher | spatial var ratio | affinity gap | L12 obj 1-16 | L12 part | L12 multi-material | L12 patterned (non-plain) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in ["bil", "H0", "H1", "H2", "H3"]:
            if c not in direct:
                continue
            d = direct[c]
            al = d["all"]
            L.append(f"| {c} | {al['L4']:.3f} | {al['L8']:.3f} | {al['L12']:.3f} | {al['cos12_raw']:.3f} | {al['pair_cos_pred_b12']:.3f} / {al['pair_cos_teacher_b12']:.3f} | "
                     f"{al['var_ratio_b12']:.3f} | {al['affinity_gap_b12']:.3f} | {d['object n_in 1-16']['L12']:.3f} | {d['part']['L12']:.3f} | "
                     f"{d['material-positive ROI']['L12']:.3f} | {d['pattern-positive ROI']['L12']:.3f} |")
        L += ["", "Lower L / affinity gap = closer to the C0 crop teacher; var ratio 1 = teacher-like spatial variation. "
              "Multi-material = >= 2 material positives; patterned = a pattern positive other than plain. Bilinear L4/L8 = bilinear read of the original block 4/8 maps; H0/H2 L4/L8 come from unsupervised aux heads.", "",
              "**Global representation change vs the original encoder** (all 2,051 val images; ImageNet is not on this server):", "",
              "| cond | CLS cos | mean-patch cos | per-patch cos |", "|---|---|---|---|"]
        for c in ("H0", "H1", "H2", "H3"):
            if c in direct and "global" in direct[c]:
                gl = direct[c]["global"]
                L.append(f"| {c} | {gl['cls_cos']:.4f} | {gl['gap_cos']:.4f} | {gl['patch_cos']:.4f} |")
        L.append("")
    else:
        L += ["direct checks missing", ""]
    # ---------------- PACO
    L += ["## 3. PACO (frozen encoder + decoder, new R2; val at the dev-selected epoch)", "",
          "| probe | sel. epoch (dev M0) | **M0 Overall** | Object | Part | Color | Material | Pattern | M2 Overall | M0 n_in 1-4 | 5-16 | 17+ | frozen check |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in PROBES:
        x = P[k]
        if x is None:
            L.append(f"| {LABEL[k]} | missing / failed | | | | | | | | | | | |")
            continue
        m0 = x["m0"]
        b = [p(float(x["bins"][nm]["overall"][0])) if x["bins"][nm] is not None else "n/a" for nm in ("1-4", "5-16", "17+")]
        fz = x["frozen"]
        fzs = "n/a" if fz is None else ("ok (no grad, checksum equal)" if fz["unchanged"] and not fz["grads_present"] and not fz["requires_grad_any"] else "FAILED")
        L.append(f"| {LABEL[k]} | {x['epoch']} ({x['dev_M0']:.2f}) | **{m0['overall'][0]:.2f}** | " + " | ".join(f"{m0[g][0]:.2f}" for g in SEC) +
                 f" | {x['m2']['overall'][0]:.2f} | " + " | ".join(b) + f" | {fzs} |")
    L += ["", "**Paired contrasts (val M0; M2 overall last column)**", "",
          "| contrast | M0 Overall | Object | Part | Color | Material | Pattern | M2 Overall |", "|---|---|---|---|---|---|---|---|"]
    rows = CONTRASTS + [(f"{c} - {nm}", c, r) for c in ("H0", "H1", "H2", "H3") for r, nm in REFS]
    out = {"contrasts": {}}
    for nm, x, y in rows:
        if P.get(x) is None or P.get(y) is None:
            continue
        d = {g: summarize(P[x]["m0"][g], P[y]["m0"][g]) for g in ["overall"] + SEC}
        d2 = summarize(P[x]["m2"]["overall"], P[y]["m2"]["overall"])
        out["contrasts"][nm] = dict(m0=d, m2=d2)
        L.append(f"| {nm} | {f(d['overall'])} | " + " | ".join(f(d[g]) for g in SEC) + f" | {f(d2)} |")
    L.append("")
    # ---------------- verdicts
    L += ["## 4. Verdicts", ""]
    for nm, x, y in CONTRASTS[:3]:
        if nm not in out["contrasts"]:
            L.append(f"- **{nm}**: not available (missing run)")
            continue
        d = out["contrasts"][nm]["m0"]["overall"]
        dx, dy = direct.get(x, {}).get("all"), direct.get(y, {}).get("all")
        fm = None if not dx or not dy else dx["L12"] < dy["L12"]
        paco = d["lo"] > 0
        L.append(f"- **{nm}**: feature matching (val L12) improved = **{fm}** ({p(dy['L12'] if dy else None, 3)} -> {p(dx['L12'] if dx else None, 3)}); "
                 f"PACO val M0 Overall {f(d)} -> improved (CI lo > 0) = **{paco}**. Verdict: **{'success' if paco else 'not a success'}**"
                 + (" (feature matching improved but PACO did not -> not a success)" if fm and not paco else "") + ".")
    best_h = [k for k in ("H0", "H1", "H2", "H3") if P.get(k)]
    if best_h and P.get("bil"):
        L.append(f"- Is the spatial feature made by the shared encoder + decoder more useful than reading the shared map directly? "
                 + "; ".join(f"{k} - Bilinear {f(out['contrasts'].get(f'{k} - Bilinear', {}).get('m0', {}).get('overall'))}" for k in best_h)
                 + "; " + "; ".join(f"{k} - Full ML+R2 {f(out['contrasts'].get(f'{k} - Full ML+R2', {}).get('m0', {}).get('overall'))}" for k in best_h) + ".")
    L.append("")
    # ---------------- cost
    L += ["## 5. Cost", "", "| run | wall-clock | note |", "|---|---|---|"]
    for c in ("H0", "H1", "H2", "H3"):
        if pre[c]["done"]:
            L.append(f"| pretrain {c} | {pre[c]['done']['time_s'] / 60:.0f} min | peak mem {pre[c]['done']['peak_mem_gb']:.1f} GB / GPU, 12k steps x 64 ROIs |")
    for k in PROBES:
        if P.get(k):
            L.append(f"| probe {k} | {P[k]['done']['time_s'] / 60:.0f} min | 10 epochs R2 (run concurrently with pre-training) |")
    L += ["", "Inference (student): one encoder forward per image (+ adapters in H2/H3: 3 x 0.30M params) and one decoder pass per ROI "
          "(CoordMLP, 576 queries); no crop or per-ROI encoder pass. The C0 reference re-encodes every ROI.", ""]
    open(os.path.join(R, "REPORT.md"), "w").write("\n".join(L) + "\n")
    json.dump(dict(contrasts=out["contrasts"], probes={k: (None if v is None else dict(epoch=v["epoch"], dev_M0=v["dev_M0"],
                                                         M0={g: float(v["m0"][g][0]) for g in v["m0"]}, M2_overall=float(v["m2"]["overall"][0])))
                                                     for k, v in P.items()}, direct=direct),
              open(os.path.join(R, "report.json"), "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
