"""R2 10-epoch completion for the decoder study (user request 2026-11-08).

Exact resume from the epoch-6 R2 checkpoints is impossible (they hold only the reader state_dict: no AdamW moments, no
torch / worker RNG for the hflip draws), so B0 / D1 / L1 / L2 R2+heads are RE-TRAINED from scratch with the same yaml,
seed and 10-epoch cosine schedule (src.decstudy.probe --stop_epoch 10) into <run>/r2full/probes/<c>. Encoders / decoders
unchanged. Existing G1-cross Encoder Only / Decoder probes (already 10 epochs) are NOT re-trained.

  python -m src.decstudy.r2full select            -> r2full/selection.json + "name epoch" lines (val inference list)
  python -m src.decstudy.r2full report            -> r2full/results_r2full.md / .json
Selection rule (fixed earlier, src.paco_devap): max (Dev Obj AP + Dev Part AP) / 2 over epochs 1..10, ties -> earlier.
"""
import glob
import json
import os
import re
import sys

import numpy as np

RR = "output/2026-11-07/decoder_study"
RF = f"{RR}/r2full"
G1 = "output/2026-11-07/g1clip"
CONDS = {  # name -> (probe dir, dev prediction root, label)
    "enc": (f"{G1}/probes/g1cross_enc", "output/2026-11-07/paco_devap/preds/g1cross_enc", "G1-cross Encoder Only"),
    "dec": (f"{G1}/probes/g1cross_dec", f"{RR}/preds_dev/g1cross_dec", "Existing G1-cross Decoder"),
    "b0": (f"{RF}/probes/b0", f"{RF}/preds_dev/b0", "B0 Bilinear Only"),
    "d1": (f"{RF}/probes/d1", f"{RF}/preds_dev/d1", "D1 Bilinear + Residual"),
    "l1": (f"{RF}/probes/l1", f"{RF}/preds_dev/l1", "L1 D1 + Covariance"),
    "l2": (f"{RF}/probes/l2", f"{RF}/preds_dev/l2", "L2 D1 + Relational"),
}
OLD6 = {"b0": f"{RR}/probes/b0", "d1": f"{RR}/probes/d1", "l1": f"{RR}/probes/l1", "l2": f"{RR}/probes/l2"}
TYPES = ("color", "pattern_marking", "material", "transparency")


def dev_curve(root):
    out = {}
    for f in glob.glob(os.path.join(root, "ep*", "official_dev.json")):
        e = int(re.search(r"ep(\d+)", f).group(1))
        o = json.load(open(f))
        out[e] = dict(obj=o["obj"], part=o["part"], avg=(o["obj"] + o["part"]) / 2,
                      **{f"obj_{t}": o["obj_types"][t] for t in TYPES}, **{f"part_{t}": o["part_types"][t] for t in TYPES})
    return dict(sorted(out.items()))


def m0_curve(pdir):
    return {json.loads(l)["epoch"]: json.loads(l)["dev"]["M0"]["overall"] for l in open(os.path.join(pdir, "r2", "curves_dev.jsonl")) if json.loads(l)["epoch"] >= 1}


def best(curve):
    return max(sorted(curve), key=lambda e: (curve[e]["avg"], -e))


def select():
    sel = {}
    for n, (pdir, droot, _) in CONDS.items():
        c = dev_curve(droot)
        assert sorted(c) == list(range(1, 11)), (n, sorted(c))
        b = best(c)
        m0 = m0_curve(pdir)
        sel[n] = dict(devap_best=b, devap_best_avg=c[b]["avg"], m0_best=max(sorted(m0), key=lambda e: (m0[e], -e)), probe=pdir)
        print(n, b)
        if b != 10:
            print(n, 10)
    json.dump(dict(rule="max (Dev Obj AP + Dev Part AP)/2 over epochs 1..10, ties -> earlier", conds=sel), open(f"{RF}/selection.json", "w"), indent=1)


def ci(boot, a, b, k):
    if not boot or a not in boot["boot"] or b not in boot["boot"]:
        return None
    d = np.array(boot["boot"][a][k]) - np.array(boot["boot"][b][k])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=boot["points"][a][k] - boot["points"][b][k], lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0))


def fci(c):
    return "n/a" if c is None else f"{c['delta']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]" + ("\\*" if c["sig"] else "")


def report():
    sel = json.load(open(f"{RF}/selection.json"))["conds"]
    bv = json.load(open(f"{RF}/boot_val.json")) if os.path.exists(f"{RF}/boot_val.json") else None
    bd = json.load(open(f"{RF}/boot_dev.json")) if os.path.exists(f"{RF}/boot_dev.json") else None
    curves = {n: dev_curve(d) for n, (_, d, _) in CONDS.items()}
    m0 = {n: m0_curve(p) for n, (p, _, _) in CONDS.items()}
    L = []
    w = L.append
    w("# Decoder study — R2 10-epoch 완료 결과 (2026-11-08)")
    w("")
    w("- B0 / D1 / L1 / L2의 R2+head를 같은 seed와 원래 10-epoch cosine 설정으로 **처음부터 재학습**했다.")
    w("  - 이유: epoch-6 checkpoint에 optimizer / RNG 상태가 없어 정확한 resume이 불가능했다.")
    w("  - Encoder와 Decoder checkpoint는 그대로 썼다.")
    w("  - 기존 Encoder Only / 기존 Decoder probe는 재학습하지 않았다(원래 10 epoch까지 학습돼 있음).")
    w("- Dev AP는 PACO train 내부 dev 1,000장에서 공식 evaluator로 계산했다(Official-protocol Dev AP).")
    w("  - Best 규칙: (Obj + Part)/2 최대, 동점이면 앞 epoch.")
    w("- Val은 공식 Val 2,410장이다. Test는 새로 실행하지 않았다.")
    w("- \\* = paired image bootstrap(B=1000) 95% CI가 0을 포함하지 않음.")
    w("")
    w("## 1. Dev AP (Obj / Part) per epoch")
    w("")
    w("| Model | " + " | ".join(f"ep{e}" for e in range(1, 11)) + " | Dev-AP best | M0 best |")
    w("|---|" + "---|" * 10 + "---|---|")
    for n, (_, _, lab) in CONDS.items():
        c = curves[n]
        cells = [(f"**{c[e]['obj']:.1f}/{c[e]['part']:.1f}**" if e == sel[n]["devap_best"] else f"{c[e]['obj']:.1f}/{c[e]['part']:.1f}") if e in c else "–" for e in range(1, 11)]
        w(f"| {lab} | " + " | ".join(cells) + f" | ep{sel[n]['devap_best']} ({sel[n]['devap_best_avg']:.2f}) | ep{sel[n]['m0_best']} |")
    w("")
    w("Dev M0 per epoch:")
    w("")
    w("| Model | " + " | ".join(f"ep{e}" for e in range(1, 11)) + " |")
    w("|---|" + "---|" * 10)
    for n, (_, _, lab) in CONDS.items():
        w(f"| {lab} | " + " | ".join(f"{m0[n].get(e, float('nan')):.2f}" for e in range(1, 11)) + " |")
    w("")
    # reproducibility of the retrained ep6
    w("## 2. 재학습 재현성: 새 run의 ep6 vs 기존 ep6 run")
    w("")
    w("| Model | 기존 ep6 Dev M0 | 새 ep6 Dev M0 | 기존 ep6 Dev AP Obj/Part | 새 ep6 Dev AP Obj/Part |")
    w("|---|---:|---:|---|---|")
    for n in ("b0", "d1", "l1", "l2"):
        o6 = m0_curve(OLD6[n]).get(6)
        od = json.load(open(f"{RR}/preds_dev/{n}/ep6/official_dev.json"))
        nd = curves[n].get(6)
        w(f"| {CONDS[n][2]} | {o6:.2f} | {m0[n].get(6, float('nan')):.2f} | {od['obj']:.2f}/{od['part']:.2f} | {nd['obj']:.2f}/{nd['part']:.2f} |")
    w("")
    for tag, title in (("ep10", "고정 Epoch 10"), ("best", "Dev AP Best Checkpoint")):
        w(f"## 3{'a' if tag == 'ep10' else 'b'}. {title} — 공식 Val AP")
        w("")
        w("| Model | epoch | Obj AP | Obj Color | Obj Pattern | Obj Material | Obj Refl.* | Part AP | Part Color | Part Pattern | Part Material | Part Refl.* |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n, (_, _, lab) in CONDS.items():
            e = 10 if tag == "ep10" else sel[n]["devap_best"]
            k = f"{n}_ep{e}"
            p = (bv or {}).get("points", {}).get(k)
            if p is None:
                w(f"| {lab} | {e} | " + " | ".join(["n/a"] * 10) + " |")
                continue
            w(f"| {lab} | {e} | {p['obj']:.2f} | " + " | ".join(f"{p['obj_' + t]:.2f}" for t in TYPES) + f" | {p['part']:.2f} | " + " | ".join(f"{p['part_' + t]:.2f}" for t in TYPES) + " |")
        w("")
        w(f"Dev AP ({title}): " + ", ".join(f"{CONDS[n][2]} ep{10 if tag == 'ep10' else sel[n]['devap_best']} {curves[n][10 if tag == 'ep10' else sel[n]['devap_best']]['obj']:.2f}/{curves[n][10 if tag == 'ep10' else sel[n]['devap_best']]['part']:.2f}" for n in CONDS))
        w("")
    w("## 4. 핵심 비교 (Val, paired bootstrap 95% CI; Dev는 Obj/Part만)")
    w("")
    pairs = [("Existing Decoder − Encoder Only", "dec", "enc"), ("D1 − Encoder Only", "d1", "enc"), ("L1 − Encoder Only", "l1", "enc"), ("L2 − Encoder Only", "l2", "enc"),
             ("B0 − Encoder Only", "b0", "enc"), ("D1 − B0", "d1", "b0"), ("L1 − D1", "l1", "d1"), ("L2 − D1", "l2", "d1"), ("D1 − Existing Decoder", "d1", "dec")]
    out = {}
    for tag in ("ep10", "best"):
        w(f"### {'고정 Epoch 10' if tag == 'ep10' else 'Dev AP Best'}")
        w("")
        w("| Contrast | Val Obj | Val Part | Val Obj Material | Val Part Pattern | Dev Obj | Dev Part |")
        w("|---|---|---|---|---|---|---|")
        for lab, a, b in pairs:
            ea = 10 if tag == "ep10" else sel[a]["devap_best"]
            eb = 10 if tag == "ep10" else sel[b]["devap_best"]
            ka, kb = f"{a}_ep{ea}", f"{b}_ep{eb}"
            r = {k: ci(bv, ka, kb, k) for k in ("obj", "part", "obj_material", "part_pattern_marking")}
            rd = {k: ci(bd, ka, kb, k) for k in ("obj", "part")}
            out[f"{tag}:{lab}"] = dict(val=r, dev=rd, epochs=[ea, eb])
            w(f"| {lab} (ep{ea} vs ep{eb}) | {fci(r['obj'])} | {fci(r['part'])} | {fci(r['obj_material'])} | {fci(r['part_pattern_marking'])} | {fci(rd['obj'])} | {fci(rd['part'])} |")
        w("")
    # effect of the incomplete R2: ep6 vs ep10 vs best, per decoder path (Dev AP; Val for ep6 = original runs)
    w("## 5. R2 미완료(ep6)의 영향")
    w("")
    w("| Model | Dev ep6 (Obj/Part) | Dev ep10 | Dev best | Val ep6 (기존 보고) | Val ep10 | Val best |")
    w("|---|---|---|---|---|---|---|")
    v6 = json.load(open(f"{RR}/boot/val.json"))["points"]
    v6k = dict(enc="g1cross_enc", dec="g1cross_dec", b0="b0", d1="d1", l1="l1", l2="l2")
    for n, (_, _, lab) in CONDS.items():
        c = curves[n]
        b = sel[n]["devap_best"]
        pv = (bv or {}).get("points", {})
        f = lambda p: f"{p['obj']:.2f}/{p['part']:.2f}" if p else "n/a"  # noqa: E731
        w(f"| {lab} | {f(c.get(6))} | {f(c.get(10))} | ep{b} {f(c.get(b))} | {f(v6.get(v6k[n]))} | {f(pv.get(f'{n}_ep10'))} | {f(pv.get(f'{n}_ep{b}'))} |")
    w("")
    w("(Val ep6 of B0/D1/L1/L2 = the ORIGINAL ep6 runs; the retrained runs' ep6 is only on Dev, section 2.)")
    w("")
    txt = "\n".join(L) + "\n"
    open(f"{RF}/results_r2full.md", "w").write(txt)
    json.dump(dict(selection=sel, dev_curves=curves, m0=m0, contrasts=out), open(f"{RF}/results_r2full.json", "w"), indent=1, default=float)
    print(txt)


if __name__ == "__main__":
    {"select": select, "report": report}[sys.argv[1]]()
