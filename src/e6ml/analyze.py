"""(copy of src.e6dir.analyze, sources / contrasts changed) E6 multi-layer memory pilot analysis (CPU) -> <run>/tables.md, paco_results.csv, bootstrap_results.csv, figures/.
Bicubic-only = IVC probe_bic (R0 + R2), E6-original = xdec probe_E6 (R0 + R2), E6-direction = this run (R0 + R2); same
src.ivc.train protocol, epoch = max PACO-dev M0. Paired image bootstrap B = 1000, seed 0, same draws.

  python -m src.e6dir.analyze --run output/2026-11-06/e6_direction
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import src.readout.common as RC
from src.ivc.analyze import dev_split, f2, fci, load_probe
from src.readout.evaluate import Boot, summarize

EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
COLS = [("Overall", "all", "overall"), ("Obj D5-16", "D5-16", "object"), ("Part D1-4", "D1-4", "part"), ("Obj D1-4", "D1-4", "object"),
        ("Obj D17+", "D17+", "object"), ("Object", "all", "object"), ("Part", "all", "part"), ("Color", "all", "color"),
        ("Material", "all", "material"), ("Pattern", "all", "pattern_marking")]
DB = (("D1-4", 1, 4), ("D5-16", 5, 16), ("D17+", 17, 10 ** 9))
CONTRASTS = [("Multi-layer Memory (B)", "Matched L12 (A)"), ("Multi-layer Memory (B)", "E6-original"), ("Multi-layer Memory (B)", "Bicubic"),
             ("Multi-layer Memory (B)", "Full Multi-layer + R2"), ("Matched L12 (A)", "E6-original"), ("Full Multi-layer + R2", "Bicubic")]
MET = ["mse", "tok_cos", "c_dir", "pool_cos", "e_mean", "sv_ratio", "e_cent", "loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = a.run
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    Ytr_by = RC.ytr_by_level(data)
    BB = {"all": Boot(dva, Yva, Ytr_by)}
    for nm, lo, hi in DB:
        BB[nm] = Boot(dva, Yva, Ytr_by, mask=(dva.n_in.values >= lo) & (dva.n_in.values <= hi))
    SRC = {"Bicubic": ("output/2026-11-05/ivc", "bic"), "Full Multi-layer + R2": ("output/2026-11-05/msd", "mlr2"),
           "E6-original": ("output/2026-11-05/xdec/results", "E6"), "Matched L12 (A)": (os.path.join(R, "results"), "A"),
           "Multi-layer Memory (B)": (os.path.join(R, "results"), "B")}
    P, RES, rows = {}, {}, []
    cfg = {}
    for m, (root, name) in SRC.items():
        if os.path.exists(os.path.join(root, f"probe_{name}", "config.json")):
            cfg[m] = json.load(open(os.path.join(root, f"probe_{name}", "config.json")))
        for rd in ("r2", "r0"):
            p = load_probe(root, name, rd)
            if p is None:
                continue
            P[(m, rd)] = p
            RES[(m, rd)] = {met: {b: getattr(bb, met)(p["Sv"]) for b, bb in BB.items()} for met in ("m0", "m2")}
            for met in ("m0", "m2"):
                for cn, b, g in COLS:
                    v = RES[(m, rd)][met][b]
                    rows.append(dict(method=m, reader=rd.upper(), metric=met.upper(), column=cn, epoch=p["epoch"], dev_M0=p["dev_M0"],
                                     value=float(v[g][0]) if v is not None and g in v else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(R, "paco_results.csv"), index=False)

    def val(m, rd, met, b, g):
        r = RES.get((m, rd))
        return None if r is None or r[met][b] is None or g not in r[met][b] else r[met][b][g]

    def pv(m, rd="r2", met="m0", b="all", g="overall"):
        v = val(m, rd, met, b, g)
        return None if v is None else float(v[0])

    brows = []
    for x, y in CONTRASTS:
        for rd in ("r2", "r0"):
            for met in ("m0", "m2"):
                for cn, b, g in COLS:
                    vx, vy = val(x, rd, met, b, g), val(y, rd, met, b, g)
                    if vx is not None and vy is not None:
                        brows.append(dict(contrast=f"{x} - {y}", reader=rd.upper(), metric=met.upper(), column=cn, **summarize(vx, vy)))
    pd.DataFrame(brows).to_csv(os.path.join(R, "bootstrap_results.csv"), index=False)
    same = {k: len({json.dumps(c.get(k)) for c in cfg.values()}) == 1 for k in ("n_train_rows", "n_val_rows", "n_val_images", "n_train_images")}
    L = ["# E6 + Centered Direction Loss: tables (auto-generated)", "",
         f"PACO **val** only (test never evaluated). Probe-config agreement across the 3 methods: {same}. Reader epoch = max PACO-dev M0.", ""]
    rec = {}
    for split in ("paco_val", "coco_dev"):
        p = os.path.join(R, "results", "recon", f"{split}_roi_metrics.parquet")
        if os.path.exists(p):
            rec[split] = pd.read_parquet(p)
    L += ["## 1. Main table", "", "| Method | Token MSE | Token Cos | **Centered Dir Cos** | Pooled Cos | Mean-feat MSE | Spatial Var / C0 | **R2 M0** | Obj D5–16 | Part D1–4 | R2 M2 | R0 M0 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in SRC:
        r = rec.get("paco_val")
        s = r[r.cond == m] if r is not None else None
        g = (lambda k: f"{s[k].mean():.4f}" if s is not None and len(s) else "n/a")
        L.append(f"| {m} | {g('mse')} | {g('tok_cos')} | {g('c_dir')} | {g('pool_cos')} | {g('e_mean')} | {g('sv_ratio')} | "
                 + " | ".join(f2(v) if (v := x) is not None else "n/a" for x in (pv(m), pv(m, b='D5-16', g='object'), pv(m, b='D1-4', g='part'),
                                                                                    pv(m, met='m2'), pv(m, rd='r0'))) + " |")
    L += ["", "Representation metrics: PACO val, all 11,109 ROIs, per-ROI coverage-weighted, mean over ROIs (c_dir = the training definition).", ""]
    for met in ("m0", "m2"):
        for rd in ("r2", "r0"):
            L += [f"## 2. {met.upper()} ({rd.upper()})", "", "| Method | epoch (dev M0) | " + " | ".join(c[0] for c in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
            for m in SRC:
                if (m, rd) in P:
                    L.append(f"| {m} | {P[(m, rd)]['epoch']} ({P[(m, rd)]['dev_M0']:.2f}) | " + " | ".join(f2(v) if (v := pv(m, rd, met, b, g)) is not None else "n/a" for _, b, g in COLS) + " |")
            L.append("")
            L += [f"Paired bootstrap ({met.upper()}, {rd.upper()}): difference [95% CI]", "", "| Contrast | " + " | ".join(c[0] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
            for x, y in CONTRASTS:
                cells = [("n/a" if (vx := val(x, rd, met, b, g)) is None or (vy := val(y, rd, met, b, g)) is None else fci(summarize(vx, vy))) for _, b, g in COLS]
                L.append(f"| {x} − {y} | " + " | ".join(cells) + " |")
            L.append("")
    for split, df in rec.items():
        dbin = np.select([df.n_in <= 4, df.n_in <= 16], ["D1-4", "D5-16"], "D17+")
        grp = [("all", np.ones(len(df), bool))]
        if split == "paco_val":
            grp += [("object", df.part.values == 0), ("part", df.part.values == 1)] + [(f"{lv} {b}", (df.part.values == k) & (dbin == b)) for k, lv in ((0, "object"), (1, "part")) for b in ("D1-4", "D5-16", "D17+")]
        else:
            grp += [(b, dbin == b) for b in ("D1-4", "D5-16", "D17+")]
        L += [f"## 3. Representation metrics: {split}", "", "| Method | " + " | ".join(MET) + " |", "|---|" + "---|" * len(MET)]
        for m in SRC:
            s = df[df.cond == m]
            if len(s):
                L.append(f"| {m} | " + " | ".join(f"{s[k].mean():.4f}" for k in MET) + " |")
        L += ["", f"Centered Dir Cos by group ({split}):", "", "| Method | " + " | ".join(gn for gn, _ in grp) + " |", "|---|" + "---|" * len(grp)]
        for m in SRC:
            mm = df.cond.values == m
            if mm.any():
                L.append(f"| {m} | " + " | ".join(f"{df.c_dir.values[mm & gm].mean():.4f}" for _, gm in grp) + " |")
        L += ["", f"Spatial Var / C0 by group ({split}):", "", "| Method | " + " | ".join(gn for gn, _ in grp) + " |", "|---|" + "---|" * len(grp)]
        for m in SRC:
            mm = df.cond.values == m
            if mm.any():
                L.append(f"| {m} | " + " | ".join(f"{df.sv_ratio.values[mm & gm].mean():.4f}" for _, gm in grp) + " |")
        L.append("")
        for xa, xb in (("Multi-layer Memory (B)", "Matched L12 (A)"), ("Multi-layer Memory (B)", "E6-original"), ("Matched L12 (A)", "E6-original")):
          if xa in set(df.cond) and xb in set(df.cond):     # paired image bootstrap of the c_dir difference
            a_ = df[df.cond == xa].reset_index(drop=True)
            b_ = df[df.cond == xb].reset_index(drop=True)
            d = a_.c_dir.values - b_.c_dir.values
            key = a_["row"].values if "row" in a_ else np.arange(len(a_))
            img = dva.image_id.values[key] if split == "paco_val" else np.arange(len(a_)) // 4
            uq, inv = np.unique(img, return_inverse=True)
            rng = np.random.default_rng(0)
            bs = [np.average(d, weights=np.bincount(rng.integers(0, len(uq), len(uq)), minlength=len(uq))[inv] + 1e-12) for _ in range(1000)]
            for k in ("c_dir", "tok_cos", "mse"):
                d = a_[k].values - b_[k].values
                rng = np.random.default_rng(0)
                bs = [np.average(d, weights=np.bincount(rng.integers(0, len(uq), len(uq)), minlength=len(uq))[inv] + 1e-12) for _ in range(1000)]
                L += [f"{k}({xa}) − {k}({xb}), {split}: {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] (image bootstrap 1000, seed 0)", ""]
    # training records
    cal = json.load(open(os.path.join(R, "calib", "calib.json"))) if os.path.exists(os.path.join(R, "calib", "calib.json")) else None
    if cal:
        L += ["## 4. lambda_dir calibration", "", f"lambda_dir = **{cal['lambda_dir']:.4f}** = 0.5 x median ratio {cal['median_ratio']:.4f} "
              f"(min {cal['ratio_min']:.3f}, max {cal['ratio_max']:.3f}, CV {cal['ratio_cv']:.3f}); median ||dL_old/dS|| {cal['g_old_median']:.3e}, "
              f"||dL_dir/dS|| {cal['g_dir_median']:.3e}, median grad cos {cal['grad_cos_median']:.3f}; teacher-energy threshold {cal['thr_eT']:.4g} "
              f"(1e-3 x median {cal['eT_median']:.4g}; min e_T {cal['eT_min']:.4g}; excluded {cal['n_excluded']} / {cal['n_roi']}); model unchanged {cal['model_unchanged']}, "
              f"RNG unchanged {cal['rng_unchanged']}, device {cal['device']}.", "",
              "| batch | n ROI | L_old | L_dir | c_dir | g_old | g_dir | ratio | grad cos |", "|---|---|---|---|---|---|---|---|---|"]
        for b in cal["batches"]:
            L.append(f"| {b['batch']} | {b['n_roi']} | {b['L_old']:.4f} | {b['L_dir']:.4f} | {b['c_dir_mean']:.4f} | {b['g_old']:.3e} | {b['g_dir']:.3e} | {b['ratio']:.3f} | {b['grad_cos']:.3f} |")
        L.append("")
    logs = {}
    for m, p in (("E6-original", "output/2026-11-05/xdec/checkpoints/E6"), ("Matched L12 (A)", os.path.join(R, "checkpoints", "A")),
                 ("Multi-layer Memory (B)", os.path.join(R, "checkpoints", "B"))):
        if os.path.exists(os.path.join(p, "train_log.jsonl")):
            logs[m] = [json.loads(l) for l in open(os.path.join(p, "train_log.jsonl"))]
            if os.path.exists(os.path.join(p, "done.json")):
                dj = json.load(open(os.path.join(p, "done.json")))
                L += [f"### {m}: COCO-dev at checkpoints (selection = min L_old): selected step {dj['selected']['step']}", "",
                      "| step | L_old (dev) | MSE | token cos | L_dir | L_new | c_dir | excluded |", "|---|---|---|---|---|---|---|---|"]
                for st_, d in sorted(dj["dev_at_ckpts"].items(), key=lambda x: int(x[0])):
                    L.append(f"| {st_} | {d['loss']:.4f} | {d['mse']:.4f} | {d['cos']:.4f} | {d.get('L_dir', float('nan')):.4f} | {d.get('L_new', float('nan')):.4f} | {d.get('c_dir', float('nan')):.4f} | {d.get('n_excluded_dir', '–')} |")
                L += ["", f"time {dj['time_s']} s, peak mem {dj['peak_mem_gb']:.2f} GB", ""]
    if "E6-direction" in logs:
        tr = [r for r in logs["E6-direction"] if "L_dir" in r and "dev" not in r]
        L += ["### E6-direction train log (50-step means, every 1,500 steps)", "", "| step | L_old | L_dir | L_new | c_dir | token cos | grad norm | excluded/step |", "|---|---|---|---|---|---|---|---|"]
        for r in tr[::30] + tr[-1:]:
            L.append(f"| {r['step']} | {r['loss']:.4f} | {r['L_dir']:.4f} | {r['L_new']:.4f} | {r['c_dir']:.4f} | {r['cos']:.4f} | {r['grad_norm']:.3f} | {r['n_excluded_dir_per_step']:.2f} |")
        L.append("")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            for m, lg in logs.items():
                t = [r for r in lg if "loss" in r and "dev" not in r]
                dv = [r for r in lg if "dev" in r]
                ax[0].plot([r["step"] for r in t], [r["loss"] for r in t], label=f"{m} train L_old", lw=1)
                ax[0].plot([r["step"] for r in dv], [r["dev"]["loss"] for r in dv], "o-", ms=3, label=f"{m} dev L_old")
                if m == "E6-direction":
                    ax[1].plot([r["step"] for r in t], [r["L_dir"] for r in t], label="train L_dir", lw=1)
                    ax[1].plot([r["step"] for r in dv], [r["dev"]["L_dir"] for r in dv], "o-", ms=3, label="dev L_dir")
                    ax[2].plot([r["step"] for r in t], [r["c_dir"] for r in t], label="train c_dir", lw=1)
                    ax[2].plot([r["step"] for r in dv], [r["dev"]["c_dir"] for r in dv], "o-", ms=3, label="dev c_dir")
            for x, tt in zip(ax, ("L_old (train, COCO dev)", "(unused)", "(unused)")):
                x.set_title(tt); x.set_xlabel("step"); x.legend(fontsize=7); x.grid(alpha=.3)
            fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "train_curves.png"), dpi=130); plt.close(fig)
        except Exception as e:
            L.append(f"(figure failed: {e})")
    open(os.path.join(R, "tables.md"), "w").write("\n".join(L) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
