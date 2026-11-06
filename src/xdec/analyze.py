"""CPU analysis of the decoder-architecture pilot -> <run>/tables.md, <run>/results/analysis.json, <run>/figures/*.png.

Rules (protocol.md, fixed before results):
  * reader epoch per (method, reader) = max PACO-dev M0 Overall over epochs 1..10; val numbers at that epoch
  * M0 / M2 = src.readout.evaluate.Boot; D-bins = manifest n_in (F336 grid) D1-4 / D5-16 / D17+ (= IVC)
  * paired image bootstrap B = 1000, seed 0, same draws for every method (contrasts and the 2x2 interaction use the
    per-draw differences); 95% percentile CI
  * reused baselines: Bicubic, C0 (IVC probes, R0 + R2), Full Multi-layer + R2 (MSD probe_mlr2, R2 only)

  python -m src.xdec.analyze --run output/2026-11-05/xdec
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import src.readout.common as RC
from src.ivc.analyze import dev_split, f2, fci, load_probe
from src.ivc.common import DBINS
from src.readout.evaluate import Boot, summarize

EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
ECONDS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
COLS = [("Overall", "all", "overall"), ("Object", "all", "object"), ("Part", "all", "part"), ("Obj D5-16", "D5-16", "object"),
        ("Part D1-4", "D1-4", "part"), ("Obj D17+", "D17+", "object"), ("All D17+", "D17+", "overall"), ("Color", "all", "color"),
        ("Material", "all", "material"), ("Pattern", "all", "pattern_marking")]
MAIN_CONTRASTS = [("E1", "Bicubic"), ("E3", "E2"), ("E4", "E3"), ("E5", "E3"), ("E6", "E4"), ("E6", "E5"), ("E7", "E3")]
REFS = ["Direct Full", "Bicubic", "R00"]
LAB = {"B0_direct": "Direct Full", "bic": "Bicubic", "c0": "C0 Oracle", "mlr2": "Full Multi-layer"}


def sources(R):
    """method -> (probe root, probe name, readers)."""
    s = {"Direct Full": (os.path.join(R, "results"), "B0_direct", ("r0", "r2")),
         "Bicubic": ("output/2026-11-05/ivc", "bic", ("r0", "r2")),
         "R00": (os.path.join(R, "results"), "R00", ("r0", "r2"))}
    for c in ECONDS:
        s[c] = (os.path.join(R, "results"), c, ("r0", "r2"))
    s["Full Multi-layer"] = ("output/2026-11-05/msd", "mlr2", ("r2",))
    s["C0 Oracle"] = ("output/2026-11-05/ivc", "c0", ("r0", "r2"))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = a.run
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    os.makedirs(os.path.join(R, "results"), exist_ok=True)
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    Ytr_by = RC.ytr_by_level(data)
    ddev, _ = dev_split(data)
    N = len(dva)
    B = Boot(dva, Yva, Ytr_by)
    BB = {nm: Boot(dva, Yva, Ytr_by, mask=(dva.n_in.values >= lo) & (dva.n_in.values <= hi)) for nm, lo, hi in DBINS}
    SRC = sources(R)
    P, RES = {}, {}
    for m, (root, name, rds) in SRC.items():
        for rd in rds:
            p = load_probe(root, name, rd)
            if p is None:
                continue
            assert p["Sv"].shape == (N, 59) and p["Sd"].shape == (len(ddev), 59), (m, rd, p["Sv"].shape)
            P[(m, rd)] = p
            RES[(m, rd)] = dict(m0={"all": B.m0(p["Sv"]), **{nm: bb.m0(p["Sv"]) for nm, bb in BB.items()}},
                                m2={"all": B.m2(p["Sv"]), **{nm: bb.m2(p["Sv"]) for nm, bb in BB.items()}})
    methods = [m for m in SRC if any((m, rd) in P for rd in ("r0", "r2"))]

    def val(m, rd, metric, bin_, g):
        r = RES.get((m, rd))
        if r is None or r[metric][bin_] is None or g not in r[metric][bin_]:
            return None
        return r[metric][bin_][g]

    def pt(m, rd, metric, bin_, g):
        v = val(m, rd, metric, bin_, g)
        return None if v is None else float(v[0])

    def ctr(x, y, rd, metric, bin_, g):
        vx, vy = val(x, rd, metric, bin_, g), val(y, rd, metric, bin_, g)
        return None if vx is None or vy is None else summarize(vx, vy)

    # ---------------------------------------------------------------- decoder training records
    TR = {}
    for c in ECONDS + ["R00"]:
        d = os.path.join(R, "checkpoints", c)
        if os.path.exists(os.path.join(d, "done.json")):
            dj = json.load(open(os.path.join(d, "done.json")))
            cj = json.load(open(os.path.join(d, "config.json")))
            log = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl"))]
            gc = json.load(open(os.path.join(d, "grad_check.json"))) if os.path.exists(os.path.join(d, "grad_check.json")) else {}
            TR[c] = dict(done=dj, config=cj, log=log, grad=gc)
    REC = {}
    for split in ("coco_dev", "paco_val"):
        p = os.path.join(R, "results", "recon", f"{split}_roi_metrics.parquet")
        if os.path.exists(p):
            REC[split] = pd.read_parquet(p)
    smoke = {}
    for sp in ("smoke_gpu", "smoke_cpu"):
        p = os.path.join(R, sp, "smoke.json")
        if os.path.exists(p):
            smoke = json.load(open(p))
            break

    L, OUT = [], dict(methods=methods)
    same = None
    cfgs = {}
    for m, (root, name, _) in SRC.items():
        p = os.path.join(root, f"probe_{name}", "config.json")
        if os.path.exists(p):
            cfgs[m] = json.load(open(p))
    same = {k: sorted({json.dumps(v[k]) for v in cfgs.values() if k in v}) for k in ("n_train_rows", "n_val_rows", "n_val_images", "n_train_images")}
    L += ["# Decoder Architecture Pilot: tables (auto-generated)", "",
          f"PACO **val** only (test never evaluated). Val rows {N}; probe-config agreement (distinct values over all methods): "
          + ", ".join(f"{k}: {len(v)}" for k, v in same.items()) + ". Reader epoch = max PACO-dev M0 (1,000 train images held out).", ""]

    # ---------------------------------------------------------------- 1. main table
    def coco_dev(m):
        if m in TR:
            d = TR[m]["done"]["selected"]["dev"]
            return d["loss"], d["cos"], TR[m]["done"]["selected"]["step"]
        if m == "Bicubic" and "coco_dev" in REC:
            r = REC["coco_dev"][REC["coco_dev"].cond == "Bicubic"]
            return float(r.loss.mean()), float(r.tok_cos.mean()), None
        return None, None, None

    L += ["## 1. Main table (R2 primary)", "",
          "| Method | ckpt step | COCO-dev Feature Loss | COCO-dev Token Cos | PACO-val Token Cos | Val M0 R2 | Obj D5-16 (M0) | Val M2 R2 | R2 epoch |",
          "|---|---|---|---|---|---|---|---|---|"]
    order = ["Direct Full", "Bicubic", "R00"] + ECONDS + ["Full Multi-layer", "C0 Oracle"]
    OUT["main"] = {}
    for m in order:
        if m not in methods and m not in TR:
            L.append(f"| {m} | not available |  |  |  |  |  |  |  |")
            continue
        dl, dc, st = coco_dev(m)
        pv = None
        if "paco_val" in REC and (REC["paco_val"].cond == m).any():
            pv = float(REC["paco_val"][REC["paco_val"].cond == m].tok_cos.mean())
        if m == "C0 Oracle":
            dl, dc, pv = 0.0, 1.0, 1.0
        row = dict(step=st, dev_loss=dl, dev_cos=dc, paco_val_tok_cos=pv, M0=pt(m, "r2", "m0", "all", "overall"),
                   obj516=pt(m, "r2", "m0", "D5-16", "object"), M2=pt(m, "r2", "m2", "all", "overall"),
                   epoch=P[(m, "r2")]["epoch"] if (m, "r2") in P else None)
        OUT["main"][m] = row
        L.append(f"| {m} | {st if st is not None else '–'} | {f2(dl, 4) if dl is not None else '–'} | {f2(dc, 4) if dc is not None else '–'} | "
                 f"{f2(pv, 4) if pv is not None else '–'} | {f2(row['M0']) if row['M0'] is not None else 'n/a'} | "
                 f"{f2(row['obj516']) if row['obj516'] is not None else 'n/a'} | {f2(row['M2']) if row['M2'] is not None else 'n/a'} | {row['epoch'] or 'n/a'} |")
    L += ["", "Direct Full / Full Multi-layer read the full-image token grid (no crop-grid feature -> no reconstruction metric). "
          "Bicubic COCO-dev numbers come from the recon pass (same loss definition). C0 = teacher itself.", ""]

    # ---------------------------------------------------------------- 2. detailed M0 / M2 tables per reader
    for metric in ("m0", "m2"):
        for rd in ("r2", "r0"):
            L += [f"## 2.{'a' if metric == 'm0' else 'b'} {metric.upper()} ({rd.upper()}), val, dev-selected epoch", "",
                  "| Method | epoch (dev M0) | " + " | ".join(c[0] for c in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
            for m in order:
                if (m, rd) not in P:
                    continue
                cells = [f2(v) if (v := pt(m, rd, metric, b, g)) is not None else "n/a" for _, b, g in COLS]
                L.append(f"| {m} | {P[(m, rd)]['epoch']} ({P[(m, rd)]['dev_M0']:.2f}) | " + " | ".join(cells) + " |")
            L.append("")
    OUT["perf"] = {f"{m}/{rd}": dict(epoch=P[(m, rd)]["epoch"], dev_M0=P[(m, rd)]["dev_M0"],
                                      M0={cn: pt(m, rd, "m0", b, g) for cn, b, g in COLS}, M2={cn: pt(m, rd, "m2", b, g) for cn, b, g in COLS})
                   for (m, rd) in P}

    # ---------------------------------------------------------------- 3. contrasts
    def contrast_table(pairs, title, metric, rd, key):
        rows = []
        for x, y in pairs:
            cells = [ctr(x, y, rd, metric, b, g) for _, b, g in COLS]
            if all(c is None for c in cells):
                continue
            OUT.setdefault(key, {})[f"{x}-{y}/{rd}/{metric}"] = {cn: c for (cn, _, _), c in zip(COLS, cells)}
            rows.append(f"| {x} − {y} | " + " | ".join("n/a" if c is None else fci(c) for c in cells) + " |")
        if rows:
            L.extend([f"### {title} ({metric.upper()}, {rd.upper()})", "", "| Contrast | " + " | ".join(c[0] for c in COLS) + " |",
                      "|---|" + "---|" * len(COLS)] + rows + [""])

    L += [f"## 3. Paired image bootstrap (B=1000, seed 0, same draws): difference [95% CI]", ""]
    for metric in ("m0", "m2"):
        for rd in ("r2", "r0"):
            contrast_table(MAIN_CONTRASTS, "3.1 Main component contrasts", metric, rd, "contrasts_main")
    for metric in ("m0", "m2"):
        for rd in ("r2", "r0"):
            contrast_table([(e, r) for e in ECONDS + ["R00"] for r in REFS if e != r] + [(e, "Full Multi-layer") for e in ECONDS] +
                           [(e, "C0 Oracle") for e in ECONDS], "3.2 Every decoder vs references", metric, rd, "contrasts_refs")

    # ---------------------------------------------------------------- 4. 2x2
    L += ["## 4. 2x2 (E3 / E4 / E5 / E6): Query (Mask vs Bicubic) x Output (Direct vs Bicubic skip)", "",
          "|  | No Skip | Bicubic Skip |", "|---|---|---|"]
    m0 = lambda m: pt(m, "r2", "m0", "all", "overall")  # noqa: E731
    L += [f"| Mask Query | E3 {f2(m0('E3')) if m0('E3') is not None else 'n/a'} | E5 {f2(m0('E5')) if m0('E5') is not None else 'n/a'} |",
          f"| Bicubic Query | E4 {f2(m0('E4')) if m0('E4') is not None else 'n/a'} | E6 {f2(m0('E6')) if m0('E6') is not None else 'n/a'} |", ""]
    EFF = [("Query effect, no skip (E4 − E3)", (("E4", 1), ("E3", -1))), ("Query effect, with skip (E6 − E5)", (("E6", 1), ("E5", -1))),
           ("Skip effect, Mask query (E5 − E3)", (("E5", 1), ("E3", -1))), ("Skip effect, Bicubic query (E6 − E4)", (("E6", 1), ("E4", -1))),
           ("Interaction (E6 − E5) − (E4 − E3)", (("E6", 1), ("E5", -1), ("E4", -1), ("E3", 1)))]
    OUT["twobytwo"] = {}
    for metric in ("m0", "m2"):
        for rd in ("r2", "r0"):
            L += [f"### 2x2 effects ({metric.upper()}, {rd.upper()})", "", "| Effect | " + " | ".join(c[0] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
            for nm, terms in EFF:
                cells = []
                for cn, b, g in COLS:
                    vs = [val(m, rd, metric, b, g) for m, _ in terms]
                    if any(v is None for v in vs):
                        cells.append(None)
                        continue
                    x = sum(s * v for (_, s), v in zip(terms, vs))
                    cells.append(summarize(x))
                OUT["twobytwo"][f"{nm}/{rd}/{metric}"] = {cn: c for (cn, _, _), c in zip(COLS, cells)}
                L.append(f"| {nm} | " + " | ".join("n/a" if c is None else fci(c) for c in cells) + " |")
            L.append("")

    # ---------------------------------------------------------------- 5. reconstruction
    MET = ["loss", "mse", "tok_cos", "pool_cos", "e_cent", "cent_cos", "sv_ratio", "ss_diff"]
    if REC:
        L += ["## 5. Teacher-feature reconstruction (vs C0, raw final-LN space; lower loss / mse / e_cent / ss_diff = closer; "
              "sv_ratio 1 = teacher-like spatial variance)", ""]
        rec_order = [m for m in ["Bicubic", "R00"] + ECONDS if m in set(REC.get("paco_val", REC.get("coco_dev")).cond)]
        OUT["recon"] = {}
        for split, df in REC.items():
            dbin = np.select([df.n_in <= 4, df.n_in <= 16], ["D1-4", "D5-16"], "D17+")
            df = df.assign(dbin=dbin)
            grp = [("all", np.ones(len(df), bool))]
            if split == "paco_val":
                grp += [("object", df.part.values == 0), ("part", df.part.values == 1)]
                grp += [(f"{lv} {b}", (df.part.values == k) & (df.dbin.values == b)) for k, lv in ((0, "object"), (1, "part")) for b in ("D1-4", "D5-16", "D17+")]
            else:
                grp += [(b, df.dbin.values == b) for b in ("D1-4", "D5-16", "D17+")] + [("GT boxes", df.is_gt.values), ("random boxes", ~df.is_gt.values)]
            L += [f"### 5.{1 if split == 'coco_dev' else 2} {split} (all ROIs)", "", "| Method | " + " | ".join(MET) + " |", "|---|" + "---|" * len(MET)]
            for m in rec_order:
                s = df[df.cond == m]
                L.append(f"| {m} | " + " | ".join(f"{s[k].mean():.4f}" for k in MET) + " |")
            L.append("")
            L += [f"Token cos by group ({split}): ", "", "| Method | " + " | ".join(g for g, _ in grp) + " |", "|---|" + "---|" * len(grp)]
            for m in rec_order:
                mm = df.cond.values == m
                L.append(f"| {m} | " + " | ".join(f"{df.tok_cos.values[mm & g].mean():.4f}" if (mm & g).any() else "n/a" for _, g in grp) + " |")
            L.append("")
            L += [f"Centred spatial difference e_cent / self-similarity diff by group ({split}): ", "",
                  "| Method | " + " | ".join(g for g, _ in grp) + " |", "|---|" + "---|" * len(grp)]
            for m in rec_order:
                mm = df.cond.values == m
                L.append(f"| {m} | " + " | ".join(f"{df.e_cent.values[mm & g].mean():.3f} / {df.ss_diff.values[mm & g].mean():.3f}" if (mm & g).any() else "n/a" for _, g in grp) + " |")
            L.append("")
            OUT["recon"][split] = {m: {gname: {k: float(df[k].values[(df.cond.values == m) & g].mean()) for k in MET} for gname, g in grp} for m in rec_order}
        # Q7: across-method link between reconstruction and PACO
        if "paco_val" in OUT["recon"]:
            ms = [m for m in rec_order if pt(m, "r2", "m0", "all", "overall") is not None]
            if len(ms) >= 4:
                from scipy.stats import spearmanr
                link = {}
                for k in ("loss", "tok_cos", "pool_cos", "e_cent", "ss_diff"):
                    x = [OUT["recon"]["paco_val"][m]["all"][k] for m in ms]
                    for rd in ("r2", "r0"):
                        y = [pt(m, rd, "m0", "all", "overall") for m in ms]
                        if None in y:
                            continue
                        rho = spearmanr(x, y).correlation
                        link[f"{k}/{rd}"] = float(rho)
                OUT["recon_paco_link"] = dict(methods=ms, spearman=link)
                L += ["### 5.3 Across-method rank correlation: PACO-val reconstruction metric vs val M0 Overall", "",
                      f"Methods (n = {len(ms)}): {', '.join(ms)}. Spearman ρ across methods (no CI; n is small).", "",
                      "| metric | R2 | R0 |", "|---|---|---|"]
                for k in ("loss", "tok_cos", "pool_cos", "e_cent", "ss_diff"):
                    L.append(f"| {k} | {link.get(k + '/r2', float('nan')):+.2f} | {link.get(k + '/r0', float('nan')):+.2f} |")
                L.append("")

    # ---------------------------------------------------------------- 6. training / cost
    L += ["## 6. Training records and cost", "",
          "| Method | params | FLOPs / ROI (G) | latency ms / 32 ROIs | infer extra mem MB | train time | train peak mem GB | dev loss 3k / 6k / 9k / 12k | selected | step-0 Z == B | trunk grad upd1 / upd2 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    OUT["cost"] = {}
    for m in ECONDS + ["R00"]:
        sm = smoke.get("conds", {}).get(m, {})
        tr = TR.get(m)
        dv = tr["done"]["dev_at_ckpts"] if tr else {}
        g = tr["grad"] if tr else {}
        z0 = g.get("step0", {}).get("Z_equals_B", "–") if g else "–"
        u1 = g.get("step1_grad_norm", {}).get("all_before_last_linear") if g else None
        u2 = g.get("step2_grad_norm", {}).get("all_before_last_linear") if g else None
        OUT["cost"][m] = dict(params=sm.get("n_params"), gflops_roi=sm.get("flops_per_roi_G"), latency_ms=sm.get("latency_ms_per_batch"),
                              time_s=tr["done"]["time_s"] if tr else None, peak_gb=tr["done"]["peak_mem_gb"] if tr else None)
        L.append(f"| {m} | {sm.get('n_params', 'n/a'):,} | {sm.get('flops_per_roi_G', float('nan')):.2f} | "
                 f"{sm.get('latency_ms_per_batch', float('nan')):.1f} | {sm.get('inference_peak_extra_mem_MB', float('nan')):.0f} | "
                 f"{(str(round(tr['done']['time_s'] / 60)) + ' min') if tr else 'n/a'} | {f2(tr['done']['peak_mem_gb']) if tr else 'n/a'} | "
                 + (" / ".join(f"{dv[k]['loss']:.4f}" for k in sorted(dv, key=int)) if dv else "n/a")
                 + f" | {tr['done']['selected']['step'] if tr else 'n/a'} | {z0} | "
                 + (f"{u1:.2g} / {u2:.2g}" if u1 is not None and u2 is not None else "n/a") + " |")
    L += ["", f"Latency / memory measured on {smoke.get('device', 'n/a')} (one H200, BF16, batch = 32 ROIs of 8 COCO images, decoder only; "
          "the frozen encoder forward is shared by all methods and not included). FLOPs = torch FlopCounterMode, decoder forward only.", ""]

    # ---------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for m, tr in TR.items():
            tl = [r for r in tr["log"] if "loss" in r and "dev" not in r]
            dl = [r for r in tr["log"] if "dev" in r]
            ax[0].plot([r["step"] for r in tl], [r["loss"] for r in tl], label=m, lw=1)
            ax[1].plot([r["step"] for r in dl], [r["dev"]["loss"] for r in dl], marker="o", ms=3, label=m)
        ax[0].set_title("train feature loss (50-step mean)"); ax[1].set_title("COCO-dev feature loss")
        for x in ax:
            x.set_xlabel("step"); x.legend(fontsize=7); x.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "train_dev_curves.png"), dpi=130); plt.close(fig)
        if OUT.get("main"):
            ms = [m for m in order if OUT["main"].get(m, {}).get("M0") is not None]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(ms)), [OUT["main"][m]["M0"] for m in ms], color=["#888" if m in ("Direct Full", "Bicubic", "R00", "Full Multi-layer", "C0 Oracle") else "#3b6ea8" for m in ms])
            ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=30, fontsize=8); ax.set_ylabel("val M0 Overall (R2)")
            lo = min(OUT["main"][m]["M0"] for m in ms) - 1.5
            ax.set_ylim(lo, max(OUT["main"][m]["M0"] for m in ms) + 1); ax.grid(axis="y", alpha=.3)
            fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "val_m0_r2.png"), dpi=130); plt.close(fig)
    except Exception as e:  # figures are optional
        L.append(f"(figure generation failed: {e})")
    open(os.path.join(R, "tables.md"), "w").write("\n".join(L) + "\n")
    json.dump(OUT, open(os.path.join(R, "results", "analysis.json"), "w"), indent=1, default=float)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
