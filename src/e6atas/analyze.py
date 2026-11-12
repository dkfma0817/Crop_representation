"""E6 ATAS-inspired loss ablation: analysis (CPU) -> <run>/tables.md, paco_results.csv, bootstrap_results.csv,
representation_results.csv, results/retrieval.json, figures/.
PACO: Bicubic = IVC probe_bic, A = E6-original = xdec probe_E6 (same model, 12k), B / C / D = this run, Full ML + R2 = MSD
probe_mlr2 (R2), C0 = IVC probe_c0. Epoch = max PACO-dev M0. Paired image bootstrap B = 1000, seed 0, same draws.
Region margin / retrieval: split-wide (PACO val 11,109 ROIs; COCO dev 1,872) with the training false-negative rule
(same image & IoU >= 0.5, or teacher-teacher cosine >= 0.95):
  margin_r = cos(s_r, t_r) - mean_{allowed negatives j} cos(s_r, t_j);  top1_r = [argmax over {r} U allowed == r].

  python -m src.e6atas.analyze --run output/2026-11-06/e6_atas_loss
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import src.readout.common as RC
from src.e6atas.losses import IOU_FN, TCOS_FN, box_iou
from src.ivc.analyze import f2, fci, load_probe
from src.readout.evaluate import Boot, summarize

EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
COLS = [("Overall", "all", "overall"), ("Obj D5-16", "D5-16", "object"), ("Part D1-4", "D1-4", "part"), ("Object", "all", "object"),
        ("Part", "all", "part"), ("Obj D17+", "D17+", "object"), ("Color", "all", "color"), ("Material", "all", "material"),
        ("Pattern", "all", "pattern_marking")]
DB = (("D1-4", 1, 4), ("D5-16", 5, 16), ("D17+", 17, 10 ** 9))
A, B_, C_, D_ = "E6-original (A)", "B Recon+Relation", "C Recon+Contrastive", "D Recon+Rel+Con"
EFFECTS = [("B − A (Relation)", ((B_, 1), (A, -1))), ("C − A (Contrastive)", ((C_, 1), (A, -1))),
           ("D − B (Con | Rel)", ((D_, 1), (B_, -1))), ("D − C (Rel | Con)", ((D_, 1), (C_, -1))),
           ("Interaction D − B − C + A", ((D_, 1), (B_, -1), (C_, -1), (A, 1)))]
for m in (B_, C_, D_):
    EFFECTS += [(f"{m.split()[0]} − Bicubic", ((m, 1), ("Bicubic", -1))), (f"{m.split()[0]} − Full ML + R2", ((m, 1), ("Full Multi-layer + R2", -1)))]


def retrieval(vec, method):
    S = torch.nn.functional.normalize(vec[method].float(), dim=-1)
    T = torch.nn.functional.normalize(vec["teacher"].float(), dim=-1)
    key, box = vec["key"].long(), vec["box"].float()
    N = len(S)
    margin, top1, n_excl = np.zeros(N), np.zeros(N), 0
    for i0 in range(0, N, 1024):
        i1 = min(N, i0 + 1024)
        sim, tt = S[i0:i1] @ T.T, T[i0:i1] @ T.T
        same = key[i0:i1, None] == key[None]
        fn = (same & (box_iou(box[i0:i1], box) >= IOU_FN)) | (tt >= TCOS_FN)
        pos = torch.arange(i0, i1)
        fn[torch.arange(i1 - i0), pos] = True
        n_excl += int(fn.sum()) - (i1 - i0)
        allowed = ~fn
        negmean = (sim * allowed).sum(1) / allowed.sum(1).clamp_min(1)
        ps = sim[torch.arange(i1 - i0), pos]
        margin[i0:i1] = (ps - negmean).numpy()
        cand = torch.where(allowed, sim, torch.full_like(sim, -9.0))
        top1[i0:i1] = (ps > cand.max(1).values).numpy()
    return margin, top1, n_excl


def img_boot(d, img, nb=1000, seed=0):
    uq, inv = np.unique(img, return_inverse=True)
    rng = np.random.default_rng(seed)
    bs = [np.average(d, weights=np.bincount(rng.integers(0, len(uq), len(uq)), minlength=len(uq))[inv] + 1e-12) for _ in range(nb)]
    return dict(point=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


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
    SRC = {"Bicubic": ("output/2026-11-05/ivc", "bic"), A: ("output/2026-11-05/xdec/results", "E6"),
           B_: (os.path.join(R, "results"), "B"), C_: (os.path.join(R, "results"), "C"), D_: (os.path.join(R, "results"), "D"),
           "Full Multi-layer + R2": ("output/2026-11-05/msd", "mlr2"), "C0 Crop Oracle": ("output/2026-11-05/ivc", "c0")}
    P, RES, rows, cfg = {}, {}, [], {}
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
                    rows.append(dict(method=m, reader=rd.upper(), metric=met.upper(), column=cn, epoch=p["epoch"],
                                     value=float(v[g][0]) if v is not None and g in v else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(R, "paco_results.csv"), index=False)

    def val(m, rd, met, b, g):
        r = RES.get((m, rd))
        return None if r is None or r[met][b] is None or g not in r[met][b] else r[met][b][g]

    def pv(m, rd="r2", met="m0", b="all", g="overall"):
        v = val(m, rd, met, b, g)
        return None if v is None else float(v[0])

    brows = []
    for nm, terms in EFFECTS:
        for rd in ("r2", "r0"):
            for met in ("m0", "m2"):
                for cn, b, g in COLS:
                    vs = [val(m, rd, met, b, g) for m, _ in terms]
                    if any(v is None for v in vs):
                        continue
                    brows.append(dict(effect=nm, reader=rd.upper(), metric=met.upper(), column=cn, **summarize(sum(s * v for (_, s), v in zip(terms, vs)))))
    bdf = pd.DataFrame(brows)
    bdf.to_csv(os.path.join(R, "bootstrap_results.csv"), index=False)
    same = {k: len({json.dumps(c.get(k)) for c in cfg.values()}) == 1 for k in ("n_train_rows", "n_val_rows", "n_val_images", "n_train_images")}

    # ---------------------------------------------------------------- representation
    REP, retr, rrows = {}, {}, []
    for split in ("paco_val", "coco_dev"):
        p = os.path.join(R, "results", "recon", f"{split}_roi_metrics.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        vec = torch.load(os.path.join(R, "results", "recon", f"{split}_region_vectors.pt"))
        conds = [c for c in df.cond.unique()]
        for c in conds:
            mg, t1, nex = retrieval(vec, c)
            df.loc[df.cond == c, "margin"] = mg
            df.loc[df.cond == c, "top1"] = t1
            retr[f"{split}/{c}"] = dict(margin=float(mg.mean()), top1=float(t1.mean()), n_fn_excluded=nex)
        Tn = torch.nn.functional.normalize(vec["teacher"].float(), dim=-1)
        idx = np.random.default_rng(0).choice(len(Tn), min(3000, len(Tn)), replace=False)
        tt = (Tn[idx] @ Tn[idx].T)[~torch.eye(len(idx), dtype=torch.bool)].numpy()
        retr[f"{split}/teacher_teacher_cos"] = dict(mean=float(tt.mean()), p50=float(np.median(tt)), p95=float(np.percentile(tt, 95)),
                                                     max=float(tt.max()), frac_ge_0p95=float((tt >= 0.95).mean()), sample=len(idx))
        tvec = vec["teacher"].float()
        mgT, t1T, _ = retrieval({"teacher": vec["teacher"], "key": vec["key"], "box": vec["box"], "T": vec["teacher"]}, "T")
        retr[f"{split}/C0 teacher itself"] = dict(margin=float(mgT.mean()), top1=float(t1T.mean()))
        REP[split] = df
        df.to_parquet(os.path.join(R, "results", f"{split}_representation_with_retrieval.parquet"))
        for c in conds:
            s = df[df.cond == c]
            rrows.append(dict(split=split, method=c, **{k: float(s[k].mean()) for k in ("mse", "tok_cos", "pool_cos", "c_dir", "sv_ratio", "rel_err", "margin", "top1", "e_mean")}))
    json.dump(retr, open(os.path.join(R, "results", "retrieval.json"), "w"), indent=1)
    pd.DataFrame(rrows).to_csv(os.path.join(R, "representation_results.csv"), index=False)

    # ---------------------------------------------------------------- tables
    L = ["# E6 + ATAS-inspired loss ablation: tables (auto-generated)", "",
         f"PACO **val** only (test never evaluated). Probe-config agreement across methods: {same}. Reader epoch = max PACO-dev M0. "
         "A = E6-original (xdec E6, 12k; identical model / data / steps / L_old, not retrained). B / C / D: fixed 12k checkpoint.", "",
         "## 1. Main table", "",
         "| Method | R2 M0 | R2 M2 | Obj D5–16 | Part D1–4 | R0 M0 | Patch Relation Error ↓ | Region Margin ↑ | Retrieval Top-1 ↑ | Spatial Var / C0 | Token Cos | Centered Dir Cos |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pvdf = REP.get("paco_val")
    for m in SRC:
        rep = pvdf[pvdf.cond == m] if pvdf is not None else pd.DataFrame()
        rg = (lambda k: f"{rep[k].mean():.4f}" if len(rep) else "–")
        L.append(f"| {m} | " + " | ".join(f2(x) if (x := pv(m, *q)) is not None else "n/a" for q in
                                          (("r2", "m0"), ("r2", "m2"), ("r2", "m0", "D5-16", "object"), ("r2", "m0", "D1-4", "part"), ("r0", "m0")))
                 + f" | {rg('rel_err')} | {rg('margin')} | {rg('top1')} | {rg('sv_ratio')} | {rg('tok_cos')} | {rg('c_dir')} |")
    L += ["", "Representation columns: PACO val (11,109 ROIs), split-wide margin / retrieval with the false-negative rule.",
          f"C0 teacher itself on PACO val: margin {retr.get('paco_val/C0 teacher itself', {}).get('margin', float('nan')):.4f}, top-1 {retr.get('paco_val/C0 teacher itself', {}).get('top1', float('nan')):.4f}; "
          f"teacher-teacher cos {retr.get('paco_val/teacher_teacher_cos')}", ""]
    for met in ("m0", "m2"):
        for rd in ("r2", "r0"):
            L += [f"## 2. {met.upper()} ({rd.upper()})", "", "| Method | epoch (dev M0) | " + " | ".join(c[0] for c in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
            for m in SRC:
                if (m, rd) in P:
                    L.append(f"| {m} | {P[(m, rd)]['epoch']} ({P[(m, rd)]['dev_M0']:.2f}) | " + " | ".join(f2(v) if (v := pv(m, rd, met, b, g)) is not None else "n/a" for _, b, g in COLS) + " |")
            L += ["", f"Paired bootstrap ({met.upper()}, {rd.upper()}): effect [95% CI]", "", "| Effect | " + " | ".join(c[0] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
            sub = bdf[(bdf.reader == rd.upper()) & (bdf.metric == met.upper())] if len(bdf) else bdf
            for nm, _ in EFFECTS:
                cells = []
                for cn, _, _ in COLS:
                    r = sub[(sub.effect == nm) & (sub.column == cn)] if len(sub) else sub
                    cells.append(fci(r.iloc[0].to_dict()) if len(r) else "n/a")
                L.append(f"| {nm} | " + " | ".join(cells) + " |")
            L.append("")
    MET = ["mse", "tok_cos", "pool_cos", "c_dir", "sv_ratio", "rel_err", "margin", "top1", "e_mean"]
    for split, df in REP.items():
        L += [f"## 3. Representation: {split}", "", "| Method | " + " | ".join(MET) + " |", "|---|" + "---|" * len(MET)]
        for c in df.cond.unique():
            s = df[df.cond == c]
            L.append(f"| {c} | " + " | ".join(f"{s[k].mean():.4f}" for k in MET) + " |")
        L.append("")
        dbin = np.select([df.n_in <= 4, df.n_in <= 16], ["D1-4", "D5-16"], "D17+")
        grp = [("all", np.ones(len(df), bool))] + ([("object", df.part.values == 0), ("part", df.part.values == 1)] if "part" in df else []) + [(b, dbin == b) for b in ("D1-4", "D5-16", "D17+")]
        for k in ("rel_err", "margin", "sv_ratio"):
            L += [f"{k} by group ({split}):", "", "| Method | " + " | ".join(g for g, _ in grp) + " |", "|---|" + "---|" * len(grp)]
            for c in df.cond.unique():
                mm = df.cond.values == c
                L.append(f"| {c} | " + " | ".join(f"{df[k].values[mm & gm].mean():.4f}" for _, gm in grp) + " |")
            L.append("")
        img = df[df.cond == "Bicubic"]["img"].values if "img" in df else (df[df.cond == "Bicubic"].index.values // 4)
        L += [f"Representation differences ({split}, image bootstrap 1000, seed 0):", "", "| Effect | rel_err | margin | top1 | sv_ratio | tok_cos | c_dir |", "|---|---|---|---|---|---|---|"]
        for nm, terms in EFFECTS[:5] + [("B − Bicubic", ((B_, 1), ("Bicubic", -1))), ("C − Bicubic", ((C_, 1), ("Bicubic", -1)))]:
            if not all(m in set(df.cond) for m, _ in terms):
                continue
            cells = []
            for k in ("rel_err", "margin", "top1", "sv_ratio", "tok_cos", "c_dir"):
                d = sum(s * df[df.cond == m][k].values for m, s in terms)
                cells.append(fci(img_boot(d, img), 4))
            L.append(f"| {nm} | " + " | ".join(cells) + " |")
        L.append("")
    cal = json.load(open(os.path.join(R, "calib", "calib.json"))) if os.path.exists(os.path.join(R, "calib", "calib.json")) else None
    if cal:
        L += ["## 4. Loss-weight calibration (once, before training)", "", "```", json.dumps({k: v for k, v in cal.items() if k != "batches"}, indent=1), "```", "",
              "| batch | L_old | L_rel | L_con | g_old | g_rel | g_con | cos(old,rel) | cos(old,con) | cos(rel,con) | margin@init | top1@init |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in cal["batches"]:
            L.append(f"| {b['batch']} | {b['L_old']:.4f} | {b['L_rel']:.4f} | {b['L_con']:.4f} | {b['g_old']:.3e} | {b['g_rel']:.3e} | {b['g_con']:.3e} | "
                     f"{b['cos_old_rel']:+.3f} | {b['cos_old_con']:+.3f} | {b['cos_rel_con']:+.3f} | {b['margin_init']:.3f} | {b['top1_init']:.3f} |")
        L.append("")
    L += ["## 5. COCO-dev monitoring at 3k / 6k / 9k / 12k (primary = 12k, fixed)", ""]
    logs = {}
    for m, p in ((A, "output/2026-11-05/xdec/checkpoints/E6"), (B_, os.path.join(R, "checkpoints", "B")), (C_, os.path.join(R, "checkpoints", "C")), (D_, os.path.join(R, "checkpoints", "D"))):
        if not os.path.exists(os.path.join(p, "done.json")):
            continue
        dj = json.load(open(os.path.join(p, "done.json")))
        logs[m] = [json.loads(l) for l in open(os.path.join(p, "train_log.jsonl"))]
        L += [f"### {m} (time {dj['time_s']} s, peak mem {dj['peak_mem_gb']:.2f} GB)", "", "| step | L_old | token cos | L_rel | L_con | margin (chunk) | top1 (chunk) |", "|---|---|---|---|---|---|---|"]
        for st_, d in sorted(dj["dev_at_ckpts"].items(), key=lambda x: int(x[0])):
            L.append(f"| {st_} | {d['loss']:.4f} | {d['cos']:.4f} | {d.get('L_rel', float('nan')):.4f} | {d.get('L_con', float('nan')):.4f} | {d.get('margin', float('nan')):.4f} | {d.get('top1', float('nan')):.4f} |")
        L.append("")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(20, 4))
        for m, lg in logs.items():
            t = [r for r in lg if "loss" in r and "dev" not in r]
            dv = [r for r in lg if "dev" in r]
            for i, k in enumerate(("loss", "L_rel", "L_con", "top1")):
                if t and k in t[0]:
                    ax[i].plot([r["step"] for r in t], [r[k] for r in t], lw=1, label=m)
                if dv and k in dv[-1]["dev"] or k == "loss":
                    ax[i].plot([r["step"] for r in dv if k in r["dev"]], [r["dev"][k] for r in dv if k in r["dev"]], "o--", ms=3, label=f"{m} dev")
        for x, tt_ in zip(ax, ("L_old", "L_rel", "L_con", "in-batch top-1")):
            x.set_title(tt_); x.set_xlabel("step"); x.legend(fontsize=6); x.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "train_curves.png"), dpi=120); plt.close(fig)
    except Exception as e:
        L.append(f"(figure failed: {e})")
    open(os.path.join(R, "tables.md"), "w").write("\n".join(L) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
