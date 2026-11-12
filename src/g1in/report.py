"""ImageNet-1K G1 generalisation report (CPU): independent linear-probe accuracy, paired image bootstrap CI vs Original,
Original-head transfer, CLS / mean-patch preservation, accuracy change by CLS-cosine quintile, verification checks.

  python -m src.g1in.report --run output/2026-11-07/g1_imagenet_generalization
"""
import argparse
import ast
import json
import os

import numpy as np

from src.g1in.common import DATA, MODELS, PACO_M0

NAME = {"original": "Original DINOv3", "g1pool": "G1-pool Encoder", "g1cross": "G1-cross Encoder Only"}
B, SEED = 1000, 0


def boot(a, b, rng_idx):
    """paired image bootstrap of mean(a) - mean(b) (in pp); a, b per-image 0/1."""
    d = (a.astype(np.float64) - b.astype(np.float64)) * 100
    pt = d.mean()
    bs = np.array([d[i].mean() for i in rng_idx])
    return dict(point=float(pt), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def fmt(s):
    sig = s["lo"] > 0 or s["hi"] < 0
    return f"{s['point']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]{'*' if sig else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R, Cc = a.run, os.path.join(a.run, "cache")
    lin = json.load(open(os.path.join(R, "linear", "linear_results.json")))
    yv = np.load(os.path.join(Cc, "val_labels.npy")).astype(np.int64)
    n = len(yv)
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    corr = {}
    for kind, pre in (("ind", "val_top5_"), ("tf", "val_top5_transfer_")):
        for m in MODELS:
            t5 = np.load(os.path.join(R, "linear", f"{pre}{m}.npy")).astype(np.int64)
            corr[(kind, m)] = (t5[:, 0] == yv, (t5 == yv[:, None]).any(1))
    out = dict(independent={}, transfer={}, ci={}, preservation={}, quintiles={}, checks={})
    for m in MODELS:
        r = lin["independent"][m]
        out["independent"][m] = dict(val_top1=r["val"]["top1"], val_top5=r["val"]["top5"], val_loss=r["val"]["loss"],
                                     train_top1=r["train"]["top1"], train_top5=r["train"]["top5"], train_loss=r["train"]["loss"])
        t = lin["transfer_original_head"][m]
        out["transfer"][m] = dict(val_top1=t["val"]["top1"], val_top5=t["val"]["top5"], val_loss=t["val"]["loss"],
                                  train_top1=t["train"]["top1"], train_loss=t["train"]["loss"])
    for kind in ("ind", "tf"):
        for x, y in (("g1pool", "original"), ("g1cross", "original"), ("g1cross", "g1pool")):
            out["ci"][f"{kind}:{x}-{y}"] = dict(top1=boot(corr[(kind, x)][0], corr[(kind, y)][0], idx),
                                                top5=boot(corr[(kind, x)][1], corr[(kind, y)][1], idx))
    stv = np.load(os.path.join(Cc, "val_stats.npy"))
    stt = np.load(os.path.join(Cc, "train_stats.npy"), mmap_mode="r")
    cols = ["norm_original", "norm_g1pool", "norm_g1cross", "clscos_g1pool", "clscos_g1cross", "patchcos_g1pool", "patchcos_g1cross"]
    for split, st in (("val", stv), ("train", stt)):
        out["preservation"][split] = {c: float(np.asarray(st[:, i], np.float64).mean()) for i, c in enumerate(cols)}
    for m, ci in (("g1pool", 3), ("g1cross", 4)):            # accuracy change by CLS-cosine quintile (val)
        q = np.quantile(stv[:, ci], [0, .2, .4, .6, .8, 1])
        rows = []
        for k in range(5):
            sel = (stv[:, ci] >= q[k]) & ((stv[:, ci] <= q[k + 1]) if k == 4 else (stv[:, ci] < q[k + 1]))
            rows.append(dict(cos_lo=float(q[k]), cos_hi=float(q[k + 1]), n=int(sel.sum()),
                             orig_top1=100 * float(corr[("ind", "original")][0][sel].mean()),
                             ind_delta=100 * float(corr[("ind", m)][0][sel].mean() - corr[("ind", "original")][0][sel].mean()),
                             tf_delta=100 * float(corr[("tf", m)][0][sel].mean() - corr[("tf", "original")][0][sel].mean())))
        out["quintiles"][m] = rows
    # checks
    ver = json.load(open(os.path.join(Cc, "verify.json")))
    infos = {f"{s}{k}": json.load(open(os.path.join(Cc, f"{s}_shard{k}_info.json"))) for s in ("train", "val") for k in (0, 1)}
    src = open(os.path.join(DATA, "classes.py")).read()
    classes = ast.literal_eval(src[src.index("{"):src.rindex("}") + 1]) if "{" in src else {}
    out["checks"] = dict(verify=ver, n_classes_classes_py=len(classes),
                         checksum_match_all=all(all(i["checksum_match"].values()) for i in infos.values()),
                         unchanged_all=all(all(i["unchanged"].values()) for i in infos.values()),
                         requires_grad_any=any(any(i["requires_grad_any"].values()) for i in infos.values()),
                         checksums=infos["train0"]["checksum_loaded"], feature=infos["train0"]["feature"],
                         preprocess=infos["train0"]["preprocess"], time_s={k: i["time_s"] for k, i in infos.items()},
                         n_images={k: i["n_images"] for k, i in infos.items()})
    out["linear_config"] = lin["config"]
    json.dump(out, open(os.path.join(R, "results.json"), "w"), indent=1)

    o, L = out["independent"], []
    L += ["## Primary: independent linear probe (ImageNet-1K val, 50,000 images, final-LN CLS @384)", "",
          "| Method | ImageNet Top-1 | Top-5 | Δ Top-1 vs Original (95% CI) | Δ Top-5 vs Original | PACO Val R2 M0 (reference) |",
          "|---|---|---|---|---|---|"]
    for m in MODELS:
        d1 = "–" if m == "original" else fmt(out["ci"][f"ind:{m}-original"]["top1"])
        d5 = "–" if m == "original" else fmt(out["ci"][f"ind:{m}-original"]["top5"])
        L.append(f"| {NAME[m]} | {o[m]['val_top1']:.2f} | {o[m]['val_top5']:.2f} | {d1} | {d5} | {PACO_M0[m]:.2f} |")
    L += ["", f"G1-cross − G1-pool (independent heads): Top-1 {fmt(out['ci']['ind:g1cross-g1pool']['top1'])}, Top-5 {fmt(out['ci']['ind:g1cross-g1pool']['top5'])}",
          "", "| Method | Train Top-1 | Train Top-5 | Train loss | Val Top-1 | Val loss |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        L.append(f"| {NAME[m]} | {o[m]['train_top1']:.2f} | {o[m]['train_top5']:.2f} | {o[m]['train_loss']:.4f} | {o[m]['val_top1']:.2f} | {o[m]['val_loss']:.4f} |")
    t = out["transfer"]
    L += ["", "## Diagnostic (separate from primary): Original head frozen, applied to each encoder's CLS", "",
          "| Encoder + Original head | Val Top-1 | Val Top-5 | Δ Top-1 vs Original (95% CI) | Val loss | Train Top-1 |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        d1 = "–" if m == "original" else fmt(out["ci"][f"tf:{m}-original"]["top1"])
        L.append(f"| {NAME[m]} | {t[m]['val_top1']:.2f} | {t[m]['val_top5']:.2f} | {d1} | {t[m]['val_loss']:.4f} | {t[m]['train_top1']:.2f} |")
    pv = out["preservation"]
    L += ["", "## Representation preservation vs Original (per-image mean)", "",
          "| Encoder | split | CLS norm | CLS cos | mean-patch cos |", "|---|---|---|---|---|"]
    for split in ("val", "train"):
        L.append(f"| Original | {split} | {pv[split]['norm_original']:.3f} | 1 | 1 |")
        for m in ("g1pool", "g1cross"):
            L.append(f"| {NAME[m]} | {split} | {pv[split]['norm_' + m]:.3f} | {pv[split]['clscos_' + m]:.4f} | {pv[split]['patchcos_' + m]:.4f} |")
    L += ["", "## Val accuracy change by CLS-cosine quintile (pp vs Original, same images)", "",
          "| Encoder | quintile | CLS cos range | n | Original Top-1 | Δ independent | Δ Original-head |", "|---|---|---|---|---|---|---|"]
    for m in ("g1pool", "g1cross"):
        for k, r in enumerate(out["quintiles"][m]):
            L.append(f"| {NAME[m]} | Q{k + 1} | {r['cos_lo']:.3f}–{r['cos_hi']:.3f} | {r['n']} | {r['orig_top1']:.2f} | {r['ind_delta']:+.2f} | {r['tf_delta']:+.2f} |")
    ck = out["checks"]
    L += ["", "## Checks", "", f"- verify: {json.dumps(ver)}", f"- classes.py classes: {ck['n_classes_classes_py']}",
          f"- encoder checksums == PACO probe frozen_check (all shards): {ck['checksum_match_all']}; unchanged after extraction: {ck['unchanged_all']}; any requires_grad: {ck['requires_grad_any']}",
          f"- feature: {ck['feature']}", f"- extraction time (s): {ck['time_s']}", f"- linear config: {json.dumps(out['linear_config'])}"]
    open(os.path.join(R, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
