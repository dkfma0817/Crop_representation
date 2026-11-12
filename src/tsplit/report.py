"""Tables for the token-splitting experiment -> <run>/results/{tables.md, results.json}. CPU only; reads official evaluator outputs.

  python -m src.tsplit.report --run <run>
"""
import argparse
import json
from pathlib import Path

CORE = ("F384", "T768", "B0", "S1", "S2", "S3")
DESC = {"F384": ("Frozen 384", "none", "-"), "T768": ("Frozen 768 (native)", "none", "-"), "B0": ("Frozen 384", "Bilinear x2", "-"),
        "S1": ("Frozen 384", "Linear Split x2", "768"), "S2": ("Frozen 384", "Split x2 + Small ViT", "768"), "S3": ("Frozen 384 + Patch Embed", "Split x2 + Small ViT", "768")}
COLS = (("overall", "Overall"), ("object", "Object"), ("part", "Part"), ("pattern", "Pattern"), ("material", "Material"), ("color", "Color"))
EXISTING_VAL = {"F384": dict(overall=54.21, object=52.27, part=56.16, src="output/2026-11-09/g1_upscaled_teacher_s0 G0-Full384 ep5"),
                "T768": dict(overall=56.40, object=54.10, part=58.69, src="output/2026-11-09/g1_native768_s0 G0-Native768 ep6")}


def row(path):
    h = json.loads(Path(path).read_text())["hierarchical"]
    o, p = h["obj-attrs"], h["obj-part-attrs"]
    return dict(overall=(h["obj-attr-AP"] + h["obj-attr-part-AP"]) / 2, object=h["obj-attr-AP"], part=h["obj-attr-part-AP"],
                pattern=(o["pattern_marking"] + p["pattern_marking"]) / 2, material=(o["material"] + p["material"]) / 2, color=(o["color"] + p["color"]) / 2)


def table(rows, sel, names, desc=True):
    head = (["Method", "Encoder", "Decoder", "Teacher"] if desc else ["Condition"]) + ["R2 epoch"] + [c for _, c in COLS]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in names:
        if n not in rows:
            continue
        lead = [n, *DESC[n]] if desc else [n]
        out.append("| " + " | ".join(lead + [str(sel[f"{n}_r2"]["epoch"])] + [f"{rows[n][k]:.2f}" for k, _ in COLS]) + " |")
    return "\n".join(out)


def contrasts(rows):
    pairs = [("B0", "F384"), ("S1", "B0"), ("S2", "B0"), ("S3", "B0"), ("S2", "S1"), ("S3", "S2"), ("S1", "F384"), ("S2", "F384"), ("S3", "F384"), ("T768", "F384"), ("T768", "B0")]
    out = ["| Contrast | " + " | ".join(c for _, c in COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for a, b in pairs:
        if a in rows and b in rows:
            out.append(f"| {a} - {b} | " + " | ".join(f"{rows[a][k] - rows[b][k]:+.2f}" for k, _ in COLS) + " |")
    gap = {k: rows["T768"][k] - rows["B0"][k] for k, _ in COLS} if "T768" in rows and "B0" in rows else None
    if gap:
        out += ["", "Gap recovered = (S - B0) / (T768 - B0) (point estimate; undefined sign if the gap is near 0):", "",
                "| Method | " + " | ".join(c for _, c in COLS[:3]) + " |", "|---|---|---|---|"]
        for s in ("S1", "S2", "S3"):
            if s in rows:
                out.append(f"| {s} | " + " | ".join(f"{100 * (rows[s][k] - rows['B0'][k]) / gap[k]:.0f}%" if abs(gap[k]) > 0.2 else "n/a" for k, _ in COLS[:3]) + " |")
        g2 = {k: rows["T768"][k] - rows["F384"][k] for k, _ in COLS}
        out += ["", "Gap recovered vs the no-decoder baseline = (S - F384) / (T768 - F384):", "", "| Method | Overall | Object | Part |", "|---|---|---|---|"]
        for s in ("S1", "S2", "S3"):
            if s in rows:
                out.append(f"| {s} | " + " | ".join(f"{100 * (rows[s][k] - rows['F384'][k]) / g2[k]:.0f}%" for k, _ in COLS[:3]) + " |")
    return "\n".join(out)


def boot_table(path, pairs):
    """Paired image bootstrap (src.g1up.support boot, B = 1000): CI of the difference per replicate; * = CI excludes 0."""
    import numpy as np
    d = json.loads(Path(path).read_text())
    bt = d["boot"]
    out = [f"Paired image bootstrap, B = {d['B']}, {d['n_images']} images, 95% percentile CI of the difference; * = CI excludes 0.", "",
           "| Contrast | Overall | Object | Part |", "|---|---|---|---|"]
    res = {}
    for a, b in pairs:
        A, B = f"{a}_r2", f"{b}_r2"
        if A not in bt or B not in bt:
            continue
        cells = []
        for k in ("overall", "obj", "part"):
            if k == "overall":
                x = (np.array(bt[A]["obj"]) + np.array(bt[A]["part"])) / 2 - (np.array(bt[B]["obj"]) + np.array(bt[B]["part"])) / 2
                p = (d["points"][A]["obj"] + d["points"][A]["part"]) / 2 - (d["points"][B]["obj"] + d["points"][B]["part"]) / 2
            else:
                x = np.array(bt[A][k]) - np.array(bt[B][k])
                p = d["points"][A][k] - d["points"][B][k]
            lo, hi = np.nanpercentile(x, [2.5, 97.5])
            res[f"{a}-{b}/{k}"] = dict(point=p, lo=float(lo), hi=float(hi))
            cells.append(f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("*" if lo > 0 or hi < 0 else ""))
        out.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
    return "\n".join(out), res


BOOT_PAIRS = (("S3", "B0"), ("S2", "B0"), ("S1", "B0"), ("S3", "S2"), ("S2", "S1"), ("B0", "F384"), ("S3", "F384"), ("T768", "F384"), ("T768", "S3"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    res, md = {}, ["# Token splitting x2 on frozen DINOv3-B/16 - PACO official GT-box + GT-category oracle attribute AP", "",
                   "R2 attention reader (not a linear probe), 10 epochs, seed 0, epoch = max (Object + Part) / 2 on official Val, frozen before Test. "
                   "Overall = (Object + Part) / 2; Pattern / Material / Color = mean of the Object and Part values.", ""]
    sel = json.loads((run / "eval/selection_frozen.json").read_text()) if (run / "eval/selection_frozen.json").exists() else None
    if sel:
        val = {n: row(run / f"eval/eval_val/{n}_r2/ep{sel[f'{n}_r2']['epoch']}/official_results.json") for n in CORE if f"{n}_r2" in sel}
        res["val"], res["selection"] = val, sel
        md += ["## Official Val", "", table(val, sel, CORE), "", contrasts(val), ""]
        test = {n: row(run / f"eval/eval_test/{n}_r2/official_results.json") for n in CORE if (run / f"eval/eval_test/{n}_r2/official_results.json").exists()}
        if test:
            res["test"] = test
            md += ["## Official Test (epochs fixed on Val)", "", table(test, sel, CORE), "", contrasts(test), ""]
        for split in ("val", "test"):
            bp = run / f"eval/boot_{split}.json"
            if bp.exists():
                t, r = boot_table(bp, BOOT_PAIRS)
                res[f"boot_{split}"] = r
                md += [f"## Bootstrap, official {split.capitalize()}", "", t, ""]
        md += ["## Re-trained F384 / T768 vs the existing G0 readers (official Val, same protocol, different code path)", "",
               "| Condition | this run Overall / Obj / Part | existing Overall / Obj / Part |", "|---|---|---|"]
        for n, e in EXISTING_VAL.items():
            if n in val:
                md.append(f"| {n} | {val[n]['overall']:.2f} / {val[n]['object']:.2f} / {val[n]['part']:.2f} | {e['overall']:.2f} / {e['object']:.2f} / {e['part']:.2f} ({e['src']}) |")
        md.append("")
    dsel = run / "eval_diag/selection.json"
    if dsel.exists():
        ds = json.loads(dsel.read_text())
        drow = {k[:-3]: row(run / f"eval_diag/eval_val/{k}/ep{v['epoch']}/official_results.json") for k, v in ds.items()}
        res["diag_val"] = drow
        names = [n for n in ("tapL0", "tapL1", "tapL3", "tapL6", "tapL9", "rerunL0", "rerunL3", "rerunL6", "rerunL9") if n in drow]
        md += ["## Layer diagnostics (official Val only)", "", "tapL* = layer-L 384 tokens (parameter-free LayerNorm) read directly (a); rerunL* = layer-L 384 tokens bilinear x2 -> remaining frozen blocks at 48 x 48 (c). "
               "F384 = tap at the final layer; B0 = rerun at L12.", "", table(drow, ds, names, desc=False), ""]
    tr = run / "train/heldout.jsonl"
    if tr.exists():
        last = [json.loads(l) for l in tr.open()][-1]
        res["heldout_feature"] = last
        md += [f"## Held-out COCO feature metrics (468 dev images, step {last['step']}; teacher = native 768 final tokens)", "",
               "| Model | bucket | n ROI | MSE | cos | within-parent residual MSE | within-parent R^2 | parent-mean MSE |", "|---|---|---|---|---|---|---|---|"]
        for k in ("NN", "B0", "S1", "S2", "S3"):
            for bk in ("train", "large", "whole"):
                m = last["metrics"][k][bk]
                md.append(f"| {k} | {bk} | {m['n_roi']} | {m['mse']:.5f} | {m['cos']:.4f} | {m['within_parent_residual_mse']:.5f} | {m['within_parent_r2']:.3f} | {m['parent_mean_mse']:.5f} |")
        md += ["", "Buckets: train = rectangles 1..6 parents per side (training range); large = 7..12; whole = every content parent of the image (up to 24 x 24). "
               "within-parent R^2 = fraction of the teacher's (child - parent-mean) energy reproduced; NN = nearest copy (R^2 = 0 by construction).", ""]
    dl = run / "diag_layers/diag_layers.json"
    if dl.exists():
        d = json.loads(dl.read_text())["metrics"]
        res["diag_layers_feature"] = d
        md += ["## Layer diagnostics, feature side (held-out COCO, 192 images)", "", "| Comparison | nMSE | cos | within-parent R^2 | parent-mean nMSE | teacher within-parent share |", "|---|---|---|---|---|---|"]
        for k, v in d.items():
            md.append(f"| {k} | {v['nmse']:.3f} | {v['cos']:.3f} | {v['within_parent_r2']:.3f} | {v['parent_mean_nmse']:.3f} | {v['teacher_within_parent_share']:.3f} |")
        md.append("")
    eff = run / "efficiency.json"
    if eff.exists():
        e = json.loads(eff.read_text())
        res["efficiency"] = e
        p = e["params"]
        md += ["## Parameters and cost", "", f"Params: S1 {p['S1']['total'] / 1e6:.2f} M, S2 {p['S2']['total'] / 1e6:.2f} M, S3 {p['S3']['total'] / 1e6:.2f} M (S3 fusion extra {p['S3_fusion_extra_vs_S2'] / 1e6:.2f} M). "
               f"Encoder per image: 384 {e['encoder_ms_per_image']['dinov3_384']:.1f} ms, 768 {e['encoder_ms_per_image']['dinov3_768']:.1f} ms (shared GPU, indicative).", "",
               "| Model | ROI (parents) | children | GFLOPs / ROI | ms / ROI |", "|---|---|---|---|---|"]
        for k in ("S2", "S3"):
            for s, v in e["decoder"][k]["per_roi"].items():
                md.append(f"| {k} | {s} | {v['children']} | {v['gflops']:.2f} | {v['ms']:.2f} |")
        md.append("")
        md.append("Split per image: " + ", ".join(f"{k} {e['decoder'][k]['split_ms_per_image']:.2f} ms / {e['decoder'][k]['split_gflops_per_image']:.2f} GFLOPs" for k in ("S1", "S2", "S3")))
        md.append("")
    out = run / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "tables.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
