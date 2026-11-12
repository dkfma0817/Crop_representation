"""Common-failure analysis on PACO official VAL (existing prediction caches, fixed R2 epoch 6; NOT a substitute for AP).

Models: Original CLIP Full, G1-cross Encoder Only, MaskEmbed Full (Full-image models) and Original CLIP C0 crop (oracle).
Groups: Object Material, Part Pattern & Markings, Object Pattern & Markings.
Unit = (level, category, attribute). Known positive = label 1, known negative = label 0 (label -1 = unknown type ->
never used). Pairs are sampled WITHIN a unit (fixed seed 0, at most K per unit; all combinations when fewer exist).
A model ranks a pair correctly iff p(pos) > p(neg) for that attribute (ties count as wrong).
  common Full failure : G1-cross Enc AND MaskEmbed both wrong
  crop-only          : common Full failure AND C0 crop correct
Outputs <out>/failure.json, failure.md, viz/ (<= 12 cases: the 336 letterboxed full input with the box + the original
tight crop, for the positive and the negative ROI).

  python -m src.decstudy.failure --out <dir> [--K 50]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

DATA = "output/2026-11-07/paco_oracle_attr/data"
RR = "output/2026-11-07/decoder_study"
PRED = {"clip_full": "output/2026-11-07/maskembed_paco/preds/clip_full/probs_val.npz",
        "g1cross_enc": f"{RR}/preds/g1cross_enc/pred_val.npz",
        "maskembed_full": "output/2026-11-07/maskembed_paco/preds/maskembed_full/probs_val.npz",
        "clip_c0": f"{RR}/preds/c0/pred_val.npz"}
NAMES = {"clip_full": "Original CLIP Full", "g1cross_enc": "G1-cross Encoder Only", "maskembed_full": "MaskEmbed Full", "clip_c0": "Original CLIP C0 crop"}
ATTR = {"color": range(0, 30), "pattern_marking": range(30, 41), "material": range(41, 55), "transparency": range(55, 59)}
GROUPS = {"obj_material": ("object", "material"), "part_pattern": ("part", "pattern_marking"), "obj_pattern": ("object", "pattern_marking")}
S = 336


def load():
    M = pd.read_parquet(os.path.join(DATA, "manifest_val.parquet"))
    Y = np.load(os.path.join(DATA, "labels_val.npy"))
    P, avail = {}, {}
    for k, f in PRED.items():
        if not os.path.exists(f):
            avail[k] = False
            continue
        z = np.load(f)
        assert np.array_equal(z["ann_id"], M.ann_id.values), f"{k}: rows != manifest"
        P[k], avail[k] = z["probs"].astype(np.float64), True
    return M, Y, P, avail


def pairs_for(M, Y, level, idxs, K, rng):
    rows = []
    lv = M.level.values == level
    for a in idxs:
        for c, g in M[lv].groupby("category_id"):
            r = g.index.values
            pos, neg = r[Y[r, a] == 1], r[Y[r, a] == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            n = len(pos) * len(neg)
            if n <= K:
                pp, nn = np.meshgrid(pos, neg, indexing="ij")
                pp, nn = pp.ravel(), nn.ravel()
            else:
                flat = rng.choice(n, K, replace=False)
                pp, nn = pos[flat // len(neg)], neg[flat % len(neg)]
            for p_, n_ in zip(pp, nn):
                rows.append((int(c), a, int(p_), int(n_), len(pos), len(neg)))
    return pd.DataFrame(rows, columns=["category_id", "attr", "pos", "neg", "n_pos_unit", "n_neg_unit"])


def side336(M, r):
    s = S / np.maximum(M.W.values[r], M.H.values[r])
    return np.sqrt(M.bw.values[r] * M.bh.values[r]) * s


def q(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), p25=float(np.percentile(x, 25)), median=float(np.median(x)), p75=float(np.percentile(x, 75))) if len(x) else None


def render(M, r, path, color):
    """336 letterboxed full input (padding gray = normalised zero) with the box, + the tight crop."""
    row = M.iloc[r]
    img = Image.open(row.path).convert("RGB")
    W, H = img.size
    s = S / max(W, H)
    nw, nh = min(S, max(1, round(W * s))), min(S, max(1, round(H * s)))
    ox, oy = (S - nw) // 2, (S - nh) // 2
    can = Image.new("RGB", (S, S), (123, 117, 104))
    can.paste(img.resize((nw, nh), Image.BICUBIC), (ox, oy))
    d = ImageDraw.Draw(can)
    x0, y0 = ox + row.l * nw / W, oy + row.t * nh / H
    x1, y1 = ox + row.r * nw / W, oy + row.b * nh / H
    d.rectangle([x0, y0, x1, y1], outline=color, width=3)
    crop = img.crop((int(row.l), int(row.t), int(row.r), int(row.b)))
    cw, ch = crop.size
    k = 240 / max(cw, ch)
    crop = crop.resize((max(1, round(cw * k)), max(1, round(ch * k))), Image.BICUBIC)
    out = Image.new("RGB", (S + 250, S), (255, 255, 255))
    out.paste(can, (0, 0))
    out.paste(crop, (S + 5, (S - crop.size[1]) // 2))
    out.save(path, quality=90)
    return dict(box_orig_px=[int(row.l), int(row.t), int(row.r), int(row.b)], box_side_336=float(np.sqrt((x1 - x0) * (y1 - y0))),
                crop_orig_px=[cw, ch], image=f"{int(row.image_id)}", W=W, H=H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--n_viz", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "viz"), exist_ok=True)
    M, Y, P, avail = load()
    import json as _j
    attrs = [x["name"] for x in _j.load(open("/home/irteam/datasets/paco/paco_lvis_v1_val.json"))["attributes"]]
    full = [k for k in ("clip_full", "g1cross_enc", "maskembed_full") if avail[k]]
    assert "g1cross_enc" in full and "maskembed_full" in full, "G1-cross / MaskEmbed val predictions are required"
    crop = avail["clip_c0"]
    res = dict(models={k: NAMES[k] for k in PRED}, available=avail, K=a.K, seed=0, rule="pair correct iff p(pos) > p(neg) (ties wrong); "
               "pairs within (level, category, attribute); known labels only", groups={})
    cases = []
    for gname, (level, typ) in GROUPS.items():
        rng = np.random.default_rng(0)
        D = pairs_for(M, Y, level, list(ATTR[typ]), a.K, rng)
        ok = {k: (P[k][D.pos.values, D.attr.values] > P[k][D.neg.values, D.attr.values]) for k in P}
        both_fail = ~ok["g1cross_enc"] & ~ok["maskembed_full"]
        g = dict(level=level, type=typ, n_units=int(D.groupby(["category_id", "attr"]).ngroups), n_pairs=len(D),
                 n_categories=int(D.category_id.nunique()), n_attributes=int(D.attr.nunique()),
                 pair_acc={k: float(v.mean()) for k, v in ok.items()},
                 unit_macro_acc={k: float(pd.Series(v).groupby([D.category_id.values, D.attr.values]).mean().mean()) for k, v in ok.items()},
                 common_full_fail_rate=float(both_fail.mean()))
        if "clip_full" in ok:
            g["all_three_full_fail_rate"] = float((both_fail & ~ok["clip_full"]).mean())
        side_all = np.concatenate([side336(M, D.pos.values), side336(M, D.neg.values)])
        g["roi_side_336_all_pairs"] = q(side_all)
        g["roi_area_orig_all_pairs"] = q(np.concatenate([M.area.values[D.pos.values], M.area.values[D.neg.values]]))
        if crop:
            co = both_fail & ok["clip_c0"]
            g["crop_only_rate"] = float(co.mean())
            g["crop_correct_given_common_fail"] = float(ok["clip_c0"][both_fail].mean()) if both_fail.any() else None
            g["crop_wrong_full_both_right_rate"] = float((~ok["clip_c0"] & ok["g1cross_enc"] & ok["maskembed_full"]).mean())
            g["n_crop_only"] = int(co.sum())
            D2 = D.assign(co=co, bf=both_fail)
            u = D2.groupby(["category_id", "attr"]).agg(n=("co", "size"), co=("co", "sum"), bf=("bf", "sum")).reset_index()
            u["co_rate"] = u.co / u.n
            rec = u[(u.n >= 5) & (u.co_rate >= 0.3)]
            g["units_with_pairs_ge5"] = int((u.n >= 5).sum())
            g["recurrent_units(n>=5, crop_only_rate>=0.3)"] = int(len(rec))
            g["crop_only_pairs_in_recurrent_units_frac"] = float(rec.co.sum() / max(1, co.sum()))
            catname = dict(zip(M.category_id, M.category))
            by_cat = D2[D2.co].groupby("category_id").size().sort_values(ascending=False)
            by_att = D2[D2.co].groupby("attr").size().sort_values(ascending=False)
            tot_cat = D2.groupby("category_id").size()
            tot_att = D2.groupby("attr").size()
            g["crop_only_by_category_top10"] = [dict(category=catname[int(c)], n=int(n), of_pairs=int(tot_cat[c]), rate=round(n / tot_cat[c], 3)) for c, n in by_cat.head(10).items()]
            g["crop_only_by_attribute"] = [dict(attribute=attrs[int(x)], n=int(n), of_pairs=int(tot_att[x]), rate=round(n / tot_att[x], 3)) for x, n in by_att.items()]
            g["roi_side_336_crop_only"] = q(np.concatenate([side336(M, D.pos.values[co]), side336(M, D.neg.values[co])]))
            g["roi_side_336_full_both_right"] = q(np.concatenate([side336(M, D.pos.values[ok["g1cross_enc"] & ok["maskembed_full"]]),
                                                                side336(M, D.neg.values[ok["g1cross_enc"] & ok["maskembed_full"]])]))
            g["roi_area_orig_crop_only"] = q(np.concatenate([M.area.values[D.pos.values[co]], M.area.values[D.neg.values[co]]]))
            # small vs large split (median of all pairs' min-side)
            ms = np.minimum(side336(M, D.pos.values), side336(M, D.neg.values))
            thr = float(np.median(ms))
            g["crop_only_rate_by_size"] = dict(threshold_side336=thr, small=float(co[ms < thr].mean()), large=float(co[ms >= thr].mean()))
            # candidates for visualisation: crop margin large, both full margins negative
            mc = P["clip_c0"][D.pos.values, D.attr.values] - P["clip_c0"][D.neg.values, D.attr.values]
            mf = np.maximum(P["g1cross_enc"][D.pos.values, D.attr.values] - P["g1cross_enc"][D.neg.values, D.attr.values],
                            P["maskembed_full"][D.pos.values, D.attr.values] - P["maskembed_full"][D.neg.values, D.attr.values])
            idx = np.nonzero(co)[0]
            idx = idx[np.argsort(-(mc[idx] - mf[idx]))]
            seen = set()
            for i in idx:
                key = (int(D.category_id.values[i]), int(D.attr.values[i]))
                if key in seen:
                    continue
                seen.add(key)
                cases.append((gname, i, D.iloc[i], {k: (float(P[k][D.pos.values[i], D.attr.values[i]]), float(P[k][D.neg.values[i], D.attr.values[i]])) for k in P}))
                if len(seen) >= 6:
                    break
        res["groups"][gname] = g
    # visualisation: round-robin over groups, at most n_viz
    viz = []
    order = []
    for gname in GROUPS:
        order += [c for c in cases if c[0] == gname][:4]
    for j, (gname, i, row, sc) in enumerate(order[: a.n_viz]):
        catname = M.category.values[row.pos]
        rp = render(M, int(row.pos), os.path.join(a.out, "viz", f"{j:02d}_{gname}_pos.jpg"), (0, 160, 0))
        rn = render(M, int(row.neg), os.path.join(a.out, "viz", f"{j:02d}_{gname}_neg.jpg"), (200, 0, 0))
        viz.append(dict(i=j, group=gname, category=catname, attribute=attrs[int(row.attr)], pos_ann=int(M.ann_id.values[row.pos]),
                        neg_ann=int(M.ann_id.values[row.neg]), scores_pos_neg=sc, pos=rp, neg=rn,
                        pos_img=f"viz/{j:02d}_{gname}_pos.jpg", neg_img=f"viz/{j:02d}_{gname}_neg.jpg"))
    res["viz"] = viz
    json.dump(res, open(os.path.join(a.out, "failure.json"), "w"), indent=1, default=float)
    # markdown
    L = ["# PACO Val common-failure analysis (pairwise, NOT official AP)", "",
         f"Models: {', '.join(NAMES[k] for k in PRED if avail[k])}. Missing: {', '.join(NAMES[k] for k in PRED if not avail[k]) or 'none'}.",
         f"Rule: {res['rule']}; at most {a.K} pairs per unit, seed 0.", "",
         "| Group | units | pairs | CLIP Full acc | G1-cross acc | MaskEmbed acc | C0 crop acc | both Full fail | crop-only | crop correct among both-fail |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for gname, g in res["groups"].items():
        pa = g["pair_acc"]
        f = lambda k: f"{100 * pa[k]:.1f}" if k in pa else "n/a"  # noqa: E731
        L.append(f"| {gname} | {g['n_units']} | {g['n_pairs']} | {f('clip_full')} | {f('g1cross_enc')} | {f('maskembed_full')} | {f('clip_c0')} | "
                 f"{100 * g['common_full_fail_rate']:.1f} | {100 * g['crop_only_rate']:.1f} | {100 * g['crop_correct_given_common_fail']:.1f} |"
                 if crop else f"| {gname} | {g['n_units']} | {g['n_pairs']} | {f('clip_full')} | {f('g1cross_enc')} | {f('maskembed_full')} | n/a | {100 * g['common_full_fail_rate']:.1f} | n/a | n/a |")
    L.append("")
    for v in viz:
        L += [f"**{v['i']}. {v['group']} — {v['category']} / {v['attribute']}** (pos ann {v['pos_ann']} green, neg ann {v['neg_ann']} red; "
              + ", ".join(f"{NAMES[k]} {p:.2f}/{n:.2f}" for k, (p, n) in v["scores_pos_neg"].items()) + ")", "",
              f"![pos]({v['pos_img']}) ![neg]({v['neg_img']})", ""]
    open(os.path.join(a.out, "failure.md"), "w").write("\n".join(L))
    print(json.dumps({k: {kk: vv for kk, vv in g.items() if not isinstance(vv, list)} for k, g in res["groups"].items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
