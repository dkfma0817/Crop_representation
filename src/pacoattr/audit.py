"""Step 0 / 2 audit (CPU): official PACO-LVIS v1 data vs the existing G1 PACO experiment, and the official-scope manifests.

Writes <out>/data:
  manifest_{val,test}.parquet : EVERY official annotation (objects + object-parts; the evaluator keeps anns with a
      non-empty segmentation = all of them) in official JSON annotation order, with the SAME ROI rule as the existing
      probes (outward-rounded integer box int_box, >= 1 px; level = object iff obj_ann_id == id) and actual file sizes.
  labels_{val,test}.npy        : (N, 59) int8 from the official fields: 1 = in attribute_ids, 0 = type known & not listed,
      -1 = type unknown (unknown_<type> = 1). Used ONLY for the sanity checks / reporting, never as model input.
  audit.json / audit.md        : split sizes, vocab / type / category checks, existing manifest + cache coverage,
      checkpoint table (paths, sha256, selection), existing subsets (probe val, test-primary).

  python -m src.pacoattr.audit --out output/2026-11-07/paco_oracle_attr
"""
import argparse
import hashlib
import json
import math
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

PACO = "/home/irteam/datasets/paco"
COCO = "/home/irteam/datasets/coco"
OLD = "third_party/research/2026-10-24/dinov2_oracle_region_decoder/perception_cache/paco_lp"
G1 = "output/2026-11-06/g1_cross_distill"
C0 = "output/2026-11-06/ivc_v3/probe_c0"
CONDS = {"frozen_full": (f"{G1}/probe_frozen_full", None, "full"),
         "g1pool_full": (f"{G1}/probe_G1pool_full", f"{G1}/G1pool/ema_last.pt", "full"),
         "g1cross_full": (f"{G1}/probe_G1cross_full", f"{G1}/G1cross/ema_last.pt", "full"),
         "g1cross_dec": (f"{G1}/probe_G1cross_dec", f"{G1}/G1cross/ema_last.pt", "dec"),
         "c0_crop": (C0, None, "c0")}
TYPES = ("color", "pattern_marking", "material", "transparency")
SEL_EPOCH = 6           # dev-M0 selected epoch of every condition (g1_cross_distill/results.json 'epochs'); fixed


def int_box(x, y, w, h, W, H):
    """= dinov2_oracle_region_decoder data.preprocessing.int_box (the existing manifests' rule)."""
    l = max(0, min(W - 1, int(math.floor(x))))
    t = max(0, min(H - 1, int(math.floor(y))))
    r = max(l + 1, min(W, int(math.ceil(x + w))))
    b = max(t + 1, min(H, int(math.ceil(y + h))))
    return l, t, r, b


def sha(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def _size(p):
    with Image.open(p) as im:
        return im.size


def build(J, sp):
    cats = {c["id"]: c for c in J["categories"]}
    imgs = {im["id"]: im for im in J["images"]}
    paths = {i: os.path.join(COCO, im["file_name"]) for i, im in imgs.items()}
    with Pool(48) as p:
        sz = dict(zip(paths, p.map(_size, list(paths.values()), chunksize=32)))
    bad = [i for i in imgs if tuple(sz[i]) != (imgs[i]["width"], imgs[i]["height"])]
    t_of = {a: t for t, ix in J["attr_type_to_attr_idxs"].items() for a in ix}
    rows, Y = [], []
    for a in J["annotations"]:
        im = imgs[a["image_id"]]
        W, H = sz[a["image_id"]]
        x, y, w, h = a["bbox"]
        l, t, r, b = int_box(x, y, w, h, W, H)
        lab = np.full(59, -1, np.int8)
        for ty in TYPES:
            if not a[f"unknown_{ty}"]:
                lab[J["attr_type_to_attr_idxs"][ty]] = 0
        for k in a["attribute_ids"]:
            lab[k] = 1 if lab[k] == 0 else lab[k]
        rows.append(dict(ann_id=a["id"], image_id=a["image_id"], obj_ann_id=a["obj_ann_id"],
                         level="object" if a["obj_ann_id"] == a["id"] else "part", category_id=a["category_id"],
                         category=cats[a["category_id"]]["name"], W=W, H=H, path=paths[a["image_id"]],
                         bx=x, by=y, bw=w, bh=h, area=a["area"], l=l, t=t, r=r, b=b,
                         n_attr_ids=len(a["attribute_ids"]), pos_in_unknown_type=int(sum(a[f"unknown_{t_of[k]}"] for k in a["attribute_ids"])),
                         **{f"unknown_{ty}": int(a[f"unknown_{ty}"]) for ty in TYPES}, split=sp))
        Y.append(lab)
    return pd.DataFrame(rows), np.stack(Y), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = os.path.join(a.out, "data")
    os.makedirs(D, exist_ok=True)
    R = dict(official={}, existing={}, vocab={}, checkpoints={}, caches={}, subsets={})
    old_attr = json.load(open(os.path.join(OLD, "attributes.json")))
    for sp in ("train", "val", "test"):
        f = os.path.join(PACO, f"paco_lvis_v1_{sp}.json")
        J = json.load(open(f))
        cats = {c["id"]: c for c in J["categories"]}
        anns = J["annotations"]
        per = Counter(x["image_id"] for x in anns)
        jp = J.get("joint_obj_attribute_categories", [])
        r = dict(file=f, md5=hashlib.md5(open(f, "rb").read()).hexdigest(), images=len(J["images"]), annotations=len(anns),
                 objects=sum(x["obj_ann_id"] == x["id"] for x in anns), parts=sum(x["obj_ann_id"] != x["id"] for x in anns),
                 seg_empty=sum(len(x["segmentation"]) == 0 for x in anns), attr_annotated=sum(len(x["attribute_ids"]) > 0 for x in anns),
                 max_anns_per_image=max(per.values()), images_gt300_anns=sum(v > 300 for v in per.values()),
                 anns_beyond_300=int(sum(max(0, v - 300) for v in per.values())),
                 categories=len(J["categories"]), obj_categories=sum(":" not in c["name"] for c in J["categories"]),
                 part_categories=sum(":" in c["name"] for c in J["categories"]), attributes=len(J["attributes"]),
                 joint_pairs=len(jp), joint_pairs_obj=sum(":" not in cats[p["obj"]]["name"] for p in jp),
                 level_vs_name_mismatch=sum((x["obj_ann_id"] == x["id"]) == (":" in cats[x["category_id"]]["name"]) for x in anns),
                 image_dirs=dict(Counter(im["file_name"].split("/")[0] for im in J["images"])))
        R["official"][sp] = r
        if sp == "test":
            R["vocab"] = dict(attributes=[(x["id"], x["name"]) for x in J["attributes"]],
                              attr_type_to_attr_idxs=J["attr_type_to_attr_idxs"],
                              old_names_equal_official=old_attr["names"] == [x["name"] for x in sorted(J["attributes"], key=lambda z: z["id"])],
                              old_type_to_idx_equal_official=old_attr["type_to_idx"] == J["attr_type_to_attr_idxs"],
                              bg_attributes_excluded_by_official_reporting=["other(pattern_marking)", "other(material)"],
                              other_attributes=[x["name"] for x in J["attributes"] if x["name"].startswith("other")],
                              pairs_equal_val=None)
            pt = {(p["obj"], p["attr"]) for p in jp}
        if sp == "val":
            pv = {(p["obj"], p["attr"]) for p in jp}
        if sp in ("val", "test"):
            df, Y, bad = build(J, sp)
            r["json_size_mismatch_images"] = len(bad)
            df.to_parquet(os.path.join(D, f"manifest_{sp}.parquet"))
            np.save(os.path.join(D, f"labels_{sp}.npy"), Y)
            # existing manifest of the same split
            old = pd.read_parquet(os.path.join(OLD, f"manifest_{sp}.parquet"))
            oY = np.load(os.path.join(OLD, f"labels_{sp}.npy"))
            m = df.set_index("ann_id")
            ok_ids = old.ann_id.isin(m.index)
            mm = m.loc[old.ann_id[ok_ids]]
            idx = df.reset_index().set_index("ann_id").loc[old.ann_id[ok_ids], "index"].values
            R["existing"][sp] = dict(
                rows=len(old), images=int(old.image_id.nunique()), degenerate_rows=int(((old.bw <= 0) | (old.bh <= 0)).sum()),
                ann_ids_in_official=int(ok_ids.sum()), image_ids_in_official=int(old.image_id.isin(set(df.image_id)).sum()),
                same_bbox=int(np.isclose(mm[["bx", "by", "bw", "bh"]].values, old.loc[ok_ids, ["bx", "by", "bw", "bh"]].values).all(1).sum()),
                same_int_box=int((mm[["l", "t", "r", "b"]].values == old.loc[ok_ids, ["l", "t", "r", "b"]].values).all(1).sum()),
                same_category=int((mm.category_id.values == old.category_id[ok_ids].values).sum()),
                same_level=int((mm.level.values == old.level[ok_ids].values).sum()),
                same_labels=int((Y[idx] == oY[ok_ids.values]).all(1).sum()),
                official_anns_not_in_existing=int((~df.ann_id.isin(set(old.ann_id))).sum()),
                of_which_no_attribute_ids=int(((~df.ann_id.isin(set(old.ann_id))) & (df.n_attr_ids == 0)).sum()),
                official_images_not_in_existing=int(len(set(df.image_id) - set(old.image_id))))
            if sp == "test":
                prim = ~old.in_decoder_pool
                R["subsets"]["test_primary"] = dict(rule="existing manifest_test rows with in_decoder_pool == False (paco_lp 'primary test')",
                                                    images=int(old.image_id[prim].nunique()), rows=int(prim.sum()))
            if sp == "val":
                R["subsets"]["probe_val"] = dict(rule="existing manifest_val minus degenerate boxes (src.readout.common.load_data)",
                                                 images=int(old[(old.bw > 0) & (old.bh > 0)].image_id.nunique()),
                                                 rows=int(((old.bw > 0) & (old.bh > 0)).sum()))
        if sp == "train":
            old = pd.read_parquet(os.path.join(OLD, "manifest_train.parquet"))
            R["existing"]["train"] = dict(rows=len(old), images=int(old.image_id.nunique()),
                                          image_ids_in_official=int(old.image_id.isin({im["id"] for im in J["images"]}).sum()),
                                          probe_rule="minus data_check exclusions and a 1,000-image PACO dev split (rng 0) -> 37,472 images")
        del J
    R["vocab"]["pairs_equal_val"] = pv == pt
    # checkpoints / caches
    for c, (pdir, enc, mode) in CONDS.items():
        cfg = json.load(open(os.path.join(pdir, "config.json")))
        r2 = os.path.join(pdir, "r2", f"ep{SEL_EPOCH}.pt")
        e = dict(mode=mode, probe_dir=pdir, r2_ckpt=r2, r2_sha256=sha(r2), selected_epoch=SEL_EPOCH,
                 selection="max PACO-dev M0 (1,000 PACO train images held out, rng 0); g1_cross_distill/results.json",
                 encoder=enc or "timm vit_base_patch16_dinov3.lvd1689m (frozen original)",
                 encoder_sha256=sha(enc) if enc else None, decoder=(enc + " ['dec']") if mode == "dec" else None,
                 probe_train_rows=cfg.get("n_train_rows"), probe_train_images=cfg.get("n_train_images"),
                 probe_val_rows=cfg.get("n_val_rows"), probe_world=cfg.get("world"), job=cfg.get("job"))
        if enc:
            gc = json.load(open(os.path.join(os.path.dirname(enc), "config.json")))
            e["distill"] = {k: gc.get(k) for k in ("kind", "steps", "seed", "excluded", "lr", "teacher", "lam_detail", "lam_global") if k in gc}
        R["checkpoints"][c] = e
        sc = os.path.join(pdir, "r2", f"scores_val_ep{SEL_EPOCH}.npy")
        R["caches"][c] = dict(file=sc, shape=list(np.load(sc, mmap_mode="r").shape), split="existing probe val (11,109 rows / 2,051 images)",
                              test_scores="none (test was never scored)")
    json.dump(R, open(os.path.join(a.out, "audit.json"), "w"), indent=1, default=str)
    print(json.dumps({k: R[k] for k in ("official", "existing", "subsets", "caches")}, indent=1, default=str))
    print(json.dumps({k: v for k, v in R["vocab"].items() if k != "attributes"}, default=str))


if __name__ == "__main__":
    main()
