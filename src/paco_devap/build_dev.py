"""Official-protocol Dev AP, step 1: build the PACO-format GT of the fixed 1,000-image train-internal dev split (CPU).

Dev images = output/2026-11-07/maskembed_paco/share/paco_dev_image_ids_1000.json (the dev split every R2 probe since the
G1 pilot used: PACO train minus data_check exclusions, sorted ids, np.random.default_rng(0).choice(ids, 1000)).
GT = the official paco_lvis_v1_train.json restricted to those images, ALL their annotations (objects + parts, as for the
official val / test runs). The train JSON lacks the three fields the attribute evaluator needs; they are added by rules
that reproduce the official val AND test JSONs exactly (every image of both splits, checked here again):
  joint_obj_attribute_categories      = the official 5,797-pair list (identical in val and test)
  neg_category_ids_attrs (per image)  = pairs of every category in neg_category_ids  ∪  pairs (c, a) for which some GT of
                                        category c in the image has a's type KNOWN and a not in attribute_ids
  not_exhaustive_category_ids_attrs   = pairs of every category in not_exhaustive_category_ids  ∪  pairs (c, a) for which
                                        some GT of category c in the image has a's type UNKNOWN
Outputs <out>/data/: paco_dev_gt.json (evaluator input), manifest_dev.parquet + labels_dev.npy (same builder and ROI rule
as the official val / test manifests, src.pacoattr.audit.build), build_dev.json (checks).

  /home/irteam/envs/paco_eval/bin/python -m src.paco_devap.build_dev --out output/2026-11-07/paco_devap
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from src.pacoattr.audit import build

PACO = "/home/irteam/datasets/paco/paco_lvis_v1_{}.json"
DEV_IDS = "output/2026-11-07/maskembed_paco/share/paco_dev_image_ids_1000.json"
DEV_ANNS = "output/2026-11-07/maskembed_paco/share/paco_dev_ann_ids.json"
PROBE_MANIFEST = "third_party/research/2026-10-24/dinov2_oracle_region_decoder/perception_cache/paco_lp/manifest_train.parquet"


def derive(J, pairs):
    """-> {image_id: (neg_attrs, not_exhaustive_attrs)} by the rules in the module docstring."""
    pairs_of = defaultdict(set)
    for p in pairs:
        pairs_of[p["obj"]].add((p["attr"], p["obj-attr"]))
    t_of = {a: t for t, ix in J["attr_type_to_attr_idxs"].items() for a in ix}
    neg, unk = defaultdict(set), defaultdict(set)
    for a in J["annotations"]:
        P = set(a["attribute_ids"])
        for at, k in pairs_of[a["category_id"]]:
            if a["unknown_" + t_of[at]]:
                unk[a["image_id"]].add(k)
            elif at not in P:
                neg[a["image_id"]].add(k)
    out = {}
    for im in J["images"]:
        i = im["id"]
        absent = {k for c in im["neg_category_ids"] for _, k in pairs_of[c]}
        nex = {k for c in im["not_exhaustive_category_ids"] for _, k in pairs_of[c]}
        out[i] = (sorted(absent | neg[i]), sorted(nex | unk[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = os.path.join(a.out, "data")
    os.makedirs(D, exist_ok=True)
    V, T = json.load(open(PACO.format("val"))), json.load(open(PACO.format("test")))
    pairs = V["joint_obj_attribute_categories"]
    assert sorted(json.dumps(x, sort_keys=True) for x in pairs) == sorted(json.dumps(x, sort_keys=True) for x in T["joint_obj_attribute_categories"])
    chk = dict(official_pairs=len(pairs))
    for nm, J in (("val", V), ("test", T)):                          # the rules reproduce the official fields exactly
        d = derive(J, pairs)
        ok = sum(set(d[im["id"]][0]) == set(im["neg_category_ids_attrs"]) and set(d[im["id"]][1]) == set(im["not_exhaustive_category_ids_attrs"])
                 for im in J["images"])
        chk[f"rule_reproduces_official_{nm}"] = f"{ok}/{len(J['images'])}"
        assert ok == len(J["images"]), (nm, ok)
    del T
    Jtr = json.load(open(PACO.format("train")))
    dev_ids = set(json.load(open(DEV_IDS)))
    assert len(dev_ids) == 1000
    for k in ("attributes", "attr_type_to_attr_idxs", "part_categories"):
        assert Jtr[k] == V[k], k
    imgs = [im for im in Jtr["images"] if im["id"] in dev_ids]
    anns = [x for x in Jtr["annotations"] if x["image_id"] in dev_ids]
    assert len(imgs) == 1000
    J = dict(images=imgs, annotations=anns, categories=Jtr["categories"], attributes=Jtr["attributes"],
             attr_type_to_attr_idxs=Jtr["attr_type_to_attr_idxs"], part_categories=Jtr["part_categories"], joint_obj_attribute_categories=pairs)
    d = derive(J, pairs)
    for im in J["images"]:
        im["neg_category_ids_attrs"], im["not_exhaustive_category_ids_attrs"] = d[im["id"]]
    json.dump(J, open(os.path.join(D, "paco_dev_gt.json"), "w"))
    df, Y, bad = build(J, "dev")
    df.to_parquet(os.path.join(D, "manifest_dev.parquet"))
    np.save(os.path.join(D, "labels_dev.npy"), Y)
    # the dev rows every probe scored (attribute-annotated ROIs) must be a subset with identical boxes
    probe_ann = set(json.load(open(DEV_ANNS)))
    old = pd.read_parquet(PROBE_MANIFEST)
    old = old[old.ann_id.isin(probe_ann)].set_index("ann_id")
    m = df.set_index("ann_id").loc[sorted(probe_ann)]
    same_box = bool((m[["l", "t", "r", "b"]].values == old.loc[sorted(probe_ann), ["l", "t", "r", "b"]].values).all())
    lab = (Y >= 0).any(1)
    chk.update(n_images=len(imgs), n_anns=len(anns), n_objects=int((df.level == "object").sum()), n_parts=int((df.level == "part").sum()),
               n_anns_with_known_attr_label=int(lab.sum()), probe_dev_rows=len(probe_ann), probe_dev_rows_subset=probe_ann <= set(df.ann_id),
               probe_dev_boxes_identical=same_box, images_over_300_anns=int((df.groupby("image_id").size() > 300).sum()),
               size_mismatch_images=len(bad), neg_attr_pairs_mean=float(np.mean([len(im["neg_category_ids_attrs"]) for im in imgs])))
    assert chk["probe_dev_rows_subset"] and same_box
    json.dump(chk, open(os.path.join(D, "build_dev.json"), "w"), indent=1)
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
