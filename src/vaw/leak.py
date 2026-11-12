"""Step 0: VAW data-leak check against the G1 distillation images and PACO train (ID level, via VG coco_id).

  G1 train  = images actually seen by G1-pool / G1-cross distillation (src.g1eval.paco_dedup.train_ids: COCO 12k stream
              rows 0..11999 minus the 2 excluded ids; identical for both runs) -> 50,489 COCO train2017 ids
  PACO train= image ids in paco_lvis_v1_train.json (COCO ids; used only for the earlier PACO R2 *probe*, never for the
              encoder)
  Mapping   : VAW image_id = VG image_id -> image_data.json coco_id (COCO 2014 id = COCO 2017 id).
              'unmapped' = VAW image_id absent from image_data.json (cannot be checked);
              'coco_id null' = VG metadata says the image has no COCO counterpart (reported separately).

  python -m src.vaw.leak --out output/2026-11-07/vaw_g1/leak
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.g1eval.paco_dedup import train_ids
from src.vaw.common import SPLITS, load_split, vg_meta, PACO_TRAIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g1, cfg = train_ids()
    paco = {int(i["id"]) for i in json.load(open(PACO_TRAIN))["images"]}
    sets = {"G1 train": g1, "PACO train": paco, "G1 ∪ PACO train": g1 | paco}
    meta = vg_meta()
    rows, inst_rows, img_split = [], [], {}
    for s in SPLITS:
        X = load_split(s)
        df = pd.DataFrame(dict(image_id=[int(x["image_id"]) for x in X], instance_id=[int(x["instance_id"]) for x in X]))
        df["split"] = s
        df["in_meta"] = df.image_id.isin(meta.keys())
        df["coco_id"] = [meta[i]["coco_id"] if i in meta and meta[i]["coco_id"] is not None else -1 for i in df.image_id]
        for k, S in sets.items():
            df[k] = df.coco_id.isin(S)
        inst_rows.append(df)
        img = df.drop_duplicates("image_id")
        img_split[s] = set(img.image_id)
        r = dict(split=s, images=len(img), rois=len(df),
                 unmapped_images=int((~img.in_meta).sum()), unmapped_rois=int((~df.in_meta).sum()),
                 coco_null_images=int((img.in_meta & (img.coco_id < 0)).sum()),
                 coco_null_rois=int((df.in_meta & (df.coco_id < 0)).sum()),
                 coco_mapped_images=int((img.coco_id >= 0).sum()), coco_mapped_rois=int((df.coco_id >= 0).sum()))
        for k in sets:
            r[f"{k} images"] = int(img[k].sum())
            r[f"{k} rois"] = int(df[k].sum())
        rows.append(r)
    D = pd.concat(inst_rows, ignore_index=True)
    D.to_parquet(os.path.join(a.out, "instances.parquet"))
    T = pd.DataFrame(rows)
    # cross-split overlap (VG image id and coco id)
    cross = {}
    for i, s in enumerate(SPLITS):
        for t in SPLITS[i + 1:]:
            cross[f"{s}∩{t} vg_image_id"] = len(img_split[s] & img_split[t])
            cs = set(D[(D.split == s) & (D.coco_id >= 0)].coco_id)
            ct = set(D[(D.split == t) & (D.coco_id >= 0)].coco_id)
            cross[f"{s}∩{t} coco_id"] = len(cs & ct)
    dup_inst = int(D.instance_id.duplicated().sum())
    res = dict(g1_train_images=len(g1), g1_rule=f"stream rows 0..{cfg['steps'] - 1} minus {cfg['excluded']['stream']}",
               paco_train_images=len(paco), g1_paco_overlap=len(g1 & paco), union=len(g1 | paco),
               vg_meta_images=len(meta), cross_split=cross, duplicate_instance_ids=dup_inst,
               table=T.to_dict("records"))
    json.dump(res, open(os.path.join(a.out, "leak.json"), "w"), indent=1)
    # markdown
    L = [f"G1 train images {len(g1):,} ({res['g1_rule']}); PACO train images {len(paco):,}; "
         f"G1∩PACO {len(g1 & paco):,}; union {len(g1 | paco):,}", ""]
    L.append("| split | images | ROIs | unmapped img (ROI) | coco_id null img (ROI) | " +
             " | ".join(f"{k} img (%) | {k} ROI (%)" for k in sets) + " |")
    L.append("|" + "---|" * (5 + 2 * len(sets)))
    for r in rows:
        c = [r["split"], f"{r['images']:,}", f"{r['rois']:,}",
             f"{r['unmapped_images']} ({r['unmapped_rois']})",
             f"{r['coco_null_images']:,} ({100 * r['coco_null_images'] / r['images']:.1f}%) / {r['coco_null_rois']:,}"]
        for k in sets:
            c.append(f"{r[f'{k} images']:,} ({100 * r[f'{k} images'] / r['images']:.2f}%)")
            c.append(f"{r[f'{k} rois']:,} ({100 * r[f'{k} rois'] / r['rois']:.2f}%)")
        L.append("| " + " | ".join(c) + " |")
    L += ["", "cross-split: " + ", ".join(f"{k} = {v}" for k, v in cross.items()), f"duplicate instance ids: {dup_inst}"]
    open(os.path.join(a.out, "leak.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
