"""Leak subsets from src.pacoattr.leak (+ visual inspection of leak/evidence/best_*.jpg / focus_*.jpg).

  exact   : official images that ARE G1 distillation images (COCO id == file md5 == pixel md5 == dHash match) -- val 146,
            test 475; none of them carries an attribute-annotated annotation (attribute_ids empty for every ann)
  near    : visually confirmed same photo / same session pairs (below); val list includes the 4 pairs of the earlier
            probe-val check (src.g1eval.paco_dedup.NEAR_DUP) + 1 new image that was outside the old probe-val set
Subsets (whole images removed with all their annotations; the official pair list is unchanged, valid pairs may change):
  test_full, test_dedup (minus exact + near) ; val_full, val_minus_prior4 (old sensitivity), val_dedup (minus exact + near)

  python -m src.pacoattr.subsets --out output/2026-11-07/paco_oracle_attr/leak
"""
import argparse
import json
import os

import pandas as pd

NEAR = {"test": {372203: (247609, "same photo"), 401589: (18529, "same photo"), 43719: (494434, "same photo, colour-processed"),
                 253480: (78336, "same photo"), 175734: (42680, "same photo (B&W)"), 204074: (444262, "same photo"),
                 368684: (247130, "same photo"), 134599: (441419, "same photo"), 20333: (119502, "same photo"),
                 283237: (524628, "same photo"), 24242: (440349, "same photo"), 2849: (402082, "same scene, adjacent frame"),
                 79268: (79191, "same scene, adjacent frame")},
        "val": {110596: (200149, "same photo, colour-processed (prior)"), 110170: (393664, "adjacent burst frame (prior)"),
                137586: (361382, "same session, different pose (prior)"), 3286: (68084, "same photo + caption overlay (prior)"),
                568071: (434072, "same location / session, different shot (new)")}}
PRIOR4 = [110596, 110170, 137586, 3286]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    auto = json.load(open(os.path.join(a.out, "leak_auto.json")))
    R = dict(near=NEAR, subsets={}, summary={})
    for sp in ("val", "test"):
        M = pd.read_parquet(f"output/2026-11-07/paco_oracle_attr/data/manifest_{sp}.parquet")
        ex = set(auto["exact"][sp]["image_id"])
        assert ex == set(auto["exact"][sp]["md5"]) == set(auto["exact"][sp]["pix_md5"])
        near = set(NEAR[sp])
        assert not (near & ex) and near <= set(M.image_id)
        allimg = sorted(int(i) for i in M.image_id.unique())
        drop = ex | near
        R["subsets"][f"{sp}_full"] = allimg
        R["subsets"][f"{sp}_dedup"] = [i for i in allimg if i not in drop]
        if sp == "val":
            R["subsets"]["val_minus_prior4"] = [i for i in allimg if i not in set(PRIOR4)]
        att = M[M.n_attr_ids > 0]
        R["summary"][sp] = dict(images=len(allimg), anns=len(M), exact_images=len(ex), exact_anns=int(M.image_id.isin(ex).sum()),
                                exact_attr_annotated_anns=int(att.image_id.isin(ex).sum()), near_images=len(near),
                                near_anns=int(M.image_id.isin(near).sum()), near_attr_annotated_anns=int(att.image_id.isin(near).sum()),
                                dedup_images=len(R["subsets"][f"{sp}_dedup"]), dedup_anns=int((~M.image_id.isin(drop)).sum()))
    for k, v in R["subsets"].items():
        json.dump(v, open(os.path.join(a.out, f"images_{k}.json"), "w"))
    json.dump(R, open(os.path.join(a.out, "subsets.json"), "w"), indent=1, default=str)
    print(json.dumps(R["summary"], indent=1))


if __name__ == "__main__":
    main()
