"""Evaluation subsets from the leak checks (src.vaw.leak + src.vaw.dhash + visual inspection of the evidence montages).

G1-exposed test images = coco_id in the G1 distillation set  ∪  visually confirmed near-duplicates of a G1 image among the
test images whose coco_id is not in that set (candidates: dHash <= 4 and an exhaustive 32x32 thumbnail correlation search
over all val/test x 50,489 G1 pairs, incl. horizontal flip; montages in leak/evidence).
Cross-split duplicate test images = test images whose coco_id also occurs in VAW train/val (51 pairs, all confirmed
visually) ∪ test images dHash-identical to a train/val image with a different VG id (28 pairs; 20 confirmed visually).

  python -m src.vaw.subsets --out output/2026-11-07/vaw_g1/leak
"""
import argparse
import json
import os

import pandas as pd

# visually confirmed (leak/evidence/unmapped_vs_g1_corr085.jpg, thumbcorr_lowmult.jpg, zoom_check.jpg)
G1_NEAR_DUP = {
    "test": {2361493: (40338, "same photo"), 2388421: (203538, "same photo"), 2411334: (209906, "same photo"),
             2393651: (542458, "same photo"), 2374998: (310052, "same photo, colour-processed"),
             2386723: (412757, "same photo, colour-processed"), 2384669: (216910, "same photo, selective-colour edit"),
             2380588: (472954, "adjacent frame"), 2401508: (267064, "same scene / session (bus present vs absent)"),
             2397515: (231104, "same photographer + airport session, different shot"),
             2410508: (64190, "same beach session, different moment"), 2415904: (487696, "adjacent frame")},
    "val": {2404741: (360525, "same photo"), 2396771: (39472, "same photo, colour-processed"),
            2383517: (301397, "same photo"), 1160046: (209388, "same location / session, different train"),
            2386362: (105582, "adjacent frame"), 2393220: (556278, "adjacent frame")},
}
# dHash-identical test vs train/val pairs (not coco_id pairs) judged NOT the same photo (generic / different scenes)
DHASH0_REJECT = {(2398717, 2389314), (2403391, 2315607), (2403391, 2372951), (2403391, 655), (2403391, 2396999),
                 (2403391, 2413048), (2407505, 2372620), (2324710, 2390308)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = pd.read_parquet(os.path.join(a.out, "instances.parquet"))
    ev = os.path.join(a.out, "evidence")
    te = D[D.split == "test"]
    by_coco = sorted(set(te[te["G1 train"]].image_id))
    near = sorted(G1_NEAR_DUP["test"])
    assert not (set(near) & set(by_coco))
    P = pd.read_csv(os.path.join(ev, "cross_split_cocoid_pairs.csv"))
    cs = set(P.a_vg[P.a_split == "test"]) | set(P.b_vg[P.b_split == "test"])
    E = pd.read_csv(os.path.join(ev, "test_vs_trainval_dhash0.csv"))
    E["ok"] = [(int(t), int(o)) not in DHASH0_REJECT for t, o in zip(E.test_vg, E.other_vg)]
    cs2 = set(E.test_vg[E.ok])
    exposed = sorted(set(by_coco) | set(near))
    dup = sorted(cs | cs2)
    n_roi = lambda ids: int(te.image_id.isin(ids).sum())  # noqa: E731
    out = dict(g1_exposed_test_images=[int(i) for i in exposed], cross_split_dup_test_images=[int(i) for i in dup],
               summary=dict(test_images=int(te.image_id.nunique()), test_rois=len(te),
                            g1_by_coco_id=dict(images=len(by_coco), rois=n_roi(by_coco)),
                            g1_near_dup_visual=dict(images=len(near), rois=n_roi(near)),
                            g1_exposed_total=dict(images=len(exposed), rois=n_roi(exposed)),
                            val_g1_near_dup_visual=len(G1_NEAR_DUP["val"]),
                            cross_split_by_coco_id=dict(test_images=len(cs), pairs=len(P), pairs_confirmed=len(P)),
                            cross_split_by_dhash0=dict(test_images=len(cs2), pairs=len(E), pairs_confirmed=int(E.ok.sum())),
                            cross_split_total=dict(images=len(dup), rois=n_roi(dup))),
               near_dup_details={sp: {str(k): v for k, v in d.items()} for sp, d in G1_NEAR_DUP.items()})
    json.dump(out, open(os.path.join(a.out, "subsets.json"), "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
