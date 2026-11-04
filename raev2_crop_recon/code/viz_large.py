"""Analysis only: same pipeline as viz_analysis.py on LARGE hval ROIs (F336 n_in >= 17), 6 ROIs by rng(0), to check whether
A/B/C reproduce shape when the full-image ROI tokens do contain it (Bilinear decode shows recognisable shapes there).
  python code/viz_large.py -> figures/reconstruction_comparison_large_rois.png, results/visualization_tiles_large.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rae_common as RC
import viz_analysis as V
hv = pd.read_parquet(os.path.join(RC.CACHE, "feat", "hval_rois.parquet"))
d = hv[hv.n_in >= 17]
idx = np.random.default_rng(0).choice(len(d), 6, replace=False)
sel = [dict(group="L", label="large object", source="hval", roi_id=str(r.roi_id), image_id=int(r.image_id),
            path=os.path.join(RC.COCO_TRAIN, f"{int(r.image_id):012d}.jpg"), ltrb=[int(r.l), int(r.t), int(r.r), int(r.b)],
            category=f"coco cat {int(r.cat_id)}", n_in_f336=int(r.n_in), attrs="") for r in d.iloc[np.sort(idx)].itertuples()]
RC.set_fp32()
tiles = V.reconstruct(sel, "cuda")
V.figure1(tiles, os.path.join(V.FIG, "reconstruction_comparison_large_rois.png"))
rows = [dict(roi_id=t["sel"]["roi_id"], n_in_f336=t["sel"]["n_in_f336"], method=k, **t["met"][k]) for t in tiles for k, _ in V.METHODS]
D = pd.DataFrame(rows); D.to_csv(os.path.join(RC.RES, "visualization_tiles_large.csv"), index=False)
print(D.groupby("method")[["l1", "lpips", "de_mean", "hp_corr", "cos"]].mean().round(3))
