"""Training stream for A/B/C: the existing category-balanced sampler (relation.build_stream, 4 categories x 4 images, seed 2027,
12,000 steps) on the COCO decoder pool (coco_rois.parquet split 'train', 60,000 images x 4 ROIs) MINUS every image that appears in
PACO val or test (any level / split dir; matched by COCO image id). PACO train images may stay (attribute labels are never used).
Saved as COCO image ids (12000, 16) -> no index-convention ambiguity.  python code/make_stream.py
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rae_common as RC
from relation import build_stream
M = "/workspace/Areum/research/2026-10-24/dinov2_oracle_region_decoder/perception_cache/paco_lp"
cid = lambda p: int(os.path.basename(p).split(".")[0].split("_")[-1])
excl = set()
for sp in ("val", "test"):
    excl |= {cid(p) for p in pd.read_parquet(os.path.join(M, f"manifest_{sp}.parquet")).path.unique()}
r = pd.read_parquet(RC.COCO_ROIS)
tr = r[r.split == "train"]
pool = tr[~tr.image_id.isin(excl)]
st, info = build_stream(pool, 12000, 2027)
groups = np.array(sorted(pool.image_id.unique()))
ids = groups[st]
os.makedirs(os.path.join(RC.CACHE, "stream"), exist_ok=True)
np.save(os.path.join(RC.CACHE, "stream", "stream_ids_12000.npy"), ids)
info.pop("cat_probs")
info.update(steps=12000, seed=2027, pool_images=int(tr.image_id.nunique()), excluded_paco_valtest_images_in_pool=int(tr.image_id.isin(excl).sum() // 4),
            pool_after=int(pool.image_id.nunique()), stream_cap_paco_valtest=int(len(set(ids.flatten()) & excl)))
json.dump(info, open(os.path.join(RC.CACHE, "stream", "stream_ids_12000.json"), "w"), indent=1)
print(info)
