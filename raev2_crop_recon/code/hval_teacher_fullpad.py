"""Correction (analysis only): recon_rae.py decoded the hval TEACHER from the content-cells-only cache, i.e. with padding cells = 0.
Here every hval GT crop is re-encoded on the fly (full 16x16 latent incl. padding cells, the true teacher) and decoded; metrics identical
to recon_rae.py. Bilinear/A/B/C rows of recon_rae are unaffected (full-image latents are cached completely). Training is unaffected
(teacher only enters the feature loss on content cells). Existing result files are NOT modified.
  python code/hval_teacher_fullpad.py -> results/recon/hval_teacher_fullpad.csv
"""
import os, sys
import numpy as np, pandas as pd, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rae_common as RC, pixel as PX
dev = "cuda"
RC.set_fp32()
rae, lp = RC.FrozenRAE(dev), PX.load_lpips(dev)
hv = pd.read_parquet(os.path.join(RC.CACHE, "feat", "hval_rois.parquet"))
rows = []
for s in range(0, len(hv), 64):
    g = hv.iloc[s:s + 64]
    crops, rects = [], []
    for r in g.itertuples():
        img = Image.open(os.path.join(RC.COCO_TRAIN, f"{int(r.image_id):012d}.jpg")).convert("RGB")
        lb = RC.Letterbox(int(r.r - r.l), int(r.b - r.t))
        crops.append(RC.u8_canvas(img.crop((int(r.l), int(r.t), int(r.r), int(r.b))), lb))
        rects.append(lb.rect())
    x = torch.stack(crops)
    with torch.no_grad():
        T = rae.encode(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = rae.decode(T).float().clamp(0, 1)
        met = PX.rgb_metrics(y, x.to(dev).float() / 255, RC.content_mask(rects, dev), lp)
    for k, r in enumerate(g.itertuples()):
        rows.append(dict(key=str(r.roi_id), rep="teacher_fullpad", n_in_f336=int(r.n_in), **{kk: float(v[k]) for kk, v in met.items()}))
pd.DataFrame(rows).to_csv(os.path.join(RC.RES, "recon", "hval_teacher_fullpad.csv"), index=False)
print(pd.DataFrame(rows)[["l1", "lpips", "de_mean", "hp_corr"]].mean())
