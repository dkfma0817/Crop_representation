"""Teacher-feature reconstruction metrics (no training): Bicubic, R00, E1-E7 (selected checkpoints) vs the C0 teacher,
on COCO dev (468 images, 1,872 ROIs: the checkpoint-selection set) and PACO val (2,051 images, 11,109 ROIs; object / part).
Every method sees the same frozen F336 forward of the same batch; the teacher is the frozen encoder on the C0 crop.

Per ROI, w = C0 valid coverage, M = w > 0, all fp32, raw final-LN space:
  loss     = MSE_w + (1 - cos)_w            (= training / dev feature loss)
  mse, tok_cos                              (coverage-weighted over M)
  pool_cos = cos of the coverage-weighted ROI means (= R0 input)
  e_cent   = sum_w mean_c ((a - a_pool) - (c - c_pool))^2 / sum_w      centred spatial difference
  cent_cos = coverage-weighted cos(a - a_pool, c - c_pool)
  sv_ratio = spatial variance(a) / spatial variance(c)
  ss_diff  = mean |S_a - S_c| over valid off-diagonal pairs, S = token cosine self-similarity on M
            (IVC used the fully covered rectangle M1 for ss_diff; here M, so values are not identical to IVC's)

  CUDA_VISIBLE_DEVICES=2 python -m src.xdec.recon --run output/2026-11-05/xdec
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import src.msd.common as MS
import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT
from src.msd.pretrain import eval_exclusions
from src.xdec import geometry as XG
from src.xdec import models as XM

CONDS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "R00"]
EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"


@torch.no_grad()
def metrics(a, c, w):
    """a, c (R, 576, D) fp32, w (R, 576) -> dict of (R,) tensors."""
    m = (w > 0).float()
    ws = w.sum(1)
    mse = ((a - c).pow(2).mean(-1) * w).sum(1) / ws
    tcos = (F.cosine_similarity(a, c, dim=-1, eps=1e-6) * w).sum(1) / ws
    pa = (a * w[..., None]).sum(1) / ws[:, None]
    pc = (c * w[..., None]).sum(1) / ws[:, None]
    ra, rc = a - pa[:, None], c - pc[:, None]
    e_cent = ((ra - rc).pow(2).mean(-1) * w).sum(1) / ws
    cent_cos = (F.cosine_similarity(ra, rc, dim=-1, eps=1e-6) * w).sum(1) / ws
    sv_a = (ra.pow(2).mean(-1) * w).sum(1) / ws
    sv_c = (rc.pow(2).mean(-1) * w).sum(1) / ws
    an, cn = F.normalize(a, dim=-1), F.normalize(c, dim=-1)
    pair = m[:, :, None] * m[:, None, :]
    pair = pair * (1 - torch.eye(a.shape[1], device=a.device))[None]
    ss = ((an @ an.transpose(1, 2) - cn @ cn.transpose(1, 2)).abs() * pair).sum((1, 2)) / pair.sum((1, 2)).clamp_min(1)
    return dict(loss=mse + 1 - tcos, mse=mse, tok_cos=tcos, pool_cos=F.cosine_similarity(pa, pc, dim=-1), e_cent=e_cent,
                cent_cos=cent_cos, sv_ratio=sv_a / sv_c.clamp_min(1e-12), ss_diff=ss, n_valid=m.sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    XG.install()
    torch.set_num_threads(96)
    out = os.path.join(a.run, "results", "recon")
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # cpu only for code tests
    decs = {}
    for c in CONDS:
        p = os.path.join(a.run, "checkpoints", c, "selected.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location="cpu")
            d = XM.build(c).to(dev)
            d.load_state_dict(ck["decoder"])
            decs[c] = d.eval()
    student = MS.Student(adapters=False).to(dev).eval()
    vit = student.vit
    info = dict(conds=list(decs), time=time.strftime("%F %T"))

    def run_split(name, ds, chunks, extra_cols):
        rows = []
        for ch in chunks:
            b = MS.to_dev(ds[ch], dev)
            with torch.no_grad():
                feats, _ = student(b["full"])
                T = torch.cat([RC.encode_layers(vit, b["crops"][i:i + 48], (12,))[12] for i in range(0, len(b["crops"]), 48)])
                g = b["geom"]
                w = g["valid_w"].float()
                Zs = {"Bicubic": XG.bicubic_roi(feats[12], g)}
                for c, d in decs.items():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        Zs[c] = d(feats, g).float()
                ex = extra_cols(b)
                for c, Z in Zs.items():
                    mt = {k: v.double().cpu().numpy() for k, v in metrics(Z, T, w).items()}
                    df = pd.DataFrame(mt)
                    df["cond"] = c
                    for k, v in ex.items():
                        df[k] = v
                    rows.append(df)
        res = pd.concat(rows, ignore_index=True)
        res.to_parquet(os.path.join(out, f"{name}_roi_metrics.parquet"))
        return res

    # ---------------------------------------------------------------- COCO dev (checkpoint-selection set)
    ex = eval_exclusions()
    dev_rois = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex["raev2_dev"])]
    ids = sorted(dev_rois.image_id.unique())[: (a.limit or None)]
    dds = MS.CocoStreamDS(dev_rois, COCO_TRAIN)
    r1 = run_split("coco_dev", dds, [ids[i:i + 8] for i in range(0, len(ids), 8)],
                   lambda b: dict(n_in=b["n_in"].cpu().numpy(), is_gt=b["is_gt"].cpu().numpy()))
    info["coco_dev_rois"] = int((r1.cond == "Bicubic").sum())
    # ---------------------------------------------------------------- PACO val (object / part)
    data = RC.load_data(EXCL, a.limit)
    dva, Yva = data["val"]
    vds = MS.MSDImageDS(dva, Yva, flip=False, want_crops=True)
    n = len(vds)
    r2 = run_split("paco_val", vds, [list(range(i, min(n, i + 8))) for i in range(0, n, 8)],
                   lambda b: dict(row=b["rows"].cpu().numpy(), n_in=b["geom"]["n_in"].cpu().numpy(), part=b["lv"].cpu().numpy()))
    info["paco_val_rois"] = int((r2.cond == "Bicubic").sum())
    assert info["paco_val_rois"] == len(dva), (info["paco_val_rois"], len(dva))
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    print("DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
