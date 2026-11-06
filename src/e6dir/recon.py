"""Representation metrics vs the C0 teacher for Bicubic, E6-original (xdec selected ckpt) and E6-direction (selected ckpt),
on PACO val (11,109 ROIs; object / part x D-bin) and COCO dev (1,872 ROIs). Same ROI lists / coverage for every method.
Per ROI (w = valid_w): token MSE_w, token cos_w, loss L_old, centred direction cos c_dir (src.e6dir.loss, the training
definition), pooled cos, mean-feature MSE e_mean = mean_c (mu_x - mu_T)^2, spatial-variance ratio, centred diff e_cent.

  CUDA_VISIBLE_DEVICES=2 python -m src.e6dir.recon --run <run>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import src.msd.common as MS
import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT
from src.e6dir.loss import centered_dir
from src.msd.pretrain import eval_exclusions
from src.xdec import geometry as XG
from src.xdec import models as XM
from src.xdec.recon import metrics

EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    XG.install()
    torch.set_num_threads(96)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(a.run, "results", "recon")
    os.makedirs(out, exist_ok=True)
    decs = {}
    for nm, p in (("E6-original", "output/2026-11-05/xdec/checkpoints/E6/selected.pt"),
                  ("E6-direction", os.path.join(a.run, "checkpoints", "E6dir", "selected.pt"))):
        if os.path.exists(p):
            ck = torch.load(p, map_location="cpu")
            d = XM.build("E6").to(dev)
            d.load_state_dict(ck["decoder"])
            decs[nm] = d.eval()
    student = MS.Student(adapters=False).to(dev).eval()
    vit = student.vit

    def run_split(name, ds, chunks, extra):
        rows = []
        for ch in chunks:
            b = MS.to_dev(ds[ch], dev)
            with torch.no_grad():
                feats, _ = student(b["full"])
                T = torch.cat([RC.encode_layers(vit, b["crops"][i:i + 48], (12,))[12] for i in range(0, len(b["crops"]), 48)])
                g = b["geom"]
                w = g["valid_w"].float()
                Zs = {"Bicubic": XG.bicubic_roi(feats[12], g)}
                for nm, d in decs.items():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        Zs[nm] = d(feats, g).float()
                ex = extra(b)
                for nm, Z in Zs.items():
                    mt = {k: v.double().cpu().numpy() for k, v in metrics(Z, T, w).items()}
                    c, eT, _ = centered_dir(Z, T, w)
                    ws = w.sum(1)
                    pa, pc = (Z * w[..., None]).sum(1) / ws[:, None], (T * w[..., None]).sum(1) / ws[:, None]
                    mt["c_dir"] = c.double().cpu().numpy()
                    mt["e_T"] = eT.double().cpu().numpy()
                    mt["e_mean"] = (pa - pc).pow(2).mean(-1).double().cpu().numpy()
                    df = pd.DataFrame(mt)
                    df["cond"] = nm
                    for k, v in ex.items():
                        df[k] = v
                    rows.append(df)
        res = pd.concat(rows, ignore_index=True)
        res.to_parquet(os.path.join(out, f"{name}_roi_metrics.parquet"))
        return res

    ex = eval_exclusions()
    dev_rois = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    dev_rois = dev_rois[~dev_rois.image_id.isin(ex["raev2_dev"])]
    ids = sorted(dev_rois.image_id.unique())[: (a.limit or None)]
    r1 = run_split("coco_dev", MS.CocoStreamDS(dev_rois, COCO_TRAIN), [ids[i:i + 8] for i in range(0, len(ids), 8)],
                   lambda b: dict(n_in=b["n_in"].cpu().numpy(), is_gt=b["is_gt"].cpu().numpy()))
    data = RC.load_data(EXCL, a.limit)
    dva, Yva = data["val"]
    vds = MS.MSDImageDS(dva, Yva, flip=False, want_crops=True)
    n = len(vds)
    r2 = run_split("paco_val", vds, [list(range(i, min(n, i + 8))) for i in range(0, n, 8)],
                   lambda b: dict(row=b["rows"].cpu().numpy(), n_in=b["geom"]["n_in"].cpu().numpy(), part=b["lv"].cpu().numpy()))
    info = dict(conds=["Bicubic"] + list(decs), coco_dev_rois=int((r1.cond == "Bicubic").sum()), paco_val_rois=int((r2.cond == "Bicubic").sum()), device=str(dev))
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    print("DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
