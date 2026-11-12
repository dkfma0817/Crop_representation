"""Official PACO VAL inference for one decoder-study path at the FIXED R2 epoch (default 6; = src.g1clip.infer row set
and output format). Rows = output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet (every official val annotation).
Output <out>/pred_val.npz (+ probs_val.npz symlink for src.pacoattr.official_eval) and infer_val.json.
Checks (--checks, first 10 batches): one encoder canvas per image (no RGB crop re-encoding for G1 paths); R2 reads only
cov > 0 cells (noise outside -> logits unchanged); crop-grid paths: decoder output replaced by zeros changes the logits;
residual paths: Delta removed (Z = B) changes the logits (the trained residual is used).

  CUDA_VISIBLE_DEVICES=<gpu> python -m src.decstudy.infer --probe <probe dir> --out <dir> [--epoch 6] [--checks]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.decstudy.paths import load_path
from src.g1clip.probe import EVAL_BS, encoder_tokens, make_ds, set_precision
from src.maskemb import common as MC
from src.maskemb.reader import build as build_r2

DATA = "output/2026-11-07/paco_oracle_attr/data"


def job_of(cfg):
    job = dict(cfg["job"])
    if "kind" not in job:                     # existing g1clip probes
        job["kind"] = {"full": "enc", "dec": "dec_existing"}[job["mode"]]
    return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epoch", type=int, default=6)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--checks", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"pred_{a.split}.npz")
    assert not os.path.exists(fn), f"{fn} exists (scored once)"
    MC.set_clip_norm()
    dev = torch.device("cuda")
    t0 = time.time()
    cfg = json.load(open(os.path.join(a.probe, "config.json")))
    job = job_of(cfg)
    vit, dec, einfo = load_path(job, dev)
    mode, prec = einfo["mode"], cfg["precision"]
    assert mode == job["mode"]
    set_precision(prec)
    rp = os.path.join(a.probe, "r2", f"ep{a.epoch}.pt")
    ck = torch.load(rp, map_location=dev, weights_only=False)
    assert ck["epoch"] == a.epoch and ck["mode"] == mode
    reader = build_r2(MC.DIM).to(dev)
    reader.load_state_dict(ck["reader"])
    reader.eval()
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    ds = make_ds(M, np.zeros((len(M), 59), np.int8), False, mode)
    bs = EVAL_BS[mode]
    chunks = [list(range(i, min(len(ds), i + bs))) for i in range(0, len(ds), bs)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
    L = torch.full((len(M), 59), float("nan"))
    st = dict(images=0, encoder_canvases=0, rois=0, sec_encode=0.0, sec_reader=0.0)
    ck_ = dict(outside_max_abs=[], inside_mean_abs=[], dec_zero_mean_abs=[], delta_removed_mean_abs=[])
    for bi, b in enumerate(dl):
        n_img = len({int(g) for g in b["group"]})
        st["images"] += n_img
        st["encoder_canvases"] += int(b["x"].shape[0])
        st["rois"] += len(b["rows"])
        if mode != "c0":
            assert b["x"].shape[0] == n_img, "one full-image encoding per image"
        b = RC.to_dev(b, dev)
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            tok = encoder_tokens(vit, b["x"], prec)
            extra = {}
            if dec is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if job["kind"] == "residual":
                        Bf, dlt = dec.parts(tok, b["img_index"], b["box_c"], b["qrel"], b["fvalid"])
                        z = Bf + dlt
                        extra["B"] = Bf
                    else:
                        z = dec(tok, b["img_index"], b["box_c"], b["qrel"], b["fvalid"]).float()
                tok = z.float()
                b["img_index"] = torch.arange(len(tok), device=dev)
            torch.cuda.synchronize()
            t2 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = reader({12: tok}, b)["logits"].float()
            torch.cuda.synchronize()
            st["sec_encode"] += t2 - t1
            st["sec_reader"] += time.time() - t2
            L[b["rows"].cpu()] = lo.cpu()
            if a.checks and bi < 10:
                t_ = tok[b["img_index"]]
                m = (b["cov"] > 0)[..., None]
                noise = torch.randn_like(t_) * t_.std()
                b2 = dict(b)
                b2["img_index"] = torch.arange(len(t_), device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo_out = reader({12: torch.where(m, t_, noise)}, b2)["logits"].float()
                    lo_in = reader({12: torch.where(m, noise, t_)}, b2)["logits"].float()
                    if dec is not None:
                        ck_["dec_zero_mean_abs"].append(float((reader({12: torch.zeros_like(t_)}, b2)["logits"].float() - lo).abs().mean()))
                    if "B" in extra:
                        ck_["delta_removed_mean_abs"].append(float((reader({12: extra["B"].float()}, b2)["logits"].float() - lo).abs().mean()))
                ck_["outside_max_abs"].append(float((lo_out - lo).abs().max()))
                ck_["inside_mean_abs"].append(float((lo_in - lo).abs().mean()))
    assert not torch.isnan(L).any()
    np.savez(fn, ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values,
             bbox_xywh=M[["bx", "by", "bw", "bh"]].values.astype(np.float64), level=M.level.values.astype(str),
             logits=L.numpy().astype(np.float32), probs=torch.sigmoid(L).numpy().astype(np.float32))
    os.symlink(os.path.basename(fn), os.path.join(a.out, f"probs_{a.split}.npz"))
    chk = {k: (dict(max=max(v), mean=float(np.mean(v))) if v else None) for k, v in ck_.items()}
    info = dict(probe=a.probe, epoch=a.epoch, reader_ckpt=rp, reader_sha256=MC.sha256(rp), job=job, path=einfo, precision=prec, split=a.split,
                n_rows=len(M), n_images=int(M.image_id.nunique()), stats=st,
                full_image_encodings_per_image=(st["encoder_canvases"] / st["images"]) if mode != "c0" else None,
                rgb_crop_encodings_per_roi=1.0 if mode == "c0" else 0.0, ms_encode_per_image=1000 * st["sec_encode"] / st["images"],
                ms_reader_per_roi=1000 * st["sec_reader"] / st["rois"], checks=chk if a.checks else None,
                gpu=torch.cuda.get_device_name(), seconds=round(time.time() - t0, 1), time=time.strftime("%F %T"))
    json.dump(info, open(os.path.join(a.out, f"infer_{a.split}.json"), "w"), indent=1, default=str)
    print(json.dumps(info, default=str), flush=True)


if __name__ == "__main__":
    main()
