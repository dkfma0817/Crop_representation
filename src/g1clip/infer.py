"""G1-CLIP official PACO inference (2026-11-07) = src.maskemb.infer (copied, not modified) for the three G1 conditions:
  g1pool_enc   : G1pool/ema_last.pt encoder, full-image tokens            (decoder never built)
  g1cross_enc  : G1cross/ema_last.pt encoder, full-image tokens           (decoder never built)
  g1cross_dec  : the SAME G1cross/ema_last.pt encoder + its decoder      (checkpoint sha256 asserted equal to g1cross_enc)
Saved per ROI (preds/<cond>/pred_<split>.npz + .json meta): image_id, ann_id, GT bbox [x, y, w, h] (official, original px),
category_id, level (object / part), logits (59, raw reader output) and probs (59, sigmoid = the attribute_probs given to
the official evaluator with detection score 1.0), plus the checkpoints (G1 + R2 epoch) and configs used.
Extra checks (--checks, val): encoder-only conditions have no decoder module; decoder condition: replacing the decoder
output by zeros changes the logits (the decoder path is used).

Original docstring follows.
Official PACO-LVIS VAL inference for one condition with its dev-selected R2 epoch (fixed before val is read).
Rows = output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet (EVERY official val annotation, objects + parts,
official JSON order; read-only). Output: <out>/probs_val.npz (ann_id, image_id, category_id, logits, probs (N, 59) in
official attribute order). No GT category / attribute enters the model (dummy zero label matrix).
Checks (--checks): (1) dev reproduction: the selected checkpoint re-scores the train-internal dev rows through THIS code
path and must match the training-time scores_dev_ep<k>.npy; (2) ROI-only reading: tokens of cells with cov == 0 are
replaced by noise -> logits must not change; ROI cells (cov > 0) replaced -> logits change; (3) encoder calls / image.

  CUDA_VISIBLE_DEVICES=<one gpu> python -m src.maskemb.infer --cond clip_full --out <dir> [--checks]
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
from src.maskemb import common as MC
from src.g1clip.probe import EVAL_BS, EXCLUDE, encoder_tokens, load_g1, make_ds, set_precision, split_dev
from src.maskemb.reader import build as build_r2

DATA = "output/2026-11-07/paco_oracle_attr/data"
RUN = "output/2026-11-07/g1clip"
CONDS = {"g1pool_enc": (f"{RUN}/G1pool/ema_last.pt", "full"), "g1cross_enc": (f"{RUN}/G1cross/ema_last.pt", "full"),
         "g1cross_dec": (f"{RUN}/G1cross/ema_last.pt", "dec")}


def load(cond, dev, pdir=None):
    ckpt, mode = CONDS[cond]
    pdir = pdir or os.path.join(RUN, "probes", cond)
    sel = json.load(open(os.path.join(pdir, "r2", "selected_epoch.json")))
    cfg = json.load(open(os.path.join(pdir, "config.json")))
    assert cfg["job"]["ckpt"] == ckpt and cfg["job"]["mode"] == mode
    vit, dec, einfo = load_g1(ckpt, mode, dev)
    r = build_r2(MC.DIM).to(dev)
    rpath = os.path.join(pdir, "r2", f"ep{sel['epoch']}.pt")
    ck = torch.load(rpath, map_location=dev, weights_only=False)
    assert ck["encoder"] == ckpt and ck["mode"] == mode and ck["epoch"] == sel["epoch"]
    einfo.update(reader_ckpt=rpath, reader_sha256=MC.sha256(rpath), probe_config=os.path.join(pdir, "config.json"))
    r.load_state_dict(ck["reader"])
    r.eval()
    for p in r.parameters():
        p.requires_grad_(False)
    return vit, dec, einfo, r, mode, cfg["precision"], sel, pdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(CONDS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--pdir", default="", help="probe dir (smoke only)")
    ap.add_argument("--limit", type=int, default=0, help="smoke only: first N val images")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"pred_{a.split}.npz")
    assert not os.path.exists(fn), f"{fn} exists (scored once)"
    MC.set_clip_norm()
    dev = torch.device("cuda")
    t0 = time.time()
    vit, dec, einfo, reader, mode, prec, sel, pdir = load(a.cond, dev, a.pdir or None)
    assert (dec is None) == (mode == "full"), "encoder-only conditions must not build a decoder"

    def enc_feats(b, zero_dec=False):
        tok = encoder_tokens(vit, b["x"], prec)
        if dec is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = dec(tok, b["img_index"], b["box_c"], b["qrel"], b["fvalid"]).float()
            if zero_dec:
                tok = torch.zeros_like(tok)
            b["img_index"] = torch.arange(len(tok), device=tok.device)
        return {12: tok}
    set_precision(prec)

    def run_rows(M, checks=False):
        ds = make_ds(M, np.zeros((len(M), 59), np.int8), False, mode)
        chunks = [list(range(i, min(len(ds), i + EVAL_BS[mode]))) for i in range(0, len(ds), EVAL_BS[mode])]
        dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
        L = torch.full((len(M), 59), float("nan"))
        st = dict(encoder_images=0, encoder_canvases=0, rois=0, sec_encoder=0.0, sec_reader=0.0)
        ck = dict(outside_max_abs=[], inside_mean_abs=[], dec_zero_max_abs=[])
        for bi, b in enumerate(dl):
            n_img = len({int(g) for g in b["group"]})
            st["encoder_canvases"] += int(b["x"].shape[0])
            st["encoder_images"] += n_img
            st["rois"] += len(b["rows"])
            assert b["x"].shape[0] == n_img, "one encoder canvas per image (full and dec)"
            b = RC.to_dev(b, dev)
            torch.cuda.synchronize()
            t1 = time.time()
            if checks and bi < 10 and dec is not None:
                b0 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
                with torch.no_grad():
                    fz = enc_feats(b0, zero_dec=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lz = reader(fz, b0)["logits"].float()
            with torch.no_grad():
                f = enc_feats(b)
            torch.cuda.synchronize()
            t2 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lo = reader(f, b)["logits"].float()
            torch.cuda.synchronize()
            st["sec_encoder"] += t2 - t1
            st["sec_reader"] += time.time() - t2
            L[b["rows"].cpu()] = lo.cpu()
            if checks and bi < 10:
                tok = f[12][b["img_index"]]
                m = (b["cov"] > 0)[..., None]
                noise = torch.randn_like(tok) * tok.std()
                b2 = dict(b)
                b2["img_index"] = torch.arange(len(tok), device=dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lo_out = reader({12: torch.where(m, tok, noise)}, b2)["logits"].float()
                    lo_in = reader({12: torch.where(m, noise, tok)}, b2)["logits"].float()
                    lo_same = reader({12: tok}, b2)["logits"].float()
                assert torch.equal(lo_same, lo)
                ck["outside_max_abs"].append(float((lo_out - lo).abs().max()))
                if dec is not None:
                    ck["dec_zero_max_abs"].append(float((lz - lo).abs().max()))
                ck["inside_mean_abs"].append(float((lo_in - lo).abs().mean()))
        assert not torch.isnan(L).any()
        return L, st, ck

    chk = {}
    if a.checks:                                    # dev reproduction through this code path
        data = RC.load_data(EXCLUDE, 0, splits=("train",))
        _, _, ddev, _ = split_dev(*data["train"], 1000)
        if a.limit:                                  # smoke: the probe smoke used --limit -> its own small dev
            data = RC.load_data(EXCLUDE, a.limit, splits=("train",))
            _, _, ddev, _ = split_dev(*data["train"], max(2, a.limit // 10))
        Ld, _, _ = run_rows(ddev)
        saved = np.load(os.path.join(pdir, "r2", f"scores_dev_ep{sel['epoch']}.npy"))
        chk["dev_repro"] = dict(rows=len(ddev), max_abs=float(np.abs(Ld.numpy() - saved).max()),
                                mean_abs=float(np.abs(Ld.numpy() - saved).mean()), corr=float(np.corrcoef(Ld.numpy().ravel(), saved.ravel())[0, 1]))
        print(json.dumps(chk), flush=True)
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    if a.limit:
        M = M[M.image_id.isin(set(M.image_id.drop_duplicates().iloc[: a.limit]))].reset_index(drop=True)
    t_val = time.time()
    L, st, ck = run_rows(M, checks=a.checks)
    sec_val = time.time() - t_val
    probs = torch.sigmoid(L).numpy().astype(np.float32)
    np.savez(fn, ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values,
             bbox_xywh=M[["bx", "by", "bw", "bh"]].values.astype(np.float64), level=M.level.values.astype(str),
             logits=L.numpy().astype(np.float32), probs=probs)
    os.symlink(os.path.basename(fn), os.path.join(a.out, f"probs_{a.split}.npz"))      # name the evaluator scripts expect
    if ck["outside_max_abs"]:
        chk["decoder_built"] = dec is not None
        if ck["dec_zero_max_abs"]:
            chk["decoder_zeroed_max_abs_logit_change"] = max(ck["dec_zero_max_abs"])
        chk["roi_only"] = dict(outside_noise_max_abs_logit_change=max(ck["outside_max_abs"]),
                               inside_noise_mean_abs_logit_change=float(np.mean(ck["inside_mean_abs"])), n_batches=len(ck["outside_max_abs"]))
    info = dict(cond=a.cond, split=a.split, mode=mode, precision=prec, selected=sel, encoder=einfo, n_rows=len(M), n_images=int(M.image_id.nunique()),
                stats=st, encoder_calls_per_image=st["encoder_canvases"] / max(1, st["encoder_images"]),
                ms_encoder_per_image=1000 * st["sec_encoder"] / st["encoder_images"], ms_encoder_per_roi=1000 * st["sec_encoder"] / st["rois"],
                ms_reader_per_roi=1000 * st["sec_reader"] / st["rois"],
                saved_fields=["ann_id", "image_id", "category_id", "bbox_xywh", "level", "logits (raw R2 output)", "probs (sigmoid = evaluator attribute_probs, score 1.0)"],
                attribute_order="official PACO attribute id order (59)", manifest=os.path.join(DATA, f"manifest_{a.split}.parquet"), val_wall_s=round(sec_val, 1), checks=chk,
                gpu=torch.cuda.get_device_name(), seconds=round(time.time() - t0, 1), time=time.strftime("%F %T"))
    json.dump(info, open(os.path.join(a.out, f"infer_{a.split}.json"), "w"), indent=1, default=str)
    print(json.dumps(info, default=str), flush=True)


if __name__ == "__main__":
    main()
