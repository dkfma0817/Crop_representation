"""Inference of the 5 FIXED existing conditions on the official PACO-LVIS val / test annotations (no training, ONE GPU).

Condition -> exactly the existing probe code path (src.g1x.probe.run encode/score, unchanged modules):
  frozen_full  : original DINOv3-B/16, full 384 letterbox, PacoDS 'full' (R2 full-canvas geometry, cov = box ∩ cell)
  g1pool_full  : G1pool/ema_last.pt ['enc'], same
  g1cross_full : G1cross/ema_last.pt ['enc'], same (decoder NOT loaded)
  g1cross_dec  : G1cross/ema_last.pt ['enc'] + ['dec'] ROIDecoder; PacoDS 'dec' (full canvas only; decoder gets the ROI box on
                 the canvas box_c + query coords qrel + valid-key mask; R2 reads the decoder output with the C0 crop geometry)
  c0_crop      : original DINOv3-B/16, BBImageDS 'c0' (GT tight crop -> 384 letterbox per ROI)
Reader = the dev-selected R2 checkpoint r2/ep6.pt of that condition (fixed, no re-selection). Output = sigmoid of the
R2 logits (59, official attribute order) for EVERY manifest row (official annotation order). No GT attribute / category
enters the model: the dataset label slot gets a dummy zero matrix.
Checks (--checks, val): reproduction of the saved probe val logits on the existing val rows, encoder calls per image,
decoder-path checks (bypass, box sensitivity).

  CUDA_VISIBLE_DEVICES=<uuid> python -m src.pacoattr.infer --cond g1cross_dec --split val --out <dir> [--checks]
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
from src.g1x.model import ROIDecoder
from src.g1x.probe import make_ds
from src.pacoattr.audit import CONDS, OLD, SEL_EPOCH
from src.readout.models import Reader
from src.viz.c0_pca import load_encoder

DATA = "output/2026-11-07/paco_oracle_attr/data"
BS = {"c0": 6, "full": 12, "dec": 12}


def build(cond, dev):
    pdir, enc, mode = CONDS[cond]
    vit = load_encoder("dinov3", dev)
    dec = None
    if enc:
        ck = torch.load(enc, map_location="cpu", weights_only=False)
        vit.load_state_dict(ck["enc"])
        if mode == "dec":
            dec = ROIDecoder(384, 16).to(dev)
            dec.load_state_dict(ck["dec"])
            dec.eval()
    vit.eval()
    torch.manual_seed(7)
    r = Reader("r2", layers=(12,)).to(dev)
    r.load_state_dict(torch.load(os.path.join(pdir, "r2", f"ep{SEL_EPOCH}.pt"), map_location=dev, weights_only=False)["reader"])
    r.eval()
    for p in list(vit.parameters()) + list(r.parameters()) + (list(dec.parameters()) if dec is not None else []):
        p.requires_grad_(False)
    return vit, dec, r, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(CONDS))
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"probs_{a.split}.npz")
    assert not os.path.exists(fn), f"{fn} exists (each condition is scored once per split)"
    dev = torch.device("cuda")
    t0 = time.time()
    M = pd.read_parquet(os.path.join(DATA, f"manifest_{a.split}.parquet"))
    if a.limit:
        M = M[M.image_id.isin(set(M.image_id.drop_duplicates().iloc[: a.limit]))].reset_index(drop=True)
    vit, dec, reader, mode = build(a.cond, dev)
    S, P = 384, 16
    ds = make_ds(M, np.zeros((len(M), 59), np.int8), False, mode, S, P)
    chunks = [list(range(i, min(len(ds), i + BS[mode]))) for i in range(0, len(ds), BS[mode])]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
    logits = torch.full((len(M), 59), float("nan"))
    stats = dict(encoder_images=0, encoder_canvases=0, rois=0, decoder_rois=0, batches=0)
    chk = dict()

    @torch.no_grad()
    def encode(b, bypass=None):
        x, toks = b["x"], []
        for i in range(0, len(x), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = vit.forward_features(x[i:i + 64])
            toks.append(t[:, vit.num_prefix_tokens:].float())
        tok = torch.cat(toks)
        if mode == "dec":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = dec(tok, b["img_index"], b["box_c"], b["qrel"], b["fvalid"]).float()
            if bypass == "zero":
                z = torch.zeros_like(z)
            b["img_index"] = torch.arange(len(z), device=z.device)
            return {12: z}
        return {12: tok}

    @torch.no_grad()
    def read(f, b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return reader(f, b)["logits"].float()

    for bi, b in enumerate(dl):
        n_img = len({int(g) for g in b["group"]})
        stats["encoder_canvases"] += int(b["x"].shape[0])
        stats["encoder_images"] += n_img
        stats["rois"] += len(b["rows"])
        if mode in ("full", "dec"):
            assert b["x"].shape[0] == n_img, "full / dec: exactly one encoder canvas per image"
        else:
            assert b["x"].shape[0] == len(b["rows"]), "c0: one crop canvas per ROI"
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        if a.checks and mode == "dec" and bi < 20:
            b0 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            o_ref = read(encode(b0), b0)
            b1 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            o_zero = read(encode(b1, bypass="zero"), b1)
            b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            b2["box_c"] = b2["box_c"].roll(1, dims=0)                       # each ROI gets another ROI's box
            o_roll = read(encode(b2), b2)
            chk.setdefault("dec_bypass_zero_max_abs_logit_change", []).append(float((o_zero - o_ref).abs().max()))
            chk.setdefault("dec_box_swap_mean_abs_logit_change", []).append(float((o_roll - o_ref).abs().mean()))
        f = encode(b)
        if mode == "dec":
            stats["decoder_rois"] += len(rows)
        logits[rows] = read(f, b).cpu()
        stats["batches"] += 1
    assert not torch.isnan(logits).any(), "missing ROI scores"
    probs = torch.sigmoid(logits).numpy().astype(np.float32)
    np.savez(fn, ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values,
             logits=logits.numpy().astype(np.float32), probs=probs)
    if a.checks and a.split == "val" and not a.limit:
        old = pd.read_parquet(os.path.join(OLD, "manifest_val.parquet"))
        old = old[(old.bw > 0) & (old.bh > 0)].reset_index(drop=True)
        saved = np.load(os.path.join(CONDS[a.cond][0], "r2", f"scores_val_ep{SEL_EPOCH}.npy"))
        pos = pd.Series(np.arange(len(M)), index=M.ann_id.values).loc[old.ann_id.values].values
        L = logits.numpy()[pos]
        chk["repro_rows"] = int(len(pos))
        chk["repro_max_abs_logit_diff"] = float(np.abs(L - saved).max())
        chk["repro_mean_abs_logit_diff"] = float(np.abs(L - saved).mean())
        chk["repro_corr"] = float(np.corrcoef(L.ravel(), saved.ravel())[0, 1])
    for k in list(chk):
        if isinstance(chk[k], list):
            chk[k] = dict(min=float(np.min(chk[k])), max=float(np.max(chk[k])), n_batches=len(chk[k]))
    info = dict(cond=a.cond, split=a.split, mode=mode, n_rows=len(M), n_images=int(M.image_id.nunique()), stats=stats,
                encoder_calls_per_image=stats["encoder_canvases"] / max(1, stats["encoder_images"]),
                checks=chk, r2=os.path.join(CONDS[a.cond][0], "r2", f"ep{SEL_EPOCH}.pt"), encoder_ckpt=CONDS[a.cond][1],
                seconds=round(time.time() - t0, 1), time=time.strftime("%F %T"))
    json.dump(info, open(os.path.join(a.out, f"infer_{a.split}.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
