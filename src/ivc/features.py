"""Streaming feature-difference pass on PACO VAL (no flip, every ROI): F336 bilinear / bicubic crop-grid features vs C0.

Per batch of images: full image 336 letterbox -> frozen DINOv2-B/14 final-LN tokens (bf16 autocast -> fp32, the probe's
exact call) -> bilinear and bicubic ROI sampling on the C0 crop grid; GT crops (C0 letterbox) -> same encoder -> C0 tokens.
Metrics (src.ivc.common.roi_metrics, fp32) are computed on the GPU and only per-ROI scalars, per-token maps (576, fp16)
and pooled vectors (768, fp16) are stored; no token tensor is written (11,109 ROIs x 576 x 768 x 3 would be ~29 GB fp16).

  CUDA_VISIBLE_DEVICES=2 python -m src.ivc.features --out output/2026-11-05/ivc/features
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.msd.common as MS
import src.readout.common as RC
from src.ivc.common import EXCL, R0, roi_metrics, sample_roi


def chunks_by_image(n_img, bs):
    return [list(range(i, min(n_img, i + bs))) for i in range(0, n_img, bs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0, help="process image chunks k with k %% nshard == shard")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="merge <out>/shard*/ into <out>")
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    if a.nshard > 1:
        a.out = os.path.join(a.out, f"shard{a.shard}")
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    data = RC.load_data(EXCL, limit=a.limit, splits=("train", "val"))
    dva, Yva = data["val"]
    ds = MS.MSDImageDS(dva, Yva, flip=False, want_crops=True)
    dl = DataLoader(ds, sampler=chunks_by_image(len(ds), a.bs)[a.shard::a.nshard], batch_size=1, num_workers=a.workers, collate_fn=MS.collate_one)
    vit = RC.load_vit().to(dev)
    from models.roi_reader import bilinear as roi_bilinear
    N = len(dva)
    maps = {k: np.zeros((N, 576), np.float16) for k in ("cos_bil", "cos_bic", "se_bil", "se_bic", "ccos_bil", "ccos_bic", "cov")}
    pooled = {k: np.zeros((N, 768), np.float16) for k in ("bil", "bic", "c0")}
    recs = [None] * N
    seen = np.zeros(N, bool)
    t0 = time.time()
    checked = False
    for bi, b in enumerate(dl):
        b = MS.to_dev(b, dev)
        with torch.no_grad():
            full = RC.encode_layers(vit, b["full"], (12,))[12]                       # (B_img, 576, 768) fp32
            crops = b["crops"]
            c0 = torch.cat([RC.encode_layers(vit, crops[i:i + 64], (12,))[12] for i in range(0, len(crops), 64)])
            feats = {"bil": sample_roi(full, b["geom"], "bilinear"), "bic": sample_roi(full, b["geom"], "bicubic")}
            if not checked:                                                          # = the probe's bilinear call, bit-exact
                ref = roi_bilinear(full, b["geom"])
                assert torch.equal(ref, feats["bil"]), "bilinear sampling differs from models.roi_reader.bilinear"
                checked = True
            cov = b["cov_c"].float()
            rows = b["rows"].cpu().numpy()
            for k, r in enumerate(rows):
                rec = dict(row=int(r))
                for nm in ("bil", "bic"):
                    m = roi_metrics(feats[nm][k], c0[k], cov[k])
                    rec.update({f"{nm}_{kk}": v for kk, v in m.items()})
                    A, Cc = feats[nm][k], c0[k]
                    maps[f"cos_{nm}"][r] = torch.nn.functional.cosine_similarity(A, Cc, dim=-1).cpu().numpy()
                    maps[f"se_{nm}"][r] = (A - Cc).pow(2).mean(-1).cpu().numpy()
                    Mk = cov[k] > 0
                    w = cov[k] / cov[k].sum()
                    pa, pc = (w[:, None] * A).sum(0), (w[:, None] * Cc).sum(0)
                    maps[f"ccos_{nm}"][r] = torch.nn.functional.cosine_similarity(A - pa, Cc - pc, dim=-1).cpu().numpy() * Mk.cpu().numpy()
                maps["cov"][r] = cov[k].cpu().numpy()
                for nm, X in (("bil", feats["bil"]), ("bic", feats["bic"]), ("c0", c0)):
                    pooled[nm][r] = R0.pooled(X[k:k + 1], cov[k:k + 1])[0].cpu().numpy()
                recs[r] = rec
                seen[r] = True
        if bi % 25 == 0:
            print(json.dumps(dict(batch=bi, n_done=int(seen.sum()), N=N, s=round(time.time() - t0))), flush=True)
    if a.nshard > 1:                                                                  # partial output, merged later
        pd.DataFrame([r for r in recs if r is not None]).to_parquet(os.path.join(a.out, "roi_metrics.parquet"))
        np.savez(os.path.join(a.out, "partial.npz"), seen=seen, **{f"map_{k}": v for k, v in maps.items()}, **{f"pool_{k}": v for k, v in pooled.items()})
        json.dump(dict(n_seen=int(seen.sum()), time_s=round(time.time() - t0), bilinear_matches_roi_reader=checked), open(os.path.join(a.out, "done.json"), "w"))
        print("SHARD DONE", int(seen.sum()), flush=True)
        return
    assert seen.all(), f"{(~seen).sum()} ROIs not processed"
    met = pd.DataFrame(recs)
    assert (met.row.values == np.arange(N)).all()
    met.insert(1, "ann_id", dva.ann_id.values)
    met.insert(2, "image_id", dva.image_id.values)
    met.insert(3, "level", dva.level.values)
    met.insert(4, "n_in", dva.n_in.values)
    met.to_parquet(os.path.join(a.out, "roi_metrics.parquet"))
    np.savez(os.path.join(a.out, "token_maps.npz"), ann_id=dva.ann_id.values, **maps)
    np.savez(os.path.join(a.out, "pooled.npz"), ann_id=dva.ann_id.values, **pooled)
    json.dump(dict(n_roi=N, n_img=len(ds), time_s=round(time.time() - t0), bilinear_matches_roi_reader=checked,
                   precision="encoder bf16 autocast -> fp32 tokens; sampling and metrics fp32", flip=False),
              open(os.path.join(a.out, "done.json"), "w"), indent=1)
    print("DONE", round(time.time() - t0), "s", flush=True)


def merge(a):
    data = RC.load_data(EXCL, splits=("val",))
    dva = data["val"][0]
    N = len(dva)
    shards = sorted(d for d in os.listdir(a.out) if d.startswith("shard"))
    seen = np.zeros(N, int)
    maps, pooled, recs, ok = {}, {}, [], []
    for d in shards:
        z = np.load(os.path.join(a.out, d, "partial.npz"))
        sm = z["seen"]
        assert not (seen.astype(bool) & sm).any(), "shards overlap"
        seen += sm
        for k in z.files:
            if k.startswith("map_") or k.startswith("pool_"):
                dst = maps if k.startswith("map_") else pooled
                kk = k.split("_", 1)[1]
                dst.setdefault(kk, np.zeros_like(z[k]))[sm] = z[k][sm]
        recs.append(pd.read_parquet(os.path.join(a.out, d, "roi_metrics.parquet")))
        ok.append(json.load(open(os.path.join(a.out, d, "done.json")))["bilinear_matches_roi_reader"])
    assert (seen == 1).all(), f"{(seen != 1).sum()} ROIs missing / duplicated"
    met = pd.concat(recs).sort_values("row").reset_index(drop=True)
    assert (met.row.values == np.arange(N)).all()
    met.insert(1, "ann_id", dva.ann_id.values)
    met.insert(2, "image_id", dva.image_id.values)
    met.insert(3, "level", dva.level.values)
    met.insert(4, "n_in", dva.n_in.values)
    met.to_parquet(os.path.join(a.out, "roi_metrics.parquet"))
    np.savez(os.path.join(a.out, "token_maps.npz"), ann_id=dva.ann_id.values, **maps)
    np.savez(os.path.join(a.out, "pooled.npz"), ann_id=dva.ann_id.values, **pooled)
    json.dump(dict(n_roi=N, shards=len(shards), bilinear_matches_roi_reader=all(ok),
                   precision="encoder bf16 autocast -> fp32 tokens; sampling and metrics fp32", flip=False),
              open(os.path.join(a.out, "done.json"), "w"), indent=1)
    print("MERGED", N)


if __name__ == "__main__":
    main()
