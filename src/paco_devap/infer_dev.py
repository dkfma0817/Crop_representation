"""Official-protocol Dev AP, step 2 (GPU, one device): score EVERY dev annotation (13,931 objects + parts of the 1,000 dev
images, data/manifest_dev.parquet) with an EXISTING probe's R2 checkpoint(s). No training, no re-selection.

Probe kinds (read from <probe>/config.json):
  maskemb probe (job.encoder clip | maskembed, mode full)   -> src.maskemb.common.build_encoder
  g1clip probe  (job.ckpt = G1 EMA checkpoint, mode full | dec) -> src.g1clip.probe.load_g1 (decoder only in 'dec')
Encoding = the probes' own code path: fp32 encoder tokens (precision from the probe config), decoder bf16 autocast (dec),
R2 bf16 autocast; inputs from the probes' datasets (PacoDS full / dec, CLIP mean / std, 336 / 14).
The frozen encoder (and decoder) output is computed ONCE and reused for every requested epoch's reader.
Per epoch -> <out>/<name>/ep<k>/pred_dev.npz: ann_id, image_id, category_id, bbox_xywh, level, logits (raw R2), probs
(sigmoid = the evaluator's attribute_probs) + meta.json (checkpoints, sha256, config) and an M0 reproduction check: the
5,411 attribute-labelled dev rows are re-scored and compared with the training-time scores_dev_ep<k>.npy and dev M0.

  CUDA_VISIBLE_DEVICES=<gpu> python -m src.paco_devap.infer_dev --probe <probe dir> --name <name> --epochs selected|all|1,2,..
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
from src.maskemb.probe import EXCLUDE, encoder_tokens, set_precision, split_dev
from src.maskemb.reader import build as build_r2
from src.g1clip.probe import make_ds, load_g1

RUN = "output/2026-11-07/paco_devap"


def load_encoder(cfg, dev):
    job = cfg["job"]
    if "ckpt" in job:
        vit, dec, info = load_g1(job["ckpt"], job["mode"], dev)
        return vit, dec, info
    vit, info = MC.build_encoder(job["encoder"], dev)
    return vit, None, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", default="selected")
    ap.add_argument("--out", default=os.path.join(RUN, "preds"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="test: official TEST manifest (all 131,899 anns, = the existing test runs), ONE epoch fixed by the dev rule")
    a = ap.parse_args()
    MC.set_clip_norm()
    dev = torch.device("cuda")
    t0 = time.time()
    cfg = json.load(open(os.path.join(a.probe, "config.json")))
    mode, prec = cfg["job"]["mode"], cfg["precision"]
    assert mode in ("full", "dec"), mode
    set_precision(prec)
    sel = json.load(open(os.path.join(a.probe, "r2", "selected_epoch.json")))
    have = sorted(int(f[2:-3]) for f in os.listdir(os.path.join(a.probe, "r2")) if f.startswith("ep") and f.endswith(".pt"))
    eps = [sel["epoch"]] if a.epochs == "selected" else have if a.epochs == "all" else [int(x) for x in a.epochs.split(",")]
    assert set(eps) <= set(have), (eps, have)
    vit, dec, einfo = load_encoder(cfg, dev)
    if a.split == "test":
        assert len(eps) == 1, "test: exactly one (dev-selected) epoch"
        M = pd.read_parquet("output/2026-11-07/paco_oracle_attr/data/manifest_test.parquet")
    else:
        M = pd.read_parquet(os.path.join(RUN, "data", "manifest_dev.parquet"))
    ds = make_ds(M, np.zeros((len(M), 59), np.int8), False, mode)
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
    cache = []          # full mode: (batch geometry, frozen encoder tokens) reused by every epoch's reader (2.4 GB fp32)
    n_img = n_can = 0   # dec mode: per-ROI decoder outputs (~33 GB) are recomputed per epoch instead (identical code path)
    with torch.no_grad():
        for b in (dl if dec is None and a.split == "dev" else []):
            n_img += len({int(g) for g in b["group"]})
            n_can += int(b["x"].shape[0])
            b = RC.to_dev(b, dev)
            tok = encoder_tokens(vit, b["x"], prec)
            del b["x"]
            cache.append((b, tok))
    assert n_can == n_img, "one encoder canvas per image"
    if a.split == "dev":                                  # M0 reproduction inputs (dev only)
        data = RC.load_data(EXCLUDE, 0, splits=("train",))
        Ytr_by = RC.ytr_by_level(data)
        _, _, ddev, Ydev = split_dev(*data["train"], 1000)
        pos = pd.Series(np.arange(len(M)), index=M.ann_id.values).loc[ddev.ann_id.values].values
        curves = {json.loads(l)["epoch"]: json.loads(l) for l in open(os.path.join(a.probe, "r2", "curves_dev.jsonl"))}
    for ep in eps:
        r = build_r2(MC.DIM).to(dev)
        rp = os.path.join(a.probe, "r2", f"ep{ep}.pt")
        ck = torch.load(rp, map_location=dev, weights_only=False)
        assert ck["epoch"] == ep and ck["mode"] == mode
        r.load_state_dict(ck["reader"])
        r.eval()
        L = torch.full((len(M), 59), float("nan"))
        with torch.no_grad():
            if dec is None and a.split == "dev":
                for b, tok in cache:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        L[b["rows"].cpu()] = r({12: tok}, b)["logits"].float().cpu()
            else:
                for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True):
                    b = RC.to_dev(b, dev)
                    z = encoder_tokens(vit, b["x"], prec)
                    if dec is not None:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            z = dec(z, b["img_index"], b["box_c"], b["qrel"], b["fvalid"]).float()
                        b["img_index"] = torch.arange(len(z), device=dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        L[b["rows"].cpu()] = r({12: z}, b)["logits"].float().cpu()
        assert not torch.isnan(L).any()
        L = L.numpy()
        if a.split == "dev":
            saved = np.load(os.path.join(a.probe, "r2", f"scores_dev_ep{ep}.npy"))
            m0 = RC.m2_eval(L[pos], ddev, Ydev, Ytr_by)["M0"]["overall"]
            repro = dict(rows=len(pos), max_abs_logit_diff=float(np.abs(L[pos] - saved).max()), m0_recomputed=m0,
                         m0_training_curve=curves[ep]["dev"]["M0"]["overall"])
        else:
            repro = None
        od = os.path.join(a.out, a.name, f"ep{ep}")
        os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, f"pred_{a.split}.npz"), ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values,
                 bbox_xywh=M[["bx", "by", "bw", "bh"]].values.astype(np.float64), level=M.level.values.astype(str),
                 logits=L.astype(np.float32), probs=(1 / (1 + np.exp(-L))).astype(np.float32))
        meta = dict(name=a.name, probe=a.probe, epoch=ep, selected_epoch_of_probe=sel["epoch"], is_selected=ep == sel["epoch"], mode=mode,
                    precision=prec, reader_ckpt=rp, reader_sha256=MC.sha256(rp), encoder=einfo, probe_config=cfg["job"],
                    n_rows=len(M), n_images=int(M.image_id.nunique()), m0_repro=repro, time=time.strftime("%F %T"))
        if a.split == "test":
            os.symlink(f"pred_{a.split}.npz", os.path.join(od, "probs_test.npz"))      # name src.pacoattr.official_eval expects
        json.dump(meta, open(os.path.join(od, f"meta{'' if a.split == 'dev' else '_test'}.json"), "w"), indent=1, default=str)
        print(json.dumps(dict(epoch=ep, split=a.split, **(repro or {}))), flush=True)
    print(f"done {a.name} epochs {eps} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
