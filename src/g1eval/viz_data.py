"""Data for the G1 leak/ADE visualisation (CPU, inference only with the SAVED ADE linear heads; nothing is trained).

1. Example images (val): 4 random (seed 0) + 3 with the largest per-image mIoU drop + 2 with the largest gain
   (drop / gain = mean of G1-pool and G1-cross per-image mIoU minus Original, from ade/val_inter_union.npz, images
   with >= 5 classes of union > 0 for Original). Same eval pipeline as src.g1eval.ade.evaluate (shorter side 384, pad
   to /16, bilinear logits -> original H x W, argmax), fp32 on CPU. Saves display-size RGB / GT / predictions and the
   per-token cosine(G1 token, Original token) maps.
2. Class-level token change: N random val images (seed 1); every 16x16 token cell whose (resized) GT has a single class
   covering >= 50 % of the cell is assigned that class; per-class mean cosine(G1 token, Original token).

  python -m src.g1eval.viz_data --run output/2026-11-07/g1_leakcheck_ade20k --n_cls 300
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.g1eval.ade import CFG, IGN, NC, NPT, P, listing, load_pair, logits, norm
from src.g1in.common import MODELS, load_encoder

DW = 320  # display width


def per_img_miou(I, U):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(U > 0, I / np.maximum(U, 1), np.nan)
    return 100 * np.nanmean(r, 1)


def prep(img):
    W, H = img.size
    s = CFG["eval_short"] / min(W, H)
    nw, nh = max(1, round(W * s)), max(1, round(H * s))
    x = norm(img.resize((nw, nh), Image.BILINEAR))
    pw, ph = math.ceil(nw / P) * P, math.ceil(nh / P) * P
    X = torch.zeros(1, 3, ph, pw)
    X[0, :, :nh, :nw] = x
    return X, nh, nw, ph, pw


@torch.no_grad()
def run(encs, heads, img, need_pred=True):
    W, H = img.size
    X, nh, nw, ph, pw = prep(img)
    h, w = ph // P, pw // P
    tok, pred = {}, {}
    for m in MODELS:
        t = encs[m].forward_features(X)[:, NPT:].float()
        assert t.shape[1] == h * w
        tok[m] = t[0].reshape(h, w, -1)
        if need_pred:
            f = t.transpose(1, 2).reshape(1, -1, h, w)
            lo = F.interpolate(logits(heads[m], f), size=(ph, pw), mode="bilinear", align_corners=False)[:, :, :nh, :nw]
            pred[m] = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False).argmax(1)[0].numpy().astype(np.uint8)
    cos = {m: F.cosine_similarity(tok[m], tok["original"], dim=-1).numpy() for m in ("g1pool", "g1cross")}
    return dict(pred=pred, cos=cos, nh=nh, nw=nw, ph=ph, pw=pw)


def nearest(a, size_wh):
    return np.array(Image.fromarray(a).resize(size_wh, Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n_cls", type=int, default=300)
    a = ap.parse_args()
    torch.set_num_threads(16)
    out = os.path.join(a.run, "viz")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    z = np.load(os.path.join(a.run, "ade", "val_inter_union.npz"))
    I = {m: z[f"I_{m}"].astype(np.int64) for m in MODELS}
    U = {m: z[f"U_{m}"].astype(np.int64) for m in MODELS}
    pm = {m: per_img_miou(I[m], U[m]) for m in MODELS}
    d = (pm["g1pool"] + pm["g1cross"]) / 2 - pm["original"]
    ok = (U["original"] > 0).sum(1) >= 5
    rng = np.random.default_rng(0)
    rand = [int(i) for i in rng.choice(len(d), 4, replace=False)]
    cand = np.where(ok)[0]
    drop = [int(i) for i in cand[np.argsort(d[cand])] if int(i) not in rand][:3]
    gain = [int(i) for i in cand[np.argsort(-d[cand])] if int(i) not in rand][:2]
    sel = [(i, "random") for i in rand] + [(i, "drop") for i in drop] + [(i, "gain") for i in gain]

    encs = {m: load_encoder(m, "cpu") for m in MODELS}
    heads = {}
    for m in MODELS:
        hd = torch.nn.Linear(768, NC)
        hd.load_state_dict(torch.load(os.path.join(a.run, "ade", f"head_{m}.pt"), map_location="cpu"))
        heads[m] = hd.eval()
    items = listing("val")
    meta = []
    for k, (i, why) in enumerate(sel):
        img, lab = load_pair(*items[i])
        W, H = img.size
        r = run(encs, heads, img)
        # sanity: re-inference reproduces the saved per-image I/U (CPU fp32 vs GPU bf16 -> report agreement)
        valid = lab != IGN
        agree = {}
        for m in MODELS:
            p, t = r["pred"][m][valid].astype(np.int64), lab[valid]
            inter = np.bincount(t[p == t], minlength=NC)[:NC]
            un = np.bincount(p, minlength=NC)[:NC] + np.bincount(t, minlength=NC)[:NC] - inter
            agree[m] = dict(miou_cpu=float(per_img_miou(inter[None], un[None])[0]), miou_saved=float(pm[m][i]))
        dh = round(H * DW / W)
        disp = dict(rgb=np.array(img.resize((DW, dh), Image.BILINEAR)), gt=nearest(lab.astype(np.uint8), (DW, dh)))
        for m in MODELS:
            disp[f"pred_{m}"] = nearest(r["pred"][m], (DW, dh))
        for m in ("g1pool", "g1cross"):
            c = np.kron(r["cos"][m], np.ones((P, P), np.float32))[: r["nh"], : r["nw"]]   # token grid -> resized image
            disp[f"cos_{m}"] = np.array(Image.fromarray(c).resize((DW, dh), Image.NEAREST))
            disp[f"cosgrid_{m}"] = r["cos"][m]
        np.savez_compressed(os.path.join(out, f"ex_{k}.npz"), **disp)
        meta.append(dict(k=k, idx=i, file=os.path.basename(items[i][0]), why=why, H=H, W=W,
                         miou={m: float(pm[m][i]) for m in MODELS}, cpu_check=agree,
                         cos_mean={m: float(r["cos"][m][: math.ceil(r["nh"] / P), : math.ceil(r["nw"] / P)].mean()) for m in ("g1pool", "g1cross")}))
        print(k, why, meta[-1]["file"], {m: round(v, 1) for m, v in meta[-1]["miou"].items()}, flush=True)

    # class-level token change
    rng = np.random.default_rng(1)
    pick = rng.choice(len(items), a.n_cls, replace=False)
    S = {m: np.zeros(NC) for m in ("g1pool", "g1cross")}
    Nc = np.zeros(NC, np.int64)
    allcos = {m: [] for m in ("g1pool", "g1cross")}
    for j, i in enumerate(pick):
        img, lab = load_pair(*items[int(i)])
        r = run(encs, heads, img, need_pred=False)
        L = np.full((r["ph"], r["pw"]), IGN, np.uint8)
        L[: r["nh"], : r["nw"]] = nearest(lab.astype(np.uint8), (r["nw"], r["nh"]))
        h, w = r["ph"] // P, r["pw"] // P
        blk = L.reshape(h, P, w, P).transpose(0, 2, 1, 3).reshape(h * w, P * P).astype(np.int64)
        cnt = np.zeros((h * w, 256), np.int64)
        np.add.at(cnt, (np.repeat(np.arange(h * w), P * P), blk.ravel()), 1)
        cnt[:, IGN] = 0
        cls, share = cnt.argmax(1), cnt.max(1) / (P * P)
        keep = share >= 0.5
        for m in ("g1pool", "g1cross"):
            cv = r["cos"][m].reshape(-1)
            np.add.at(S[m], cls[keep], cv[keep])
            allcos[m].append(cv[keep])
        np.add.at(Nc, cls[keep], 1)
        if j % 50 == 0:
            print("cls", j, round(time.time() - t0), flush=True)
    cls_out = dict(n_images=int(a.n_cls), seed=1, n_tokens=Nc.tolist(),
                   cos={m: [float(S[m][c] / Nc[c]) if Nc[c] else None for c in range(NC)] for m in S},
                   hist={m: np.histogram(np.concatenate(allcos[m]), bins=40, range=(0.0, 1.0))[0].tolist() for m in S},
                   mean={m: float(np.concatenate(allcos[m]).mean()) for m in S})
    json.dump(dict(examples=meta, cls=cls_out, select_rule=__doc__.split("\n\n")[1], time_s=round(time.time() - t0)),
              open(os.path.join(out, "viz_data.json"), "w"), indent=1)
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
