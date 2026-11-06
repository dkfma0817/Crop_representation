"""RAE reconstruction diagnosis (no training): RGB fidelity vs latent fidelity of GT-crop / full-image / ROI-sampled latents.

Frozen official RAEv2 pair (src.rae.common.FrozenRAE: dinov2_vitb14_reg encoder at 224 from the 256 letterbox canvas, raw latent
(256 tokens x 768, final LN without affine), ViT-XL pixel decoder on the raw latent -> 256 x 256 RGB). PACO VAL ROIs (manifest of
src.readout.common.load_data), no flip, encoder fp32 (TF32 off), decoder / LPIPS bf16 autocast (= src.eval.recon).

Conditions per ROI (all RGB compared with the GT crop canvas = 256 letterbox of the original-pixel GT crop, content pixels only):
  A     GT crop canvas -> encoder -> T (= C0 latent) -> decoder                       (decoder reference)
  B     full canvas -> encoder -> Zf -> decoder -> full RGB 256 -> GT ROI resampled onto the crop canvas pixel grid
  B_in  the same resampling applied to the full INPUT canvas (no encoder / decoder): input-resolution reference
  C     Zf -> bilinear ROI sampling on the crop grid (src.rae.common.bilinear = roi_reader.bilinear, G=16) -> decoder
  D     the same grid with mode='bicubic' -> decoder
  E     diagnostic (uses T): T low-passed to the ROI's full-grid footprint (content block area-pooled to round(h / s) x
        round(w / s) cells, s = crop.sx / full.sx = crop cells per full-grid cell, bilinear back), padding cells = T -> decoder
  F     diagnostic (uses T): C with its coverage-weighted content mean replaced by T's (C - mean_w(C) + mean_w(T) on content
        cells, padding cells = C) -> decoder
  MB    existing Pixel-only CoordMLP (R00 structure, output/2026-11-04/B/ckpt_12000.pt) on Zf -> decoder
Re-encoding: every reconstructed crop canvas (content = clamped RGB rounded to uint8, padding = the ImageNet-mean colour of
the GT pipeline) -> the same encoder -> R_x, compared with T on content cells (coverage weights).

  CUDA_VISIBLE_DEVICES=2 python -m src.raediag.run --out output/2026-11-05/rae_diag --shard 0 --nshard 2
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import src.readout.common as RD
import src.rae.common as RC
import src.losses.pixel as PX
import src.engine.train as TR

EXCL = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
MB_CKPT = "output/2026-11-04/B/ckpt_12000.pt"
G, S, PATCH = RC.G, RC.S, RC.PATCH
DEC_CONDS = ("A", "C", "D", "E", "F", "MB")          # latents fed to the pixel decoder
ALL_CONDS = ("A", "B", "B_in", "C", "D", "E", "F", "MB")


# ------------------------------------------------------------------------------------------------ data
class ValImages(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.groups = [g.index.values for _, g in df.groupby("image_id", sort=False)]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        fulls, crops, geoms, img_index, rects, frect, rows = [], [], [], [], [], [], []
        for k, gi in enumerate(idx):
            rws = self.groups[gi]
            r0 = self.df.loc[rws[0]]
            img = Image.open(r0.path).convert("RGB")
            assert img.size == (int(r0.W), int(r0.H)), r0.path
            lbf = RC.Letterbox(*img.size)
            fulls.append(RC.u8_canvas(img, lbf))
            for r in rws:
                d = self.df.loc[r]
                g = RC.roi_geometry(img.size[0], img.size[1], (int(d.l), int(d.t), int(d.r), int(d.b)))
                crops.append(RC.u8_canvas(img.crop(g["ltrb"]), g["crop"]))
                geoms.append({kk: g[kk] for kk in RC.GEOM_KEYS})
                img_index.append(k)
                cl = g["crop"]
                rects.append(cl.rect())
                # crop-canvas pixel -> full-canvas pixel affine map: X = ax * x + bx  (pixel-centre convention)
                ax = lbf.sx / cl.sx
                bx = lbf.ox + (int(d.l) - cl.ox / cl.sx) * lbf.sx
                ay = lbf.sy / cl.sy
                by = lbf.oy + (int(d.t) - cl.oy / cl.sy) * lbf.sy
                frect.append((ax, bx, ay, by, cl.sx / lbf.sx, cl.sy / lbf.sy))
                rows.append(r)
        return dict(full_u8=torch.stack(fulls), crops=torch.stack(crops), geom=RC.collate_geoms(geoms, img_index),
                    rects=torch.as_tensor(np.array(rects)), affine=torch.as_tensor(np.array(frect), dtype=torch.float64),
                    rows=torch.as_tensor(rows))


def resample_full_to_crop(full_rgb, img_index, affine):
    """full_rgb (n_img,3,256,256) -> (R,3,256,256): value at crop-canvas pixel centre (x+.5, y+.5) taken from full canvas
    position X = ax (x + .5) + bx (bilinear, align_corners=False, border). Padding pixels get extrapolated values (masked later)."""
    R = len(img_index)
    dev = full_rgb.device
    c = torch.arange(S, device=dev, dtype=torch.float64) + 0.5
    ax, bx, ay, by = (affine[:, i].to(dev)[:, None] for i in range(4))
    X = ax * c[None] + bx                         # (R, 256) per column
    Y = ay * c[None] + by
    gx = (X / (S / 2) - 1).float()
    gy = (Y / (S / 2) - 1).float()
    grid = torch.stack([gx[:, None, :].expand(R, S, S), gy[:, :, None].expand(R, S, S)], -1)
    return F.grid_sample(full_rgb[img_index].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)


# ------------------------------------------------------------------------------------------------ latents
def bicubic(full_tokens, b):
    maps = full_tokens.transpose(1, 2).reshape(full_tokens.shape[0], -1, G, G)[b["img_index"]]
    out = F.grid_sample(maps.float(), b["bil_grid"].view(-1, G, G, 2), mode="bicubic", padding_mode="border", align_corners=False)
    return out.flatten(2).transpose(1, 2)


def lowpass_T(T, valid_w, scale):
    """E: per ROI, area-pool the content-cell block of T to the full-grid footprint and bilinear-upsample it back."""
    out = T.clone()
    for k in range(len(T)):
        c = valid_w[k].view(G, G) > 0
        rr, cc = torch.nonzero(c.any(1)).flatten(), torch.nonzero(c.any(0)).flatten()
        r0, r1, c0, c1 = int(rr[0]), int(rr[-1]) + 1, int(cc[0]), int(cc[-1]) + 1
        hc, wc = r1 - r0, c1 - c0
        sx, sy = float(scale[k, 0]), float(scale[k, 1])
        h2, w2 = max(1, round(hc / sy)), max(1, round(wc / sx))
        blk = T[k].view(G, G, -1)[r0:r1, c0:c1].permute(2, 0, 1)[None]
        lo = F.adaptive_avg_pool2d(blk, (h2, w2))
        up = F.interpolate(lo, size=(hc, wc), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        o = out[k].view(G, G, -1)
        o[r0:r1, c0:c1] = up
    return out


def wmean(Z, w):
    w = w.float()
    return (Z * w[..., None]).sum(1) / w.sum(1, keepdim=True)


def mean_shift(Cz, T, w):
    sh = (wmean(T, w) - wmean(Cz, w))[:, None]
    return torch.where((w > 0)[..., None], Cz + sh, Cz)


# ------------------------------------------------------------------------------------------------ metrics
def ssim(p, y, m, win=11, sigma=1.5):
    """Mean SSIM (per channel, gaussian window) over content pixels whose whole window lies in the content area."""
    dev = p.device
    ax = torch.arange(win, device=dev, dtype=torch.float32) - win // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None]).expand(3, 1, win, win).contiguous()
    f = lambda x: F.conv2d(x, k, padding=win // 2, groups=3)  # noqa: E731
    mu_p, mu_y = f(p), f(y)
    s_pp, s_yy, s_py = f(p * p) - mu_p ** 2, f(y * y) - mu_y ** 2, f(p * y) - mu_p * mu_y
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_p * mu_y + C1) * (2 * s_py + C2)) / ((mu_p ** 2 + mu_y ** 2 + C1) * (s_pp + s_yy + C2))
    mm = (F.avg_pool2d(m, win, 1, win // 2) > 0.999).float()
    n = mm.sum((1, 2, 3))
    v = (smap.mean(1, keepdim=True) * mm).sum((1, 2, 3)) / n.clamp_min(1)
    return torch.where(n > 0, v, torch.full_like(v, float("nan")))


def latent_metrics(Z, T, w, prefix):
    """Coverage-weighted token cos / MSE, pooled cos / L2, spatial variance ratio, self-similarity |dS| (content cells)."""
    w = w.float()
    ws = w / w.sum(1, keepdim=True)
    tc = F.cosine_similarity(Z, T, dim=-1)
    se = (Z - T).pow(2).mean(-1)
    pz, pt = wmean(Z, w), wmean(T, w)
    rz, rt = Z - pz[:, None], T - pt[:, None]
    svz, svt = (ws * rz.pow(2).mean(-1)).sum(1), (ws * rt.pow(2).mean(-1)).sum(1)
    out = {f"{prefix}tok_cos": (ws * tc).sum(1), f"{prefix}tok_mse": (ws * se).sum(1),
           f"{prefix}pool_cos": F.cosine_similarity(pz, pt, dim=-1), f"{prefix}pool_l2": (pz - pt).norm(dim=-1),
           f"{prefix}svar_ratio": svz / svt.clamp_min(1e-12)}
    ss = []
    for k in range(len(Z)):
        mk = w[k] > 0
        if mk.sum() < 2:
            ss.append(float("nan"))
            continue
        a, c = F.normalize(Z[k][mk], dim=-1), F.normalize(T[k][mk], dim=-1)
        Sa, Sc = a @ a.T, c @ c.T
        off = ~torch.eye(len(a), dtype=torch.bool, device=a.device)
        ss.append(float((Sa - Sc)[off].abs().mean()))
    out[f"{prefix}ss_diff"] = torch.tensor(ss, device=Z.device)
    return out


def input_stats(Z, w, mean, var):
    """Decoder-input distribution: token L2 norm, normalised-latent energy mean(z^2) (official per-position stats; ~1 on the
    training distribution), share of |z| > 3, on content and on padding cells; adjacent-token cosine on content cells."""
    zn = (Z.transpose(1, 2).reshape(len(Z), -1, G, G) - mean) / torch.sqrt(var + 1e-5)
    zn = zn.flatten(2).transpose(1, 2)
    cm = (w > 0).float()
    pm = 1 - cm
    nrm = Z.norm(dim=-1)
    z2 = zn.pow(2).mean(-1)
    big = (zn.abs() > 3).float().mean(-1)
    red = lambda x, mm: (x * mm).sum(1) / mm.sum(1).clamp_min(1)  # noqa: E731
    Zg = Z.view(len(Z), G, G, -1)
    cg = cm.view(len(Z), G, G)
    h = F.cosine_similarity(Zg[:, :, 1:], Zg[:, :, :-1], dim=-1)
    v = F.cosine_similarity(Zg[:, 1:], Zg[:, :-1], dim=-1)
    hm, vm = cg[:, :, 1:] * cg[:, :, :-1], cg[:, 1:] * cg[:, :-1]
    adj = ((h * hm).sum((1, 2)) + (v * vm).sum((1, 2))) / (hm.sum((1, 2)) + vm.sum((1, 2))).clamp_min(1)
    return dict(in_norm=red(nrm, cm), in_z2=red(z2, cm), in_z3=red(big, cm), in_norm_pad=red(nrm, pm), in_z2_pad=red(z2, pm), in_adj_cos=adj)


def to_u8_canvas(y, m):
    """Reconstructed crop canvas -> uint8 canvas for re-encoding: content = round(clamp(y) * 255), padding = PAD_RGB."""
    pad = torch.tensor(RC.PAD_RGB, device=y.device, dtype=torch.float32).view(1, 3, 1, 1)
    u = torch.round(y.clamp(0, 1) * 255)
    return (u * m + pad * (1 - m)).to(torch.uint8)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--ann_ids", default="", help="smoke test: restrict to these ann ids")
    ap.add_argument("--viz_json", default="configs/raediag/sample_rois.json")
    ap.add_argument("--bs", type=int, default=6, help="images per batch")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    RC.set_fp32()
    dva, _ = RD.load_data(EXCL, splits=("val",))["val"]
    if a.ann_ids:
        dva = dva[dva.ann_id.isin([int(x) for x in a.ann_ids.split(",")])]
    viz_ids = set(json.load(open(a.viz_json))["viz_ann_ids"]) if os.path.exists(a.viz_json) else set()
    ds = ValImages(dva)
    chunks = [list(range(i, min(len(ds), i + a.bs))) for i in range(0, len(ds), a.bs)][a.shard::a.nshard]
    dl = torch.utils.data.DataLoader(ds, batch_size=None, sampler=chunks, num_workers=a.workers)
    rae = RC.FrozenRAE(dev)
    lp = PX.load_lpips(dev)
    mb = TR.load_model(MB_CKPT, dev).eval()
    P = torch.as_tensor(RC.query_coords(), device=dev)
    fp0 = dict(decoder=RC.param_hash(rae.dec), encoder=RC.param_hash(rae.enc.model), mb=RC.param_hash(mb))
    rec_rows, lat_rows, viz = [], [], {}
    t0 = time.time()
    for bi, b in enumerate(dl):
        g = RC.to_device(b["geom"], dev)
        m = RC.content_mask(b["rects"], dev)
        w = g["valid_w"]
        rows = b["rows"].numpy()
        gt = b["crops"].to(dev).float() / 255
        with torch.no_grad():
            Zf = rae.encode(b["full_u8"])
            T = rae.encode(b["crops"])
            Cz = RC.bilinear(Zf, g)
            lat = dict(A=T, C=Cz, D=bicubic(Zf, g), E=lowpass_T(T, w, b["affine"][:, 4:6].float()), F=mean_shift(Cz, T, w))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lat["MB"] = mb(RC.gather_roi(Zf, g), None, g, P).float()
            rgb = {}
            for c in DEC_CONDS:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rgb[c] = rae.decode(lat[c], chunk=48).float().clamp(0, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                full_rec = rae.decode(Zf, chunk=48).float().clamp(0, 1)
            ii = g["img_index"]
            rgb["B"] = resample_full_to_crop(full_rec, ii, b["affine"]).clamp(0, 1)
            rgb["B_in"] = resample_full_to_crop(b["full_u8"].to(dev).float() / 255, ii, b["affine"]).clamp(0, 1)
            for c in ALL_CONDS:
                y = rgb[c]
                met = PX.rgb_metrics(y, gt, m, lp)
                met["ssim"] = ssim(y, gt, m)
                Rx = rae.encode(to_u8_canvas(y, m))
                lm = latent_metrics(Rx, T, w, "re_")
                if c in lat:
                    lm.update(latent_metrics(lat[c], T, w, "in_"))
                    lm.update({f"cyc_{k[3:]}": v for k, v in latent_metrics(Rx, lat[c], w, "in_").items() if k in ("in_tok_cos", "in_pool_cos")})
                    lm.update(input_stats(lat[c], w, rae.mean, rae.var))
                for k in range(len(rows)):
                    r = int(rows[k])
                    base = dict(row=r, ann_id=int(dva.ann_id[r]), image_id=int(dva.image_id[r]), level=dva.level[r], n_in=int(dva.n_in[r]),
                                n_in_rae=int(g["n_in"][k]), cond=c)
                    rec_rows.append(dict(base, **{kk: float(v[k]) for kk, v in met.items()}))
                    lat_rows.append(dict(base, **{kk: float(v[k]) for kk, v in lm.items()}))
            for k in range(len(rows)):
                aid = int(dva.ann_id[rows[k]])
                if aid in viz_ids:
                    viz[aid] = dict(gt=gt[k].cpu(), mask=m[k].cpu(), **{c: rgb[c][k].cpu() for c in ALL_CONDS}, full=(b["full_u8"][int(ii[k])].float() / 255),
                                    full_rec=full_rec[int(ii[k])].cpu())
        if bi % 20 == 0:
            print(json.dumps(dict(batch=bi, of=len(chunks), rois=len(rec_rows) // len(ALL_CONDS), s=round(time.time() - t0))), flush=True)
    sfx = f"_shard{a.shard}" if a.nshard > 1 else ""
    pd.DataFrame(rec_rows).to_csv(os.path.join(a.out, f"reconstruction_results{sfx}.csv"), index=False)
    pd.DataFrame(lat_rows).to_csv(os.path.join(a.out, f"latent_cycle_results{sfx}.csv"), index=False)
    torch.save(viz, os.path.join(a.out, f"viz{sfx}.pt"))
    fp1 = dict(decoder=RC.param_hash(rae.dec), encoder=RC.param_hash(rae.enc.model), mb=RC.param_hash(mb))
    json.dump(dict(n_roi=len(rec_rows) // len(ALL_CONDS), time_s=round(time.time() - t0), frozen_unchanged=fp0 == fp1, fingerprints=fp1,
                   mb_ckpt=MB_CKPT), open(os.path.join(a.out, f"done{sfx}.json"), "w"), indent=1)
    print("DONE", len(rec_rows) // len(ALL_CONDS), flush=True)


if __name__ == "__main__":
    main()
