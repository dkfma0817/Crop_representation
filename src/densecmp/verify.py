"""Checks for the EVA-family PACO comparison (writes JSON + PNG into --out).

  geometry : (1) parametric geometry with S=336/P=14 + ImageNet norm == the original DINOv2 functions (bit-exact) on 3,000
             val ROIs / 40 canvases; (2) S=1024/P=16 alignment: rasterise each GT box in ORIGINAL pixels, letterbox it with the
             same rule (nearest), per-cell box fraction must equal the analytic cov used by R2 (max abs err), ROI token counts.
  params   : visual-tower parameter differences EVA-CLIP vs CLIPSelf / DeCLIP checkpoints (+ text tower), per block.
  samples  : (GPU) feature shape, token grid, ROI token count, ROI canvas coords, extraction path for fixed sample ROIs;
             SDPA shim vs the repo's own non-xformers attention branch (max abs diff); figure of canvas + box + cov>0 cells.
  python -m src.densecmp.verify geometry --out <dir>   (CPU; any env)
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import src.readout.common as RC
from src.densecmp import geom
from src.densecmp.encoders import CK, ENC, state_dict_of
from src.ivc.common import EXCL
from src.readout.common import C, RR, roi_geometry


def sample_rows(df, n, seed=0):
    return np.sort(np.random.default_rng(seed).choice(len(df), n, replace=False))


def check_geometry(out):
    from common import Letterbox as LB0, to_tensor_canvas as TTC0   # DINOv2 originals (path set by src.readout.common)
    data = RC.load_data(EXCL, splits=("train", "val"))
    dva, _ = data["val"]
    res = dict()
    rows = sample_rows(dva, 3000)
    bad = dict(letterbox=0, roi_geo=0, box4=0, cov=0, crop_geo=0)
    for r in rows:
        W, H = int(dva.W[r]), int(dva.H[r])
        bx = tuple(int(v) for v in dva.loc[r, ["l", "t", "r", "b"]])
        a, b = LB0(W, H), geom.Letterbox(W, H, 336)
        bad["letterbox"] += int((a.nw, a.nh, a.ox, a.oy, a.sx, a.sy) != (b.nw, b.nh, b.ox, b.oy, b.sx, b.sy) or not np.array_equal(a.coverage(), b.coverage(14)))
        g0, b40, c0 = RR.roi_geo(a, bx)
        g1, b41, c1 = geom.roi_geo(b, bx, 14)
        bad["roi_geo"] += int(not np.array_equal(g0, g1))
        bad["box4"] += int(not np.array_equal(b40, b41))
        bad["cov"] += int(not np.array_equal(c0, c1) or not np.array_equal(C.box_cell_weights(a, bx), geom.box_cell_weights(b, bx, 14)))
        rg = roi_geometry(W, H, bx)
        assert tuple(rg["ltrb"]) == bx
        x0 = RC.crop_canvas_geo(rg["crop"])
        x1 = geom.crop_canvas_geo(geom.Letterbox(bx[2] - bx[0], bx[3] - bx[1], 336), 14)
        bad["crop_geo"] += int(not all(np.array_equal(p, q) for p, q in zip(x0, x1)))
    imgs = dva.drop_duplicates("image_id").path.values[:40]
    mx = 0.0
    for p in imgs:
        im = Image.open(p).convert("RGB")
        t0 = TTC0(im, LB0(*im.size))
        t1 = geom.to_tensor_canvas(im, geom.Letterbox(*im.size, 336), geom.IMAGENET)
        mx = max(mx, float((t0 - t1).abs().max()))
    res["dinov2_reproduction"] = dict(n_rois=len(rows), n_mismatch=bad, canvas_max_abs_diff=mx, n_canvases=len(imgs))
    # 1024 / 16 alignment by rasterisation (nearest) vs analytic coverage
    S, P = 1024, 16
    G = S // P
    errs, counts, nins = [], [], []
    for r in rows[:600]:
        W, H = int(dva.W[r]), int(dva.H[r])
        bx = tuple(int(v) for v in dva.loc[r, ["l", "t", "r", "b"]])
        lb = geom.Letterbox(W, H, S)
        _, _, cov = geom.roi_geo(lb, bx, P)
        # supersample canvas pixels (4x4 per px) -> original px -> inside box?
        k = 4
        X0, Y0 = lb.to_canvas(bx[0], bx[1])
        X1, Y1 = lb.to_canvas(bx[2], bx[3])
        i0, i1 = int(np.floor(Y0 / P)), int(np.ceil(Y1 / P))
        j0, j1 = int(np.floor(X0 / P)), int(np.ceil(X1 / P))
        ys = (np.arange(i0 * P * k, i1 * P * k) + 0.5) / k
        xs = (np.arange(j0 * P * k, j1 * P * k) + 0.5) / k
        xo = (xs - lb.ox) / lb.sx                    # canvas -> original px
        yo = (ys - lb.oy) / lb.sy
        mx_ = (xo >= bx[0]) & (xo < bx[2])
        my_ = (yo >= bx[1]) & (yo < bx[3])
        frac_x = mx_.reshape(-1, P * k).mean(1)
        frac_y = my_.reshape(-1, P * k).mean(1)
        ras = np.zeros((G, G))
        ras[i0:i1, j0:j1] = frac_y[:, None] * frac_x[None, :]
        errs.append(float(np.abs(ras.reshape(-1) - cov).max()))
        counts.append(int((cov > 0).sum()))
        nins.append(int(dva.n_in[r]))
    counts, nins = np.array(counts), np.array(nins)
    res["align_1024"] = dict(n_rois=len(errs), max_abs_err_cov=max(errs), mean_abs_err_cov=float(np.mean(errs)),
                             roi_tokens_cov_gt0={nm: dict(n=int(((nins >= lo) & (nins <= hi)).sum()),
                                                          min=int(counts[(nins >= lo) & (nins <= hi)].min()),
                                                          median=float(np.median(counts[(nins >= lo) & (nins <= hi)])),
                                                          max=int(counts[(nins >= lo) & (nins <= hi)].max()))
                                                 for nm, lo, hi in (("D1-4", 1, 4), ("D5-16", 5, 16), ("D17+", 17, 10 ** 9))},
                             zero_token_rois=int((counts == 0).sum()))
    json.dump(res, open(os.path.join(out, "geometry_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def check_params(out):
    ref, _ = state_dict_of(os.path.join(CK, ENC["eva"]["ckpt"]))
    res = {}
    for nm in ("clipself", "declip", "declip_csa"):
        sd, meta = state_dict_of(os.path.join(CK, ENC[nm]["ckpt"]))
        keys = [k for k in ref if "freqs_" not in k]
        assert set(keys) <= set(sd), "key set differs"
        per = {}
        for k in keys:
            a, b = ref[k].double(), sd[k].double()
            per[k] = dict(changed=not torch.equal(ref[k], sd[k]), rel=float((a - b).norm() / (a.norm() + 1e-12)))
        vis = {k: v for k, v in per.items() if k.startswith("visual.")}
        txt = {k: v for k, v in per.items() if k.startswith("text.")}
        blocks = {}
        for i in range(12):
            ks = [k for k in vis if k.startswith(f"visual.blocks.{i}.")]
            blocks[i] = dict(n=len(ks), n_changed=sum(vis[k]["changed"] for k in ks), max_rel=max(vis[k]["rel"] for k in ks),
                             mean_rel=float(np.mean([vis[k]["rel"] for k in ks])))
        other = [k for k in vis if not k.startswith("visual.blocks.")]
        res[nm] = dict(ckpt=ENC[nm]["ckpt"], meta={k: str(v) for k, v in meta.items()},
                       visual=dict(n=len(vis), n_changed=sum(v["changed"] for v in vis.values()), max_rel=max(v["rel"] for v in vis.values())),
                       text=dict(n=len(txt), n_changed=sum(v["changed"] for v in txt.values())), blocks=blocks,
                       non_block_visual={k: vis[k] for k in other})
    a, b = state_dict_of(os.path.join(CK, ENC["declip"]["ckpt"]))[0], state_dict_of(os.path.join(CK, ENC["clipself"]["ckpt"]))[0]
    res["declip_vs_clipself_visual_identical"] = all(torch.equal(a[k], b[k]) for k in a if k.startswith("visual.") and "freqs_" not in k)
    json.dump(res, open(os.path.join(out, "param_diff.json"), "w"), indent=1)
    print(json.dumps({k: (dict(visual=v["visual"], text=v["text"], meta=v["meta"]) if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))


def check_samples(out, enc_name, side, mode):
    """GPU, evaclip env. One encoder per process."""
    from src.densecmp.data import EvaPacoDS
    from src.densecmp.encoders import Encoder
    import src.densecmp.encoders as E
    dev = torch.device("cuda")
    data = RC.load_data(EXCL, splits=("train", "val"))
    dva, Yva = data["val"]
    enc = Encoder(enc_name).to(dev)
    ds = EvaPacoDS(dva, Yva, flip=False, mode=mode, side=side)
    groups = C.image_groups(dva)
    gsel = [int(i) for i in np.random.default_rng(0).choice(len(groups), 4, replace=False)]
    b = RC.to_dev(ds[gsel], dev)
    f = torch.cat([enc(b["x"][i:i + 8]) for i in range(0, len(b["x"]), 8)])
    rep = dict(encoder=enc_name, mode=mode, side=side, path=enc.path, repo=enc.repo_src, input=tuple(b["x"].shape),
               feature=tuple(f.shape), token_grid=int(round(f.shape[1] ** 0.5)), feat_norm_mean=float(f.norm(dim=-1).mean()), rois=[])
    for k in range(min(8, len(b["rows"]))):
        r = int(b["rows"][k])
        cov = b["cov"][k]
        rep["rois"].append(dict(row=r, ann_id=int(dva.ann_id[r]), level=dva.level[r], box_ltrb=[int(v) for v in dva.loc[r, ["l", "t", "r", "b"]]],
                                WH=[int(dva.W[r]), int(dva.H[r])], n_in_dinov2_grid=int(dva.n_in[r]), roi_tokens_cov_gt0=int((cov > 0).sum()),
                                cov_sum=round(float(cov.sum()), 3), box4_canvas=[round(float(v) * side, 1) for v in b["box4"][k]],
                                img_index=int(b["img_index"][k])))
    # SDPA shim vs the repo's own non-xformers branch (xattn False) on the same input
    x = b["x"][:2]
    with torch.no_grad():
        f1 = enc(x)
        for blk in enc.model.visual.blocks:
            blk.attn.xattn = False
        f2 = enc(x)
        for blk in enc.model.visual.blocks:
            blk.attn.xattn = True
    rep["sdpa_shim_vs_repo_math_attention"] = dict(max_abs=float((f1 - f2).abs().max()), min_cos=float((f1 * f2).sum(-1).min()))
    rep["load_report"] = {k: v for k, v in enc.load_report.items() if k != "rope_buffer_maxdiff"}
    rep["load_report"]["rope_buffer_maxdiff"] = max(enc.load_report["rope_buffer_maxdiff"].values())
    json.dump(rep, open(os.path.join(out, f"samples_{enc_name}_{mode}{side}.json"), "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
    if mode == "full":                          # figure: canvas + GT boxes + cov>0 cells
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mean = torch.tensor(geom.OPENAI_CLIP[0])[:, None, None]
        std = torch.tensor(geom.OPENAI_CLIP[1])[:, None, None]
        img0 = (b["x"][0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ks = [k for k in range(len(b["rows"])) if int(b["img_index"][k]) == 0][:4]
        fig, axs = plt.subplots(1, len(ks), figsize=(5 * len(ks), 5))
        axs = np.atleast_1d(axs)
        G = side // E.PATCH
        for ax, k in zip(axs, ks):
            ax.imshow(img0)
            cov = b["cov"][k].cpu().numpy().reshape(G, G)
            ii, jj = np.nonzero(cov > 0)
            for i, j in zip(ii, jj):
                ax.add_patch(plt.Rectangle((j * E.PATCH, i * E.PATCH), E.PATCH, E.PATCH, fill=True, alpha=0.25 * cov[i, j] + 0.05, color="yellow", lw=0))
            X0, Y0, w, h = (float(v) * side for v in b["box4"][k])
            ax.add_patch(plt.Rectangle((X0, Y0), w, h, fill=False, color="red", lw=1.5))
            r = int(b["rows"][k])
            ax.set_title(f"ann {int(dva.ann_id[r])} {dva.level[r]} n_in(24)={int(dva.n_in[r])} tok={len(ii)}", fontsize=8)
            ax.set_xlim(max(0, X0 - 80), min(side, X0 + w + 80))
            ax.set_ylim(min(side, Y0 + h + 80), max(0, Y0 - 80))
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"align_{enc_name}_{side}.png"), dpi=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["geometry", "params", "samples"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--enc", default="eva")
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--mode", default="full")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"geometry": lambda: check_geometry(a.out), "params": lambda: check_params(a.out),
     "samples": lambda: check_samples(a.out, a.enc, a.side, a.mode)}[a.what]()


if __name__ == "__main__":
    main()
