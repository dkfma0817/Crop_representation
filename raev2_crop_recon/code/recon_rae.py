"""Reconstruction evaluation (no training): representations -> frozen RAE decoder -> RGB vs the GT crop canvas, and latent vs teacher.

Sets
  hval : rae_crop_pilot head_val, 3,194 GT ROIs from 1,000 COCO images outside every training pool (cached latents)
  pacot: PACO test-primary ROIs, up to 1,000 per F336 bin (D1-4 part, D5-16 object, D17+ object; rng 0), encoded on the fly
Representations: teacher (GT crop latent = upper bound of the decoder), bil (bilinear), A/B/C at the given steps.
Metrics per ROI (content pixels, clamped RGB): l1, psnr, lpips, dE_mean (mean colour), dE_8x8, hp_corr (GT-aligned detail), hf_ratio;
latent: mse, cos vs teacher (content cells, coverage-weighted), spatial-variance ratio. Paired image bootstrap (1,000) for C-A, B-A, C-B.
  python code/recon_rae.py --steps 4000,12000   -> results/recon/*.csv, results/viz/recon_*.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rae_common as RC  # noqa: E402
import pixel as PX  # noqa: E402
import data_rae as DR  # noqa: E402
import train_rae as TR  # noqa: E402

RUNS = os.path.join(RC.CACHE, "runs")
OUTD = os.path.join(RC.RES, "recon")
VIZ = os.path.join(RC.RES, "viz")
PACO = "/workspace/Areum/research/2026-10-24/dinov2_oracle_region_decoder/perception_cache/paco_lp"


def paco_batches(n_per=1000, per=48):
    df = pd.read_parquet(os.path.join(PACO, "manifest_test.parquet"))
    df = df[(df.bw > 0) & (df.bh > 0) & ~df.in_decoder_pool].reset_index(drop=True)
    rng = np.random.default_rng(0)
    picks = []
    for name, lv, lo, hi in (("D1-4", "part", 1, 4), ("D5-16", "object", 5, 16), ("D17+", "object", 17, 10 ** 9)):
        cand = df[(df.level == lv) & (df.n_in >= lo) & (df.n_in <= hi)]
        sel = cand.iloc[np.sort(rng.choice(len(cand), min(n_per, len(cand)), replace=False))].copy()
        sel["bin"] = name
        picks.append(sel)
    sel = pd.concat(picks).sort_values(["image_id"]).reset_index(drop=True)
    return sel, [list(range(i, min(len(sel), i + per))) for i in range(0, len(sel), per)]


class PacoRecon(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fulls, slot, crops, geoms, img_index, rects = [], {}, [], [], [], []
        for i in idx:
            r = self.df.iloc[i]
            img = Image.open(r.path).convert("RGB")
            if r.path not in slot:
                slot[r.path] = len(fulls)
                fulls.append(RC.u8_canvas(img, RC.Letterbox(*img.size)))
            g = RC.roi_geometry(img.size[0], img.size[1], (int(r.l), int(r.t), int(r.r), int(r.b)))
            crops.append(RC.u8_canvas(img.crop(g["ltrb"]), g["crop"]))
            geoms.append({k: g[k] for k in RC.GEOM_KEYS})
            img_index.append(slot[r.path])
            rects.append(g["crop"].rect())
        return dict(full_u8=torch.stack(fulls), crops=torch.stack(crops), geom=RC.collate_geoms(geoms, img_index),
                    rects=torch.as_tensor(np.array(rects)), rows=torch.as_tensor(idx))


def run_set(name, it, rae, lp, decs, P, dev, keep_viz):
    recs, viz = [], {}
    for batch in it:
        if name == "hval":
            b = DR.to_dev(batch, dev)
            full, T, gt = b["full"], b["teacher"], b["rgb"]
            img_ids = [int(x.split("_")[0]) if isinstance(x, str) else -1 for x in batch["roi_ids"]]
            keys = list(batch["roi_ids"])
        else:
            g = RC.to_device(batch["geom"], dev)
            b = dict(geom=g, mask=RC.content_mask(batch["rects"], dev))
            full = rae.encode(batch["full_u8"])
            T = rae.encode(batch["crops"])
            gt = batch["crops"].to(dev).float() / 255
            keys = batch["rows"].tolist()
        g, m = b["geom"], b["mask"]
        w = g["valid_w"]
        reps = {"teacher": T, "bil": RC.bilinear(full, g)}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for tag, dm in decs.items():
                reps[tag] = dm(RC.gather_roi(full, g), None, g, P).float()
        for tag, Z in reps.items():
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = rae.decode(Z, chunk=48).float().clamp(0, 1)
                met = PX.rgb_metrics(y, gt, m, lp)
                mse, cos = PX.feat_mse_cos(Z, T, w)
                sv = PX.svar_ratio(Z, T, w)
            for k in range(len(keys)):
                recs.append(dict(set=name, key=keys[k], rep=tag, n_in_rae=int(g["n_in"][k]), mse=float(mse[k]), cos=float(cos[k]), svar=float(sv[k]),
                                 **{kk: float(v[k]) for kk, v in met.items()}))
            for k in range(len(keys)):
                if keys[k] in keep_viz:
                    viz.setdefault(keys[k], {})[tag] = y[k].cpu()
                    viz[keys[k]]["gt"], viz[keys[k]]["mask"] = gt[k].cpu(), m[k].cpu()
    return pd.DataFrame(recs), viz


def boot_diff(df, a, b, metric, group_col="img", nb=1000):
    x = df[df.rep == a].set_index("key")[metric]
    y = df[df.rep == b].set_index("key")[metric]
    gcol = df[df.rep == a].set_index("key")[group_col]
    d = (x - y.loc[x.index]).to_frame("d")
    d["g"] = gcol
    gs = d.groupby("g").d.agg(["sum", "count"])
    rng = np.random.default_rng(0)
    S, N = gs["sum"].values, gs["count"].values
    bs = []
    for _ in range(nb):
        ix = rng.integers(0, len(S), len(S))
        bs.append(S[ix].sum() / N[ix].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.d.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="4000,12000")
    ap.add_argument("--n_per", type=int, default=1000)
    ap.add_argument("--runs", default="A,B,C")
    ap.add_argument("--out", default="", help="output subfolder name (default results/recon, results/viz)")
    a = ap.parse_args()
    global OUTD, VIZ
    if a.out:
        OUTD, VIZ = os.path.join(RC.RES, a.out), os.path.join(RC.RES, a.out, "viz")
    dev = "cuda"
    RC.set_fp32()
    os.makedirs(OUTD, exist_ok=True)
    os.makedirs(VIZ, exist_ok=True)
    rae = RC.FrozenRAE(dev)
    lp = PX.load_lpips(dev)
    P = torch.as_tensor(RC.query_coords(), device=dev)
    decs = {}
    for st in a.steps.split(","):
        for run in a.runs.split(","):
            p = os.path.join(RUNS, run, f"ckpt_{st}.pt")
            if os.path.exists(p):
                decs[f"{run}_{st}"] = TR.load_model(p, dev).eval()
    print("decoders", list(decs), flush=True)
    # hval (cached)
    ds = DR.CachedSet("hval", rgb=True)
    hv_rois = ds.rois
    rng = np.random.default_rng(1)
    keep_h = set(hv_rois.roi_id.iloc[rng.choice(len(hv_rois), 8, replace=False)])
    dfh, vizh = run_set("hval", DR.loader(ds, DR.seq_slots(ds, 12), 6), rae, lp, decs, P, dev, keep_h)
    dfh["img"] = dfh.key.map(dict(zip(hv_rois.roi_id, hv_rois.image_id)))
    dfh["bin"] = "all"
    # PACO test-primary
    sel, batches = paco_batches(a.n_per)
    keep_p = set()
    for bn in ("D1-4", "D5-16", "D17+"):
        keep_p |= set(sel.index[sel.bin == bn][np.random.default_rng(2).choice((sel.bin == bn).sum(), 4, replace=False)])
    dl = torch.utils.data.DataLoader(PacoRecon(sel), batch_size=None, sampler=batches, num_workers=8)
    dfp, vizp = run_set("pacot", dl, rae, lp, decs, P, dev, keep_p)
    dfp["img"] = dfp.key.map(dict(zip(sel.index, sel.image_id)))
    dfp["bin"] = dfp.key.map(dict(zip(sel.index, sel.bin)))
    D = pd.concat([dfh, dfp])
    D.to_csv(os.path.join(OUTD, "per_roi.csv.gz"), index=False)
    mets = ("l1", "psnr", "lpips", "de_mean", "de_8x8", "hp_corr", "hf_ratio", "mse", "cos", "svar")
    summ = D.groupby(["set", "bin", "rep"])[list(mets)].mean().reset_index()
    summ.to_csv(os.path.join(OUTD, "summary.csv"), index=False)
    drows = []
    for (st, bn), d in D.groupby(["set", "bin"]):
        for s in a.steps.split(","):
            for x, y in (("C", "A"), ("B", "A"), ("C", "B"), ("C_L1", "C"), ("C_L1", "A")):
                if f"{x}_{s}" in set(d.rep) and f"{y}_{s}" in set(d.rep):
                    for mt in mets:
                        v, lo, hi = boot_diff(d, f"{x}_{s}", f"{y}_{s}", mt)
                        drows.append(dict(set=st, bin=bn, comparison=f"{x}-{y} @{s}", metric=mt, value=v, ci_lo=lo, ci_hi=hi, sig=bool(lo > 0 or hi < 0)))
    pd.DataFrame(drows).to_csv(os.path.join(OUTD, "paired_diffs.csv"), index=False)
    # visualisation: GT / teacher / bilinear / A / B / C (last step)
    last = a.steps.split(",")[-1]
    order = ["gt", "teacher", "bil"] + [f"{r}_{last}" for r in a.runs.split(",") if f"{r}_{last}" in decs]
    for nm, vz, keys in (("hval", vizh, sorted(keep_h)), ("paco", vizp, sorted(keep_p, key=lambda k: (sel.bin[k], k)))):
        keys = [k for k in keys if k in vz]
        rows = [torch.stack([vz[k][o] for k in keys]) for o in order]
        mk = torch.stack([vz[k]["mask"] for k in keys])
        PX.save_grid(os.path.join(VIZ, f"recon_{nm}_{last}.png"), rows, mk, order)
        if nm == "paco":
            json.dump([dict(row=int(k), bin=sel.bin[k], image_id=int(sel.image_id[k]), level=sel.level[k], category=sel.category[k]) for k in keys],
                      open(os.path.join(VIZ, f"recon_{nm}_{last}_examples.json"), "w"), indent=1)
    print(summ.to_string(), flush=True)


if __name__ == "__main__":
    main()
