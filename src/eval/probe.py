"""PACO probes on the RAEv2 latent: R2 AttnReader (lr 3e-4) and R0 cov-mean Linear (lr 1e-4), existing protocol unchanged
(paco_region_reader common_rr readers; probe_vr / train_rd schedule: image packs of ~128 ROIs from full_epoch_batches (data seed 0),
10 epochs, image hflip p=0.5, masked BCE, AdamW wd 0.05, 1-epoch warmup + cosine, clip 1.0, head lr 1e-3, reader init
torch.manual_seed(1000 seed + 7), best epoch = val pooled level-mean mAP, R0 z-score stats from 4,000 train images (rng 0, no flip)
of THIS representation). Each condition gets its own readers; several conditions share one frozen-encoder forward per batch.

Conditions (all from the official RAEv2 encoder, 256 letterbox canvases -> official 224 preprocess -> 16x16 raw latent):
  full : full-image 16x16 tokens; GT ROI given as cov = |box ∩ cell| / cell (256 canvas, 16 px cells); R2 attends to cov > 0
         (= the existing F336 "direct read-out"), geo/box4 from the GT box on the full canvas
  bil  : full map bilinearly sampled at the 16x16 teacher (crop-canvas) cell centres; crop-grid geometry
  crop : GT-crop oracle (original crop -> 256 letterbox -> encoder); crop-grid geometry
  <tag>: CoordMLP output (BF16 autocast as in training) from a checkpoint; crop-grid geometry
Crop-grid geometry = C0 convention: cov = crop letterbox coverage, geo/box4 = the crop content rect on the 256 canvas.
geo = [rel_x, rel_y, cov, log(box_w / 16), log(box_h / 16)] (box size in cell units, = common_tu.canvas_geo with PATCH 16).

  python -m src.eval.probe --conds full,bil,crop --seeds 0 --out cache/probe
  python -m src.eval.probe --dec A_4000=<ckpt>,... --seeds 0 --out ...      (val only; --test scores test once from best-val)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import src.rae.common as RC  # noqa: E402
import src.engine.train as TR  # noqa: E402
from src.config import PACO_READER_CODE  # noqa: E402
sys.path.insert(0, PACO_READER_CODE)   # earlier project (not on this server)
import common_rr as RD  # noqa: E402
C = RD.C
LEVELS = RD.LEVELS


def canvas_geo(X0, Y0, X1, Y1):
    e = np.arange(RC.G + 1) * RC.PATCH
    cx = np.clip(np.minimum(e[1:], X1) - np.maximum(e[:-1], X0), 0, None) / RC.PATCH
    cy = np.clip(np.minimum(e[1:], Y1) - np.maximum(e[:-1], Y0), 0, None) / RC.PATCH
    cov = (cy[:, None] * cx[None, :]).reshape(-1).astype(np.float32)
    bw, bh = X1 - X0, Y1 - Y0
    ci, cj = np.divmod(np.arange(RC.NCELL), RC.G)
    px, py = (cj + 0.5) * RC.PATCH, (ci + 0.5) * RC.PATCH
    geo = np.stack([(px - X0) / bw, (py - Y0) / bh, cov, np.full(RC.NCELL, math.log(bw / RC.PATCH)), np.full(RC.NCELL, math.log(bh / RC.PATCH))], 1)
    return geo.astype(np.float32), np.array([X0 / RC.S, Y0 / RC.S, bw / RC.S, bh / RC.S], np.float32), cov


class PacoDS(Dataset):
    def __init__(self, df, Y, flip, want_crop):
        self.groups, self.flip, self.want_crop = C.image_groups(df), flip, want_crop
        # manifest paths are old-server /dataset/... files -> same COCO id in this server's train2017 / val2017
        paths = {p: RC.coco_image_path(p) for p in df.path.unique()}
        self.p, self.box = df.path.map(paths).values, df[["l", "t", "r", "b"]].values.astype(int)
        self.WH = df[["W", "H"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.Y = torch.as_tensor(Y)
        self.n_roi = np.array([len(g) for g in self.groups])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        full, crops, geoms, img_index, rows = [], [], [], [], []
        per = {k: [] for k in ("geo_f", "box4_f", "cov_f", "geo_c", "box4_c", "cov_c")}
        for k, gi in enumerate(idx):
            rws = self.groups[gi]
            img = Image.open(self.p[rws[0]]).convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[rws[0]]), self.p[rws[0]]
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            if f:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lb = RC.Letterbox(W, H)
            full.append(RC.u8_canvas(img, lb))
            for r in rws:
                bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                g = RC.roi_geometry(W, H, bx)
                geoms.append({kk: g[kk] for kk in RC.GEOM_KEYS})
                l, t, rr, b = g["ltrb"]
                X0, Y0 = lb.to_canvas(l, t)
                X1, Y1 = lb.to_canvas(rr, b)
                for key, v in zip(("geo_f", "box4_f", "cov_f"), canvas_geo(float(X0), float(Y0), float(X1), float(Y1))):
                    per[key].append(torch.from_numpy(v))
                cl = g["crop"]
                for key, v in zip(("geo_c", "box4_c", "cov_c"), canvas_geo(cl.ox, cl.oy, cl.ox + cl.nw, cl.oy + cl.nh)):
                    per[key].append(torch.from_numpy(v))
                if self.want_crop:
                    crops.append(RC.u8_canvas(img.crop(g["ltrb"]), cl))
                img_index.append(k)
                rows.append(r)
        rows = np.array(rows)
        out = {k: torch.stack(v) for k, v in per.items()}
        out.update(full=torch.stack(full), geom=RC.collate_geoms(geoms, img_index), rows=torch.from_numpy(rows),
                   lv=torch.from_numpy(self.lv[rows]), Y=self.Y[rows])
        if self.want_crop:
            out["crops"] = torch.stack(crops)
        return out


def to_dev(b, dev):
    o = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items() if k != "geom"}
    o["geom"] = RC.to_device(b["geom"], dev)
    return o


@torch.no_grad()
def cond_inputs(rae, decs, conds, b, P):
    full = rae.encode(b["full"])
    g = b["geom"]
    out = {}
    crop_kw = dict(geo=b["geo_c"], box4=b["box4_c"], cov=b["cov_c"])
    for c in conds:
        if c == "full":
            tok, kw = full[g["img_index"]], dict(geo=b["geo_f"], box4=b["box4_f"], cov=b["cov_f"])
        elif c == "bil":
            tok, kw = RC.bilinear(full, g), crop_kw
        elif c == "crop":
            tok, kw = rae.encode(b["crops"]), crop_kw
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = decs[c](RC.gather_roi(full, g), None, g, P).float()
            kw = crop_kw
        cov = kw["cov"]
        out[c] = dict(tok=tok, mean=(tok * cov[..., None]).sum(1) / cov.sum(1, keepdim=True), **kw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="")
    ap.add_argument("--dec", default="", help="tag=ckpt,...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readers", default="r0,r2")
    ap.add_argument("--r2_lr", type=float, default=3e-4)
    ap.add_argument("--r0_lr", type=float, default=1e-4)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_bs", type=int, default=24)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--stats_images", type=int, default=4000)
    ap.add_argument("--limit_steps", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    RC.set_fp32()
    data = C.load_data()
    Ytr_lv = {lv: data["train"][1][(data["train"][0].level == lv).values] for lv in LEVELS}
    rae = RC.FrozenRAE(dev, decoder=False)
    P = torch.as_tensor(RC.query_coords(), device=dev)
    conds = [c for c in a.conds.split(",") if c]
    decs = {}
    for item in [x for x in a.dec.split(",") if x]:
        tag, path = item.split("=")
        m = TR.load_model(path, dev).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        decs[tag] = m
        conds.append(tag)
    want_crop = "crop" in conds
    for c in conds:
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
    ev_ds = {sp: PacoDS(*data[sp], False, want_crop) for sp in data}

    def loader(ds, idx_lists, workers=a.workers):
        return DataLoader(ds, batch_size=None, sampler=idx_lists, num_workers=workers, pin_memory=True,
                          prefetch_factor=a.prefetch if workers else None)

    need = [c for c in conds if not os.path.exists(os.path.join(a.out, c, "mean_stats.pt"))]
    if need:
        idx = np.sort(np.random.default_rng(0).choice(len(ev_ds["train"]), a.stats_images, replace=False)).tolist()
        feats, lvs = {c: [] for c in need}, []
        for b in loader(ev_ds["train"], [idx[i:i + 32] for i in range(0, len(idx), 32)]):
            b = to_dev(b, dev)
            inp = cond_inputs(rae, decs, need, b, P)
            for c in need:
                feats[c].append(inp[c]["mean"].cpu())
            lvs.append(b["lv"].cpu())
        lv = torch.cat(lvs)
        for c in need:
            f = torch.cat(feats[c])
            st = [(f[lv == k].mean(0), f[lv == k].std(0) + 1e-6) for k in range(len(LEVELS))]
            torch.save(dict(mu=torch.stack([s[0] for s in st]), sd=torch.stack([s[1] for s in st])), os.path.join(a.out, c, "mean_stats.pt"))
    stats = {}
    for c in conds:
        z = torch.load(os.path.join(a.out, c, "mean_stats.pt"))
        stats[c] = [(z["mu"][k], z["sd"][k]) for k in range(len(LEVELS))]
    readers, opts, names = [], [], []
    for c in conds:
        for kind in a.readers.split(","):
            lr = {"r0": a.r0_lr, "r2": a.r2_lr}[kind]
            for seed in [int(x) for x in a.seeds.split(",")]:
                torch.manual_seed(1000 * seed + 7)
                rd = RD.make_reader(kind, stats[c]).to(dev)
                nm = f"{kind}_lr{lr:g}_s{seed}"
                readers.append(rd)
                names.append((c, nm))
                opts.append(torch.optim.AdamW(RD.reader_param_groups(rd, lr, a.head_lr, a.wd), lr=lr, betas=(0.9, 0.999)))
                os.makedirs(os.path.join(a.out, c, nm), exist_ok=True)
    rdir = lambda n: os.path.join(a.out, n[0], n[1])
    tagkey = "_".join(conds) + f"_s{a.seeds.replace(',', '')}"

    @torch.no_grad()
    def score(sp):
        for rd in readers:
            rd.eval()
        ds = ev_ds[sp]
        Sc = {n: np.zeros((len(data[sp][0]), 59), np.float32) for n in names}
        for b in loader(ds, [list(range(i, min(len(ds), i + a.eval_bs))) for i in range(0, len(ds), a.eval_bs)]):
            b = to_dev(b, dev)
            inp = cond_inputs(rae, decs, conds, b, P)
            rows = b["rows"].cpu().numpy()
            for n, rd in zip(names, readers):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Sc[n][rows] = rd(inp[n[0]], b["lv"]).float().cpu().numpy()
        for rd in readers:
            rd.train()
        return Sc

    def test_best():
        for n, rd in zip(names, readers):
            rd.load_state_dict(torch.load(os.path.join(rdir(n), "best.pt"), map_location=dev)["reader"])
        Sc = score("test")
        res = {}
        for n in names:
            ep = torch.load(os.path.join(rdir(n), "best.pt"), map_location="cpu")["epoch"]
            np.save(os.path.join(rdir(n), f"scores_test_ep{ep}.npy"), Sc[n])
            res["/".join(n)] = dict(best_epoch=ep, test_pooled=C.level_metrics(Sc[n], *data["test"], Ytr_lv))
        json.dump(res, open(os.path.join(a.out, f"test_once_{tagkey}.json"), "w"), indent=1)
        print(json.dumps({n: dict(ep=r["best_epoch"], test=round(r["test_pooled"]["mAP_levelmean"], 2)) for n, r in res.items()}), flush=True)

    if a.test_only:
        test_best()
        return
    tr_ds = PacoDS(*data["train"], True, want_crop)
    steps_of = lambda e: [[i for sh in st for i in sh] for st in C.full_epoch_batches(tr_ds.n_roi, e, a.data_seed)]
    steps_per_ep = [len(C.full_epoch_batches(tr_ds.n_roi, e, a.data_seed)) for e in range(1, a.epochs + 1)]
    total, warm = sum(steps_per_ep), steps_per_ep[0]
    cfg = dict(vars(a), readers=["/".join(n) for n in names], n_params={"/".join(n): RD.n_params(r) for n, r in zip(names, readers)},
               steps_per_epoch=steps_per_ep, total_steps=total, gpu=torch.cuda.get_device_name(0), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    json.dump(cfg, open(os.path.join(a.out, f"config_{tagkey}.json"), "w"), indent=1)
    best = {n: -1.0 for n in names}
    step, t0, start_ep = 0, time.time(), 1
    state_f = os.path.join(a.out, f"state_{tagkey}.pt")
    if a.resume and os.path.exists(state_f):
        st = torch.load(state_f, map_location=dev)
        for n, rd, opt in zip(names, readers, opts):
            rd.load_state_dict(st["readers"]["/".join(n)])
            opt.load_state_dict(st["opts"]["/".join(n)])
        best = {n: st["best"]["/".join(n)] for n in names}
        step, start_ep, t0 = st["step"], st["epoch"] + 1, time.time() - st["elapsed_s"]

    def evaluate(ep, run):
        te = time.time()
        Sc = score("val")
        for n, rd in zip(names, readers):
            np.save(os.path.join(rdir(n), f"scores_val_ep{ep}.npy"), Sc[n])
            rec = dict(epoch=ep, step=step, elapsed_s=time.time() - t0, val=C.level_metrics(Sc[n], *data["val"], Ytr_lv), eval_s=time.time() - te, **run.get(n, {}))
            with open(os.path.join(rdir(n), "curves.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["val"]["mAP_levelmean"] > best[n]:
                best[n] = rec["val"]["mAP_levelmean"]
                torch.save(dict(reader=rd.state_dict(), epoch=ep, val=rec["val"]), os.path.join(rdir(n), "best.pt"))
        print(json.dumps(dict(ep=ep, val={"/".join(n): round(best[n], 2) for n in names}, eval_s=round(time.time() - te))), flush=True)

    if start_ep == 1 and not a.limit_steps:
        evaluate(0, {})
    for ep in range(start_ep, a.epochs + 1):
        dl = DataLoader(tr_ds, batch_size=None, sampler=steps_of(ep), num_workers=a.workers, pin_memory=True, prefetch_factor=a.prefetch)
        run_l, run_n, t_ep = {n: 0.0 for n in names}, 0.0, time.time()
        for b in dl:
            b = to_dev(b, dev)
            inp = cond_inputs(rae, decs, conds, b, P)
            lv, Y = b["lv"], b["Y"]
            nt = float((Y >= 0).sum())
            for n, rd, opt in zip(names, readers, opts):
                for g in opt.param_groups:
                    g["lr"] = C.lr_at(step, g["base"], warm, total)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = rd(inp[n[0]], lv)
                ls, _ = C.masked_bce_sum(o, Y)
                opt.zero_grad(set_to_none=True)
                (ls / nt).backward()
                torch.nn.utils.clip_grad_norm_(rd.parameters(), 1.0)
                opt.step()
                run_l[n] += float(ls.detach())
            run_n += nt
            step += 1
            if step % 400 == 0 or (a.limit_steps and step % 5 == 0):
                print(json.dumps(dict(step=step, epoch=ep, s=round(time.time() - t0), mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                                      loss_run={"/".join(n): round(run_l[n] / run_n, 4) for n in names})), flush=True)
            if a.limit_steps and step >= a.limit_steps:
                print("SMOKE_DONE", flush=True)
                return
        evaluate(ep, {n: dict(train_running_loss=run_l[n] / run_n, train_s=time.time() - t_ep) for n in names})
        torch.save(dict(readers={"/".join(n): rd.state_dict() for n, rd in zip(names, readers)}, opts={"/".join(n): o.state_dict() for n, o in zip(names, opts)},
                        best={"/".join(n): best[n] for n in names}, step=step, epoch=ep, elapsed_s=time.time() - t0), state_f)
    if a.test:
        test_best()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
