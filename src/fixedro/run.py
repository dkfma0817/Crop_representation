"""Fixed-readout substitution for every 48 x 48 method (no training): the F768 R2 reader of output/2026-11-11/token_split_x2_s0
(probes/T768/r2, epoch frozen on official Val = 6) is loaded once and never trained; only its 24 x 24 ROI input changes.

Read contract (identical for every method; = how the reader was trained on F768):
  feature map on the 48 x 48 grid of the 384 canvas (child (ci, cj) = 8 px cell, centre (4 + 8 cj, 4 + 8 ci)), DINOv3 final-token space;
  24 x 24 read points = the C0 crop-grid points of the GT box (bil_grid, continuous box geometry), F.grid_sample(bilinear, border,
  align_corners=False); R2 geometry / coverage mask (cov > 0 keys) unchanged.
  Global maps (one 48 x 48 map per image, read directly):
    F768      frozen DINOv3 on the native 768 canvas (the reader's own input; reproduction check)
    F384NN    F384 24 x 24 final tokens repeated 2 x 2 (nearest)
    F384BIL   F384 bilinear x2 (24 -> 48, align_corners=False) == B0 (the same tensor)
    S1        S1 child map (linear split of every parent; generated everywhere, no fill)
    T768M     T768-MeanOnly (2 x 2 mean of the F768 map, repeated)
  ROI-generated (decoder run per GT box on its read rectangle = the smallest parent rectangle containing every bilinear corner of
  every valid read point, i.e. the box's parents plus the 0/1-cell margin the bilinear read needs; no cap on large boxes):
    S2, S3L2, S3KL, FAA, FAB, P1, P2: all children of the rectangle are decoder outputs (letterbox-invalid children included, as
    generated); local 2H x 2W map -> read points in local coordinates (src.fte.common.local_read: equal to the global read at
    every valid point, checked). Nothing outside the rectangle is reachable from a valid read point; nothing is filled.
Not evaluated: RSP-Masked / noPix (see the README: 96 x 96 4-px grid of a different dense p4 teacher; their read keeps F384 values at
cov = 0 points), so they do not meet the 48 x 48 F768 contract without an extra pooling step that is not part of the method.

  python -m src.fixedro.run --method P2 --split val --root output/2026-11-12/fixed_readout_all
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.g1native.bbdata as BD
import src.readout.common as RC
from src.fte.common import local_read
from src.readout.models import Reader
from src.tsplit.features import PREFIX, ContentDS, frozen_vit, mean_only, read_rects, sample_maps, teacher, up2, valid48
from src.tsplit.model import G, G2, rect_index

DATA = "output/2026-11-07/paco_oracle_attr/data"
T768 = Path("output/2026-11-11/token_split_x2_s0")
CK = dict(S1=T768 / "train/S1.pt", S2=T768 / "train/S2.pt", S3L2=T768 / "train/S3.pt", S3KL=Path("output/2026-11-11/tsplit_distillv3_patchkl_s0/train/S3.pt"),
          FAA=Path("output/2026-11-11/fine_aware_mean_residual_s0/train/model.pt"), FAB=Path("output/2026-11-11/fine_aware_patchce_sigreg_s0/train/model.pt"),
          P1=Path("output/2026-11-12/p1_smallvit_latesplit_s0/train/model.pt"), P2=Path("output/2026-11-12/p2_dinov3_pretrained_earlysplit_s0/train/model.pt"))
GLOBAL = ("F768", "F384NN", "F384BIL", "S1", "T768M")
ROI = ("S2", "S3L2", "S3KL", "FAA", "FAB", "P1", "P2")


@torch.no_grad()
def student_all(vit, x, chunk=64):
    """One original frozen forward: final patch tokens, patch_embed output (early), final-norm input (X_pre), blocks[7] output (h8)."""
    outs = dict(final=[], early=[], xpre=[], h8=[])
    for i in range(0, len(x), chunk):
        st = {}
        hs = [vit.patch_embed.register_forward_hook(lambda m, a, o: st.__setitem__("early", o)),
              vit.norm.register_forward_pre_hook(lambda m, a: st.__setitem__("xpre", a[0])),
              vit.blocks[7].register_forward_hook(lambda m, a, o: st.__setitem__("h8", o))]
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = vit.forward_features(x[i:i + chunk])
        finally:
            for h in hs:
                h.remove()
        outs["final"].append(t[:, PREFIX:].float())
        outs["early"].append(st["early"].flatten(1, 2).float())
        outs["xpre"].append(st["xpre"][:, PREFIX:].float())
        outs["h8"].append(st["h8"])
    return {k: torch.cat(v) for k, v in outs.items()}


def to_parent_major(Y, rects):
    """(R, N, D) children in rect_index order (row-major over the 2H x 2W child rectangle) -> (R, K, 4, D) parent-major, slot 2a + b."""
    R, N, D = Y.shape
    H, W = rects[:, 2], rects[:, 3]
    K = int((H * W).max())
    t = torch.arange(K, device=Y.device)[None]
    li = torch.div(t, W[:, None], rounding_mode="floor")
    lj = t % W[:, None]
    a = torch.arange(2, device=Y.device)
    rows = 2 * li[..., None, None] + a[:, None]
    cols = 2 * lj[..., None, None] + a[None, :]
    pos = (rows * (2 * W)[:, None, None, None] + cols).reshape(R, K * 4)
    pos = pos.clamp(max=N - 1)                                  # padded parents (t >= H*W) are never read by local_read
    return Y.gather(1, pos[..., None].expand(-1, -1, D)).view(R, K, 4, D)


class Method:
    def __init__(self, name, dev):
        self.name, self.dev = name, dev
        self.vit = frozen_vit(dev)
        self.m = None
        if name in ("S1", "S2", "S3L2", "S3KL"):
            from src.tsplit.model import TokenSplit
            ck = torch.load(CK[name], map_location="cpu", weights_only=False)
            self.m = TokenSplit(ck["kind"]).to(dev).eval()
            self.m.load_state_dict(ck["model"])
            self.kind = ck["kind"]
        elif name in ("FAA", "FAB"):
            from src.finedec.model import FineAware
            self.m = FineAware(self.vit).to(dev).eval()
            self.m.load_state_dict(torch.load(CK[name], map_location="cpu", weights_only=False)["model"])
        elif name in ("P1", "P2"):
            from src.fte.models import P1, P2
            self.m = (P1 if name == "P1" else P2)(self.vit).to(dev).eval()
            self.m.load_state_dict(torch.load(CK[name], map_location="cpu", weights_only=False)["model"])
        if self.m is not None:
            for p in self.m.parameters():
                p.requires_grad_(False)
        self.native = name in ("F768", "T768M")

    @torch.no_grad()
    def tokens(self, b):
        x, grid, ii = b["x"], b["bil_grid"], b["img_index"]
        R = len(ii)
        n = self.name
        if n in ("F768", "T768M"):
            full = teacher(self.vit, x, chunk=24)
            maps = full if n == "F768" else mean_only(full)
        else:
            f = student_all(self.vit, x)
            if n == "F384NN":
                maps = f["final"].view(-1, G, 1, G, 1, 768).expand(-1, G, 2, G, 2, 768).reshape(-1, G2 * G2, 768)
            elif n == "F384BIL":
                maps = up2(f["final"])
            elif n == "S1":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    maps = self.m.child_map(f["final"], f["early"]).float()
            else:
                return self.roi_tokens(f, b)
        return torch.cat([sample_maps(maps[ii[i:i + 96]], grid[i:i + 96]) for i in range(0, R, 96)])

    def roi_tokens(self, f, b):
        grid, ii = b["bil_grid"], b["img_index"]
        R = len(ii)
        rects = read_rects(grid, b["cov"])
        v48 = valid48(b["content"])
        n = self.name
        Kl = (rects[:, 2] * rects[:, 3]).tolist()
        cost = (lambda k: 581 + 4 * k) if n == "P2" else (lambda k: 4 * k)
        budget = 40000
        if n in ("S2", "S3L2", "S3KL"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cmap = self.m.child_map(f["final"], f["early"]).float()
        if n in ("FAA", "FAB"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fine, kvs = self.m.image_side(b["x"], f["final"])
        order = sorted(range(R), key=lambda r: Kl[r])
        out = [None] * R
        i = 0
        while i < R:
            j = i + 1
            while j < R and cost(Kl[order[j]]) * (j + 1 - i) <= budget:
                j += 1
            sel = torch.tensor(order[i:j], device=grid.device)
            rs, ri = rects[sel], ii[sel]
            if n in ("P1", "P2"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Y, _, _, _ = self.m(f["final"] if n == "P1" else f["h8"], ri, rs)
            else:
                idx, cy, cx, ok = rect_index(rs)
                key_ok = ok & v48[ri[:, None], idx]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if n in ("FAA", "FAB"):
                        Yc, _, _, _ = self.m.decode(fine, kvs, f["xpre"], ri, idx, cy, cx, ok, key_ok, rs[:, 0], rs[:, 1], rs[:, 3])
                    else:
                        Yc = self.m.refine(cmap, ri, idx, cy, cx, key_ok)
                Y = to_parent_major(Yc.float(), rs)
            tok = local_read(Y.float(), rs, grid[sel])
            for k, r in enumerate(order[i:j]):
                out[r] = tok[k]
            i = j
        return torch.stack(out)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=GLOBAL + ROI)
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    t0 = time.time()
    ep = json.loads((T768 / "eval/selection_frozen.json").read_text())["T768_r2"]["epoch"]
    reader = Reader("r2", layers=(12,)).to(dev)
    reader.load_state_dict(torch.load(T768 / f"probes/T768/r2/ep{ep}.pt", map_location=dev, weights_only=False)["reader"])
    reader.eval().requires_grad_(False)
    meth = Method(a.method, dev)
    BD.NATIVE = meth.native
    M = pd.read_parquet(f"{DATA}/manifest_{a.split}.parquet")
    ds = ContentDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    L = torch.full((len(M), 59), float("nan"))
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=8, collate_fn=RC.collate_one, pin_memory=True):
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        tok = meth.tokens(b)
        b["img_index"] = torch.arange(len(tok), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            L[rows] = reader({12: tok}, b)["logits"].float().cpu()
    assert not torch.isnan(L).any()
    out = Path(a.root) / "preds" / a.method
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"probs_{a.split}.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=L.numpy(), probs=torch.sigmoid(L).numpy())
    (out / f"info_{a.split}.json").write_text(json.dumps(dict(method=a.method, split=a.split, reader=str(T768 / f"probes/T768/r2/ep{ep}.pt"), reader_epoch=ep, n_rows=len(M),
                                                               ckpt=str(CK.get(a.method, "")), seconds=round(time.time() - t0))))
    print("done", a.method, a.split, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
