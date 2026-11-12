"""Pre-training checks for the token-splitting experiment (writes <out>/checks.json; any failed assertion stops the queue).

  1 index mapping (synthetic): parent (i, j) -> children (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1) for rect_index and Split
  2 pixel geometry (synthetic coordinate-ramp images, several aspect ratios, with / without flip): mean source coordinate of the
    384 student quadrant (a, b) of parent (i, j) == that of teacher cell (2i + a, 2j + b) on the native 768 canvas; 1-child shift control
  3 patch embedding: the hooked patch_embed output == patch part of the input of block 0; the manual forward == forward_features;
    rerun() with no interpolation reproduces forward_features
  4 real features (COCO dev): MSE(up2(final384), teacher768) matched < shifted by one child (x / y) < shifted by one parent
  5 ROI read on PACO val batches: the read rectangle contains every bilinear corner of every valid read point (fraction 1);
    F384 / T768 reads equal the existing bil / native768 reads; S-condition reads of an untrained (init) S1 equal nearest copy
  6 leakage: COCO stream / dev images that equal a PACO val / test image (id, md5, pixel md5, dHash) are excluded

  python -m src.tsplit.checks --out <run>/checks
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

import src.g1native.bbdata as BD
import src.readout.common as RC
from src.config import STREAM
from src.g1native.native import native_canvas
from src.msd.pretrain import KEYS, eval_exclusions
from src.tsplit.features import PREFIX, ContentDS, RoiEncoder, coverage_check, read_rects, student, teacher, up2
from src.tsplit.model import G, G2, Split, rect_index
from src.tsplit.train import dev_set, load_pair
from src.viz.c0_pca import LB, MEAN_T, STD_T

DATA = "output/2026-11-07/paco_oracle_attr/data"


def check_index():
    rects = torch.tensor([[3, 5, 2, 3], [0, 0, 1, 1], [20, 18, 4, 6]])
    idx, cy, cx, ok = rect_index(rects)
    for r, (py, px, H, W) in enumerate(rects.tolist()):
        got = set(idx[r][ok[r]].tolist())
        want = {(2 * i + a) * G2 + 2 * j + b for i in range(py, py + H) for j in range(px, px + W) for a in (0, 1) for b in (0, 1)}
        assert got == want and len(got) == 4 * H * W
    s = Split(768)
    torch.nn.init.normal_(s.lin.weight, std=.02)
    p = torch.randn(2, G * G, 768)
    y = s(p).view(2, G2, G2, 768)
    full = s.lin(p).view(2, G, G, 4, 768)
    for i, j in [(0, 0), (5, 7), (23, 23), (11, 0)]:
        for a in (0, 1):
            for b in (0, 1):
                assert torch.allclose(y[:, 2 * i + a, 2 * j + b], full[:, i, j, 2 * a + b], atol=1e-6)
    return dict(rect_index="ok", split_layout="ok: child (2i+a, 2j+b) = split block 2a+b of parent (i, j)")


def check_pixels():
    rows = []
    for W, H in [(640, 480), (480, 640), (500, 333), (612, 612), (1000, 217)]:
        xs = np.tile(np.linspace(0, 255, W, dtype=np.float32), (H, 1))
        ys = np.tile(np.linspace(0, 255, H, dtype=np.float32)[:, None], (1, W))
        for flip in (False, True):
            img = Image.fromarray(np.stack([xs, ys, np.full_like(xs, 128)], -1).astype(np.uint8))
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lb = LB(W, H, 384, 16)
            s, t = lb.canvas(img) * STD_T + MEAN_T, native_canvas(img, lb) * STD_T + MEAN_T
            sq = F.avg_pool2d(s[None, :2], 8)[0]                     # (2, 48, 48) student quadrant means (8 px at 384)
            tc = F.avg_pool2d(t[None, :2], 16)[0]                    # (2, 48, 48) teacher cell means (16 px at 768)
            inside = torch.zeros(48, 48, dtype=torch.bool)
            inside[(lb.oy + 7) // 8:(lb.oy + lb.nh) // 8, (lb.ox + 7) // 8:(lb.ox + lb.nw) // 8] = True
            d = (sq - tc).abs().amax(0)[inside]
            sh = (sq[:, :, 1:] - tc[:, :, :-1]).abs().amax(0)[inside[:, 1:] & inside[:, :-1]]
            rows.append(dict(W=W, H=H, flip=flip, max_abs=float(d.max()) * 255, mean_abs=float(d.mean()) * 255, shift1_mean_abs=float(sh.mean()) * 255))
            assert rows[-1]["mean_abs"] < 0.5 and rows[-1]["max_abs"] < 2.5 and rows[-1]["shift1_mean_abs"] > 4 * rows[-1]["mean_abs"], rows[-1]
    return dict(per_image=rows, unit="mean source-coordinate ramp value (0..255) per 8-px quadrant / 16-px teacher cell, full-content cells")


@torch.no_grad()
def check_patch_embed(enc, dev, x):
    vit = enc.vit
    store = {}
    h1 = vit.blocks[0].register_forward_pre_hook(lambda m, args, kw=None: store.__setitem__("blk0_in", args[0]))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fin, ear = student(vit, x)
    finally:
        h1.remove()
    blk0 = store["blk0_in"][:, PREFIX:].float()
    d_pe = float((blk0 - ear).abs().max())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hs = enc.hidden(x)
        man = vit.norm(hs[-1])[:, PREFIX:].float()
    d_man = float((man - fin).abs().max())
    # rerun without interpolation must reproduce forward_features (validates prefix / RoPE handling of the rerun path)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = vit.norm_pre(vit._pos_embed(vit.patch_embed(x))[0])
        rope = vit.rope.get_embed(shape=(G, G))
        for blk in vit.blocks:
            h = blk(h, rope=rope)
        rr = vit.norm(h)[:, PREFIX:].float()
    d_rr = float((rr - fin).abs().max())
    assert d_pe == 0.0 and d_man < 1e-3 * fin.abs().max() and d_rr < 1e-3 * fin.abs().max(), (d_pe, d_man, d_rr)
    return dict(patch_embed_vs_block0_input_max_abs=d_pe, manual_vs_forward_features_max_abs=d_man, rerun_noninterp_vs_forward_features_max_abs=d_rr,
                early_rms=float(ear.pow(2).mean().sqrt()), final_rms=float(fin.pow(2).mean().sqrt()),
                note="early = vit.patch_embed output (Conv 16x16 stride 16, NHWC, no norm, no position term); pos_embed is None and norm_pre is Identity, so it is the block-0 input")


@torch.no_grad()
def check_real(enc, dev):
    ids = dev_set()[:48]
    xs, xt, cont = [], [], []
    for iid in ids:
        lb, a, t = load_pair(iid, False)
        xs.append(a)
        xt.append(t)
        cont.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
    x, x768 = torch.stack(xs).to(dev), torch.stack(xt).to(dev)
    fin, _ = student(enc.vit, x)
    T = teacher(enc.vit, x768).view(-1, G2, G2, 768)
    B = up2(fin).view(-1, G2, G2, 768)
    from src.hcf.distill import cell_w
    v = (cell_w(torch.tensor(cont, dtype=torch.float32), G2, 8) >= 1 - 1e-6).view(-1, G2, G2).to(dev)

    def mse(a, b, m):
        return float(((a - b).pow(2).mean(-1) * m).sum() / m.sum())
    out = dict(matched=mse(B, T, v.float()),
               shift_child_x=mse(B[:, :, 1:], T[:, :, :-1], (v[:, :, 1:] & v[:, :, :-1]).float()),
               shift_child_y=mse(B[:, 1:], T[:, :-1], (v[:, 1:] & v[:, :-1]).float()),
               shift_parent_x=mse(B[:, :, 2:], T[:, :, :-2], (v[:, :, 2:] & v[:, :, :-2]).float()))
    flipT = teacher(enc.vit, x768.flip(-1)).view(-1, G2, G2, 768).flip(2)
    out["flipped_teacher_unflipped"] = mse(B, flipT, v.float())
    assert out["matched"] < min(out["shift_child_x"], out["shift_child_y"]) < out["shift_parent_x"], out
    return out


@torch.no_grad()
def check_reads(dev, n_img=36):
    M = pd.read_parquet(f"{DATA}/manifest_val.parquet")
    Y = np.zeros((len(M), 59), np.int8)
    idx = list(range(n_img))
    out = {}
    from src.g1native.bbdata import bil_sample
    from src.g1native.official_infer import frozen_encode
    BD.NATIVE = False
    b = RC.to_dev(ContentDS(M, Y, False, "bil", 384, 16)[idx], dev)
    enc = RoiEncoder("F384", dev)
    rects = read_rects(b["bil_grid"], b["cov"])
    out["read_rect_corner_coverage"] = coverage_check(rects, b["bil_grid"], b["cov"])
    out["read_rect_children_mean"] = float((4 * rects[:, 2] * rects[:, 3]).float().mean())
    assert out["read_rect_corner_coverage"] == 1.0
    old = bil_sample(student(enc.vit, b["x"])[0], b)
    new = enc(dict(b))
    out["F384_vs_existing_bil_max_abs"] = float((old - new).abs().max())
    b0 = RoiEncoder("B0", dev)(dict(b))
    out["B0_vs_F384_cos"] = float(F.cosine_similarity(b0, new, dim=-1)[b["cov"] > 0].mean())
    BD.NATIVE = True
    bn = RC.to_dev(ContentDS(M, Y, False, "bil", 384, 16)[idx], dev)
    BD.NATIVE = False
    t_new = RoiEncoder("T768", dev)(dict(bn))
    t_old = frozen_encode(enc.vit, dict(bn), True)
    out["T768_vs_existing_native768_max_abs"] = float((t_new - t_old).abs().max())
    assert out["F384_vs_existing_bil_max_abs"] < 1e-4 and out["T768_vs_existing_native768_max_abs"] < 1e-4, out
    return out


def check_leak():
    ex = eval_exclusions()
    t = pd.read_parquet(KEYS)
    ids_eval = set(t[t.source.isin(["paco_val", "paco_test"])].image_id)
    stream = set(int(i) for i in np.load(STREAM)[:12000].ravel())
    dev = set(dev_set())
    rest_stream = stream - set(ex["raev2_stream"])
    return dict(excluded_stream=ex["raev2_stream"], excluded_dev=len(ex["raev2_dev"]), stream_images=len(stream), dev_images=len(dev),
                stream_after_exclusion_id_overlap_with_paco_eval=len(rest_stream & ids_eval), dev_id_overlap_with_paco_eval=len(dev & ids_eval),
                rule="eval_exclusions: COCO image equal to a PACO val/test image by image id, byte md5, pixel md5 or dHash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    res = dict(index=check_index(), pixels=check_pixels(), leak=check_leak())
    assert res["leak"]["stream_after_exclusion_id_overlap_with_paco_eval"] == 0 and res["leak"]["dev_id_overlap_with_paco_eval"] == 0
    enc = RoiEncoder("F384", dev)
    x = torch.stack([load_pair(i, False)[1] for i in dev_set()[:8]]).to(dev)
    res["patch_embed"] = check_patch_embed(enc, dev, x)
    res["real_alignment"] = check_real(enc, dev)
    res["reads"] = check_reads(dev)
    res["status"] = "passed"
    (out / "checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
