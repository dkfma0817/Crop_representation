"""Verification of the direct dense-teacher read (run before any R2 probe; GPU). -> <out>/unit_tests.json

  python -m src.dtev.checks --out output/2026-11-12/dense_teacher_eval/checks
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.dtev.encoder import corner_rects, dt_tokens, sample_local
from src.dtev.teacher import DT
from src.rsp.common import PREFIX
from src.rsp.data import RSPContentDS
from src.tsplit.features import frozen_vit, sample_maps
from src.tsplit.official_infer import DATA

RES = {}


def test(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                r = fn(*a, **k)
                RES[name] = dict(status="PASS", **(r or {}), sec=round(time.time() - t0, 1))
            except Exception as e:                                           # noqa: BLE001
                RES[name] = dict(status="FAIL", error=repr(e), tb=traceback.format_exc()[-1500:])
            print(name, json.dumps(RES[name])[:800], flush=True)
        return run
    return deco


def mx(a, b):
    return float((a.float() - b.float()).abs().max())


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def paco_batches(split, n_batches, flip, dev, seed=0):
    RSPContentDS.WANT_EV = False
    M = pd.read_parquet(f"{DATA}/manifest_{split}.parquet")
    ds = RSPContentDS(M, np.zeros((len(M), 59), np.int8), flip, "bil", 384, 16)
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(ds) - 12, n_batches, replace=False)
    for s in starts:
        torch.manual_seed(int(s))
        yield RC.to_dev(RC.collate_one([ds[list(range(s, s + 12))]]), dev)


@test("pi_resize_identity")
def t_pi(vit, dts, x):
    """fine token of a p_T x p_T canvas patch == DINOv3 patch embedding of that patch bilinearly upsampled to 16 x 16 (patch-wise)."""
    out = {}
    for pt, dt in dts.items():
        e = dt.fine_embed(x)
        gf = dt.gf
        ps = x.unfold(2, pt, pt).unfold(3, pt, pt)                         # (B, 3, gf, gf, pt, pt)
        sel = [(0, 3, 5), (0, gf // 2, gf // 3), (1, gf - 1, 0), (1, 7, gf - 2)]
        errs = []
        for bi, i, j in sel:
            up = F.interpolate(ps[bi, :, i, j][None].double(), size=(16, 16), mode="bilinear", align_corners=False)
            ref = F.conv2d(up, vit.patch_embed.proj.weight.double(), vit.patch_embed.proj.bias.double())[0, :, 0, 0]
            errs.append(rel(e[bi, i * gf + j].double(), ref))
        out[f"pt{pt}_max_rel_err"] = max(errs)
        assert max(errs) < 1e-5, errs
    return out


@test("coarse_equals_timm")
def t_coarse(vit, dts, x):
    final, _ = dts[8].coarse(x)
    d = mx(final, vit.forward_features(x))
    assert d < 1e-4, d
    return dict(fp32_max_abs=d)


def synth_rects():
    return torch.tensor([[3, 4, 3, 3], [0, 0, 3, 4], [10, 9, 6, 8], [5, 6, 4, 5], [20, 21, 4, 3]])   # incl. edge, overlap


@test("masked_single_vs_multi_vs_efficient")
def t_masked(vit, dts, x):
    out = {}
    pr = synth_rects().to(x.device)
    for pt, dt in dts.items():
        x1 = x[:1]
        singles = [dt.ref_masked(x1, pr[m:m + 1])[1][0] for m in range(len(pr))]
        cm, multi = dt.ref_masked(x1, pr)
        final, kv = dt.coarse(x1)
        T, ok = dt.fine_masked(dt.fine_embed(x1), kv, pr, torch.zeros(len(pr), dtype=torch.long, device=x.device))
        d_sm = max(mx(s, m) for s, m in zip(singles, multi))
        d_se = max(mx(s, T[m, :len(s)]) for m, s in enumerate(singles))
        d_c = mx(cm, final[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fb, kvb = dt.coarse(x1)
            Tb, _ = dt.fine_masked(dt.fine_embed(x1), kvb, pr, torch.zeros(len(pr), dtype=torch.long, device=x.device))
        cos = min(float(F.cosine_similarity(Tb[m, :len(s)].float(), s, dim=-1).mean()) for m, s in enumerate(singles))
        out[f"pt{pt}"] = dict(single_vs_multi_literal=d_sm, single_literal_vs_efficient=d_se, coarse_rows_vs_forward_features=d_c,
                              bf16_efficient_vs_fp32_single_min_cos=cos, token_abs_max=float(max(s.abs().max() for s in singles)))
        assert d_sm < 1e-3 and d_se < 1e-3 and d_c < 1e-3 and cos > 0.995, out
    return out


@test("full_single_literal_vs_batched_padded")
def t_full(vit, dts, x):
    out = {}
    pr = synth_rects().to(x.device)
    rimg = torch.tensor([0, 1, 0, 1, 0], device=x.device)
    for pt, dt in dts.items():
        refs = [dt.ref_full(x[int(rimg[m])][None], pr[m]) for m in range(len(pr))]
        h0, _ = dt.embed0(x)
        T, ok = dt.fine_full(h0, dt.fine_embed(x), pr, rimg)
        d = max(mx(r, T[m, :len(r)]) for m, r in enumerate(refs))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h0b, _ = dt.embed0(x)
            Tb, _ = dt.fine_full(h0b, dt.fine_embed(x), pr, rimg)
        cos = min(float(F.cosine_similarity(Tb[m, :len(r)].float(), r, dim=-1).mean()) for m, r in enumerate(refs))
        # unrestricted differs from masked (sanity: the restriction matters)
        _, kv = dt.coarse(x)
        Tm, _ = dt.fine_masked(dt.fine_embed(x), kv, pr, rimg)
        cfm = float(np.mean([float(F.cosine_similarity(Tm[m, :len(r)], r, dim=-1).mean()) for m, r in enumerate(refs)]))
        out[f"pt{pt}"] = dict(literal_vs_batched_max_abs=d, bf16_min_cos=cos, mean_cos_full_vs_masked=cfm)
        assert d < 1e-3 and cos > 0.995, out
    return out


@test("parent_rects_and_read_geometry_on_paco")
def t_read(vit, dts, dev):
    """Real PACO batches (val no flip, val + test with flip): every valid read corner inside the rectangle; local read == grid_sample on
    the full fine map; rectangle = box parents + 1 margin."""
    out = {}
    n_roi = 0
    worst = {}
    for split, flip, seed in (("val", False, 0), ("val", True, 1), ("test", True, 2)):
        for b in paco_batches(split, 6, flip, dev, seed):
            grid, cov, boxc = b["bil_grid"], b["cov"], b["boxc"]
            n_roi += len(cov)
            for pt, dt in dts.items():
                pr = dt.parent_rects(boxc, 1)
                fr = dt.fine_rects(pr)
                lo, hi, anyv = corner_rects(grid, cov, dt.gf)
                ins = ((lo >= fr[:, :2]) & (hi <= fr[:, :2] + fr[:, 2:] - 1)).all(1)
                assert bool(ins[anyv].all())
                # rectangle covers the box (box parents within rect) and has <= 1 cell margin
                l, t, r, bb = boxc.unbind(1)
                assert bool(((pr[:, 0] * 16 <= t.clamp(0, 383)) & (pr[:, 1] * 16 <= l.clamp(0, 383))).all())
                assert bool((((pr[:, 0] + pr[:, 2]) * 16 >= bb.clamp(1, 384)) & ((pr[:, 1] + pr[:, 3]) * 16 >= r.clamp(1, 384))).all())
                full = torch.randn(len(cov), 8, dt.gf, dt.gf, device=dev)
                hm, wm = int(fr[:, 2].max()), int(fr[:, 3].max())
                loc = torch.zeros(len(cov), 8, hm, wm, device=dev)
                for k in range(len(cov)):
                    fy, fx, h, w = fr[k].tolist()
                    loc[k, :, :h, :w] = full[k, :, fy:fy + h, fx:fx + w]
                a = sample_local(loc, fr, grid, dt.gf)
                ref = F.grid_sample(full, grid.view(-1, 24, 24, 2), mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)
                v = cov > 0
                worst[pt] = max(worst.get(pt, 0.0), mx(a[v], ref[v]))
                assert worst[pt] < 1e-4, worst
    out.update(n_roi=n_roi, local_vs_full_read_max_abs=worst)
    return out


@test("dt_tokens_end_to_end_vs_literal_single_roi")
def t_e2e(vit, dts, dev):
    """dt_tokens (bf16, batched, chunked) on a real PACO val batch vs the fp32 literal single-ROI teacher read for 6 ROIs; chunk-budget
    invariance (small vs default budget) holds up to bf16 kernel differences."""
    out = {}
    b = next(paco_batches("val", 1, False, dev, seed=3))
    for pt, dt in dts.items():
        for attn in ("masked", "full"):
            bb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            st = {}
            tok = dt_tokens(dt, attn, bb, 1, stats=st)
            bb2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            tok2 = dt_tokens(dt, attn, bb2, 1, budget=3000)
            v0 = b["cov"] > 0
            inv = mx(tok[v0], tok2[v0])
            inv_cos = float(F.cosine_similarity(tok[v0], tok2[v0], dim=-1).min())
            pr = st["prects"]
            order = np.argsort(st["n_fine"])
            pick = [int(order[0]), int(order[len(order) // 2]), int(order[-1])] + [int(i) for i in np.random.default_rng(0).choice(len(order), 3, replace=False)]
            cs = []
            for k in pick:
                ii = int(b["img_index"][k])
                x1 = b["x"][ii:ii + 1]
                T = dt.ref_masked(x1, pr[k:k + 1])[1][0] if attn == "masked" else dt.ref_full(x1, pr[k])
                fr = dt.fine_rects(pr[k:k + 1])
                rd = sample_local(T.T.reshape(1, -1, int(fr[0, 2]), int(fr[0, 3])), fr, b["bil_grid"][k:k + 1], dt.gf)[0]
                v = b["cov"][k] > 0
                if v.any():
                    cs.append(float(F.cosine_similarity(tok[k][v], rd[v], dim=-1).mean()))
            out[f"pt{pt}_{attn}"] = dict(chunk_invariance_bf16_max_abs=inv, chunk_invariance_min_cos=inv_cos, min_cos_vs_fp32_literal=min(cs), n_roi=len(order),
                                         fine_tokens_min_med_max=[int(np.min(st["n_fine"])), int(np.median(st["n_fine"])), int(np.max(st["n_fine"]))])
            assert min(cs) > 0.99 and inv_cos > 0.999, out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vit = frozen_vit(dev)
    dts = {8: DT(vit, 8).to(dev), 4: DT(vit, 4).to(dev)}
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 384, 384, generator=g).to(dev)
    b = next(paco_batches("val", 1, False, dev, seed=7))
    x = torch.cat([b["x"][:1], x[:1]])                                     # one real canvas + one noise canvas
    with torch.no_grad():
        t_pi(vit, dts, x)
        t_coarse(vit, dts, x)
        t_masked(vit, dts, x)
        t_full(vit, dts, x)
        t_read(vit, dts, dev)
        t_e2e(vit, dts, dev)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    allpass = all(r["status"] == "PASS" for r in RES.values())
    json.dump(dict(all_pass=allpass, tests=RES, torch=torch.__version__, time=time.strftime("%F %T")), open(Path(a.out) / "unit_tests.json", "w"), indent=1)
    print("ALL_PASS" if allpass else "SOME_FAILED", flush=True)
    raise SystemExit(0 if allpass else 1)


if __name__ == "__main__":
    main()
