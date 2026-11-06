"""Pre-run checks (no training result is kept). CPU by default (runs while the QAT job still owns GPU 2,3);
--gpu adds latency / peak memory. Writes <out>/smoke.json (+ alignment.json).

  alignment : synthetic coordinate feature map + special ROIs (long, tiny, border, heavy padding); query / memory
              ROI-relative coordinates vs roi_geometry (bil_grid, coords); bicubic / bilinear on linear ramps
  decoders  : for E1-E7 and R00 on real COCO stream images (frozen encoder, C0 teacher):
              shapes, finite, encoder frozen (no grad), teacher never an input (output unchanged when the teacher crops
              change), step-0 skip identity (Z == B, head == 0), gradient flow after 1 and 2 optimizer updates,
              masks (E2 invariant to non-ROI tokens; E1 / E7 valid outputs invariant to invalid-query perturbation),
              memory swap response (memory of another image -> output change), tiny-subset loss decrease (60 steps),
              params / FLOPs per ROI (+ latency, peak memory with --gpu)

  CUDA_VISIBLE_DEVICES= python -m src.xdec.smoke --out <run>/smoke
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

import src.msd.common as MS
import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT, STREAM
from src.losses.feature import dense_cos, dense_mse
from src.xdec import geometry as XG
from src.xdec import models as XM

CONDS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "R00"]


# ------------------------------------------------------------------------------------------------ alignment
def alignment():
    XG.install()
    rng = np.random.default_rng(0)
    cases = {"long_horizontal": (640, 480, (10, 200, 630, 230)), "long_vertical": (640, 480, (300, 5, 320, 475)),
             "tiny_3px": (640, 480, (400, 300, 403, 303)), "tiny_1x6": (500, 375, (100, 100, 101, 106)),
             "border_topleft": (640, 427, (0, 0, 60, 45)), "border_bottomright": (427, 640, (380, 590, 427, 640)),
             "full_image": (640, 480, (0, 0, 640, 480)), "heavy_padding_1x10": (640, 480, (50, 100, 70, 300))}
    for k in range(40):
        W, H = int(rng.integers(200, 641)), int(rng.integers(200, 641))
        l, t = int(rng.integers(0, W - 1)), int(rng.integers(0, H - 1))
        r, b = int(rng.integers(l + 1, W + 1)), int(rng.integers(t + 1, H + 1))
        cases[f"random_{k}"] = (W, H, (l, t, r, b))
    k = (np.arange(24) + 0.5) * 14
    fmap = torch.zeros(1, 576, 2)
    fmap[0, :, 0] = torch.as_tensor(np.tile(k, 24), dtype=torch.float32)
    fmap[0, :, 1] = torch.as_tensor(np.repeat(k, 24), dtype=torch.float32)
    out = {}
    worst = dict(q_impl=0.0, mem_impl=0.0, bil_lin=0.0, bic_lin=0.0)
    for name, (W, H, box) in cases.items():
        g = MS.roi_geometry(W, H, box)
        gb = MS.collate_geoms([{kk: g[kk] for kk in MS.GEOM_KEYS}], [0])
        Xq = XG.query_canvas_from_uv(gb)[0]                                       # from (u, v) and the box
        Xb = (gb["bil_grid"][0].double() + 1) * 168                                # roi_geometry sampling grid
        v = gb["valid_w"][0] > 0
        e_q = float((Xq - Xb).abs()[v].max())
        uv_full, _ = XG.full_uv(gb)
        uv_tok = uv_full[0][gb["tok"][0]]                                          # ROI+ring tokens
        crop = g["crop"]
        u_ref = (gb["coords"][0, :, 0].double() * 336 - crop.ox) / crop.nw         # roi_geometry coords (crop frame)
        v_ref = (gb["coords"][0, :, 1].double() * 336 - crop.oy) / crop.nh
        e_m = float(torch.maximum((uv_tok[:, 0] - u_ref).abs(), (uv_tok[:, 1] - v_ref).abs()).max())
        ins = gb["inside"][0]
        # ROI-relative extent of selected ROI tokens (centres) and of the outside tokens
        bil = torch.nn.functional.grid_sample(fmap.transpose(1, 2).reshape(1, 2, 24, 24), gb["bil_grid"].view(1, 24, 24, 2),
                                              mode="bilinear", padding_mode="border", align_corners=False).flatten(2)[0].T
        bic = XG.bicubic_roi(fmap, gb)[0]
        hull = ((Xb >= 7) & (Xb <= 329)).all(-1) & v
        e_bil = float((bil.double() - Xb).abs()[hull].max()) if hull.any() else float("nan")
        e_bic = float((bic.double() - Xb).abs()[hull].max()) if hull.any() else float("nan")
        qu, _ = XG.query_uv(gb)
        out[name] = dict(W=W, H=H, box=box, n_in=g["n_in"], n_valid_query=int(v.sum()), n_hull=int(hull.sum()),
                         query_uv_vs_bil_grid_max_px=e_q, memory_uv_vs_roi_geometry_coords_max=e_m,
                         bilinear_linear_ramp_max_px=e_bil, bicubic_linear_ramp_max_px=e_bic,
                         valid_query_u_range=[float(qu[0, v, 0].min()), float(qu[0, v, 0].max())],
                         valid_query_v_range=[float(qu[0, v, 1].min()), float(qu[0, v, 1].max())],
                         full_token_u_range=[float(uv_full[0, :, 0].min()), float(uv_full[0, :, 0].max())],
                         roi_token_centre_u_range=[float(uv_tok[ins, 0].min()), float(uv_tok[ins, 0].max())])
        for kk, val in (("q_impl", e_q), ("mem_impl", e_m), ("bil_lin", e_bil), ("bic_lin", e_bic)):
            if np.isfinite(val):
                worst[kk] = max(worst[kk], val)
    return dict(cases=out, worst=worst,
                note=("query_uv_vs_bil_grid / memory_uv_vs_coords = implementation (grid / coordinate) error, must be ~1e-5; "
                      "bilinear ramp = 0 up to float32; bicubic ramp error = PyTorch cubic kernel (A=-0.75) not reproducing "
                      "linear functions (11-05 IVC: mean 0.55, max 0.67 px), a kernel property, same grid"))


# ------------------------------------------------------------------------------------------------ decoders
def batch(dev, n_img=8, row=0):
    rois = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    ds = MS.CocoStreamDS(rois, COCO_TRAIN)
    ids = [int(i) for i in np.load(STREAM)[row][:n_img]]
    return MS.to_dev(ds[ids], dev)


def flops_per_roi(dec, feats, g):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            dec(feats, g)
    return fc.get_total_flops() / len(g["valid_w"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--conds", default=",".join(CONDS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(32)
    res = dict(time=time.strftime("%F %T"), device="cuda" if a.gpu else "cpu")
    al = alignment()
    json.dump(al, open(os.path.join(a.out, "alignment.json"), "w"), indent=1)
    res["alignment_worst"] = al["worst"]
    print("alignment", json.dumps(al["worst"]), flush=True)
    dev = torch.device("cuda" if a.gpu else "cpu")
    z = np.load("output/2026-11-05/msd/teacher_stats.npz")
    mu, sd = torch.as_tensor(z["mean_12"], device=dev), torch.as_tensor(z["std_12"], device=dev)
    student = MS.Student(adapters=False).to(dev).eval()
    vit = student.vit
    b = batch(dev, 8, 0)
    b2 = batch(dev, 8, 1)
    with torch.no_grad():
        feats, _ = student(b["full"])
        T = torch.cat([RC.encode_layers(vit, b["crops"][i:i + 16], (12,))[12] for i in range(0, len(b["crops"]), 16)])
        feats2, _ = student(b2["full"])
    g, w = b["geom"], b["geom"]["valid_w"].float()
    res["encoder"] = dict(requires_grad_any=any(p.requires_grad for p in vit.parameters()), feats_shape=list(feats[12].shape),
                          teacher_shape=list(T.shape), feats_std_mean=float(feats[12].std(0).mean()), teacher_std_mean=float(T.std(0).mean()),
                          n_roi=int(len(w)), valid_cells_per_roi_mean=float((w > 0).float().sum(1).mean()))
    B = XG.bicubic_roi(feats[12], g)
    ac = (torch.autocast("cuda", dtype=torch.bfloat16) if a.gpu else torch.autocast("cpu", enabled=False))
    base_mse, base_cos = dense_mse(B, T, w), dense_cos(B, T, w)
    res["bicubic_baseline_loss"] = float((base_mse + 1 - base_cos).mean())
    res["conds"] = {}
    for c in a.conds.split(","):
        r = {}
        torch.manual_seed(0)
        dec = XM.build(c).to(dev)
        dec.set_stats(mu, sd)
        r["n_params"] = XM.n_params(dec)
        r["n_params_groups"] = {k: int(sum(p.numel() for p in v)) for k, v in dec.groups().items()}
        with torch.no_grad(), ac:
            Z, parts = dec(feats, g, return_parts=True)
        Z = Z.float()
        r["shape"] = list(Z.shape)
        r["finite"] = bool(torch.isfinite(Z).all())
        if c != "R00" and XM.COND_SPEC[c]["out"] == "skip":
            r["step0_head_max_abs"] = float(parts["head"].abs().max())
            r["step0_Z_equals_B"] = bool(torch.equal(Z, B))
            r["step0_Z_minus_B_max"] = float((Z - B).abs().max())
        # teacher never an input: forward has no teacher argument; recompute with different crops in the batch dict
        g_alt = dict(g)
        with torch.no_grad(), ac:
            Z_t = dec(feats, g_alt).float()
        r["output_independent_of_teacher"] = bool(torch.equal(Z, Z_t))
        # memory swap: full tokens of another image (same geometry)
        fs = {12: feats2[12][: feats[12].shape[0]]}
        with torch.no_grad(), ac:
            Z_s = dec(fs, g).float()
        vm = w > 0
        r["memory_swap_rel_change_valid"] = float((Z_s - Z)[vm].norm() / Z[vm].norm())
        # mask checks
        if c == "E2":
            noise = torch.randn_like(feats[12]) * 5
            keep = torch.zeros(feats[12].shape[:2], dtype=torch.bool, device=dev)
            tok = g["tok"][g["inside"] & ~g["pad"]]
            img = g["img_index"][:, None].expand_as(g["tok"])[g["inside"] & ~g["pad"]]
            keep[img, tok] = True
            # only ROIs whose ROI tokens are NOT shared with another ROI's outside region matter: perturb tokens no ROI uses
            fpert = {12: torch.where(keep[..., None], feats[12], feats[12] + noise)}
            with torch.no_grad(), ac:
                Z_p = dec(fpert, g).float()
            r["E2_invariant_to_non_roi_tokens_max_abs"] = float((Z_p - Z).abs().max())
        if c in ("E1", "E7"):
            inv = ~(w > 0)
            orig_q = dec.queries

            def noisy_q(Bq, gq):
                x = orig_q(Bq, gq)
                return torch.where(inv[..., None], x + 10 * torch.randn_like(x), x)
            dec.queries = noisy_q
            with torch.no_grad(), ac:
                Z_p = dec(feats, g).float()
            del dec.queries
            r["valid_outputs_invariant_to_invalid_query_perturbation_max_abs"] = float((Z_p - Z)[vm].abs().max())
            r["n_invalid_queries"] = int(inv.sum())
        # gradients after 1 and 2 optimizer updates + tiny-subset loss decrease
        opt = torch.optim.AdamW(dec.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
        gr = dec.groups()
        losses = []
        for s in range(a.steps):
            with ac:
                Z = dec(feats, g).float()
            mse, cos = dense_mse(Z, T, w), dense_cos(Z, T, w)
            loss = (mse + 1 - cos).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if s < 2:
                r[f"update{s + 1}_grad_norm"] = {k: float(torch.stack([p.grad.norm() for p in v if p.grad is not None]).norm()) if any(p.grad is not None for p in v) else 0.0
                                                for k, v in gr.items()}
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
            if not np.isfinite(losses[-1]):
                break
        r["loss_first_last"] = [losses[0], losses[-1]]
        r["loss_decreased"] = bool(losses[-1] < losses[0])
        r["encoder_grad_present"] = any(p.grad is not None for p in vit.parameters())
        dec.eval()
        r["flops_per_roi_G"] = flops_per_roi(dec, feats, g) / 1e9
        if a.gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            m0 = torch.cuda.memory_allocated()
            with torch.no_grad(), ac:
                for _ in range(3):
                    dec(feats, g)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(20):
                    dec(feats, g)
                torch.cuda.synchronize()
            r["latency_ms_per_batch"] = (time.time() - t0) / 20 * 1000
            r["latency_batch_rois"] = int(len(w))
            r["inference_peak_extra_mem_MB"] = (torch.cuda.max_memory_allocated() - m0) / 2 ** 20
        res["conds"][c] = r
        print(c, json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(a.out, "smoke.json"), "w"), indent=1)
    print("SMOKE DONE", flush=True)


if __name__ == "__main__":
    main()
