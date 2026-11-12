"""CPU checks for the decoder study (no encoder needed) -> <out>/checks_cpu.json. Asserts:
 1. grid alignment (real COCO stream ROIs): for every crop-grid content cell, the C0 crop-canvas cell centre mapped back to
    ORIGINAL image pixels and then onto the FULL canvas == roi_points (the bilinear sampling point), max error in px;
    + pixel check: RGB of the C0 crop canvas at the cell centres vs RGB of the full canvas sampled (bilinear) at the
    corresponding points -> correlation (alignment of real pixels, both letterboxes kept)
 2. synthetic token map whose channels encode the cell-centre (x, y): bilinear_roi returns exactly the sampling point
    (= the token grid convention (k + .5) P with align_corners=False)
 3. ResidualDecoder: shared parameters initialised == G1-cross ROIDecoderD (seed 7); Delta == 0 and Z == B exactly at init
    (fp32 and bf16 autocast); one SGD step on L_detail -> gradients on `out` at step 0, on the body after the step
 4. padding exclusion: replacing Z / T at cells with crop coverage 0 by noise changes neither L_detail, L_cov nor L_rel
 5. L_cov == naive per-ROI loop (torch.cov on normalised valid tokens); L_cov(T, T) == 0; L_rel(T, T) == 0, KL == naive
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import COCO_TRAIN, FEAT, STREAM
from src.decstudy import losses as LS
from src.decstudy.model import ResidualDecoder, bilinear_roi, build_residual, roi_points
from src.g1clip.model import ROIDecoderD
from src.g1x.data import CocoDS, G, roi_geom
from src.maskemb import common as MC

S, P, D = MC.S, MC.P, MC.DIM


def main(out):
    torch.set_num_threads(32)
    MC.set_clip_norm()
    os.makedirs(out, exist_ok=True)
    res = {}
    rois = pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet"))
    ids = [int(i) for i in np.load(STREAM)[:4].ravel()][:24]
    ds = CocoDS(rois, COCO_TRAIN, S, P)
    b = ds[ids]
    # 1. analytic mapping crop cell -> original px -> full canvas
    errs, corr = [], []
    by = ds.by
    k = 0
    for ii, iid in enumerate(ids):
        g = by[iid]
        W, H = int(g.iloc[0].W), int(g.iloc[0].H)
        for r in g.itertuples():
            bx = (int(r.l), int(r.t), int(r.r), int(r.b))
            gg = roi_geom(W, H, bx, S, P)
            cl, fl = gg["crop_lb"], gg["full_lb"]
            ci, cj = np.divmod(np.arange(G * G), G)
            cx, cy = (cj + 0.5) * P, (ci + 0.5) * P
            inside = (cx >= cl.ox) & (cx <= cl.ox + cl.nw) & (cy >= cl.oy) & (cy <= cl.oy + cl.nh)
            ox_ = bx[0] + (cx - cl.ox) / cl.nw * (bx[2] - bx[0])            # original px
            oy_ = bx[1] + (cy - cl.oy) / cl.nh * (bx[3] - bx[1])
            fx, fy = fl.ox + ox_ * fl.nw / W, fl.oy + oy_ * fl.nh / H       # full canvas px
            pts = roi_points(torch.from_numpy(gg["box_c"]).float()[None], torch.from_numpy(gg["qrel"])[None])[0].numpy()
            e = np.abs(np.stack([fx, fy], 1) - pts)[inside]
            errs.append(float(e.max()))
            # pixel check
            crop = b["crops"][k]                                             # (3, S, S) normalised
            full = b["full"][ii]
            cpx = crop[:, cy.astype(int)[inside], cx.astype(int)[inside]].T  # nearest pixel at the cell centre
            grid = torch.from_numpy(pts[inside] / S * 2 - 1).float().view(1, 1, -1, 2)
            fpx = F.grid_sample(full[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T
            corr.append(float(np.corrcoef(cpx.numpy().ravel(), fpx.numpy().ravel())[0, 1]))
            k += 1
    res["alignment"] = dict(n_rois=k, max_px_error=max(errs), pixel_corr_median=float(np.median(corr)), pixel_corr_min=float(np.min(corr)),
                            note="crop canvas cell centre (nearest px) vs full canvas bilinear at the mapped point; resolutions differ, so corr < 1")
    assert max(errs) < 1e-3 and np.median(corr) > 0.9, res["alignment"]
    # 2. synthetic coordinate map
    ci, cj = torch.div(torch.arange(G * G), G, rounding_mode="floor"), torch.arange(G * G) % G
    tok = torch.zeros(len(ids), G * G, D)
    tok[..., 0], tok[..., 1] = ((cj + 0.5) * P).float(), ((ci + 0.5) * P).float()
    Bs = bilinear_roi(tok, b["roi_img"], b["box_c"], b["qrel"], S)
    pts = roi_points(b["box_c"], b["qrel"])
    inner = (pts >= P / 2).all(-1) & (pts <= S - P / 2).all(-1)                # border padding clamps outside the centres
    e2 = float((Bs[..., :2] - pts).abs()[inner].max())
    res["synthetic_coordinate_map_max_err_px"] = e2
    assert e2 < 1e-3
    # 3. residual decoder init / zero-init / gradient flow
    torch.manual_seed(7)
    old = ROIDecoderD(S, P, D)
    new = build_residual(S, P, D)
    so, sn = old.state_dict(), new.state_dict()
    shared_equal = all(torch.equal(so[k_], sn[k_]) for k_ in so if not k_.startswith("out."))
    res["residual_init"] = dict(shared_params_equal_g1cross_init=shared_equal, new_params=sorted(set(sn) - set(so)),
                                out_zero=bool((sn["out.weight"] == 0).all() and (sn["out.bias"] == 0).all()),
                                n_params=sum(p.numel() for p in new.parameters()), n_params_g1cross_decoder=sum(p.numel() for p in old.parameters()))
    assert shared_equal and res["residual_init"]["out_zero"]
    torch.manual_seed(0)
    Fm = torch.randn(len(ids), G * G, D) * 2
    T = torch.randn(len(b["roi_img"]), G * G, D)
    for amp in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            Bf, dl = new.parts(Fm, b["roi_img"], b["box_c"], b["qrel"], b["fvalid"])
        res[f"init_delta_max_abs_amp{int(amp)}"] = float(dl.abs().max())
        res[f"init_z_equals_b_amp{int(amp)}"] = bool(torch.equal(Bf + dl, Bf))
        assert float(dl.abs().max()) == 0.0
    cc = b["ccov"]
    opt = torch.optim.SGD(new.parameters(), lr=1e-2)
    gstat = []
    for it in range(2):
        Bf, dl = new.parts(Fm, b["roi_img"], b["box_c"], b["qrel"], b["fvalid"])
        tk, dp = LS.detail_terms(Bf + dl, T, cc)
        loss = tk.mean() + dp.mean()
        opt.zero_grad()
        loss.backward()
        gstat.append(dict(step=it, out_grad=float(new.out.weight.grad.norm()), body_grad=float(sum(p.grad.norm() ** 2 for n_, p in new.named_parameters()
                                                                                                    if not n_.startswith("out.") and p.grad is not None) ** 0.5)))
        opt.step()
    res["grad_flow"] = gstat
    assert gstat[0]["out_grad"] > 0 and gstat[1]["body_grad"] > 0
    # 4. padding exclusion
    Z = Bf.detach() + 0.1 * torch.randn_like(Bf)
    pad = (cc == 0)[..., None]
    res["padding_cells_fraction"] = float((cc == 0).float().mean())
    Zn, Tn = torch.where(pad, torch.randn_like(Z) * 5, Z), torch.where(pad, torch.randn_like(T) * 5, T)

    def all_losses(z, t):
        tk, dp = LS.detail_terms(z, t, cc)
        e, ok, _ = LS.cov_terms(z, t, cc)
        return torch.stack([tk.mean() + dp.mean(), e[ok].mean(), LS.rel_kl(LS.pool(z, cc), LS.pool(t, cc)).mean()])

    l0, l1 = all_losses(Z, T), all_losses(Zn, Tn)
    res["padding_invariance_max_abs"] = float((l0 - l1).abs().max())
    assert res["padding_invariance_max_abs"] < 1e-4, (l0, l1)
    # 5. reference implementations
    e, ok, _ = LS.cov_terms(Z, T, cc)
    naive = []
    for r in range(len(Z)):
        m = cc[r] > 0
        cz = torch.cov(F.normalize(Z[r][m], dim=-1).T.double())
        ct = torch.cov(F.normalize(T[r][m], dim=-1).T.double())
        naive.append(float((cz - ct).pow(2).sum() / ct.pow(2).sum()))
    res["cov_vs_naive_max_rel_err"] = float(np.max(np.abs(e.numpy() - np.array(naive)) / np.array(naive)))
    assert res["cov_vs_naive_max_rel_err"] < 1e-3
    res["cov_self_zero"] = float(LS.cov_terms(T, T, cc)[0].abs().max())
    pz, pt = LS.pool(Z, cc), LS.pool(T, cc)
    res["rel_self_zero"] = float(LS.rel_kl(pt, pt).abs().max())
    kl = LS.rel_kl(pz, pt)
    sz, st = LS.rel_matrix(pz).double(), LS.rel_matrix(pt).double()
    n = len(pz)
    ref = []
    for i in range(n):
        j = [x for x in range(n) if x != i]
        a_ = torch.softmax(st[i, j] / LS.TAU, 0)
        b_ = torch.softmax(sz[i, j] / LS.TAU, 0)
        ref.append(float((a_ * (a_.log() - b_.log())).sum()))
    res["rel_vs_naive_max_abs"] = float(np.max(np.abs(kl.numpy() - np.array(ref))))
    assert res["cov_self_zero"] < 1e-6 and res["rel_self_zero"] < 1e-6 and res["rel_vs_naive_max_abs"] < 1e-4
    res["shapes"] = dict(full_tokens=list(Fm.shape), B=list(Bf.shape), Delta=list(dl.shape), teacher=list(T.shape), n_rois=len(b["roi_img"]))
    json.dump(res, open(os.path.join(out, "checks_cpu.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("CPU CHECKS OK")


if __name__ == "__main__":
    main(sys.argv[1])
