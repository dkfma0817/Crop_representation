"""RSP unit tests (run before training; results -> <out>/unit_tests.json; exit code 1 on any failure).

  python -m src.rsp.checks --shared <shared> --out <dir>
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from src.config import STREAM
from src.msd.pretrain import eval_exclusions
from src.rsp.common import (GF, K_TRAIN, PREFIX, PT, coarse_centres, dino_rope, fine_rect_index, fine_valid, pi_resize, read_frects,
                            sample_local, upsample_matrix)
from src.rsp.data import RSPStreamDS, fine_valid_np, frects_of, rect_box, sample_queries, evidence
from src.rsp.model import LV_CTX, LV_L1, LV_Q, RSPPredictor, level_mask
from src.rsp.teacher import DenseTeacher
from src.rsp.train import query_centres, roi_l1, teacher_targets
from src.tsplit.features import frozen_vit
from src.tsplit.train import STEPS, StreamDS
from src.viz.c0_pca import LB

R = {}


def test(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                info = fn(*a, **k) or {}
                R[name] = dict(ok=True, **info, seconds=round(time.time() - t0, 2))
            except Exception:
                R[name] = dict(ok=False, error=traceback.format_exc())
            print(name, "OK" if R[name]["ok"] else "FAIL", json.dumps({k: v for k, v in R[name].items() if k != "error"})[:400], flush=True)
            if not R[name]["ok"]:
                print(R[name]["error"], flush=True)
        return run
    return deco


def mx(a, b):
    return float((a.float() - b.float()).abs().max())


def batch(ds, s, dev):
    b = ds[s]
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@test("dinov3_structure_and_rope_coords")
def t_structure(vit, dev):
    assert vit.num_prefix_tokens == PREFIX and vit.cls_token is not None and vit.reg_token.shape[1] == 4 and vit.pos_embed is None
    e24 = vit.rope.get_embed(shape=(24, 24)).to(dev)
    e96 = vit.rope.get_embed(shape=(GF, GF)).to(dev)
    i, j = torch.meshgrid(torch.arange(GF, device=dev), torch.arange(GF, device=dev), indexing="ij")
    fc = torch.stack([(i.flatten() + .5) * PT, (j.flatten() + .5) * PT], -1).float()
    d24, d96 = mx(dino_rope(vit, coarse_centres(dev)), e24), mx(dino_rope(vit, fc), e96)
    assert d24 < 1e-5 and d96 < 1e-5, (d24, d96)
    # parent (i, j) centre == mean of its 16 fine centres (same full-image frame)
    fcv = fc.view(24, 4, 24, 4, 2).mean((1, 3)).reshape(-1, 2)
    dpar = mx(fcv, coarse_centres(dev))
    assert dpar == 0.0
    return dict(prefix_tokens=PREFIX, prefix="CLS + 4 registers (no RoPE, as in timm DINOv3)", rope_coarse_vs_timm_24=d24, rope_fine_vs_timm_96x96=d96,
                fine_centre_mean_minus_parent_centre=dpar, rope_period_buffer=vit.rope.periods.tolist())


@test("coarse_manual_forward_equals_timm")
def t_coarse(vit, teach, x):
    with torch.no_grad():
        ref = vit.forward_features(x)
        out, kv = teach.coarse(x)
        d32 = mx(out, ref)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            refb = vit.forward_features(x)
            outb, _ = teach.coarse(x)
    db = mx(outb, refb)
    assert d32 < 1e-3 and len(kv) == 12 and kv[0][0].shape[2] == PREFIX + 576, (d32, kv[0][0].shape)
    return dict(fp32_max_abs=d32, bf16_max_abs=db, ref_abs_max=float(ref.abs().max()))


@test("pi_resize_flexivit")
def t_pi(vit, teach, x):
    """Teacher kernel w4 = U^T w (U = bilinear 4 -> 16 upsampling): <U y, w> == <y, w4> must hold for EVERY 4 x 4 patch y (exact).
    Literal FlexiViT down-LS form pinv(B^T) w: exact on B's row space only. Real-canvas fidelity of every variant vs the zoomed token."""
    w = vit.patch_embed.proj.weight.detach().double().cpu()
    w4 = teach.w4.double().cpu()
    U = upsample_matrix(4, 16, "bilinear")
    g = torch.Generator().manual_seed(0)
    y = torch.randn(64, 3, 16, generator=g, dtype=torch.float64)
    lhs = torch.einsum("ncq,ocq->no", torch.einsum("pq,ncq->ncp", U, y), w.view(768, 3, 256))
    rhs = torch.einsum("ncq,ocq->no", y, w4.view(768, 3, 16))
    exact_up = float((lhs - rhs).norm() / lhs.norm())
    wd, B = pi_resize(w, 4, "bilinear")
    xs = torch.einsum("pq,ncp->ncq", B, y)                                               # x = B^T y (row space of B)
    l2 = torch.einsum("ncq,ocq->no", xs, w.view(768, 3, 256))
    r2 = torch.einsum("ncp,ocp->no", torch.einsum("pq,ncq->ncp", B, xs), wd.double().view(768, 3, 16))
    exact_down_rowspace = float((l2 - r2).norm() / l2.norm())
    assert exact_up < 1e-6 and exact_down_rowspace < 1e-6, (exact_up, exact_down_rowspace)       # w4 is a float32 buffer
    dev, bias = x.device, vit.patch_embed.proj.bias.detach().double()
    xd = x.double()
    refs = {m: F.conv2d(F.interpolate(xd, scale_factor=4, mode=m, align_corners=False) if m != "nearest" else F.interpolate(xd, scale_factor=4, mode="nearest"),
                        w.to(dev), bias, stride=16).flatten(2) for m in ("bilinear", "bicubic", "nearest")}
    cands = {"used_up_bilinear_UTw": w4, "flexivit_down_bilinear_pinv": wd.double(), "area_blocksum": pi_resize(w, 4, "area")[0].double()}
    table = {}
    for n, k in cands.items():
        e4 = F.conv2d(xd, k.to(dev), bias, stride=4).flatten(2)
        table[n] = {f"vs_{m}_zoom": dict(cos=float(F.cosine_similarity(e4, r, dim=1).mean()), rel_err=float((e4 - r).norm() / r.norm())) for m, r in refs.items()}
    assert table["used_up_bilinear_UTw"]["vs_bilinear_zoom"]["cos"] > 0.98
    return dict(used="w4 = U^T w, U = bilinear 4->16 upsampling (align_corners False); bias copied; frozen buffer",
                exact_identity_rel_err_used=exact_up, flexivit_down_rowspace_rel_err=exact_down_rowspace, real_canvas_patch_embed_fidelity=table,
                note="fidelity measured on whole-canvas 4x zooms (neighbour pixels enter the zoom at patch borders, so the used form is exact per patch, not per canvas)")


@test("teacher_single_vs_multi_vs_efficient")
def t_teacher(teach, x, content):
    """fp32: literal single-ROI (timm blocks + explicit mask) == literal multi-ROI == efficient batched multi-ROI. Coarse rows of the
    literal sequence == plain forward_features (coarse context unaffected by fine tokens)."""
    dev = x.device
    frs = torch.tensor([[8, 12, 4, 4], [10, 14, 12, 8], [0, 0, 24, 24], [40, 40, 4, 16]], device=dev)          # overlapping, 1x1 parent, 6x6
    c1 = content[0]
    with torch.no_grad():
        ref_full = teach.vit.forward_features(x[:1])
        singles = [teach.ref_forward(x[:1], frs[m:m + 1], c1)[1][0] for m in range(len(frs))]
        cm, multi, mask, owner = teach.ref_forward(x[:1], frs, c1)
        final, kv = teach.coarse(x[:1])
        T, valid, idx = teach.fine(teach.fine_embed(x[:1]), kv, frs, torch.zeros(len(frs), dtype=torch.long, device=dev), c1[None].expand(len(frs), 4))
    d_coarse = mx(cm, ref_full[0])
    d_sm = max(mx(s, m) for s, m in zip(singles, multi))
    d_se = max(mx(s, T[k, :len(s)]) for k, s in enumerate(singles))
    scale = float(torch.cat(singles).abs().max())
    assert d_coarse < 1e-3 and d_sm < 2e-3 and d_se < 2e-3, (d_coarse, d_sm, d_se)
    # perturb ROI 1's fine embedding: coarse rows and ROI 0 outputs unchanged
    Nc = PREFIX + 576
    rows1 = (owner == 1).nonzero().squeeze(1)
    cross = bool(mask[owner == 0][:, rows1].any()) or bool(mask[:Nc][:, Nc:].any())
    assert not cross
    # bf16 efficient vs fp32 reference (training precision)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fb, kvb = teach.coarse(x[:1])
        Tb, _, _ = teach.fine(teach.fine_embed(x[:1]), kvb, frs, torch.zeros(len(frs), dtype=torch.long, device=dev), c1[None].expand(len(frs), 4))
    cos_b = float(torch.stack([F.cosine_similarity(Tb[k, :len(s)].float(), s, dim=-1).mean() for k, s in enumerate(singles)]).mean())
    # fine tokens differ from the coarse tokens they replace (they are a different, denser tokenisation)
    par = (idx[0] // GF // 4) * 24 + (idx[0] % GF) // 4
    cos_parent = float(F.cosine_similarity(T[0, :16], final[0, PREFIX:][par[:16]], dim=-1).mean())
    return dict(rois=frs.tolist(), coarse_rows_vs_forward_features=d_coarse, single_vs_multi_literal=d_sm, single_literal_vs_efficient=d_se, token_abs_max=scale,
                cross_roi_or_coarse_to_fine_mask_entries=int(cross), bf16_efficient_vs_fp32_single_cos=cos_b, fine_vs_own_parent_coarse_cos=cos_parent,
                mask_rule="coarse/prefix rows: prefix + coarse; ROI-m fine rows: prefix + coarse outside ROI-m parents + valid ROI-m fine")


@test("predictor_mask_matches_design_doc")
def t_mask():
    lv = torch.tensor([LV_CTX] * 3 + [LV_L1] * 2 + [LV_Q] * 3 + [LV_L1] * 2 + [LV_Q] * 2)
    crop = torch.tensor([-1] * 3 + [0] * 5 + [1] * 4)
    valid = torch.ones(len(lv), dtype=torch.bool)
    valid[-1] = False                                                                       # one padding query of ROI 1
    m = level_mask(lv, crop, valid)
    # design-doc table (rows: context / evidence,z / query; cols: context / evidence / z / query) + crop separation + padding
    exp = torch.zeros_like(m)
    for i in range(len(lv)):
        for j in range(len(lv)):
            if not valid[j]:
                continue
            if lv[i] == LV_CTX:
                exp[i, j] = lv[j] == LV_CTX
            elif lv[i] == LV_L1:
                exp[i, j] = lv[j] == LV_CTX or (lv[j] == LV_L1 and crop[j] == crop[i])
            else:
                exp[i, j] = lv[j] == LV_CTX or (lv[j] == LV_L1 and crop[j] == crop[i]) or i == j
    assert torch.equal(m, exp)
    qq = m[lv == LV_Q][:, lv == LV_Q]
    assert torch.equal(qq, torch.eye(len(qq), dtype=torch.bool) & valid[lv == LV_Q][None])
    return dict(matrix=m.int().tolist(), query_query_offdiag=int((qq & ~torch.eye(len(qq), dtype=torch.bool)).sum()))


@test("predictor_efficient_equals_masked_reference")
def t_pred_ref(vit, x, dev):
    out = {}
    with torch.no_grad():
        feats = vit.forward_features(x[:1])
    g = torch.Generator(device="cpu").manual_seed(1)
    box = torch.tensor([[40., 60., 104., 120.], [100., 100., 116., 116.], [10., 200., 300., 380.]], device=dev)
    ev = torch.randn(3, 3, 64, 64, generator=g).to(dev)
    K = 20
    cen = torch.stack([box[:, 1:2] + torch.rand(3, K, generator=g).to(dev) * (box[:, 3:4] - box[:, 1:2]),
                       box[:, 0:1] + torch.rand(3, K, generator=g).to(dev) * (box[:, 2:3] - box[:, 0:1])], -1)
    qvalid = torch.ones(3, K, dtype=torch.bool, device=dev)
    qvalid[1, 15:] = False
    for pix in (True, False):
        torch.manual_seed(0)
        m = RSPPredictor(pix=pix).to(dev).eval()
        with torch.no_grad():
            for p in m.parameters():                                                       # non-trivial weights
                p.add_(torch.randn(p.shape, generator=g).to(dev) * .02)
            e = ev if pix else None
            eff = m(feats, torch.zeros(3, dtype=torch.long, device=dev), box, e, cen)
            ref, mask, lv, crop = m.ref_forward(feats, box, e, cen, qvalid)
            d = mx(eff[qvalid], ref[qvalid])
            # query independence: drop half of the queries -> remaining outputs unchanged
            eff_half = m(feats, torch.zeros(3, dtype=torch.long, device=dev), box, e, cen[:, ::2])
            d_ind = mx(eff_half, eff[:, ::2])
            # cross-ROI: change ROI 2's box / evidence -> ROI 0 outputs unchanged
            box2, ev2 = box.clone(), (ev.clone() if pix else None)
            box2[2] = torch.tensor([0., 0., 50., 50.], device=dev)
            if pix:
                ev2[2] = 0
            eff2 = m(feats, torch.zeros(3, dtype=torch.long, device=dev), box2, ev2, cen)
            d_cross = mx(eff2[0], eff[0])
        assert d < 1e-3 and d_ind < 1e-4 and d_cross < 1e-4, (pix, d, d_ind, d_cross)
        out["pix" if pix else "nopix"] = dict(efficient_vs_reference_max_abs=d, out_abs_max=float(ref.abs().max()), query_subset_invariance=d_ind, cross_roi_invariance=d_cross,
                                             seq_len=int(len(lv)), n_l1_tokens=int((lv == LV_L1).sum() // 3))
    return out


@test("training_step_grads_frozen_and_nopix_path")
def t_step(vit, teach, ds_pix, ds_nopix, shared, dev):
    st = torch.load(Path(shared) / "norm_stats.pt", map_location=dev)
    mu, sd = st["mu"].to(dev), st["sd"].to(dev)
    out = {}
    vit_sum = float(sum(p.double().sum() for p in vit.parameters()))
    w4_sum = float(teach.w4.double().sum())
    for pix, ds in ((True, ds_pix), (False, ds_nopix)):
        torch.manual_seed(0)
        m = RSPPredictor(pix=pix).to(dev)
        m.load_state_dict(torch.load(Path(shared) / f"init_{'pix' if pix else 'nopix'}.pt", map_location=dev))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        b = batch(ds, 0, dev)
        assert ("ev" in b) == pix
        fr = frects_of(b["rects"])
        final, tgt, bad = teacher_targets(teach, b["x"], fr, b["rimg"], b["content"], b["q"], b["q_ok"])
        assert bad == 0 and not tgt.requires_grad and not final.requires_grad
        cen, flat = query_centres(fr, b["q"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pn = m(final, b["rimg"], b["box"], b.get("ev"), cen)
        loss = roi_l1(pn, (tgt - mu) / sd, b["q_ok"])
        loss.backward()
        gs = {n: float(torch.stack([p.grad.float().norm() for p in ps]).norm()) for n, ps in m.groups().items()}
        none = [n for n, p in m.named_parameters() if p.grad is None]
        # blocks.11 attention q/k of context/L1 matter only through the queries: still every parameter must receive a gradient
        assert not none and all(v > 0 and np.isfinite(v) for v in gs.values()), (none, gs)
        assert all(p.grad is None for p in vit.parameters())
        opt.step()
        assert float(sum(p.double().sum() for p in vit.parameters())) == vit_sum and float(teach.w4.double().sum()) == w4_sum
        out["pix" if pix else "nopix"] = dict(loss=float(loss), grad_norms=gs, params_without_grad=none, n_roi=len(b["rects"]), n_query=int(b["q_ok"].sum()),
                                             has_ev_embed=m.ev_embed is not None, trainable_params=sum(p.numel() for p in m.parameters()))
    assert out["nopix"]["has_ev_embed"] is False and "evidence" not in out["nopix"]["grad_norms"] and out["pix"]["grad_norms"]["evidence"] > 0
    out["frozen_student_teacher_unchanged"] = True
    out["teacher_tensors_require_grad"] = False
    out["teacher_fine_in_predictor_inputs"] = "no: RSPPredictor.forward(feats=student 384 final tokens, rimg, box, ev, query centres); fine tokens only enter the loss"
    return out


@test("query_sampling_matches_teacher_gather")
def t_query(teach, ds_pix, dev):
    b = batch(ds_pix, 3, dev)
    fr = frects_of(b["rects"])
    cen, flat = query_centres(fr, b["q"])
    idx, uy, ux, ok = fine_rect_index(fr)
    gi = torch.gather(idx, 1, b["q"])
    assert torch.equal(gi[b["q_ok"]], flat[b["q_ok"]])
    # centre -> fine cell -> parent rect & box containment
    u = torch.div(cen, PT, rounding_mode="floor").long()
    assert torch.equal((u[..., 0] * GF + u[..., 1])[b["q_ok"]], flat[b["q_ok"]])
    bx = b["box"][:, None]
    inside = (cen[..., 1] > bx[..., 0] - PT) & (cen[..., 1] < bx[..., 2] + PT) & (cen[..., 0] > bx[..., 1] - PT) & (cen[..., 0] < bx[..., 3] + PT)
    assert bool(inside[b["q_ok"]].all())
    # numpy validity == torch validity; teacher gather equals a per-query lookup in the full fine output
    v_t = ok & fine_valid(b["content"][b["rimg"]], uy, ux)
    for r in range(len(fr)):
        n = int(fr[r, 2] * fr[r, 3])
        lbv = fine_valid_np(b["rects"][r].tolist(), _lb_from_content(b["content"][b["rimg"][r]]))
        assert np.array_equal(lbv, v_t[r, :n].cpu().numpy())
    with torch.no_grad():
        final, tgt, bad = teacher_targets(teach, b["x"], fr, b["rimg"], b["content"], b["q"], b["q_ok"])
        with torch.autocast("cuda", dtype=torch.bfloat16):                                     # same precision as training
            res, _ = teach.targets(b["x"], fr, b["rimg"], b["content"], budget=10 ** 9)     # different chunking / padding
    d = 0.0
    for r in range(len(fr)):
        T, v, ix = res[r]
        k = b["q_ok"][r]
        pos = {int(f): i for i, f in enumerate(ix.tolist())}
        sel = torch.tensor([pos[int(f)] for f in flat[r][k].tolist()], device=dev)
        d = max(d, mx(T[sel], tgt[r][k]))
    n_small = int(((b["rects"][:, 2] == 1) & (b["rects"][:, 3] == 1)).sum())
    assert d < 0.1, d
    return dict(bad=bad, gather_max_abs_vs_lookup_bf16_other_chunking=d, k_per_roi=b["q_ok"].sum(1).tolist(), n_1x1_rois=n_small, K=K_TRAIN)


def _lb_from_content(c):
    lb = LB(1, 1, 384, 16)
    lb.ox, lb.oy, lb.nw, lb.nh = int(c[0]), int(c[1]), int(c[2] - c[0]), int(c[3] - c[1])
    return lb


@test("small_roi_and_invalid_boxes")
def t_small(vit, teach, dev, x, content):
    torch.manual_seed(0)
    m = RSPPredictor(pix=True).to(dev).eval()
    c = _lb_from_content(content[0])
    # 1x1 parent ROI at the letterbox edge
    r = [c.oy // 16, c.ox // 16, 1, 1]
    v = fine_valid_np(r, c)
    q, qok = sample_queries(np.random.default_rng(0), v)
    ok = qok
    fr = torch.tensor([r], device=dev) * 4
    with torch.no_grad():
        final, tgt, bad = teacher_targets(teach, x[:1], fr, torch.zeros(1, dtype=torch.long, device=dev), content[:1], torch.from_numpy(q)[None].to(dev), torch.from_numpy(ok)[None].to(dev))
        cen, _ = query_centres(fr, torch.from_numpy(q)[None].to(dev))
        pn = m(final, torch.zeros(1, dtype=torch.long, device=dev), torch.tensor([rect_box(r, c)], device=dev, dtype=torch.float32), torch.zeros(1, 3, 64, 64, device=dev), cen)
    assert bad == 0 and torch.isfinite(pn).all() and int(ok.sum()) == int(v.sum()) <= 16
    # tiny / degenerate GT boxes on the R2 read path
    from src.viz.c0_pca import geometry
    W, H = 640, 480
    boxes = [(100, 100, 103, 103), (100, 100, 100, 140), (0, 0, 640, 480), (630, 470, 640, 480)]
    grids, covs = [], []
    from src.g1native.bbdata import crop_geo
    for bx in boxes:
        crop, full, grid = geometry(W, H, bx, 384, 16) if bx[2] > bx[0] and bx[3] > bx[1] else geometry(W, H, (bx[0], bx[1], bx[0] + 1, bx[3]), 384, 16)
        geo, _, cov = crop_geo(crop, 384, 16)
        grids.append(torch.from_numpy(grid))
        covs.append(torch.from_numpy(cov))
    grid, cov = torch.stack(grids).to(dev), torch.stack(covs).to(dev)
    frs = read_frects(grid, cov)
    nocov = read_frects(grid[:1], torch.zeros_like(cov[:1]))
    # evidence for a zero-width box does not crash
    from PIL import Image
    img = Image.new("RGB", (W, H), (120, 30, 200))
    e = evidence(img, LB(W, H, 384, 16), [60.0, 60.0, 60.0, 90.0])
    assert torch.isfinite(e).all() and e.shape == (3, 64, 64)
    # local read == full-map read (border clamp included) on random maps
    mapf = torch.randn(len(boxes), 8, GF, GF, device=dev)
    full_read = F.grid_sample(mapf, grid.view(-1, 24, 24, 2), mode="bilinear", padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)
    idx, uy, ux, ok = fine_rect_index(frs)
    hm, wm = int(frs[:, 2].max()), int(frs[:, 3].max())
    loc = torch.zeros(len(boxes), 8, hm, wm, device=dev)
    for k in range(len(boxes)):
        h, w = int(frs[k, 2]), int(frs[k, 3])
        loc[k, :, :h, :w] = mapf[k, :, frs[k, 0]:frs[k, 0] + h, frs[k, 1]:frs[k, 1] + w]
    lr = sample_local(loc, frs, grid)
    vv = cov > 0
    d_local = mx(lr[vv], full_read[vv])
    assert d_local < 1e-4, d_local
    return dict(roi_1x1_rect=r, roi_1x1_valid_fine=int(v.sum()), roi_1x1_K=int(qok.sum()), tiny_box_fine_rects=frs.tolist(), no_coverage_fallback=nocov.tolist(), local_vs_full_read_max_abs=d_local)


@test("data_identical_across_runs_and_to_S3")
def t_data():
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:STEPS]
    a, b, s3 = RSPStreamDS(stream, ex, want_ev=True), RSPStreamDS(stream, ex, want_ev=False), StreamDS(stream, ex)
    for s in (0, 777, 11999):
        x, y, z = a[s], b[s], s3[s]
        for k in ("x", "content", "rects", "rimg", "q", "q_ok", "box", "ids"):
            assert torch.equal(x[k], y[k]), (s, k)
        for k in ("x", "content", "rects", "rimg", "ids"):
            assert torch.equal(x[k], z[k]), (s, k)
        assert "ev" in x and "ev" not in y
    return dict(steps_checked=[0, 777, 11999], same_as_S3=["x", "content", "rects", "rimg", "ids"])


@test("paco_dataset_flip_replay_and_evidence")
def t_paco():
    import src.readout.common as RC
    from src.rsp.data import RSPContentDS
    from src.tsplit.features import ContentDS
    data = RC.load_data("output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json", 0)
    dtr, Ytr = data["train"]
    torch.manual_seed(5)
    a = RSPContentDS(dtr, Ytr, True, "bil", 384, 16)[[0, 1, 2, 3, 4, 5]]
    torch.manual_seed(5)
    c = ContentDS(dtr, Ytr, True, "bil", 384, 16)[[0, 1, 2, 3, 4, 5]]
    for k in ("x", "bil_grid", "cov", "rows", "content"):
        assert torch.equal(a[k], c[k]), k
    assert a["ev"].shape[1:] == (3, 64, 64) and len(a["boxc"]) == len(a["rows"])
    w = a["boxc"][:, 2] - a["boxc"][:, 0]
    return dict(n_roi=len(a["rows"]), box_w_canvas_px_min=float(w.min()), box_w_canvas_px_max=float(w.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vit = frozen_vit(dev)
    teach = DenseTeacher(vit).to(dev)
    ex = eval_exclusions()["raev2_stream"]
    stream = np.load(STREAM)[:STEPS]
    ds_pix, ds_nopix = RSPStreamDS(stream, ex, want_ev=True), RSPStreamDS(stream, ex, want_ev=False)
    b0 = batch(ds_pix, 1, dev)
    x, content = b0["x"], b0["content"]
    t_structure(vit, dev)
    t_coarse(vit, teach, x[:4])
    t_pi(vit, teach, x)
    t_teacher(teach, x, content)
    t_mask()
    t_pred_ref(vit, x, dev)
    t_step(vit, teach, ds_pix, ds_nopix, a.shared, dev)
    t_query(teach, ds_pix, dev)
    t_small(vit, teach, dev, x, content)
    t_data()
    t_paco()
    ok = all(v["ok"] for v in R.values())
    (out / "unit_tests.json").write_text(json.dumps(dict(all_ok=ok, time=time.strftime("%F %T"), tests=R), indent=1, default=str))
    print("ALL_OK" if ok else "SOME_FAILED", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
