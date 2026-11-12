"""FA pre-training checks (spec items 1-13) -> <out>/checks.json; any failed assertion stops the run.

  python -m src.finedec.checks --shared <shared> --out <shared>/checks
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import src.readout.common as RC
from src.config import STREAM
from src.finedec.feats import student_pre
from src.finedec.losses import TeacherQueue, fp32, mean_residual, patch_ce, sample_latents, sigreg
from src.finedec.model import FineAware
from src.finedec.train import batch_tensors, build, forward, make_opt, state_sha
from src.msd.pretrain import eval_exclusions
from src.tsplit.features import ContentDS, read_rects, sample_maps, student
from src.tsplit.model import G, G2
from src.tsplit.train import STEPS, StreamDS

DATA = "output/2026-11-07/paco_oracle_attr/data"


def randomize_heads(m, g):
    m = copy.deepcopy(m)
    with torch.no_grad():
        for h in (m.head_mean, m.head_res):
            h[2].weight.copy_(torch.randn(h[2].weight.shape, generator=g) .to(h[2].weight) * 0.02)
            h[2].bias.copy_(torch.randn(h[2].bias.shape, generator=g).to(h[2].bias) * 0.02)
    return m


def centers():
    """Mean pixel-centre coordinate of each output window (uniform kernel on a coordinate ramp) for both fine convs."""
    x = torch.arange(384).float().view(1, 1, 1, 384).expand(1, 1, 384, 384) + 0.5
    ov = F.conv2d(x, torch.ones(1, 1, 16, 16) / 256, stride=8, padding=4)      # zero padding only at the two border cells
    lo = F.conv2d(x, torch.ones(1, 1, 8, 8) / 64, stride=8)
    want = 4 + 8 * torch.arange(48).float()
    d_ov = float((ov[0, 0, 10, 1:-1] - want[1:-1]).abs().max())
    d_lo = float((lo[0, 0, 10] - want).abs().max())
    assert ov.shape[-1] == 48 and lo.shape[-1] == 48 and d_ov < 1e-4 and d_lo < 1e-4, (d_ov, d_lo)
    return dict(overlap_conv_center_max_dev_px=d_ov, local_conv_center_max_dev_px=d_lo, rule="cell u centre = 4 + 8u px on the 384 canvas (interior cells; the overlapping conv's border cells see 4 px of zero padding)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    vit, m = build(dev, a.shared)
    _, m2 = build(dev, a.shared)
    res["10_same_init"] = dict(sha_A=state_sha(m), sha_B=state_sha(m2), equal=state_sha(m) == state_sha(m2) == json.loads((Path(a.shared) / "prep_manifest.json").read_text())["init_sha256"])
    assert res["10_same_init"]["equal"]
    ex = eval_exclusions()["raev2_stream"]
    ds = StreamDS(np.load(STREAM)[:STEPS], ex)
    b0, b0b = ds[0], ds[0]
    res["10_same_data"] = dict(rects_equal=bool(torch.equal(b0["rects"], b0b["rects"])), x_equal=bool(torch.equal(b0["x"], b0b["x"])), note="StreamDS is deterministic per (step, slot); FA-A and FA-B use the same code, stream and steps (= S3)")
    d = batch_tensors(vit, m, b0, dev)
    # 1, 2
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fine, kvs = m.image_side(d["x"], d["final"])
    res["1_2_shapes"] = dict(global_final=list(d["final"].shape), x_pre=list(d["xpre"].shape), fine=list(fine.shape), teacher_children=list(d["Tt"].shape), memory_k=list(kvs[0][0].shape))
    assert d["final"].shape[1:] == (576, 768) and fine.shape[1:] == (2304, 384) and kvs[0][0].shape[2] == 576
    res["1_fine_centers"] = centers()
    with torch.no_grad():
        ov = m.ov(d["x"][:1])
        loc = m.local(d["x"][:1])
    assert ov.shape[-2:] == (48, 48) and loc.shape[-2:] == (48, 48)
    # 3 parent <-> child
    par = (d["cy"] // 2) * G + d["cx"] // 2
    want = (d["rects"][:, 0:1] + d["lp"] // d["rects"][:, 3:4]) * G + d["rects"][:, 1:2] + d["lp"] % d["rects"][:, 3:4]
    assert torch.equal(par[d["ok"]], want[d["ok"]])
    cnt = torch.zeros(len(par), G * G, device=dev).scatter_add_(1, torch.where(d["ok"], par, 0), d["ok"].float())
    assert set(cnt[cnt > 0].unique().tolist()) == {4.0}
    res["3_parent_child"] = dict(ok=True, children_per_parent=4, geometry="pixel / index checks of output/2026-11-11/token_split_x2_s0/checks/checks.json reused (same canvases and rect_index)")
    # 4 norm copy
    eq = all(torch.equal(p, q) for p, q in zip(m.norm.parameters(), vit.norm.parameters()))
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        nd = float((m.norm(d["xpre"]) - d["final"]).abs().max())
    assert eq and nd < 1e-4 and all(not p.requires_grad for p in m.norm.parameters())
    assert torch.equal(m.ov.weight, vit.patch_embed.proj.weight) and torch.equal(m.ov.bias, vit.patch_embed.proj.bias)
    res["4_norm_copy"] = dict(params_equal=eq, norm_xpre_vs_final_max_abs=nd, patch_embed_copy_equal=True, eps=m.norm.eps)
    # 5 zero correction
    Y, Z, dm, dr = forward(m, d)
    rep = d["final"][d["rimg"][:, None], par]
    z_dev = float((Y - rep)[d["ok"]].abs().max())
    assert z_dev < 1e-4 and float(dm.abs().max()) == 0 and float(dr.abs().max()) == 0
    res["5_zero_correction"] = dict(max_abs_Y_minus_F384_repeat=z_dev)
    # 6 residual sums to zero (random non-zero heads)
    g = torch.Generator().manual_seed(0)
    mr = randomize_heads(m, g)
    Yr, Zr, dmr, drr = forward(mr, d)
    s = torch.zeros(len(par), G * G, 768, device=dev).scatter_add_(1, torch.where(d["ok"], par, 0)[..., None].expand(-1, -1, 768), drr * d["ok"][..., None].float())
    res["6_residual_sum"] = dict(max_abs_parent_sum=float(s.abs().max()), delta_res_mean_norm=float(drr.norm(dim=-1)[d["ok"]].mean()))
    assert res["6_residual_sum"]["max_abs_parent_sum"] < 1e-4
    # 7 post-norm mean/residual, losses at equality
    const = json.loads((Path(a.shared) / "constants.json").read_text())
    l0, _, _, _ = mean_residual(d["Tt"], d["Tt"], d["lp"], d["key_ok"], const["v_mean"], const["v_res"])
    Q = TeacherQueue(torch.load(Path(a.shared) / "queue_prefill.pt", weights_only=False)["features"].to(dev))
    snap = Q.snapshot()
    ce0, st0 = patch_ce(d["Tt"], d["Tt"], d["key_ok"], d["rimg"], d["n_img"], snap)
    res["7_equality"] = dict(L_A_at_equality=float(l0), patch_kl_at_equality=st0["patch_kl"], patch_ce_equals_teacher_entropy=abs(st0["patch_kl"]) < 1e-4)
    assert float(l0) < 1e-10 and abs(st0["patch_kl"]) < 1e-4
    # 8 letterbox / padding invariance
    lA, _, _, _ = mean_residual(Yr, d["Tt"], d["lp"], d["key_ok"], const["v_mean"], const["v_res"])
    Yp = Yr.clone()
    Yp[~d["key_ok"]] = 1e3
    lA2, _, _, _ = mean_residual(Yp, d["Tt"], d["lp"], d["key_ok"], const["v_mean"], const["v_res"])
    cB, _ = patch_ce(Yr, d["Tt"], d["key_ok"], d["rimg"], d["n_img"], snap)
    cB2, _ = patch_ce(Yp, d["Tt"], d["key_ok"], d["rimg"], d["n_img"], snap)
    n_invalid = int((d["ok"] & ~d["key_ok"]).sum())
    res["8_mask"] = dict(invalid_children_in_rects=n_invalid, padded=int((~d["ok"]).sum()), L_A_invariant=abs(float(lA) - float(lA2)) < 1e-6, CE_invariant=abs(float(cB) - float(cB2)) < 1e-6)
    assert res["8_mask"]["L_A_invariant"] and res["8_mask"]["CE_invariant"]
    # 9 cross-attention sees all 576 tokens: perturb the memory of a parent outside every ROI of image 0
    r0 = torch.nonzero(d["rimg"] == 0).flatten()
    inside = set(par[r0][d["ok"][r0]].tolist())
    far = next(p for p in range(G * G) if p not in inside)
    d2 = dict(d)
    d2["final"] = d["final"].clone()
    d2["final"][0, far] += 5.0
    Y2, _, _, _ = forward(mr, d2)
    ch = float((Y2[r0] - Yr[r0])[d["ok"][r0]].abs().max())
    res["9_global_cross_attention"] = dict(memory_tokens=576, perturbed_parent_outside_rois=far, max_change_in_roi_outputs=ch)
    assert ch > 0
    # 11, 12 one real optimisation step per loss + gradient flow after a few updates
    flows = {}
    for loss_kind in ("A", "B"):
        _, mm = build(dev, a.shared)
        mm.train()
        opt = make_opt(mm)
        q = TeacherQueue(torch.load(Path(a.shared) / "queue_prefill.pt", weights_only=False)["features"].to(dev))
        hist = []
        for st in range(4):
            dd = batch_tensors(vit, mm, ds[st], dev)
            Yk, Zk, _, _ = forward(mm, dd)
            if loss_kind == "A":
                L, _, _, _ = mean_residual(Yk, dd["Tt"], dd["lp"], dd["key_ok"], const["v_mean"], const["v_res"])
                parts = {"loss": L}
            else:
                ce, _ = patch_ce(Yk, dd["Tt"], dd["key_ok"], dd["rimg"], dd["n_img"], q.snapshot())
                zs = sample_latents(Zk.float(), dd["key_ok"], dd["rimg"], dd["idx"], 1024, seed=1_000_000 + st)
                sg = sigreg(zs, seed=2_000_000 + st)
                L = ce + 0.05 * sg
                parts = {"ce": ce, "sigreg": sg}
            assert torch.isfinite(L) and L.dtype == torch.float32
            rec = {}
            for pn, pv in parts.items():
                gs = torch.autograd.grad(pv, mm.trainable(), retain_graph=True, allow_unused=True)
                names = {id(p): n for n, ps in mm.groups().items() for p in ps}
                rec[pn] = {gn: float(torch.cat([g.flatten() for p, g in zip(mm.trainable(), gs) if g is not None and names[id(p)] == gn] or [torch.zeros(1, device=dev)]).norm()) for gn in mm.groups()}
            opt.zero_grad(set_to_none=True)
            L.backward()
            assert all(p.grad is None for p in vit.parameters()) and all(p.grad is None for p in mm.parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_(mm.trainable(), 1.0)
            opt.step()
            hist.append(dict(step=st + 1, loss=float(L), grads=rec, **({"sigreg_samples": len(zs)} if loss_kind == "B" else {})))
        last = hist[-1]["grads"]
        for pn, parts_ in last.items():
            need = [g for g in parts_ if not (pn == "sigreg" and g.startswith("head_"))]      # SIGReg acts on Z, before the heads (by design)
            assert all(parts_[g] > 0 for g in need), (pn, parts_)
            if pn == "sigreg":
                assert parts_["head_mean"] == 0 and parts_["head_res"] == 0
        flows[loss_kind] = hist
    res["11_12_gradients"] = dict(per_step=flows, note="step 1: only the zero-initialised last head layers receive prediction gradient (expected); by step 4 every trainable group receives nonzero gradient from every prediction loss term, and SIGReg reaches every group before Z (not the heads, by design); frozen DINOv3 / copied conv / copied norm never get .grad")
    with fp32():
        tf32_inside = torch.backends.cuda.matmul.allow_tf32
    res["13_precision"] = dict(decoder="bf16 autocast (as S3)", tf32_outside_loss=torch.backends.cuda.matmul.allow_tf32, tf32_inside_loss=tf32_inside, loss_dtype="float32")
    assert tf32_inside is False
    # R2 input consistency at zero correction (read path of src.tsplit.features RoiEncoder mode FA)
    M = __import__("pandas").read_parquet(f"{DATA}/manifest_val.parquet")
    bb = RC.to_dev(ContentDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)[list(range(24))], dev)
    from src.tsplit.features import RoiEncoder
    enc = RoiEncoder("FA", dev, ckpt=str(Path(a.shared) / "init_as_ckpt.pt") if False else None, fa_model=m)
    t_fa = enc(dict(bb))
    fin, _ = student(vit, bb["x"])
    nn_map = fin.view(-1, G, 1, G, 1, 768).expand(-1, G, 2, G, 2, 768).reshape(-1, G2 * G2, 768)
    t_nn = sample_maps(nn_map[bb["img_index"]], bb["bil_grid"])
    t_f384 = sample_maps(fin[bb["img_index"]], bb["bil_grid"])
    c = bb["cov"] > 0
    res["5b_r2_input_at_init"] = dict(FA_vs_nearest_repeat_F384_read_max_abs=float((t_fa - t_nn)[c].abs().max()), FA_vs_F384_bilinear_read_cos=float(F.cosine_similarity(t_fa, t_f384, dim=-1)[c].mean()),
                                      note="at zero correction the FA read equals a read of the 2x nearest-repeated F384 map, not the 24x24 bilinear F384 read")
    assert res["5b_r2_input_at_init"]["FA_vs_nearest_repeat_F384_read_max_abs"] < 1e-3
    res["status"] = "passed"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "checks.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "11_12_gradients"}, indent=1, default=str), flush=True)
    for k, h in flows.items():
        print(k, [{kk: {g: round(x, 5) for g, x in vv.items()} for kk, vv in s["grads"].items()} for s in (h[0], h[-1])], flush=True)


if __name__ == "__main__":
    main()
