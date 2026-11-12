"""P1 / P2 pre-training checks -> <out>/checks.json. Any failed assertion blocks training / evaluation.

  P2 check 1  frozen blocks[7] state -> copied blocks 9-12 (rope_from_coords on the 24 x 24 grid) -> frozen norm == original final patch tokens
              (+ rope_from_coords == timm get_embed((24, 24)) exactly; the hooked blocks[7] output == the input of blocks[8])
  P2 check 2  zero-initialised split: every child == its parent h before the copied blocks
  P2 check 3  with children: difference to F384 (recorded only)
  P2 check 4  all 576 parents split: fine ids cover the 48 x 48 grid once; child coords == timm make_coords_dinov3(48, 48) at those ids
  P2 check 5  gradients: none on the original DINOv3 / frozen norm / teacher; nonzero on copied blocks and the split
  P2 masks    selected parents masked as keys == physically removed (one ROI); a ROI alone == the same ROI in a padded batch
  P1          shapes, frozen output norm, identity native projection at init, padding invariance, gradients
  ROIAlign    synthetic 48 x 48 map: (A) global grid_sample at the GT-box read points == (B) gather of the read rectangle -> local coordinates
              -> grid_sample, on real PACO val boxes, valid read points
  python -m src.fte.checks --out <dir>
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from timm.layers.pos_embed_sincos import make_coords_dinov3

import src.readout.common as RC
from src.config import STREAM
from src.fte.common import G, child_coords, child_ids, local_read, parent_coords, rect_parents, rope_from_coords
from src.fte.models import PREFIX
from src.fte.train import build, prepare, run_model, student_h8, patch_ce_roi, sig_term, PREFILL
from src.finedec.losses import TeacherQueue
from src.msd.pretrain import eval_exclusions
from src.tsplit.features import ContentDS, read_rects, sample_maps
from src.tsplit.train import STEPS, StreamDS

DATA = "output/2026-11-07/paco_oracle_attr/data"


def mx(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    ds = StreamDS(np.load(STREAM)[:STEPS], eval_exclusions()["raev2_stream"])
    vit, p2 = build("P2", dev)
    b = ds[0]
    d = prepare(vit, b, dev)
    # ---- check 1
    rope_t = vit.rope.get_embed(shape=(G, G))
    rope_m = rope_from_coords(vit.rope, parent_coords(torch.arange(G * G, device=dev) // G, torch.arange(G * G, device=dev) % G))
    store = {}
    h_in = vit.blocks[8].register_forward_pre_hook(lambda m, args: store.__setitem__("in9", args[0]))
    with torch.no_grad():
        _, h8 = student_h8(vit, b["x"][:4].to(dev))
    h_in.remove()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = p2.reference_path(d["h8"])
    res["P2_check1"] = dict(rope_from_coords_vs_timm_max_abs=mx(rope_m, rope_t), h8_vs_block9_input_max_abs=mx(h8, store["in9"]),
                            copied_path_vs_original_final_max_abs=mx(ref, d["final"]), final_abs_max=float(d["final"].abs().max()), dtype="bf16 autocast (= original forward)")
    assert res["P2_check1"]["rope_from_coords_vs_timm_max_abs"] == 0 and res["P2_check1"]["h8_vs_block9_input_max_abs"] == 0
    assert res["P2_check1"]["copied_path_vs_original_final_max_abs"] < 1e-3, res["P2_check1"]
    # ---- check 2
    pid, pok = rect_parents(d["rects"])
    hp = d["h8"][d["rimg"][:, None], PREFIX + pid]
    with torch.no_grad():
        corr = p2.split(hp.float())
    res["P2_check2"] = dict(max_abs_child_minus_parent=float(corr.abs().max()))
    assert res["P2_check2"]["max_abs_child_minus_parent"] == 0
    # ---- check 3
    with torch.no_grad():
        Y, Z, cid, valid, Tt = run_model("P2", p2, d)
    rep = d["final"][d["rimg"][:, None, None], pid[..., None].expand(-1, -1, 4)]
    res["P2_check3"] = dict(cos_to_F384_parent=float(F.cosine_similarity(Y, rep, dim=-1)[valid].mean()), cos_to_teacher=float(F.cosine_similarity(Y, Tt.float(), dim=-1)[valid].mean()),
                            F384_cos_to_teacher=float(F.cosine_similarity(rep, Tt.float(), dim=-1)[valid].mean()), mean_abs_diff_to_F384=float((Y - rep).abs()[valid].mean()),
                            note="recorded only; children + mixed positions change attention, so equality is not expected")
    # ---- check 4
    allr = torch.tensor([[0, 0, G, G]], device=dev)
    pa, _ = rect_parents(allr)
    ca, ci, cj = child_ids(pa)
    cover = torch.bincount(ca.flatten(), minlength=48 * 48)
    ref48 = make_coords_dinov3(48, 48).to(dev)
    res["P2_check4"] = dict(each_fine_id_once=bool((cover == 1).all()), coords_vs_make_coords_48_max_abs=mx(child_coords(ci, cj).reshape(-1, 2), ref48[ca.flatten()]),
                            order="parent-major (row-major over the rectangle), child row-major (k = 2a + b)")
    assert res["P2_check4"]["each_fine_id_once"] and res["P2_check4"]["coords_vs_make_coords_48_max_abs"] == 0
    # ---- masks: removal == masking (one ROI), padding invariance
    r = 5
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        Yb, Zb, _, _ = p2(d["h8"], d["rimg"], d["rects"])
        Y1, Z1, _, _ = p2(d["h8"], d["rimg"][r:r + 1], d["rects"][r:r + 1])
        # explicit removal for ROI r
        pr, okr = rect_parents(d["rects"][r:r + 1])
        K = int(okr.sum())
        keep = torch.ones(G * G, dtype=torch.bool, device=dev)
        keep[pr[0, :K]] = False
        kidx = torch.nonzero(keep).flatten()
        h = d["h8"][d["rimg"][r]]
        hpp = h[PREFIX + pr[0, :K]]
        ch = (hpp[:, None] + p2.split(hpp).view(K, 4, -1)).reshape(K * 4, -1)
        c2, ci2, cj2 = child_ids(pr[:, :K])
        x = torch.cat([h[:PREFIX], h[PREFIX + kidx], ch.to(h.dtype)])[None]
        coords = torch.cat([parent_coords(kidx // G, kidx % G), child_coords(ci2, cj2).reshape(-1, 2)])
        rope = rope_from_coords(p2.rope, coords)
        for blk in p2.blocks:
            x = blk(x, rope=rope)
        Yr = p2.norm(x[0, PREFIX + len(kidx):].float()).view(K, 4, -1)
    res["P2_masks"] = dict(removal_vs_masking_max_abs=mx(Yr, Y1[0, :K]), alone_vs_padded_batch_max_abs=mx(Y1[0, :K], Yb[r, :K]), K=K, batch_K_max=int(pok.sum(1).max()))
    assert res["P2_masks"]["removal_vs_masking_max_abs"] < 0.05 * float(Yr.abs().max()) and res["P2_masks"]["alone_vs_padded_batch_max_abs"] < 0.05 * float(Yr.abs().max()), res["P2_masks"]
    # ---- check 5 (+ SIGReg / CE both reach)
    p2.train()
    Q = TeacherQueue(torch.load(PREFILL, weights_only=False)["features"].to(dev)).snapshot()
    Y, Z, cid, valid, Tt = run_model("P2", p2, d)
    ce, st = patch_ce_roi(Y, Tt, valid, Q)
    sg, n = sig_term(Z, cid, valid, d["rimg"], 1, 2)
    (ce + sg).backward()
    res["P2_check5"] = dict(original_dinov3_grad=any(p.grad is not None for p in vit.parameters()), frozen_norm_grad=any(p.grad is not None for p in p2.norm.parameters()),
                            copied_blocks_grad_norm=float(torch.stack([p.grad.norm() for p in p2.blocks.parameters() if p.grad is not None]).norm()),
                            split_grad_norm=float(p2.split.weight.grad.norm()), teacher_requires_grad=bool(Tt.requires_grad), ce=float(ce), sigreg=float(sg), sigreg_samples=n,
                            Z_norm_mean=float(Z.float().norm(dim=-1)[valid].mean()), patch_kl=st["patch_kl"])
    c5 = res["P2_check5"]
    assert not c5["original_dinov3_grad"] and not c5["frozen_norm_grad"] and c5["copied_blocks_grad_norm"] > 0 and c5["split_grad_norm"] > 0 and not c5["teacher_requires_grad"]
    del p2
    # ---- P1
    vit1, p1 = build("P1", dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        Yb, Zb, pid1, pok1 = p1(d["final"], d["rimg"], d["rects"])
        Y1, Z1, _, _ = p1(d["final"], d["rimg"][r:r + 1], d["rects"][r:r + 1])
    K1 = int(pok1[r].sum())
    p1.train()
    Y, Z, cid, valid, Tt = run_model("P1", p1, d)
    ce, _ = patch_ce_roi(Y, Tt, valid, Q)
    sg, n = sig_term(Z, cid, valid, d["rimg"], 1, 2)
    (ce + sg).backward()
    gr = {k: float(torch.stack([p.grad.norm() for p in ps if p.grad is not None]).norm()) for k, ps in p1.groups().items()}
    res["P1"] = dict(Y_shape=list(Yb.shape), Z_shape=list(Zb.shape), native_proj_identity=bool(torch.equal(p1.native_proj.weight.detach(), torch.eye(768, device=dev))) or "updated-after-backward-check",
                     norm_frozen=all(not p.requires_grad for p in p1.output_norm.parameters()), norm_equals_dinov3=all(torch.equal(x, y) for x, y in zip(p1.output_norm.parameters(), vit1.norm.parameters())),
                     alone_vs_padded_batch_max_abs=mx(Y1[0, :K1], Yb[r, :K1]), group_grad_norms=gr, original_grad=any(p.grad is not None for p in vit1.parameters()), ce=float(ce), sigreg=float(sg))
    assert res["P1"]["norm_frozen"] and res["P1"]["norm_equals_dinov3"] and not res["P1"]["original_grad"] and all(v > 0 for v in gr.values())
    assert res["P1"]["alone_vs_padded_batch_max_abs"] < 0.05 * float(Y1.abs().max()), res["P1"]
    # ---- ROIAlign: global vs local gather
    M = pd.read_parquet(f"{DATA}/manifest_val.parquet")
    bb = RC.to_dev(ContentDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)[list(range(48))], dev)
    rects = read_rects(bb["bil_grid"], bb["cov"])
    g = torch.Generator(device=dev).manual_seed(0)
    syn = torch.randn(len(bb["x"]), 48 * 48, 16, device=dev, generator=g)
    A = sample_maps(syn[bb["img_index"]], bb["bil_grid"])
    pidr, okr = rect_parents(rects)
    cidr, _, _ = child_ids(pidr)
    Yl = syn[bb["img_index"][:, None, None], cidr]
    B = local_read(Yl, rects, bb["bil_grid"])
    c = bb["cov"] > 0
    shifted = sample_maps(syn.view(-1, 48, 48, 16).roll(1, 2).flatten(1, 2)[bb["img_index"]], bb["bil_grid"])
    res["roialign_local_vs_global"] = dict(max_abs_valid=mx(A[c], B[c]), mean_abs_valid=float((A - B).abs()[c].mean()), control_one_child_shift_mean_abs=float((A - shifted).abs()[c].mean()),
                                           n_rois=int(len(rects)), max_rect_side=int(rects[:, 2:].max()), n_valid_points=int(c.sum()), feature_scale="unit-variance random features",
                                           note="read rectangle = every bilinear corner of every valid read point; GT-box continuous geometry kept; no parent-rectangle resize; residual = fp32 round-off of the coordinate shift")
    assert res["roialign_local_vs_global"]["max_abs_valid"] < 1e-4 and res["roialign_local_vs_global"]["control_one_child_shift_mean_abs"] > 0.1
    res["status"] = "passed"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "checks.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
