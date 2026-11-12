"""Pre-training checks for RSP-Residual (GPU). Writes <run>/checks/unit_tests.json and the Step-0 model (<step0>/model.pt).

  python -m src.rspres.checks --run output/2026-11-12/rsp_residual_s0 --step0 output/2026-11-12/rsp_residual_step0_s0 \
      --shared output/2026-11-12/rsp_masked_comparison/shared
"""
import argparse
import inspect
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import src.readout.common as RC
import src.rsp.train as RT
import src.rspres.train as XT
from src.config import STREAM
from src.msd.pretrain import eval_exclusions
from src.rsp.data import RSPContentDS, RSPStreamDS, frects_of
from src.rsp.teacher import DenseTeacher
from src.rspres.encoder import res_tokens
from src.rspres.model import RSPResidual, f384_bil
from src.tsplit.features import RoiEncoder, frozen_vit
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
            print(name, json.dumps(RES[name], default=float)[:900], flush=True)
        return run
    return deco


def build(shared, dev):
    st = torch.load(shared / "norm_stats.pt", map_location=dev)
    mu, sd = st["mu"].to(dev), st["sd"].to(dev)
    torch.manual_seed(0)
    m = RSPResidual(mu, sd).to(dev)
    init_pix = torch.load(shared / "init_pix.pt", map_location=dev)
    m.load_state_dict(init_pix)
    m.zero_delta()
    return m, mu, sd, init_pix


def train_batch(s, dev):
    ds = RSPStreamDS(np.load(STREAM)[:RT.STEPS], eval_exclusions()["raev2_stream"], want_ev=True)
    b = ds[s]
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def step_inputs(teach, b):
    fr = frects_of(b["rects"])
    final, tgt, bad = XT.teacher_targets(teach, b["x"], fr, b["rimg"], b["content"], b["q"], b["q_ok"])
    assert bad == 0
    cen, _ = XT.query_centres(fr, b["q"])
    return final, tgt, cen


@test("step0_delta_zero_and_yhat_equals_f384_bil")
def t_step0(m, teach, mu, sd, init_pix, dev):
    b = train_batch(0, dev)
    final, tgt, cen = step_inputs(teach, b)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_n, base, delta_n = m.parts(final, b["rimg"], b["box"], b["ev"], cen)
    yhat = pred_n * sd + mu
    # the same F384-bilinear read as RSP's training statistic 'ref_f384bil' (src.rsp.train.feature_stats)
    fm = final[:, 5:].transpose(1, 2).reshape(final.shape[0], 768, 24, 24)
    fb = F.grid_sample(fm[b["rimg"]], (cen.flip(-1) / 384 * 2 - 1)[:, :, None], mode="bilinear", padding_mode="border", align_corners=False)[..., 0].transpose(1, 2)
    others_equal = all(torch.equal(v, init_pix[k]) for k, v in m.state_dict().items() if k not in ("out.weight", "out.bias"))
    r = dict(delta_n_max_abs=float(delta_n.abs().max()), out_weight_abs_max=float(m.out.weight.abs().max()), out_bias_abs_max=float(m.out.bias.abs().max()),
             yhat_minus_f384bil_max_abs=float((yhat - base).abs().max()), yhat_minus_f384bil_rel=float((yhat - base).norm() / base.norm()),
             base_minus_rsp_ref_f384bil_max_abs=float((base - fb).abs().max()), all_other_init_tensors_equal_init_pix=others_equal,
             n_roi=len(b["rimg"]), n_query=int(b["q_ok"].sum()))
    assert r["delta_n_max_abs"] == 0.0 and r["yhat_minus_f384bil_max_abs"] < 1e-4 and r["base_minus_rsp_ref_f384bil_max_abs"] == 0.0 and others_equal, r
    return r


@test("step0_r2_input_vs_f384_baseline")
def t_r2(m, vit, dev):
    """PACO val batches through the R2 read: Step-0 (no predictor) vs the F384 baseline read, and Step-0 vs the full RSPRES path with the
    zero-init model (must be identical)."""
    RSPContentDS.WANT_EV = True
    M = pd.read_parquet(f"{DATA}/manifest_val.parquet")
    ds = RSPContentDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    f384 = RoiEncoder("F384", dev)
    rng = np.random.default_rng(0)
    d_abs, cos, rel_num, rel_den, d_path, n = [], [], 0.0, 0.0, [], 0
    for s in rng.choice(len(ds) - 12, 8, replace=False):
        b = RC.to_dev(RC.collate_one([ds[list(range(s, s + 12))]]), dev)
        ii = b["img_index"].clone()
        t_base = f384(dict(b))
        b["img_index"] = ii
        t0 = res_tokens(m, vit, b, step0=True)
        t1 = res_tokens(m, vit, b, step0=False)
        v = b["cov"] > 0
        d_abs.append(float((t0[v] - t_base[v]).abs().max()))
        cos.append(F.cosine_similarity(t0[v], t_base[v], dim=-1).cpu())
        rel_num += float((t0[v] - t_base[v]).pow(2).sum())
        rel_den += float(t_base[v].pow(2).sum())
        d_path.append(float((t0[v] - t1[v]).abs().max()))
        n += int(v.any(1).sum())
    c = torch.cat(cos)
    r = dict(n_roi=n, n_read_points=len(c), step0_vs_f384_max_abs=max(d_abs), step0_vs_f384_rel_l2=(rel_num / rel_den) ** .5,
             step0_vs_f384_cos_mean=float(c.mean()), step0_vs_f384_cos_min=float(c.min()), step0_vs_f384_cos_p01=float(c.quantile(0.01)),
             step0_path_vs_zero_init_predictor_path_max_abs=max(d_path),
             identical_to_f384=max(d_abs) < 1e-5,
             note="Step-0 = F384 map sampled at the 4 px fine-cell centres, then bilinearly re-read at the R2 points; F384 = direct bilinear read. "
                  "They differ where the coarse-grid kinks fall between fine centres and at borders -> Step-0 is evaluated as its own R2 control.")
    assert r["step0_path_vs_zero_init_predictor_path_max_abs"] < 1e-5, r
    return r


@test("gradient_flow_zero_init")
def t_grad(m, teach, mu, sd, dev):
    """Step 0: only the output Linear gets gradient (Delta_n = W h + b with W = 0 blocks every other path - expected). After ONE
    AdamW update (on a copy, schedule step 0) every predictor group gets non-zero gradient."""
    import copy
    mm = copy.deepcopy(m).train()
    opt = XT.make_opt(mm)
    out = {}
    for it, s in enumerate((0, 1)):
        b = train_batch(s, dev)
        final, tgt, cen = step_inputs(teach, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_n = mm(final, b["rimg"], b["box"], b["ev"], cen)
        loss = XT.roi_l1(pred_n, (tgt - mu) / sd, b["q_ok"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = {}
        for gname, ps in mm.groups().items():
            gg = [p.grad.float().norm() for p in ps if p.grad is not None]
            g[gname] = float(torch.stack(gg).norm()) if gg else 0.0
        g["out.weight"] = float(mm.out.weight.grad.norm())
        g["out.bias"] = float(mm.out.bias.grad.norm())
        out[f"backward_{it}"] = dict(loss=float(loss), grad_norms=g, frozen_vit_grads_none=all(p.grad is None for p in teach.vit.parameters()))
        for gr in param_groups_lr(opt):
            gr["lr"] = XT.lr_at(it)
        opt.step()
    g0, g1 = out["backward_0"]["grad_norms"], out["backward_1"]["grad_norms"]
    assert g0["out.weight"] > 0 and g0["out.bias"] > 0
    assert all(g0[k] == 0.0 for k in ("ctx_in", "size_z", "mask_token", "blocks", "evidence")), g0
    assert all(g1[k] > 0 for k in ("ctx_in", "size_z", "mask_token", "blocks", "evidence", "out")), g1
    assert out["backward_0"]["frozen_vit_grads_none"] and out["backward_1"]["frozen_vit_grads_none"]
    out["note"] = "step 0: body grads are exactly 0 because the zero output weight blocks them; after one update they flow"
    return out


def param_groups_lr(opt):
    return opt.param_groups


@test("teacher_target_normalisation_data_recipe_identical_to_rsp")
def t_same(shared, teach, mu, sd, dev):
    man = json.loads((shared / "prep_manifest.json").read_text())
    import hashlib
    h = hashlib.sha256()
    h.update(torch.load(shared / "norm_stats.pt", map_location="cpu")["mu"].numpy().tobytes())
    same_src = {}
    for fn in ("teacher_targets", "roi_l1", "query_centres", "lr_at", "make_opt", "state_sha"):
        same_src[fn] = inspect.getsource(getattr(RT, fn)) == inspect.getsource(getattr(XT, fn))
    consts = {k: (getattr(RT, k), getattr(XT, k)) for k in ("STEPS", "WARM", "LR", "WD", "CLIP")}
    rsp_cfg = json.loads(Path("output/2026-11-12/rsp_masked_pix_s0/train/config.json").read_text())
    b = train_batch(777, dev)
    fr = frects_of(b["rects"])
    a1 = RT.teacher_targets(teach, b["x"], fr, b["rimg"], b["content"], b["q"], b["q_ok"])
    a2 = XT.teacher_targets(teach, b["x"], fr, b["rimg"], b["content"], b["q"], b["q_ok"])
    r = dict(function_source_identical=same_src, constants=consts, norm_stats_sha_rsp_config=rsp_cfg["norm_stats_sha256"],
             norm_stats_sha_manifest=man["norm_stats_sha256"], teacher_targets_max_abs_diff=float((a1[1] - a2[1]).abs().max()),
             data_class="src.rsp.data.RSPStreamDS(stream[:12000], raev2_stream exclusions, want_ev=True) = RSP-Masked pix",
             teacher="src.rsp.teacher.DenseTeacher(vit) p_T 4, U^T w kernel (same class / args as RSP)")
    assert all(same_src.values()) and all(x == y for x, y in consts.values())
    assert rsp_cfg["norm_stats_sha256"] == man["norm_stats_sha256"] and r["teacher_targets_max_abs_diff"] == 0.0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step0", required=True)
    ap.add_argument("--shared", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    shared = Path(a.shared)
    vit = frozen_vit(dev)
    teach = DenseTeacher(vit).to(dev)
    m, mu, sd, init_pix = build(shared, dev)
    m.eval()
    with torch.no_grad():
        t_step0(m, teach, mu, sd, init_pix, dev)
        t_r2(m, vit, dev)
    t_grad(m, teach, mu, sd, dev)
    with torch.no_grad():
        t_same(shared, teach, mu, sd, dev)
    allpass = all(r["status"] == "PASS" for r in RES.values())
    out = Path(a.run) / "checks"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(all_pass=allpass, tests=RES, time=time.strftime("%F %T")), open(out / "unit_tests.json", "w"), indent=1, default=float)
    if allpass:
        s0 = Path(a.step0) / "model"
        s0.mkdir(parents=True, exist_ok=True)
        torch.save(dict(kind="RSPRES", pix=True, model=m.state_dict(), step=0, config=dict(condition="RSP-Residual Step-0 (Delta = 0)"),
                        mu=mu.cpu(), sd=sd.cpu()), s0 / "model.pt")
    print("ALL_PASS" if allpass else "SOME_FAILED", flush=True)
    raise SystemExit(0 if allpass else 1)


if __name__ == "__main__":
    main()
