"""CPU / GPU smoke for the E6 multi-layer memory pilot -> <out>/smoke.json (no result kept)."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import src.msd.common as MS
import src.readout.common as RC
from src.config import COCO_TRAIN, FEAT, STREAM
from src.losses.feature import dense_cos, dense_mse
from src.xdec import geometry as XG
from src.xdec import models as XM
from src.e6ml import models as EM
from src.xdec.pretrain import STATS


def gn(ps):
    g = [p.grad.norm() for p in ps if p.grad is not None]
    return float(torch.stack(g).norm()) if g else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(64)
    XG.install()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(STATS)
    mu, sd = torch.as_tensor(z["mean_12"], device=dev), torch.as_tensor(z["std_12"], device=dev)
    student = MS.Student(adapters=False).to(dev).eval()
    ds = MS.CocoStreamDS(pd.read_parquet(os.path.join(FEAT, "stream_rois.parquet")), COCO_TRAIN)
    b = MS.to_dev(ds[[int(i) for i in np.load(STREAM)[0][:8]]], dev)
    res = {}
    with torch.no_grad():
        feats, _ = student(b["full"])
        ref = RC.encode_layers(student.vit, b["full"], (4, 8, 12))          # = Full Multi-layer + R2 encoder call
        T = RC.encode_layers(student.vit, b["crops"], (12,))[12]
    res["feature_shapes"] = {d: list(v.shape) for d, v in feats.items()}
    res["student_vs_fullML_encoder_max_abs"] = {d: float((feats[d] - ref[d]).abs().max()) for d in (4, 8, 12)}
    res["feature_stats_mean_std_tokennorm"] = {d: [float(feats[d].mean()), float(feats[d].std()), float(feats[d].norm(dim=-1).mean())] for d in (4, 8, 12)}
    g, w = b["geom"], b["geom"]["valid_w"].float()
    models = {"E6-original": XM.build("E6"), "A": EM.build("A"), "B": EM.build("B")}
    for m in models.values():
        m.to(dev).set_stats(mu, sd)
    res["n_params"] = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    res["n_params_P4P8"] = {k: sum(p.numel() for p in m.mem_in4.parameters()) + sum(p.numel() for p in m.mem_in8.parameters()) for k, m in models.items() if k != "E6-original"}
    sdA, sdB = models["A"].state_dict(), models["B"].state_dict()
    res["A_B_identical_init_state"] = all(torch.equal(sdA[k], sdB[k]) for k in sdA)
    sd0 = models["E6-original"].state_dict()
    res["A_shares_E6_original_init_for_common_keys"] = all(torch.equal(sd0[k], sdA[k]) for k in sd0)
    with torch.no_grad():
        Z = {k: m(feats, g).float() for k, m in models.items()}
        B_ = XG.bicubic_roi(feats[12], g)
        # memory tokens before PE: A, B, E6-original identical at init
        m0 = models["E6-original"].mem_in(feats[12])
        mA = models["A"].mem_in(feats[12]) + models["A"].mem_in4(feats[12]) + models["A"].mem_in8(feats[12])
        mB = models["B"].mem_in(feats[12]) + models["B"].mem_in4(feats[4]) + models["B"].mem_in8(feats[8])
    res["init_memory_max_abs"] = {"A-E6orig": float((mA - m0).abs().max()), "B-E6orig": float((mB - m0).abs().max())}
    res["init_output_max_abs"] = {"A-B": float((Z["A"] - Z["B"]).abs().max()), "A-E6orig": float((Z["A"] - Z["E6-original"]).abs().max()),
                                  "B-Bicubic": float((Z["B"] - B_).abs().max())}
    # gradient flow over a few AdamW updates (L_old)
    for k in ("A", "B"):
        m = models[k]
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
        gr = m.groups()
        hist = []
        for s in range(a.steps):
            Zk = m(feats, g).float()
            loss = (dense_mse(Zk, T, w) + 1 - dense_cos(Zk, T, w)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            hist.append(dict(step=s + 1, loss=float(loss), **{f"grad_{n}": gn(gr[n]) for n in ("head_last_linear", "trunk", "P12 (mem_in)", "P4P8_linear", "P4P8_ln")}))
            opt.step()
        res[f"grad_flow_{k}"] = hist
        res[f"P4P8_weight_norm_after_{k}"] = float(torch.cat([p.detach().reshape(-1) for p in gr["P4P8_linear"]]).norm())
    with torch.no_grad():
        res["after_updates_A_vs_B_output_max_abs"] = float((models["A"](feats, g) - models["B"](feats, g)).abs().max())
    res["checks"] = dict(
        encoder_match=max(res["student_vs_fullML_encoder_max_abs"].values()) < 1e-3,
        same_grid=all(v == [8, 576, 768] for v in res["feature_shapes"].values()),
        params_equal_A_B=res["n_params"]["A"] == res["n_params"]["B"],
        init_equal=res["init_output_max_abs"]["A-B"] == 0 and res["init_output_max_abs"]["A-E6orig"] == 0 and res["init_output_max_abs"]["B-Bicubic"] == 0,
        P4P8_linear_gets_grad=all(any(h["grad_P4P8_linear"] > 0 for h in res[f"grad_flow_{k}"]) for k in ("A", "B")),
        P4P8_ln_gets_grad=all(any(h["grad_P4P8_ln"] > 0 for h in res[f"grad_flow_{k}"]) for k in ("A", "B")),
        A_B_diverge_after_updates=res["after_updates_A_vs_B_output_max_abs"] > 0)
    res["ALL_PASS"] = all(res["checks"].values())
    json.dump(res, open(os.path.join(a.out, "smoke.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
