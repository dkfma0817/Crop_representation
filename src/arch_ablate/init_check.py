"""CPU-only initialization and structure verification for A0 / A1 / A2. No training, no PACO labels.

Checks: (1) lineage of the common initial checkpoint; (2) per-tensor SHA-256 equality of every shared tensor across A0 / A1 / A2 and against the checkpoint;
(3) forward equivalences: A0 == existing CoarseFine; A0 with s_f2c = 0 == A1; A0 with all exchange scales = 0 == A2; A1 and A2 Coarse == the plain ViT.

  CUDA_VISIBLE_DEVICES= python -m src.arch_ablate.init_check --out <run>/init_check.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import timm
import torch

from src.arch_ablate.model import MODES, from_initial
from src.hcf.model import CoarseFine, load_vit

INITIAL = "output/2026-11-08/hcf_material_causal/initial.pt"
A0_CONFIG = "output/2026-11-08/hcf_material_causal/r384_cosine/P/config.json"
G1POOL = "output/2026-11-06/g1_cross_distill/G1pool/ema_last.pt"
TRAINED_BIDIRECTIONAL = ("output/2026-11-08/hcf_pilot/P/checkpoints/ema_last.pt", "output/2026-11-08/hcf_material_causal/r384_cosine/P/checkpoints/ema_last.pt")


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def matches(a, b, prefix=""):
    """(tensors equal, tensors different) over the keys of b present in a"""
    same = diff = 0
    for k, v in b.items():
        if prefix + k in a:
            same, diff = (same + 1, diff) if torch.equal(a[prefix + k].float(), v.float()) else (same, diff + 1)
    return same, diff


def equal_out(a, b, keys=("coarse", "cls", "reg", "fine")):
    get = lambda o, k: o["mem"]["fine"] if k == "fine" else o[k]  # noqa: E731
    return {k: torch.equal(get(a, k), get(b, k)) for k in keys}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cpu")
    report = dict()
    ck = torch.load(INITIAL, map_location="cpu", weights_only=False)
    sha = file_sha(INITIAL)
    cfg = json.load(open(A0_CONFIG))
    report["initial"] = dict(path=INITIAL, sha256=sha, step=ck["step"], cond=ck["cond"], kind=ck["kind"], a0_used_same_initial=cfg["initial_sha256"] == sha, a0_args_init=cfg["args"]["init"], a0_steps=cfg["steps"], a0_images_per_step=cfg["images_per_step"], a0_world=cfg["world"], a0_global_weight=cfg["global_weight"], a0_centered_weight=cfg["centered_weight"], a0_size=cfg["input_size"])
    assert report["initial"]["a0_used_same_initial"] and ck["step"] == 0 and cfg["centered_weight"] == 0 and cfg["input_size"] == 384
    # Lineage: which tensors of the common initial state were ever trained, and with what
    vit_pre = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, img_size=384, dynamic_img_size=True).state_dict()
    g1pool = torch.load(G1POOL, map_location="cpu", weights_only=False)
    new = {k: v for k, v in ck["model"].items() if not k.startswith("vit.")}
    report["lineage"] = dict(coarse_vs_pretrained_dinov3_same_diff=matches(ck["model"], vit_pre, "vit."), coarse_vs_g1pool_ema_same_diff=matches(ck["model"], g1pool["enc"], "vit."), n_new_tensors=len(new))
    assert report["lineage"]["coarse_vs_g1pool_ema_same_diff"] == (162, 0)
    for path in TRAINED_BIDIRECTIONAL:
        trained = torch.load(path, map_location="cpu", weights_only=False)["model"]
        same = matches(new, {k: v for k, v in trained.items() if not k.startswith("vit.")})
        report["lineage"][f"fine_and_interaction_vs_trained:{path}"] = same
        assert same[0] == 0, (path, same)                                   # no Fine/Interaction tensor comes from a trained bidirectional model
    # Same shared weights across structures
    models = {m: from_initial(INITIAL, m, dev).eval() for m in MODES}
    hashes = {m: {k: tensor_sha(v) for k, v in models[m].state_dict().items()} for m in MODES}
    saved = {k: tensor_sha(v) for k, v in ck["model"].items()}
    shared = set(hashes["A0"]) & set(hashes["A1"]) & set(hashes["A2"])
    report["tensors"] = {m: dict(n_tensors=len(hashes[m]), n_scalars=sum(p.numel() for p in models[m].parameters()), all_equal_to_initial=all(saved[k] == v for k, v in hashes[m].items())) for m in MODES}
    report["tensors"].update(n_shared_all_three=len(shared), shared_equal_all_three=all(hashes["A0"][k] == hashes["A1"][k] == hashes["A2"][k] for k in shared), shared_sha256_of_hashes=hashlib.sha256("".join(hashes["A0"][k] for k in sorted(shared)).encode()).hexdigest(), a1_removed=sorted(set(hashes["A0"]) - set(hashes["A1"])), a2_removed=sorted(set(hashes["A0"]) - set(hashes["A2"])))
    assert report["tensors"]["shared_equal_all_three"] and all(report["tensors"][m]["all_equal_to_initial"] for m in MODES)
    assert all(k.endswith(("f2c.weight", "f2c.bias", "s_f2c")) for k in report["tensors"]["a1_removed"]) and len(report["tensors"]["a1_removed"]) == 12
    assert all(k.startswith("inter.") for k in report["tensors"]["a2_removed"]) and len(report["tensors"]["a2_removed"]) == 24
    # Forward equivalences
    torch.manual_seed(0)
    x = torch.randn(2, 3, 384, 384)
    base = CoarseFine(load_vit(dev), ck["s0"]).to(dev).eval()
    base.load_state_dict(ck["model"], strict=True)
    O = {m: models[m].forward_all(x) for m in MODES}
    eq = dict(A0_equals_existing_CoarseFine=equal_out(O["A0"], base.forward_all(x)))
    zero = from_initial(INITIAL, "A0", dev).eval()
    for n, p in zero.named_parameters():
        if n.endswith("s_f2c"):
            p.zero_()
    eq["A0_with_s_f2c_0_equals_A1"] = equal_out(zero.forward_all(x), O["A1"])
    for n, p in zero.named_parameters():
        if n.endswith("s_c2f"):
            p.zero_()
    eq["A0_with_all_scales_0_equals_A2"] = equal_out(zero.forward_all(x), O["A2"])
    tok = models["A1"].vit.forward_features(x)[:, models["A1"].npt:]
    eq["A1_coarse_equals_plain_vit"] = bool(torch.equal(O["A1"]["coarse"], tok))
    eq["A2_coarse_equals_plain_vit"] = bool(torch.equal(O["A2"]["coarse"], tok))
    eq["A1_fine_differs_from_A2"] = not torch.equal(O["A1"]["mem"]["fine"], O["A2"]["mem"]["fine"])
    eq["A1_coarse_differs_from_A0"] = not torch.equal(O["A1"]["coarse"], O["A0"]["coarse"])
    report["forward"] = eq
    for name, v in eq.items():
        assert all(v.values()) if isinstance(v, dict) else v, (name, v)
    report["status"] = "passed"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print(json.dumps(report, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
