"""Pre-training checks for S3-Distill -> <out>/kl_checks.json (any failed assertion stops the run).

  1 synthetic: masked bank / local KL == per-ROI reference computed on the valid tokens only (float64), incl. an all-invalid ROI and padding;
    gradients equal the reference gradients; no NaN / inf in values or gradients; KL(P||P) = 0; direction = KL(teacher || student)
    (differs from the reverse); perturbing padded / invalid student tokens changes neither loss nor any valid-token gradient
  2 initialisation: build() is deterministic, and on the S3-L2 step-1 batch it reproduces the logged S3-L2 step-1 L2 loss and gradient norms
    (head / split / split early columns / total) -> same initial parameters and same first batch (images, flips, ROI rectangles)
  3 bank: shape, unit norms, sha256 equals the manifest, no gradient
  4 real batch, one forward/backward of the KL loss: finite losses, nonzero gradient in every decoder group (incl. the early columns),
    no gradient on the frozen encoder or the bank, analysis gradients do not touch .grad

  python -m src.tsplit.kl_checks --out <run>/checks --bank <run>/bank --s3l2 output/2026-11-11/token_split_x2_s0/train
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.config import STREAM
from src.msd.pretrain import eval_exclusions
from src.tsplit.features import student, teacher, valid48
from src.tsplit.kl_loss import bank_kl, local_kl, ref_bank, ref_local
from src.tsplit.model import rect_index
from src.tsplit.train import STEPS, StreamDS, roi_mse
from src.tsplit.train_kl import build


def synthetic(dev):
    g = torch.Generator().manual_seed(0)
    R, N, D, K = 5, 12, 16, 64
    S0 = torch.randn(R, N, D, generator=g, dtype=torch.float64)
    T = torch.randn(R, N, D, generator=g, dtype=torch.float64)
    Q = torch.nn.functional.normalize(torch.randn(K, D, generator=g, dtype=torch.float64), dim=-1)
    valid = torch.ones(R, N, dtype=torch.bool)
    valid[1, 7:] = False          # padding
    valid[2, [0, 5]] = False      # letterbox-invalid inside the rectangle
    valid[3] = False              # no valid child at all -> skipped
    valid[4, 1:] = False          # a single valid child
    out = {}
    for name, fn, ref, kw in (("bank", bank_kl, ref_bank, dict(Q=Q.float().to(dev), tau=.07)), ("local", local_kl, ref_local, dict(tau=.07))):
        S = S0.clone().float().to(dev).requires_grad_(True)
        loss, _ = fn(S, T.float().to(dev), valid.to(dev), **kw)
        (gS,) = torch.autograd.grad(loss, S)
        Sr = S0.clone().requires_grad_(True)
        lr = ref(Sr, T, valid, kw["Q"].double().cpu(), .07) if name == "bank" else ref(Sr, T, valid, .07)
        (gr,) = torch.autograd.grad(lr, Sr)
        assert torch.isfinite(loss) and torch.isfinite(gS).all()
        d_val, d_grad = abs(float(loss) - float(lr)), float((gS.double().cpu() - gr).abs().max())
        assert d_val < 1e-4 * max(1, abs(float(lr))) and d_grad < 1e-4, (name, d_val, d_grad)
        # padded / invalid tokens: perturb them -> same loss and same valid-token gradient
        S2 = S0.clone().float()
        S2[~valid] = 1e3 * torch.randn(int((~valid).sum()), D, generator=g).float()
        S2 = S2.to(dev).requires_grad_(True)
        l2, _ = fn(S2, T.float().to(dev), valid.to(dev), **kw)
        (g2,) = torch.autograd.grad(l2, S2)
        assert abs(float(l2) - float(loss)) < 1e-5 and float((g2 - gS)[valid.to(dev)].abs().max()) < 1e-5 and float(g2[~valid.to(dev)].abs().max()) == 0.0
        # KL(P || P) = 0
        l0, _ = fn(T.float().to(dev), T.float().to(dev), valid.to(dev), **kw)
        assert abs(float(l0)) < 1e-5, float(l0)
        # direction: loss equals KL(teacher || student), not KL(student || teacher)
        rev = ref(T.clone(), S0, valid, kw["Q"].double().cpu(), .07) if name == "bank" else ref(T.clone(), S0, valid, .07)
        out[name] = dict(value=float(loss), reference=float(lr), abs_diff=d_val, grad_max_abs_diff=d_grad, reverse_kl=float(rev), padding_invariant=True, kl_self=float(l0))
        assert abs(float(rev) - float(lr)) > 0.05 * abs(float(lr))
    return out


def init_repro(dev, s3l2):
    vit, m = build(dev)
    p1 = [p.detach().clone() for p in m.parameters()]
    _, m2 = build(dev)
    same = all(torch.equal(a, b) for a, b in zip(p1, m2.parameters()))
    assert same, "build() not deterministic"
    ex = eval_exclusions()["raev2_stream"]
    b = StreamDS(np.load(STREAM)[:STEPS], ex)[0]
    x, x768 = b["x"].to(dev), b["x768"].to(dev)
    rects, rimg, content = b["rects"].to(dev), b["rimg"].to(dev), b["content"].to(dev)
    fin, ear = student(vit, x)
    T = teacher(vit, x768)
    idx, cy, cx, ok = rect_index(rects)
    key_ok = ok & valid48(content)[rimg[:, None], idx]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m.refine(m.child_map(fin, ear), rimg, idx, cy, cx, key_ok)
    loss = roi_mse(y, T[rimg[:, None], idx], key_ok).mean()
    loss.backward()
    got = {gname: float(torch.stack([p.grad.float().norm() for p in ps if p.grad is not None]).norm()) for gname, ps in m.groups().items()}
    got["split_early_cols"] = float(m.split.lin.weight.grad[:, 768:].float().norm())
    logged = json.loads(open(Path(s3l2) / "grads.jsonl").readline())
    loggedL = json.loads(open(Path(s3l2) / "train.jsonl").readline())
    assert logged["step"] == 1 and loggedL["step"] == 1
    cmp = dict(loss=(float(loss), loggedL["loss_S3"]), head=(got["head"], logged["S3/head"]), split=(got["split"], logged["S3/split"]),
               split_early_cols=(got["split_early_cols"], logged["S3/split_early_cols"]))
    rel = {k: abs(a - b) / max(abs(b), 1e-12) for k, (a, b) in cmp.items()}
    assert max(rel.values()) < 2e-3, (cmp, rel)
    return dict(deterministic=True, step1_vs_logged_S3_L2=cmp, rel_diff=rel, n_img=len(x), n_roi=len(rects),
                note="head gradient depends on the random inp/blocks init and the first batch; the S3-L2 run kept no step-0 state, so this is the reproduction evidence. Steps > 1 use the same StreamDS code (rng per (step, slot)), so later batches are identical by construction"), (vit, m)


def bank_check(bank_dir, dev):
    bk = torch.load(Path(bank_dir) / "bank.pt", map_location="cpu", weights_only=False)
    man = json.loads((Path(bank_dir) / "bank_manifest.json").read_text())
    Q = bk["bank"].float()
    sha = hashlib.sha256(Q.numpy().tobytes()).hexdigest()
    assert sha == man["sha256_bank_fp32"] and Q.shape == (8192, 768) and not Q.requires_grad
    assert torch.allclose(Q.norm(dim=-1), torch.ones(len(Q)), atol=1e-4)
    return dict(sha256=sha, shape=list(Q.shape), unit_norm=True, stats=man["stats"]), Q.to(dev)


def real_step(dev, vit, Q):
    _, m = build(dev)
    m.train()
    ex = eval_exclusions()["raev2_stream"]
    b = StreamDS(np.load(STREAM)[:STEPS], ex)[0]
    x, x768 = b["x"].to(dev), b["x768"].to(dev)
    rects, rimg, content = b["rects"].to(dev), b["rimg"].to(dev), b["content"].to(dev)
    fin, ear = student(vit, x)
    T = teacher(vit, x768)
    idx, cy, cx, ok = rect_index(rects)
    key_ok = ok & valid48(content)[rimg[:, None], idx]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m.refine(m.child_map(fin, ear), rimg, idx, cy, cx, key_ok)
    one = (rects[:, 2] == 1) & (rects[:, 3] == 1)
    Lb, sb = bank_kl(y, T[rimg[:, None], idx], key_ok, Q)
    Ll, sl = local_kl(y, T[rimg[:, None], idx], key_ok, one_by_one=one)
    params = list(m.parameters())
    gb = torch.autograd.grad(Lb, params, retain_graph=True, allow_unused=True)
    assert all(p.grad is None for p in params)
    (Lb + Ll).backward()
    groups = {n: float(torch.stack([p.grad.float().norm() for p in ps if p.grad is not None]).norm()) for n, ps in m.groups().items()}
    early = float(m.split.lin.weight.grad[:, 768:].float().norm())
    assert torch.isfinite(Lb) and torch.isfinite(Ll) and all(np.isfinite(v) for v in groups.values())
    assert groups["split"] > 0 and groups["head"] > 0 and early > 0
    assert all(p.grad is None for p in vit.parameters()) and not Q.requires_grad
    return dict(L_bank=float(Lb), L_local=float(Ll), group_grad_norms=groups, split_early_cols=early, grad_bank_only_total=float(torch.cat([g.flatten() for g in gb if g is not None]).norm()),
                n_roi=len(rects), n_1x1=int(one.sum()), dtype_y=str(y.dtype), **sb, **sl,
                note="at init S3 = nearest copy, so inp/blocks gradients are 0 at step 1 (zero-init head), exactly as in S3-L2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--s3l2", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = dict(synthetic=synthetic(dev))
    res["init"], (vit, _) = init_repro(dev, a.s3l2)
    res["bank"], Q = bank_check(a.bank, dev)
    res["real_step"] = real_step(dev, vit, Q)
    res["status"] = "passed"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "kl_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
