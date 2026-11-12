"""Layer diagnostic (b) and feature side of (c), held-out COCO dev images only (no PACO, no training).

(b) same-layer comparison: for L in {0 (patch embed), 1, 3, 6, 9, 12 (final norm)}: the 384 student layer-L patch tokens upsampled x2
    (bilinear) vs the 768 teacher layer-L patch tokens (both parameter-free LayerNorm per token for L < 12, final norm for 12).
    Metrics over child cells fully inside the content: normalised MSE (MSE / teacher per-dim variance), cosine, within-parent R^2
    (fraction of the teacher's within-parent residual energy that the upsampled student reproduces), parent-mean normalised MSE.
(c) rerun feature side: layer-L 384 tokens x2 -> remaining frozen blocks at 48 x 48 -> final norm, vs the teacher final tokens
    (same metrics), L in {0, 3, 6, 9}; L = 12 row = B0 (final x2).
The PACO side of (a) / (c) is the R2 probes tapL* / rerunL* of the queue.

  python -m src.tsplit.diag_layers --out <run>/diag_layers
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.hcf.distill import cell_w
from src.tsplit.features import PREFIX, RoiEncoder, up2
from src.tsplit.model import G, G2
from src.tsplit.train import dev_set, load_pair

LAYERS = (0, 1, 3, 6, 9, 12)
RERUN = (0, 3, 6, 9)


def metrics(S, T, m):
    """S, T (B, 48, 48, D); m (B, 48, 48) bool."""
    var = T[m].var(0).mean()
    mse = (S - T).pow(2).mean(-1)
    cos = F.cosine_similarity(S, T, dim=-1)
    B = S.shape[0]
    s4, t4 = S.view(B, G, 2, G, 2, -1), T.view(B, G, 2, G, 2, -1)
    pm = m.view(B, G, 2, G, 2).all(4).all(2)
    rs, rt = s4 - s4.mean((2, 4), keepdim=True), t4 - t4.mean((2, 4), keepdim=True)
    res = (rs - rt).pow(2).mean(-1).sum((2, 4))[pm].sum()
    en = rt.pow(2).mean(-1).sum((2, 4))[pm].sum()
    pmse = (s4.mean((2, 4)) - t4.mean((2, 4))).pow(2).mean(-1)[pm].mean()
    return dict(nmse=float(mse[m].mean() / var), cos=float(cos[m].mean()), within_parent_r2=float(1 - res / en), parent_mean_nmse=float(pmse / var),
                teacher_within_parent_share=float(en / 4 / pm.sum() / var))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=192)
    a = ap.parse_args()
    dev = torch.device("cuda")
    enc = RoiEncoder("F384", dev)
    vit = enc.vit
    acc = {}
    ids = dev_set()[:a.n]
    for i in range(0, len(ids), 16):
        xs, xt, c = [], [], []
        for iid in ids[i:i + 16]:
            lb, s, t = load_pair(iid, False)
            xs.append(s)
            xt.append(t)
            c.append([lb.ox, lb.oy, lb.ox + lb.nw, lb.oy + lb.nh])
        x, x768 = torch.stack(xs).to(dev), torch.stack(xt).to(dev)
        m = (cell_w(torch.tensor(c, dtype=torch.float32), G2, 8) >= 1 - 1e-6).view(-1, G2, G2).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs, ht = enc.hidden(x), enc.hidden(x768)
        final_t = vit.norm(ht[-1])[:, PREFIX:].float()
        for L in LAYERS:
            if L == 12:
                s, t = vit.norm(hs[-1])[:, PREFIX:].float(), final_t
            else:
                s, t = F.layer_norm(hs[L][:, PREFIX:].float(), (768,)), F.layer_norm(ht[L][:, PREFIX:].float(), (768,))
            acc.setdefault(f"b_same_layer_L{L}", []).append(metrics(up2(s).view(-1, G2, G2, 768), t.view(-1, G2, G2, 768), m))
        for L in RERUN:
            r = enc.rerun_final(x, L)
            acc.setdefault(f"c_rerun_L{L}_vs_teacher_final", []).append(metrics(r.view(-1, G2, G2, 768), final_t.view(-1, G2, G2, 768), m))
        acc.setdefault("c_rerun_L12_vs_teacher_final(=B0)", []).append(metrics(up2(vit.norm(hs[-1])[:, PREFIX:].float()).view(-1, G2, G2, 768), final_t.view(-1, G2, G2, 768), m))
    out = {k: {mk: sum(d[mk] for d in v) / len(v) for mk in v[0]} for k, v in acc.items()}
    res = dict(n_images=len(ids), metrics=out, note="nmse = MSE / mean per-dim teacher variance over valid cells; within_parent_r2 = 1 - sum ||r_S - r_T||^2 / sum ||r_T||^2 with r = child - parent mean; teacher_within_parent_share = within-parent residual energy / total teacher variance; batch means over 16-image chunks")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "diag_layers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
