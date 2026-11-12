"""Width-parametric copy of the existing R2 reader (src.readout.models.Reader kind 'r2', layers=[12], whose pieces
RR.CrossBlock / RR.Heads hard-code D = 768). Structure, module order (= parameter creation / RNG order), names, init,
masking and heads are unchanged; only the width D follows the encoder feature dim (1024 for CLIP ViT-L).
`python -m src.maskemb.reader` asserts that R2(D=768) == Reader('r2', [12]) bit for bit (same seed -> same state_dict,
same logits on a real PACO batch) and prints the parameter counts at D = 768 / 1024.

  geo = [rel_x, rel_y, cov, log(bw/P), log(bh/P)] -> geo_mlp (5 -> D -> D); box4 -> box_mlp (4 -> D -> D)
  q0 = learned query + box_mlp(box4); kv = LN(token + geo_mlp(geo)); 2 x CrossBlock (8 heads, MLP x2, pre-LN), attention
  restricted to cov > 0 (the ROI cells); LN -> level head Linear(D -> 59) (object / part).
"""
import math

import torch
import torch.nn as nn

LEVELS = ("object", "part")


class CrossBlock(nn.Module):
    def __init__(self, D, heads=8, mlp_ratio=2):
        super().__init__()
        self.D = D
        self.ln_q, self.ln_m = nn.LayerNorm(D), nn.LayerNorm(D)
        self.q, self.kv, self.o = nn.Linear(D, D), nn.Linear(D, 2 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, mlp_ratio * D), nn.GELU(), nn.Linear(mlp_ratio * D, D))
        self.h = heads

    def attend(self, q, kv, mask):
        D = self.D
        n, T, _ = kv.shape
        Q = self.q(self.ln_q(q)).view(n, 1, self.h, D // self.h).transpose(1, 2)
        K, V = self.kv(kv).view(n, T, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = (Q @ K.transpose(-1, -2)) / math.sqrt(D // self.h)
        if mask is not None:
            att = att.masked_fill(~mask[:, None, None, :], float("-inf"))
        att = att.softmax(-1)
        out = (att @ V).transpose(1, 2).reshape(n, 1, D)
        return self.o(out), att

    def forward(self, q, kv, mask):
        a, att = self.attend(q, kv, mask)
        q = q + a
        q = q + self.mlp(self.ln_m(q))
        return q, att


class Heads(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(D, 59) for _ in LEVELS])

    def forward(self, f, lv):
        o = torch.stack([h(f).float() for h in self.heads], 1)
        return o[torch.arange(len(f), device=f.device), lv]


class _Fuse(nn.Module):
    """= src.readout.models.Fuse with one layer (no parameters)."""

    def __init__(self, layers=(12,)):
        super().__init__()
        self.layers = tuple(layers)

    def forward(self, feats):
        return feats[self.layers[0]]


class R2(nn.Module):
    kind = "r2"

    def __init__(self, D, heads=8, mlp_ratio=2):
        super().__init__()
        self.D = D
        self.fuse = _Fuse((12,))
        self.geo_mlp = nn.Sequential(nn.Linear(5, D), nn.GELU(), nn.Linear(D, D))
        self.box_mlp = nn.Sequential(nn.Linear(4, D), nn.GELU(), nn.Linear(D, D))
        self.query = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.ln_kv = nn.LayerNorm(D)
        self.blocks = nn.ModuleList([CrossBlock(D, heads, mlp_ratio) for _ in range(2)])
        self.ln_out, self.heads = nn.LayerNorm(D), Heads(D)

    def forward(self, feats, b, return_att=False):
        tok = self.fuse(feats)[b["img_index"]]
        mask = b["cov"] > 0
        kv = self.ln_kv(tok + self.geo_mlp(b["geo"]))
        q = self.query.expand(len(b["geo"]), -1, -1) + self.box_mlp(b["box4"])[:, None]
        atts = []
        for blk in self.blocks:
            q, att = blk(q, kv, mask)
            atts.append(att)
        out = dict(logits=self.heads(self.ln_out(q[:, 0]), b["lv"]), extra={})
        if return_att:
            out["att"] = atts
        return out


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build(D, seed=0):
    torch.manual_seed(1000 * seed + 7)           # = src.g1x.probe.build_readers
    return R2(D)


def check():
    import numpy as np
    import src.readout.common as RC
    from src.readout.models import Reader
    from src.g1x.data import PacoDS
    torch.manual_seed(7)
    ref = Reader("r2", layers=(12,))
    new = build(768)
    sr, sn = ref.state_dict(), new.state_dict()
    assert list(sr) == list(sn), (set(sr) ^ set(sn))
    assert all(torch.equal(sr[k], sn[k]) for k in sr), "init differs"
    dtr, Ytr = RC.load_data(splits=("train",), limit=8)["train"]
    b = PacoDS(dtr, Ytr, False, "full", 384, 16)[list(range(4))]
    tok = torch.randn(4, 576, 768)
    with torch.no_grad():
        a, c = ref({12: tok}, b)["logits"], new({12: tok}, b)["logits"]
    assert torch.equal(a, c), (a - c).abs().max()
    pr = RC.RR.reader_param_groups(new, 3e-4, 1e-3, 0.05)
    pr0 = RC.RR.reader_param_groups(ref, 3e-4, 1e-3, 0.05)
    assert [len(g["params"]) for g in pr] == [len(g["params"]) for g in pr0]
    r1024 = build(1024)
    out = dict(equal_to_existing_R2_at_768=True, n_rois_checked=int(len(b["rows"])),
               params_768=dict(total=n_params(new), reader_only=n_params(new) - n_params(new.heads), heads=n_params(new.heads)),
               params_1024=dict(total=n_params(r1024), reader_only=n_params(r1024) - n_params(r1024.heads), heads=n_params(r1024.heads)))
    print(out)
    return out


if __name__ == "__main__":
    check()
