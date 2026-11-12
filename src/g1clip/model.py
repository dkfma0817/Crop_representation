"""Width-parametric copy of the G1-cross ROI cross-attention decoder (src.g1x.model.ROIDecoder hard-codes D = 768).
Everything else (queries from ROI-relative + absolute Fourier coordinates + size, learned 576-cell grid embedding, keys =
LN(token + MLP(key geometry)), 2 pre-LN cross blocks with 8 heads and FFN x4, SDPA, padding keys masked, output Linear)
is unchanged, in the same module order (= the same RNG consumption / parameter names). Only D follows the encoder width
(1024 for CLIP ViT-L) and S / P are passed through (336 / 14, 24 x 24 grid as before).
`python -m src.g1clip.model` asserts ROIDecoderD(D=768, 384, 16) == src.g1x.model.ROIDecoder bit for bit (seeded init and
outputs) and prints the parameter count at D = 1024.
"""
import torch
import torch.nn as nn

from src.g1x.model import G, XBlock, fourier, mlp, squash


class ROIDecoderD(nn.Module):
    NF_Q, NF_K = 16, 8

    def __init__(self, S, P, D, depth=2, heads=8):
        super().__init__()
        self.S, self.P, self.D = S, P, D
        dq = 2 * (2 * self.NF_Q + 1) * 2 + 3
        dk = 2 * (2 * self.NF_K + 1) * 2 + 4
        self.q_mlp, self.k_mlp = mlp(dq, D), mlp(dk, D)
        self.q_grid = nn.Parameter(torch.zeros(1, G * G, D))
        nn.init.trunc_normal_(self.q_grid, std=0.02)
        self.ln_k = nn.LayerNorm(D)
        self.blocks = nn.ModuleList([XBlock(D, heads) for _ in range(depth)])
        self.ln_out, self.out = nn.LayerNorm(D), nn.Linear(D, D)
        ci, cj = torch.div(torch.arange(G * G), G, rounding_mode="floor"), torch.arange(G * G) % G
        self.register_buffer("cell_c", torch.stack([(cj + 0.5) * P, (ci + 0.5) * P], -1).float(), persistent=False)
        e = torch.arange(G + 1).float() * P
        self.register_buffer("edges", e, persistent=False)

    def queries(self, box_c, qrel):
        X0, Y0, X1, Y1 = box_c.unbind(-1)
        bw, bh = (X1 - X0).clamp_min(1e-3), (Y1 - Y0).clamp_min(1e-3)
        ax = (X0[:, None] + qrel[..., 0] * bw[:, None]) / self.S
        ay = (Y0[:, None] + qrel[..., 1] * bh[:, None]) / self.S
        size = torch.stack([torch.log(bw / self.P), torch.log(bh / self.P), torch.log(bw / bh)], -1)[:, None].expand(-1, qrel.shape[1], -1)
        f = torch.cat([fourier(qrel, self.NF_Q, 6.0), fourier(torch.stack([ax, ay], -1), self.NF_Q, 6.0), size], -1)
        return self.q_mlp(f) + self.q_grid

    def key_geo(self, box_c):
        X0, Y0, X1, Y1 = box_c.unbind(-1)
        bw, bh = (X1 - X0).clamp_min(1e-3), (Y1 - Y0).clamp_min(1e-3)
        c = self.cell_c
        absxy = (c / self.S)[None].expand(len(box_c), -1, -1)
        rx = (c[None, :, 0] - (X0 + X1)[:, None] / 2) / bw[:, None]
        ry = (c[None, :, 1] - (Y0 + Y1)[:, None] / 2) / bh[:, None]
        e = self.edges
        ox = (torch.minimum(e[None, 1:], X1[:, None]) - torch.maximum(e[None, :-1], X0[:, None])).clamp_min(0) / self.P
        oy = (torch.minimum(e[None, 1:], Y1[:, None]) - torch.maximum(e[None, :-1], Y0[:, None])).clamp_min(0) / self.P
        ov = (oy[:, :, None] * ox[:, None, :]).flatten(1)
        size = torch.stack([torch.log(bw / self.P), torch.log(bh / self.P)], -1)[:, None].expand(-1, G * G, -1)
        f = torch.cat([fourier(absxy, self.NF_K, 5.0), fourier(squash(torch.stack([rx, ry], -1)), self.NF_K, 3.0),
                       ov[..., None], (ov > 0).float()[..., None], size], -1)
        return f, ov

    def forward(self, full_tok, roi_img, box_c, qrel, fvalid, chunk=64):
        outs = []
        for i in range(0, len(roi_img), chunk):
            sl = slice(i, i + chunk)
            bc, ri = box_c[sl].float(), roi_img[sl]
            q = self.queries(bc, qrel[sl].float())
            kg, _ = self.key_geo(bc)
            kv = self.ln_k(full_tok[ri] + self.k_mlp(kg))
            m = fvalid[ri].bool()
            for blk in self.blocks:
                q = blk(q, kv, m)
            outs.append(self.out(self.ln_out(q)))
        return torch.cat(outs)


def check():
    from src.g1x.model import ROIDecoder
    torch.manual_seed(7)
    a = ROIDecoder(384, 16)
    torch.manual_seed(7)
    b = ROIDecoderD(384, 16, 768)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    tok = torch.randn(2, 576, 768)
    ri = torch.tensor([0, 1, 1])
    box = torch.tensor([[10., 20., 200., 300.], [0., 0., 50., 60.], [100., 100., 384., 384.]])
    qrel = torch.rand(3, 576, 2) * 1.2 - 0.1
    fv = torch.ones(2, 576, dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(a(tok, ri, box, qrel, fv), b(tok, ri, box, qrel, fv))
    torch.manual_seed(7)
    c = ROIDecoderD(336, 14, 1024)
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    out = dict(equal_to_g1x_decoder_at_768=True, params_768=n(b), params_1024=n(c))
    print(out)
    return out


if __name__ == "__main__":
    check()
