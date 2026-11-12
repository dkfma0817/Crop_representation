"""G1-cross with Bilinear-initialized decoder queries (2026-11-08).

= src.g1clip.model.ROIDecoderD (G1-cross decoder: key geometry / k_mlp / ln_k / 2 cross blocks / ln_out / out, created
under the same seed 7) with ONE change in the query:
  G1-cross   q0 = q_mlp(position features) + q_grid                      (learned "mask" query, no image content)
  G1-bq      q0 = LN_qin(B + q_mlp(position features)) + q_grid
             B = bilinear_roi(F, ROI): the TRAINABLE student encoder's full-image tokens F bilinearly sampled at the C0
             crop-grid cell centres mapped to the full canvas (src.decstudy.model.bilinear_roi; aspect-preserving
             letterbox alignment verified in src.decstudy.checks). Not detached -> L_detail reaches the encoder through
             the queries AND the keys.
Output unchanged: Z = out(ln_out(q)) (no residual skip, default init). Padding (cells outside the crop content rect) is
excluded by the loss weights (crop coverage) exactly as in G1-cross; keys keep the full-canvas valid mask.
The decoder is used only during training; PACO evaluation reads the encoder only.
"""
import torch.nn as nn

from src.decstudy.model import bilinear_roi
from src.g1x.model import fourier
from src.g1clip.model import ROIDecoderD


class BQDecoder(ROIDecoderD):
    def __init__(self, S, P, D, depth=2, heads=8):
        super().__init__(S, P, D, depth, heads)
        self.ln_qin = nn.LayerNorm(D)

    def pos_feat(self, box_c, qrel):
        import torch
        X0, Y0, X1, Y1 = box_c.unbind(-1)
        bw, bh = (X1 - X0).clamp_min(1e-3), (Y1 - Y0).clamp_min(1e-3)
        ax = (X0[:, None] + qrel[..., 0] * bw[:, None]) / self.S
        ay = (Y0[:, None] + qrel[..., 1] * bh[:, None]) / self.S
        size = torch.stack([torch.log(bw / self.P), torch.log(bh / self.P), torch.log(bw / bh)], -1)[:, None].expand(-1, qrel.shape[1], -1)
        return torch.cat([fourier(qrel, self.NF_Q, 6.0), fourier(torch.stack([ax, ay], -1), self.NF_Q, 6.0), size], -1)

    def forward(self, full_tok, roi_img, box_c, qrel, fvalid, chunk=64):
        import torch
        Bf = bilinear_roi(full_tok, roi_img, box_c, qrel, self.S, chunk)
        outs = []
        for i in range(0, len(roi_img), chunk):
            sl = slice(i, i + chunk)
            bc, ri = box_c[sl].float(), roi_img[sl]
            q = self.ln_qin(Bf[sl] + self.q_mlp(self.pos_feat(bc, qrel[sl].float()))) + self.q_grid
            kg, _ = self.key_geo(bc)
            kv = self.ln_k(full_tok[ri] + self.k_mlp(kg))
            m = fvalid[ri].bool()
            for blk in self.blocks:
                q = blk(q, kv, m)
            outs.append(self.out(self.ln_out(q)))
        return torch.cat(outs)
