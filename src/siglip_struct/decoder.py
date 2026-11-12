import torch
from torch import nn
from src.g1x.model import XBlock
from src.siglip_struct.geometry import native_tokens, query_geometry


class ROIDecoder(nn.Module):
    """16 queries, two768-wide pre-LN cross blocks,8heads, MLPx2."""
    def __init__(self):
        super().__init__()
        self.q_grid = nn.Parameter(torch.empty(1, 16, 768))
        nn.init.trunc_normal_(self.q_grid, std=.02)
        self.q_mlp = nn.Sequential(nn.Linear(8, 768), nn.GELU(), nn.Linear(768, 768))
        self.k_mlp = nn.Sequential(nn.Linear(6, 768), nn.GELU(), nn.Linear(768, 768))
        self.type_embed = nn.Embedding(2, 768)
        self.ln_k, self.ln_out = nn.LayerNorm(768), nn.LayerNorm(768)
        self.blocks = nn.ModuleList([XBlock(768, 8, ratio=2) for _ in range(2)])
        self.out = nn.Linear(768, 768)

    def forward(self, o, encoder, image_index, boxes, valid_rect, chunk=16):
        outs = []
        for lo in range(0, len(boxes), chunk):
            sl = slice(lo, lo+chunk)
            # Context is fixed at25% ROI extent on each side. Content-padding
            # keys are always masked; partial cells retain real coverage.
            tok, geo, typ, mask = native_tokens(o, encoder, image_index[sl], boxes[sl], valid_rect, .25)
            qp, _ = query_geometry(boxes[sl])
            q = self.q_grid+self.q_mlp(qp)
            kg = torch.cat([geo, typ.clamp_min(0)[..., None].float()], -1)
            kv = self.ln_k(tok+self.k_mlp(kg)+self.type_embed(typ.clamp_min(0)))
            for block in self.blocks:
                q = block(q, kv, mask)
            outs.append(self.out(self.ln_out(q)))
        return torch.cat(outs)
