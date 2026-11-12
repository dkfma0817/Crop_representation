import torch
from torch import nn
from torch.nn import functional as F

from src.hcf.model import FineBlock, sincos_2d
from src.siglip_pm.model import CoarseFine, load_vit, load_hcf, MODEL_ID


class Interaction4(nn.Module):
    def __init__(self, s0=.1):
        super().__init__()
        self.f2c, self.c2f = nn.Linear(384, 768), nn.Linear(768, 384)
        self.s_f2c = nn.Parameter(torch.tensor(float(s0)))
        self.s_c2f = nn.Parameter(torch.tensor(float(s0)))

    def forward(self, coarse, fine, collect=False):
        assert fine.shape[1:3] == (coarse.shape[1]*4, coarse.shape[2]*4)
        pooled = F.avg_pool2d(fine.permute(0, 3, 1, 2), 4).permute(0, 2, 3, 1)
        broadcast = coarse.repeat_interleave(4, 1).repeat_interleave(4, 2)
        dc = self.s_f2c*self.f2c(pooled)
        df = self.s_c2f*self.c2f(broadcast)
        ratios = {}
        if collect:
            ratios = dict(c=float(dc.norm(dim=-1).mean()/coarse.norm(dim=-1).mean().clamp_min(1e-8)),
                          f=float(df.norm(dim=-1).mean()/fine.norm(dim=-1).mean().clamp_min(1e-8)))
        return coarse+dc, fine+df, ratios


class P4(CoarseFine):
    """Same four stages and fusion; stride4, window16/shift8 =64px/32px."""
    def __init__(self, vit, s0=.1):
        nn.Module.__init__(self)
        self.vit, self.npt = vit, 0
        self.fine_embed = nn.Conv2d(3, 384, 4, 4)
        self.fine_blocks = nn.ModuleList([FineBlock(384, 6, 16, 0 if i%2==0 else 8) for i in range(4)])
        self.inter = nn.ModuleList([Interaction4(s0) for _ in range(4)])
        self.fine_norm, self.Wd = nn.LayerNorm(384), nn.Linear(384, 768)
        self.s_d = nn.Parameter(torch.tensor(float(s0)))
        self.diag = {}

    def forward_all(self, x, collect=False):
        assert x.shape[-2:] == (384, 384), 'Only384 inputs are authorized'
        t = self.vit.patch_embed(x)
        B = len(x)
        f = self.fine_embed(x).permute(0, 2, 3, 1)
        assert f.shape[1:] == (96, 96, 384)
        f = f+sincos_2d(96, 96, 384, x.device)[None]
        diagnostic = {}
        for stage in range(4):
            for i in range(stage*3, stage*3+3):
                t = self.vit.blocks[i](t, attention_mask=None)[0]
            f = self.fine_blocks[stage](f)
            c, f, ratios = self.inter[stage](t.reshape(B, 24, 24, 768), f, collect)
            t = c.reshape(B, 576, 768)
            if collect:
                diagnostic.update({f'stage{stage+1}_res_c': ratios['c'], f'stage{stage+1}_res_f': ratios['f']})
        t, f = self.vit.norm(t), self.fine_norm(f)
        if collect:
            self.diag = diagnostic
        return dict(cls=self.vit.forward_head(t), coarse=t, H=24, W=24,
                    mem=dict(fine=f.reshape(B, 9216, 384)), grids=dict(fine=4))


def build_p4(device):
    torch.manual_seed(0)
    vit = load_vit(device)
    torch.manual_seed(0)
    return P4(vit).to(device)


def load_p4(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['backbone']==MODEL_ID and ck['architecture']=='P4' and ck['kind']=='ema'
    model = build_p4(device)
    model.load_state_dict(ck['model'], strict=True)
    return model.eval().requires_grad_(False), ck


def load_p8(path, device):
    model, ck = load_hcf(path, device)
    assert ck['cond']=='P' and ck['step']==12000 and ck['kind']=='ema'
    return model.eval().requires_grad_(False)
