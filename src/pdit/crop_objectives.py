"""Compact crop-learning heads inspired by MAE, I-JEPA and PixelDiT.

These are ROI-conditional adaptations, not reproductions of the original papers.
The common CoarseFine encoder is retained for inference; all heads are removed.
PixelDiT-style head has patch-level conditional DiT blocks and a local compact
pixel pathway with patch-conditioned AdaLN. MAE uses a dense, pixel-masked full
encoder rather than dropping visible tokens. JEPA predicts EMA crop-view targets.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.hcf.model import sincos_2d


def patchify(x,p=16):
    return F.unfold(x,kernel_size=p,stride=p).transpose(1,2)


def unpatchify(x,p=16):
    g=math.isqrt(x.shape[1]);assert g*g==x.shape[1]
    return F.fold(x.transpose(1,2),output_size=(g*p,g*p),kernel_size=p,stride=p)


def compact_pixels(x):
    """384 RGB -> 576 patches x 16 groups x (4*4*RGB). Exact, no pooling."""
    r,c,h,w=x.shape;assert c==3 and h==w and h%16==0
    g=h//16
    fine=patchify(x,4).reshape(r,g,4,g,4,48)
    return fine.permute(0,1,3,2,4,5).reshape(r,g*g,16,48)


def uncompact_pixels(x):
    r,n,k,c=x.shape;g=math.isqrt(n);assert n==g*g and k==16 and c==48
    tokens=x.reshape(r,g,g,4,4,48).permute(0,1,3,2,4,5).reshape(r,g*4*g*4,48)
    return unpatchify(tokens,4)


def block_mask(x,step,ratio=.75):
    """Mask before both embeddings; fixed 4x4 coarse-patch blocks, same per step."""
    generator=torch.Generator(device=x.device).manual_seed(100000+step)
    n=x.shape[0];scores=torch.rand(n,36,device=x.device,generator=generator)
    ids=scores.argsort(1)[:,:round(36*ratio)]
    small=torch.zeros(n,36,device=x.device);small.scatter_(1,ids,1.)
    mask=small.reshape(n,6,6).repeat_interleave(4,1).repeat_interleave(4,2)
    pixels=mask.repeat_interleave(16,1).repeat_interleave(16,2)
    return x*(1-pixels[:,None]),mask


def coarse_mask_on_crop(mask,b):
    return F.grid_sample(mask[b['aux_ri'],None],b['aux_grid'].reshape(-1,24,24,2),mode='nearest',padding_mode='zeros',align_corners=False).flatten(1)


def roi_tokens(model,o,b):
    def sample(tokens):
        g=math.isqrt(tokens.shape[1]);assert g*g==tokens.shape[1]
        maps=tokens.transpose(1,2).reshape(tokens.shape[0],-1,g,g)[b['aux_ri']]
        return F.grid_sample(maps.float(),b['aux_grid'].reshape(-1,24,24,2),mode='bilinear',padding_mode='border',align_corners=False).flatten(2).transpose(1,2)
    a=sample(o['coarse']);mem={k:sample(v) for k,v in o['mem'].items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):q=model.fuse(a,mem)
    return q.float()


def pooled_grid(tokens,cov,g=8):
    x=tokens.transpose(1,2).reshape(-1,768,24,24);w=cov.reshape(-1,1,24,24)
    den=F.adaptive_avg_pool2d(w,g)
    value=F.adaptive_avg_pool2d(x*w,g)/den.clamp_min(1e-12)
    return value.flatten(2).transpose(1,2),den.flatten(1)


class ConditionalBlock(nn.Module):
    def __init__(self,dim=384,heads=6):
        super().__init__()
        self.n1=nn.LayerNorm(dim);self.n2=nn.LayerNorm(dim);self.nm=nn.LayerNorm(dim);self.n3=nn.LayerNorm(dim)
        self.self_attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.cross_attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
        self.mod=nn.Sequential(nn.SiLU(),nn.Linear(dim,6*dim))
        nn.init.zeros_(self.mod[-1].weight);nn.init.zeros_(self.mod[-1].bias)

    def forward(self,x,memory,valid,time=None):
        if time is None:
            a=self.n1(x);x=x+self.self_attn(a,a,a,need_weights=False)[0]
            x=x+self.cross_attn(self.n2(x),self.nm(memory),self.nm(memory),key_padding_mask=~valid,need_weights=False)[0]
            return x+self.ffn(self.n3(x))
        s1,c1,g1,s2,c2,g2=self.mod(time).chunk(6,-1)
        a=self.n1(x)*(1+c1[:,None])+s1[:,None]
        x=x+g1.sigmoid()[:,None]*self.self_attn(a,a,a,need_weights=False)[0]
        x=x+self.cross_attn(self.n2(x),self.nm(memory),self.nm(memory),key_padding_mask=~valid,need_weights=False)[0]
        a=self.n3(x)*(1+c2[:,None])+s2[:,None]
        return x+g2.sigmoid()[:,None]*self.ffn(a)


class CropPredictor(nn.Module):
    """Same width/depth/output size for masked-pixel and EMA-feature prediction."""
    def __init__(self,dim=384,depth=2):
        super().__init__();self.dim=dim
        self.memory=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,dim))
        self.query=nn.Parameter(torch.randn(1,1,dim)*.02)
        self.blocks=nn.ModuleList([ConditionalBlock(dim) for _ in range(depth)])
        self.output=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,768))

    def forward(self,memory,cov,grid):
        mg=math.isqrt(memory.shape[1]);m=self.memory(memory)+sincos_2d(mg,mg,self.dim,memory.device).reshape(1,mg*mg,self.dim)
        x=self.query+sincos_2d(grid,grid,self.dim,memory.device).reshape(1,grid*grid,self.dim)
        x=x.expand(memory.shape[0],-1,-1);valid=cov>0
        assert valid.any(1).all()
        for block in self.blocks:x=block(x,m,valid)
        return self.output(x)


class CompactPixelBlock(nn.Module):
    def __init__(self,patch_dim=384,dim=64):
        super().__init__();self.norm1=nn.LayerNorm(dim,elementwise_affine=False);self.norm2=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim,4,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
        self.mod=nn.Linear(patch_dim,2*dim);self.condition=nn.Linear(patch_dim,dim)

    def forward(self,x,patch):
        r,n,k,d=x.shape
        x=x+self.condition(patch)[:,:,None]
        shift,scale=self.mod(patch).chunk(2,-1)
        y=self.norm1(x)*(1+scale[:,:,None])+shift[:,:,None]
        y=y.reshape(r*n,k,d);y=self.attn(y,y,y,need_weights=False)[0].reshape(r,n,k,d)
        x=x+y
        return x+self.ffn(self.norm2(x))


class CropPixelDiT(nn.Module):
    """Patch DiT -> compact local pixel transformer -> RGB velocity. No VAE."""
    def __init__(self,dim=384,depth=2):
        super().__init__();self.dim=dim
        self.memory=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,dim))
        self.patch=nn.Linear(768,dim);self.time=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.blocks=nn.ModuleList([ConditionalBlock(dim) for _ in range(depth)])
        self.pixel_embed=nn.Linear(48,64);self.pixel=CompactPixelBlock(dim,64);self.output=nn.Linear(64,48)

    def forward(self,noisy,t,memory,cov):
        r=memory.shape[0];mg=math.isqrt(memory.shape[1]);m=self.memory(memory)+sincos_2d(mg,mg,self.dim,memory.device).reshape(1,mg*mg,self.dim)
        freq=torch.exp(-math.log(10000)*torch.arange(self.dim//2,device=t.device)/(self.dim//2-1))
        angle=t[:,None]*freq[None]*1000
        time=self.time(torch.cat([angle.cos(),angle.sin()],-1))
        x=self.patch(patchify(noisy))+sincos_2d(24,24,self.dim,noisy.device).reshape(1,576,self.dim)
        valid=cov>0;assert valid.any(1).all()
        for block in self.blocks:x=block(x,m,valid,time)
        p=self.pixel_embed(compact_pixels(noisy))+sincos_2d(4,4,64,noisy.device).reshape(1,1,16,64)
        return uncompact_pixels(self.output(self.pixel(p,x)))
