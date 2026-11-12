"""SigLIP2-B/16 fixed384 adapters. Reuse the P/M pilot fine path, interactions and fusion.
No prefix tokens, no RoPE, no CLS. Global embedding = the native pretrained MAP head.
"""
import os
import torch
from torch import nn
from transformers import SiglipVisionModel
from src.hcf.model import (FineBlock, Interaction, CoarseFine as OriginalP,
                           MultiLayer as OriginalM, sincos_2d, ML, SCALARS, checksum, n_params)
MODEL_ID='google/siglip2-base-patch16-384'
REVISION='f775b65a79762255128c981547af89addcfe0f88'
MODEL_PATH=os.environ.get('SIGLIP2_MODEL_PATH', os.path.expanduser(f'~/.cache/huggingface/hub/models--google--siglip2-base-patch16-384/snapshots/{REVISION}'))

class Vision(nn.Module):
    num_prefix_tokens=0
    def __init__(self):
        super().__init__()
        v=SiglipVisionModel.from_pretrained(MODEL_PATH,attn_implementation='sdpa').vision_model
        self.patch_embed=v.embeddings
        self.blocks=v.encoder.layers
        self.norm=v.post_layernorm
        self.head=v.head
        self.config=v.config
        assert (v.config.hidden_size,v.config.num_hidden_layers,v.config.patch_size,v.config.image_size)==(768,12,16,384)
    def forward_features(self,x):
        t=self.patch_embed(x)
        for block in self.blocks: t=block(t,attention_mask=None)[0]
        return self.norm(t)
    def forward_head(self,t,pre_logits=False): return self.head(t)
    def forward(self,x): return self.forward_head(self.forward_features(x))

class CoarseFine(OriginalP):
    def forward_all(self,x,collect=False):
        t=self.vit.patch_embed(x)
        B,N,D=t.shape
        H=W=24
        f=self.fine_embed(x).permute(0,2,3,1)
        f=f+sincos_2d(48,48,384,x.device)[None]
        diag={}
        for st in range(4):
            for i in range(st*3,st*3+3): t=self.vit.blocks[i](t,attention_mask=None)[0]
            f=self.fine_blocks[st](f)
            if collect:
                c,f,r=self.inter[st](t.reshape(B,H,W,D),f)
            else:
                # Same simultaneous residual interaction; avoid diagnostic .item() GPU syncs.
                inter=self.inter[st];c=t.reshape(B,H,W,D)
                fp=torch.nn.functional.avg_pool2d(f.permute(0,3,1,2),2).permute(0,2,3,1)
                cu=c.repeat_interleave(2,1).repeat_interleave(2,2)
                rc=inter.s_f2c*inter.f2c(fp);rf=inter.s_c2f*inter.c2f(cu)
                c,f=c+rc,f+rf
            t=c.reshape(B,N,D)
            if collect: diag.update({f'stage{st+1}_res_c':r['c'],f'stage{st+1}_res_f':r['f']})
        t=self.vit.norm(t); f=self.fine_norm(f)
        if collect: self.diag=diag
        return dict(cls=self.vit.forward_head(t),coarse=t,H=24,W=24,
                    mem=dict(fine=f.reshape(B,2304,384)),grids=dict(fine=2))
    def forward_features(self,x): return self.forward_all(x)['coarse']

class MultiLayer(OriginalM):
    def forward_all(self,x,collect=False):
        t=self.vit.patch_embed(x);raw={}
        for i,b in enumerate(self.vit.blocks):
            t=b(t,attention_mask=None)[0]
            if i+1 in self.layers[:-1]:raw[i+1]=t
        t=self.vit.norm(t)
        mem={f'l{l}':self.ln[str(l)](raw[l]) for l in raw}
        return dict(cls=self.vit.forward_head(t),coarse=t,H=24,W=24,mem=mem,grids={k:1 for k in mem},raw=raw)
    def forward_features(self,x):return self.forward_all(x)['coarse']

def load_vit(dev,ckpt=None):
    assert ckpt is None, 'G1/other initialization is forbidden'
    return Vision().to(dev)

def build_student(kind,dev,ckpt=None,s0=.1):
    assert ckpt is None
    torch.manual_seed(0)
    vit=load_vit(dev)
    torch.manual_seed(0)
    return (CoarseFine(vit,s0) if kind=='P' else MultiLayer(vit,s0)).to(dev)

def load_hcf(path,dev):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['backbone']==MODEL_ID
    m=build_student(ck['cond'],dev,s0=ck.get('s0',.1))
    m.load_state_dict(ck['model'],strict=True);m.eval().requires_grad_(False)
    return m,ck

@torch.no_grad()
def run_enc(vit,x,chunk=96):
    ts,gs=[],[]
    for i in range(0,len(x),chunk):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            t=vit.forward_features(x[i:i+chunk]);g=vit.forward_head(t)
        ts.append(t.float());gs.append(g.float())
    return torch.cat(ts),torch.cat(gs)
