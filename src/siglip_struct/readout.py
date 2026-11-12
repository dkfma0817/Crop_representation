"""Adapters to the unchanged seed0 R2 learner/inference; representation only."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.siglip_struct.model import load_p4, load_p8
from src.siglip_struct.decoder import ROIDecoder
from src.siglip_struct.data import BBImageDS
from src.siglip_struct.geometry import native_tokens, query_geometry


class Frozen(nn.Module):
    def __init__(self, job, device):
        super().__init__()
        self.roi_mode = job['cond']
        if self.roi_mode=='P4':
            self.encoder,_=load_p4(job['ckpt'],device)
        else:
            self.encoder=load_p8(job['p8'],device)
        if self.roi_mode=='Decode':
            self.decoder=ROIDecoder().to(device)
            ck=torch.load(job['ckpt'],map_location='cpu',weights_only=False)
            assert ck['architecture']=='Decode' and ck['kind']=='ema' and ck['step']==12000
            self.decoder.load_state_dict(ck['model'],strict=True)
        self.eval().requires_grad_(False)


def load_frozen(job, device):
    return Frozen(job,device)


def align(tokens,g,b,chunk=24):
    maps=tokens.transpose(1,2).reshape(len(tokens),-1,g,g)
    result=[]
    for lo in range(0,len(b['img_index']),chunk):
        sl=slice(lo,lo+chunk)
        result.append(F.grid_sample(maps[b['img_index'][sl]].float(),b['bil_grid'][sl].reshape(-1,24,24,2),mode='bilinear',padding_mode='border',align_corners=False).flatten(2).transpose(1,2))
    return torch.cat(result)


@torch.no_grad()
def encode(model,b,keys,drop_fine=False):
    encoder=model.encoder
    parts=[]
    for lo in range(0,len(b['x']),4):
        with torch.autocast('cuda',dtype=torch.bfloat16):parts.append(encoder.forward_all(b['x'][lo:lo+4]))
    o=dict(coarse=torch.cat([p['coarse'] for p in parts]),mem=dict(fine=torch.cat([p['mem']['fine'] for p in parts])))
    if model.roi_mode=='P4':
        a,f=align(o['coarse'],24,b),align(o['mem']['fine'],96,b)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=encoder.fuse(a,dict(fine=f)).float()
    else:
        with torch.autocast('cuda',dtype=torch.bfloat16):o['fine_projected']=encoder.Wd(o['mem']['fine'])
        if model.roi_mode=='Native':
            z,geo,typ,mask=native_tokens(o,encoder,b['img_index'],b['box_c'],b['valid_rect'])
            b.update(geo=geo,cov=geo[...,2],native_type=typ,native_mask=mask)
            assert torch.equal(b['cov']>0,mask)
            z=z.float()
        else:
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model.decoder(o,encoder,b['img_index'],b['box_c'],b['valid_rect']).float()
            _,geo=query_geometry(b['box_c']);b.update(geo=geo,cov=geo[...,2])
    b['img_index']=torch.arange(len(z),device=z.device)
    return {'Q':z,12:z}


def main():
    import sys
    action=sys.argv.pop(1)
    from src.siglip_extra.resources import root_from_argv,gpu_lease,output_lease,complete
    if action=='probe':
        import src.siglip_pm.probe as base
        base.load_frozen=load_frozen;base.encode=encode;base.BBImageDS=BBImageDS
        ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--out',required=True);args,_=ap.parse_known_args()
        with output_lease(args.out):
            if complete(args.out):return
            with gpu_lease(root_from_argv('--out')):base.main()
    elif action=='infer':
        import src.siglip_pm.infer as base
        base.load_frozen=load_frozen;base.encode=encode;base.BBImageDS=BBImageDS
        with gpu_lease(root_from_argv('--run')):base.main()
    else:raise ValueError(action)


if __name__=='__main__':main()
