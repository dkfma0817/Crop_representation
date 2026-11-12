"""Geometry and variable-length native/decoder gradient tests; CPU safe."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from src.siglip_pm.distill import cell_w
from src.siglip_struct.geometry import native_tokens, box_from_bilinear
from src.siglip_struct.decoder import ROIDecoder
from src.siglip_struct.model import Interaction4
from src.siglip_pm.data import geometry
from src.siglip_pm.bbdata import crop_geo
from src.readout.models import Reader


def run():
    torch.manual_seed(0)
    crops, grid = [], []
    expected = []
    for w,h,box in [(640,480,(31,20,221,207)),(480,640,(1,3,100,44)),(640,480,(2,2,3,3))]:
        crop,full,points=geometry(w,h,box,384,16)
        crops.append(crop_geo(crop,384,16)[0]);grid.append(points)
        expected.append([full.ox+box[0]*full.nw/w,full.oy+box[1]*full.nh/h,full.ox+box[2]*full.nw/w,full.oy+box[3]*full.nh/h])
    import numpy as np
    recovered=box_from_bilinear(dict(geo=torch.tensor(np.stack(crops)),bil_grid=torch.tensor(np.stack(grid))))
    err=float((recovered-torch.tensor(expected)).abs().max());assert err<2e-3,err
    coarse=torch.randn(2,576,768);fine=torch.randn(2,2304,384)
    encoder=SimpleNamespace(Wd=torch.nn.Linear(384,768).requires_grad_(False))
    o=dict(coarse=coarse,mem=dict(fine=fine))
    boxes=torch.tensor([[0.,0.,16.,16.],[20.,30.,120.,160.],[3.,3.,4.,4.]])
    ri=torch.tensor([0,1,0]);rect=torch.tensor([[0.,0.,384.,384.],[0.,20.,384.,360.]])
    tok,geo,types,mask=native_tokens(o,encoder,ri,boxes,rect)
    assert int(mask[0].sum())==5 and int(mask[2].sum())==2
    assert torch.equal((geo[...,2]>0),mask)
    assert set(types[mask].tolist())=={0,1}
    # Geometry exposes native stride exactly; no learned token-type module is
    # inserted into the unchanged R2 architecture.
    assert torch.allclose(torch.exp(geo[0,:5,3]),torch.tensor([1.,2.,2.,2.,2.]))
    b=dict(geo=geo,cov=geo[...,2],box4=torch.tensor([[0.,0.,1.,1.]]).repeat(3,1),lv=torch.tensor([0,1,0]),img_index=torch.arange(3),side=384)
    reader=Reader('r2',layers=('Q',));out=reader({'Q':tok},b)['logits']
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reader.parameters())
    dec=ROIDecoder();z=dec(o,encoder,ri,boxes,rect)
    assert z.shape==(3,16,768)
    torch.nn.functional.cosine_similarity(z.mean(1),torch.randn(3,768),dim=-1).mean().backward()
    bad=[n for n,p in dec.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or not bool(p.grad.abs().sum()>0)]
    assert not bad,bad
    inter=Interaction4();c=torch.randn(1,2,2,768,requires_grad=True);f=torch.randn(1,8,8,384,requires_grad=True)
    nc,nf,_=inter(c,f);(nc.square().mean()+nf.square().mean()).backward()
    assert all(p.grad is not None and bool(p.grad.abs().sum()>0) for p in inter.parameters())
    # Fractional area sums agree for physical4px and16px cells.
    area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    assert torch.allclose(cell_w(boxes,96,4).sum(1)*16,area)
    return dict(status='passed',ROI_affine_max_error_px=err,native_cell_counts=mask.sum(1).tolist(),decoder_tokens=16,
                native_R2_all_gradients=True,decoder_all_gradients=True,interaction4_all_gradients=True,fractional_coverage_4px=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    result=run();Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
