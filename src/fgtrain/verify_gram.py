"""CPU verification of the weighted Gram extension before any GPU is reserved."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import torch
from src.fgtrain.gram import gram_loss,GramData
from src.fgtrain.data import Data
from src.fgtrain.common import RUN,write_json


def verify(out):
    reference=Path('third_party/BRIXEL/src/brixel/dinov3_main/dinov3/loss/gram_loss.py')
    spec=importlib.util.spec_from_file_location('official_gram_reference',reference)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(9)
    s=torch.randn(3,7,11,requires_grad=True);t=torch.randn(3,7,11,requires_grad=True)
    plain=gram_loss(s,t,torch.ones(3,7))[0]
    official=module.GramLoss(apply_norm=True,remove_neg=False)(s,t.detach())
    assert torch.allclose(plain,official,atol=1e-7),'unweighted official equivalence'
    area=torch.tensor([[1,.25,0,1,0,1,.5],[0,0,0,0,0,0,0],[.1,1,1,0,0,1,1]])
    loss,_,_=gram_loss(s,t,area)
    sn=s.detach().numpy().copy();tn=t.detach().numpy().copy()
    sn/=np.linalg.norm(sn,axis=-1,keepdims=True);tn/=np.linalg.norm(tn,axis=-1,keepdims=True)
    expected=[]
    for i in [0,2]:
        w=area[i].numpy();pair=w[:,None]*w[None,:]
        expected.append(float((((sn[i]@sn[i].T-tn[i]@tn[i].T)**2)*pair).sum()/pair.sum()))
    error=abs(float(loss)-float(np.mean(expected)))
    assert error<1e-7,('fractional area reference',error)
    sp=s.detach().clone();tp=t.detach().clone()
    sp[area==0]=float('nan');tp[area==0]=float('nan')
    assert torch.equal(loss.detach(),gram_loss(sp,tp,area)[0]),'padding invariance'
    loss.backward()
    assert t.grad is None and s.grad.norm()>0,'teacher detach and student gradient'
    assert torch.equal(s.grad[area==0],torch.zeros_like(s.grad[area==0])),'zero padding gradient'
    z=torch.randn(2,3,4,requires_grad=True)
    zero=gram_loss(z,z.detach(),torch.zeros(2,3))[0];zero.backward()
    assert float(zero)==0 and z.grad.abs().sum()==0,'zero-valid handling'
    negative=gram_loss(torch.tensor([[[1.,0],[1.,0]]]),torch.tensor([[[1.,0],[-1.,0]]]),torch.ones(1,2))[0]
    assert float(negative)==2.,'negative similarities must be kept'
    base=Data(RUN/'teacher');extended=GramData(RUN/'teacher');ids=list(base.by)[:2]
    a,b=base[ids],extended[ids]
    assert all(torch.equal(a[k],b[k]) for k in ['full','fvalid','tpool','g_cls','g_mean','wfull','ltrb']),'B0 data unchanged'
    result=dict(status='passed',official_unweighted_error=float((plain-official).abs()),numpy_weighted_error=error,
                padding_pair_invariance_including_nonfinite=True,fractional_pair_weight_verified=True,
                zero_valid_image_excluded=True,all_zero_differentiable_zero=True,negative_similarity_kept=True,
                teacher_gradient=None,student_gradient=float(s.grad.norm()),B0_input_and_target_bit_equal=True,
                gram_weight_source='LB.coverage area; original bool fvalid remains unchanged for B0 losses')
    write_json(Path(out)/'unit_verification.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();verify(a.out)
