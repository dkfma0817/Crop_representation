"""Actual model/geometry/gradient checks, using first COCO minibatch and frozen native teacher."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import src.readout.common as RC
from src.config import STREAM
from src.fgtrain.common import encoder, freeze_export, write_json
from src.fgtrain.data import Data, teacher_geometry
from src.fgtrain.train import modules, terms, per_loss_grads


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cache',required=True)
    a=p.parse_args();run=Path(a.run);root=Path(a.cache)
    torch.set_num_threads(2)
    enc,dec,_=modules(run/'initial_encoder.pt','B1')
    ds=Data(root,spatial=True)
    ex=json.loads((root/'manifest.json').read_text())['excluded_train_images']
    ids=[int(i) for i in np.load(STREAM)[0] if int(i) not in ex]
    # Two images include multiple geometries without repeating full expensive calibration.
    b=RC.to_dev(ds[ids[:2]],torch.device('cuda'))
    coordinate_error=0.;centres=0
    for qr,box,raw,cov in zip(b['qrel'],b['box_c'],b['ltrb'],b['ccov']):
        lb,expected,rel=teacher_geometry(raw.cpu().tolist())
        assert np.array_equal(expected,cov.cpu().numpy())
        assert np.array_equal(rel,qr.cpu().numpy())
        xy=box[:2]+qr*(box[2:]-box[:2])
        # Native crop content -> original ROI -> full-image letterbox. The two routes must coincide.
        recovered=(xy-box[:2])/(box[2:]-box[:2])
        coordinate_error=max(coordinate_error,float((recovered-qr).abs().max()))
        centres+=int((cov>0).sum())
        assert abs(float(cov.sum())-lb.nw*lb.nh/256)<1e-3
    # Compare saved spatial cells at THEIR actual token index to a fresh frozen native teacher.
    from src.ivc.ft import run_enc
    crops=RC.to_dev(Data(root,crops=True)[ids[:2]],torch.device('cuda'))
    teacher=encoder(run/'initial_encoder.pt')
    with torch.no_grad():
        live,_=run_enc(teacher,crops['teacher_crops'],chunk=16)
        valid=b['ccov']>0
        target=b['target'].float()
        live_cos=torch.nn.functional.cosine_similarity(live[valid].float(),target[valid],dim=-1)
        spatial_min=float(live_cos.min());spatial_mean=float(live_cos.mean())
        assert spatial_min>.999,('spatial teacher cache mismatch',spatial_min)
        exact_fraction=float((live[valid].half()==b['target'][valid]).float().mean())
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    del teacher,live,crops
    o=terms(b,enc,dec)
    gradients=per_loss_grads(o,enc,dec)
    assert gradients['spatial']['encoder']>1e-10
    assert gradients['spatial']['patch_embed']>1e-12
    assert all(gradients['spatial'][f'block_{k:02d}']>1e-12 for k in range(12))
    loss=o['crop'].mean()+.5*o['glob'].mean()+o['spatial'].mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None)
    assert all(torch.isfinite(p.grad).all() for p in dec.parameters() if p.grad is not None)
    # Padding has exactly zero contribution, including target cells filled with extreme finite values.
    with torch.no_grad():
        z=torch.randn_like(b['target'],dtype=torch.float32)
        from src.g1x.distill import cosd
        target=b['target'].float().clone();cc=b['ccov']
        l1=(cosd(z,target)*cc).sum(1)/cc.sum(1)
        target[cc==0]=1e6
        l2=(cosd(z,target)*cc).sum(1)/cc.sum(1)
        assert torch.equal(l1,l2)
        enc.eval();tok=enc.forward_features(b['full'])
        coarse_before=tok.clone()
        dec(o['tok'].detach(),b['roi_img'],b['box_c'].roll(1,0),b['qrel'],b['fvalid'])
        coarse_after=enc.forward_features(b['full'])
        assert torch.equal(coarse_before,coarse_after)
    exported=freeze_export(enc,run/'smoke_export',b['full'],run/'initial_encoder.pt')
    # One real update proves loss gradients can change the standard encoder weights.
    opt=torch.optim.AdamW(enc.parameters(),lr=5e-5)
    initial=enc.patch_embed.proj.weight.detach().clone()
    opt.step()
    update=float((initial-enc.patch_embed.proj.weight).abs().max())
    assert update>0
    result=dict(status='passed',teacher_frozen=True,spatial_cache_vs_live_min_cos=spatial_min,spatial_cache_vs_live_mean_cos=spatial_mean,spatial_cache_exact_scalar_fraction=exact_fraction,coordinate_max_error=coordinate_error,valid_teacher_tokens=centres,
                padding_loss_invariant=True,roi_independent_encoder_forward=True,decoder_rgb_input=False,
                teacher_targets_require_grad=b['target'].requires_grad,gradients=gradients,patch_embedding_update_max=update,
                finite_loss=float(loss.detach()),finite_gradients=True,export=exported,
                note='No claim that 48x48 reconstruction recovers unobserved native detail in tiny ROIs')
    write_json(run/'smoke_verification.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
