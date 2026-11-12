"""Fixed N1 ROI-level ordering; no learned module and no attribute labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F

CONFIG=dict(positive_topk=4,negative_ranks=[16,32],teacher_gap_min=.02,
            temperature=.07,exclude_same_image_iou_above=.5,gradient_ratio=.25,
            unit='Pooled ROIs across the entire image minibatch; not patches',
            original_NeCo_reproduction=False)


@torch.no_grad()
def neighbors(teacher,boxes,image):
    target=F.normalize(teacher.detach().float(),dim=-1)
    t=target@target.T
    n=len(target)
    box=boxes.detach().float()
    lo=torch.maximum(box[:,None,:2],box[None,:,:2])
    hi=torch.minimum(box[:,None,2:],box[None,:,2:])
    intersection=(hi-lo).clamp_min(0).prod(-1)
    area=(box[:,2:]-box[:,:2]).clamp_min(0).prod(-1)
    iou=intersection/(area[:,None]+area[None,:]-intersection).clamp_min(1e-12)
    same=image[:,None]==image[None,:]
    self_pair=torch.eye(n,dtype=torch.bool,device=t.device)
    overlap=same & (iou>.5) & ~self_pair
    candidate=~(self_pair | overlap)
    # Stable ties use the ROI stream order. Ranks16..32 are inclusive, 1-based.
    order=t.masked_fill(~candidate,-torch.inf).argsort(dim=1,descending=True,stable=True)
    pos,neg=order[:,:4],order[:,15:32]
    gap=t.gather(1,pos)[:,:,None]-t.gather(1,neg)[:,None,:]
    possible=candidate.gather(1,pos)[:,:,None] & candidate.gather(1,neg)[:,None,:]
    valid=possible & (gap>.02)
    return t,pos,neg,gap,possible,valid,candidate,overlap


def ordering_loss(student,teacher,boxes,image,details=False):
    t,pos,neg,gap,possible,valid,candidate,overlap=neighbors(teacher,boxes,image)
    z=F.normalize(student.float(),dim=-1)
    s=z@z.T
    difference=s.gather(1,neg)[:,None,:]-s.gather(1,pos)[:,:,None]
    selected=difference[valid]
    # A rare empty minibatch contributes zero; preflight never accepts zero support.
    loss=F.softplus(selected/.07).mean() if selected.numel() else student.sum()*0
    with torch.no_grad():
        count=int(valid.sum());potential=int(possible.sum());n=len(student)
        cross=(image[:,None]!=image[neg])[:,None,:].expand_as(valid)
        stats=dict(rois=n,valid_pairs=count,possible_pairs=potential,
                   pair_ratio=count/max(1,potential),anchors=int(valid.any((1,2)).sum()),
                   anchor_coverage=float(valid.any((1,2)).float().mean()),
                   excluded_overlap_pairs=int(overlap.sum()),candidate_pairs=int(candidate.sum()),
                   cross_image_negative_pair_fraction=float((valid & cross).sum())/max(1,count),
                   order_accuracy=float((selected<0).float().mean()) if count else 0.,
                   teacher_gap_mean=float(gap[valid].mean()) if count else 0.)
        if details:
            q=t.new_tensor([0.,.1,.25,.5,.75,.9,1.])
            stats['quantile_probabilities']=q.cpu().tolist()
            for name,values in [('teacher_gap_candidates',gap[possible]),('teacher_gap_valid',gap[valid]),
                                ('teacher_similarity',t[candidate])]:
                stats[name]=torch.quantile(values,q).cpu().tolist() if values.numel() else []
            stats['candidate_counts_per_anchor']=candidate.sum(1).cpu().tolist()
            stats['valid_pairs_per_anchor']=valid.sum((1,2)).cpu().tolist()
    return loss,stats


def preflight(run,base):
    from src.fgtrain.common import encoder,state_sha,write_json
    from src.fgtrain.data import Data
    from src.fgtrain.train import terms,per_loss_grads
    from src.fgtrain.audit import target_audit
    from src.config import STREAM
    import src.readout.common as RC
    run,base=Path(run),Path(base)
    path=run/'lambda_rank.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['ranking_config']==CONFIG
        assert old['status']=='passed'
        return old
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    target_check=target_audit(base)
    enc=encoder(base/'initial_encoder.pt',train=True)
    initial_sha=state_sha(enc.state_dict())
    ds=Data(base/'teacher',spatial=False)
    excluded=set(json.loads((base/'teacher/manifest.json').read_text())['excluded_train_images'])
    stream=np.load(STREAM)
    ids=[int(i) for i in stream[0] if int(i) not in excluded]
    b=RC.to_dev(ds[ids],torch.device('cuda'))
    o=terms(b,enc,None,rank=True,rank_details=True)
    reference=terms(b,enc,None)
    assert torch.equal(o['crop'],reference['crop']) and torch.equal(o['glob'],reference['glob'])
    norms=per_loss_grads(o,enc,None)
    gc,gr=norms['crop']['encoder'],norms['rank']['encoder']
    stats=o['rank_stats']
    support=stats['valid_pairs']>0 and stats['anchor_coverage']>=.5 and stats['pair_ratio']>=.01
    result=dict(status='passed' if support else 'insufficient_neighbor_support',ranking_config=CONFIG,
                image_ids=ids,roi_kinds=dict(Counter(k for iid in ids for k in ds.by[iid]['kind'])),
                initial_sha=initial_sha,gradients=norms,pair_statistics=stats,targets=target_check,
                procedure='Initial identical B0 stream minibatch0; no optimizer update; lambda fixed for all12k steps',
                B0_crop_global_bit_equal=True,teacher_detached=not o['target_pool'].requires_grad,
                support_rule='At least one pair, >=50% anchors, >=1% eligible pairs; report without relaxing fixed loss thresholds')
    write_json(run/'preflight_rank_support.json',result)
    assert support, 'Insufficient fixed neighbor support; inspect preflight_rank_support.json; do not relax thresholds'
    assert math.isfinite(gc) and math.isfinite(gr) and gr>1e-10 and gc>1e-10
    for loss in ['crop','rank']:
        assert norms[loss]['patch_embed']>1e-12
        assert all(norms[loss][f'block_{k:02d}']>1e-12 for k in range(12))
    lam=.25*gc/gr
    assert state_sha(enc.state_dict())==initial_sha
    result.update(lambda_rank=lam,measured_gradient_ratio=lam*gr/gc,
                  initial_weights_unchanged=True,standard_encoder_parameters=sum(p.numel() for p in enc.parameters()))
    write_json(path,result)
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--base-run',required=True);a=p.parse_args()
    preflight(a.run,a.base_run)


if __name__=='__main__':main()
