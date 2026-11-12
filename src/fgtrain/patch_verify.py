"""CPU geometry/provenance checks, then one P2-only gradient calibration."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.config import STREAM
from src.fgtrain.common import encoder, state_sha, write_json
from src.fgtrain.patch_preserve import PatchData, union_coverage, weights, patch_loss, weight_stats, gradient_trace
from src.fgtrain.p1 import sha, read
from src.ivc.ft import run_enc
from src.fgtrain.train import terms
import src.readout.common as RC


def batch0(base, device='cpu'):
    ds = PatchData(Path(base)/'teacher')
    excluded = set(read(Path(base)/'teacher/manifest.json')['excluded_train_images'])
    ids = [int(i) for i in np.load(STREAM)[0] if int(i) not in excluded]
    return RC.to_dev(ds[ids], device)


def reference_area(boxes, cell):
    """Independent sweep-line union area, for testing inclusion/exclusion."""
    rs = []
    for r in boxes:
        a, b = np.maximum(r[:2], cell[:2]), np.minimum(r[2:], cell[2:])
        if (b > a).all(): rs.append([*a, *b])
    xs = sorted({x for r in rs for x in [r[0], r[2]]})
    area = 0.
    for left, right in zip(xs, xs[1:]):
        ys = sorted((r[1], r[3]) for r in rs if r[0] < right and r[2] > left)
        length = 0.; end = -math.inf
        for low, high in ys:
            length += max(0., high-max(low, end)); end = max(end, high)
        area += (right-left)*length
    return area


def units(root):
    rng = np.random.default_rng(0)
    max_error = 0.
    for _ in range(50):
        xy = rng.uniform(-5, 36, (4, 4))
        boxes = np.concatenate([np.minimum(xy[:,:2], xy[:,2:]), np.maximum(xy[:,:2], xy[:,2:])], -1)
        content = [3., 5., 29., 31.]
        v, m = union_coverage(boxes, content, size=32, patch=8)
        for p in range(16):
            y, x = divmod(p, 4)
            cell = [max(x*8,3), max(y*8,5), min((x+1)*8,29), min((y+1)*8,31)]
            expected = reference_area(boxes, cell) if cell[2] > cell[0] and cell[3] > cell[1] else 0.
            error = abs(float(v[p]*m[p]*64)-expected)
            max_error = max(max_error, error)
            assert error < 1e-5, (boxes, p, error)
    v, m = union_coverage([[0,0,8,16],[4,0,12,16]], [0,0,16,16], 16,16)
    assert np.allclose(m,.75) # sum=1, max=.5 would both be wrong.
    assert np.array_equal(union_coverage([[0,0,8,16]]*2, [0,0,16,16],16,16)[1], np.array([.5],np.float32))
    v = torch.tensor([[1.,.5,0.],[0.,0.,0.]])
    m = torch.tensor([[1.,.5,0.],[0.,0.,0.]])
    w = weights(v,m,True)
    assert torch.allclose((v*w).sum(1),v.sum(1))
    assert torch.allclose(weights(v,torch.ones_like(m),True),weights(v,m,False))
    s = torch.randn(2,3,5,requires_grad=True); t = torch.randn_like(s,requires_grad=True)
    a, per, _ = patch_loss(s,t,v,m,True)
    a.backward(); assert t.grad is None and s.grad[0,:2].abs().sum()>0
    assert s.grad[0,2].abs().sum()==0 and s.grad[1].abs().sum()==0
    with torch.no_grad():s[0,2]=float('nan');t[0,2]=float('nan')
    changed,_,_=patch_loss(s,t,v,m,True)
    assert torch.allclose(a,changed)
    z,_,_=patch_loss(s,t,torch.zeros_like(v),m,True)
    assert float(z)==0 and z.requires_grad; z.backward()
    same,_,_=patch_loss(torch.randn(2,3,5),torch.randn(2,3,5),v,m,False)
    assert torch.isfinite(same)
    write_json(Path(root)/'unit_verification.json',dict(status='passed',random_union_cases=50,
               max_area_error=max_error,overlap_analytic=True,duplicates_count_once=True,
               normalized_mean_one=True,full_union_equals_uniform=True,padding_nan_ignored=True,
               teacher_detached=True,student_gradient=True,zero_valid_graph_zero=True))


def preflight(root, base):
    root, base = Path(root),Path(base)
    ds = PatchData(base/'teacher')
    b = batch0(base)
    ids = list(dict.fromkeys(b['image_ids'].tolist()+sorted(ds.by)[:512]))
    rows = []
    for iid in ids:
        v,m,_ = ds.geometry(iid)
        q=weight_stats(torch.from_numpy(v)[None],torch.from_numpy(m)[None])[0]
        rows.append(dict(image_id=iid,**q))
    write_json(root/'roi_union_preflight.json',dict(
        status='passed',method='Float64 exact rectangle union by inclusion/exclusion, tested against independent sweep-line',
        sample_rule='All first-batch images plus first512 sorted B0 training images',images=rows,
        union_fraction_quantiles={str(q):float(np.quantile([r['union_fraction'] for r in rows],q)) for q in [0,.25,.5,.75,.95,1]},
        nearly_uniform_fraction=float(np.mean([r['p2_p3_mean_absolute_difference']<.01 for r in rows])),
        warning='A large GT+Random union can make selective weights almost uniform; coefficient stays0.75',
        geometry='Exact float64 B0 canvas boxes before existing float32 transport; same content LB coverage',
        raw_weight_examples={'outside':1.,'half':.625,'inside':.25}))
    assert all(abs(r['mean']-1)<1e-6 for r in rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    folder=root/'preflight';folder.mkdir(exist_ok=True)
    # Pick low/median/high union coverage from the actual first calibration batch.
    stats=weight_stats(b['patch_area'],b['roi_union'])
    order=np.argsort([q['union_fraction'] for q in stats])
    for j in [int(order[0]),int(order[len(order)//2]),int(order[-1])]:
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        rgb=(b['full'][j]*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]).permute(1,2,0).clamp(0,1)
        axes[0].imshow(rgb)
        for box in b['box_c'][b['roi_img']==j]:
            x,y,r,d=box.tolist();axes[0].add_patch(Rectangle((x,y),r-x,d-y,fill=False,edgecolor='red',linewidth=1))
        axes[0].set_title('Same Full384 + actual GT/random ROIs')
        matrices=[b['patch_area'][j],b['roi_union'][j],weights(b['patch_area'][j:j+1],b['roi_union'][j:j+1],True)[0]]
        for ax,z,title in zip(axes[1:],matrices,['Valid fractional coverage v','Exact ROI union / valid area m','P3 normalized weight (P2=1)']):
            im=ax.imshow(z.reshape(24,24),extent=(0,384,384,0),interpolation='nearest');ax.set_title(title);fig.colorbar(im,ax=ax,fraction=.046)
        fig.suptitle(f"Image {int(b['image_ids'][j])}, union={stats[j]['union_fraction']:.3f}, mean w={stats[j]['mean']:.6f}")
        fig.tight_layout();fig.savefig(folder/f"union_{int(b['image_ids'][j])}.png",dpi=130);plt.close(fig)


def calibrate(root, base, checkpoint):
    root,base,checkpoint=map(Path,[root,base,checkpoint])
    assert read(root/'unit_verification.json')['status']=='passed'
    record=read(root/'calibration_source_audit.json')
    assert sha(checkpoint)==record['checkpoint_sha256']
    if (root/'lambda_preserve.json').exists():return
    torch.set_num_threads(2);torch.manual_seed(0)
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and torch.cuda.device_count()==1
    props=torch.cuda.get_device_properties(0)
    write_json(root/'calibration_device.json',dict(physical_gpu=0,CUDA_VISIBLE_DEVICES='0',internal_device='cuda:0',uuid=str(getattr(props,'uuid','unavailable'))))
    torch.backends.cuda.matmul.allow_tf32=False
    teacher=encoder(base/'initial_encoder.pt')
    student=encoder(base/'initial_encoder.pt',train=True)
    assert state_sha(teacher.state_dict())==record['initial_state_sha256']
    b=batch0(base,'cuda')
    with torch.no_grad():target,_=run_enc(teacher,b['full'])
    o=terms(b,student,None)
    initial,_,_=patch_loss(o['tok'],target,b['patch_area'],b['roi_union'],False)
    assert abs(float(initial.detach()))<5e-7
    del o,initial
    ck=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    assert ck['step']==6000 and ck['condition']=='B0' and ck['total_steps']==12000
    student.load_state_dict(ck['ema'],strict=True)
    calibration_sha=state_sha(student.state_dict())
    assert calibration_sha==record['calibration_state_sha256']
    o=terms(b,student,None)
    preserve,_,_=patch_loss(o['tok'],target,b['patch_area'],b['roi_union'],False)
    trace=gradient_trace(o['crop'].mean(),preserve,student,1.)
    gc,gp=trace['crop']['encoder'],trace['preserve']['encoder']
    write_json(root/'calibration_measurement.json',dict(loss=float(preserve.detach()),trace=trace))
    assert math.isfinite(gp) and gp>1e-8 and math.isfinite(gc) and gc>1e-8,trace
    for key in ['crop','preserve']:
        assert trace[key]['patch_embed']>1e-12
        assert all(trace[key][f'block_{i:02d}']>1e-12 for i in range(12))
    frozen=all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert frozen and torch.isfinite(preserve)
    lam=.5*gc/gp
    result=dict(status='passed',lambda_preserve=lam,P2_P3_same_lambda=True,target_ratio=.5,
                measured_ratio=lam*gp/gc,unweighted_gradient_trace=trace,teacher_frozen=True,
                checkpoint=str(checkpoint),checkpoint_kind='B0 replay EMA6000, full12000 schedule',
                checkpoint_sha256=record['checkpoint_sha256'],calibration_state_sha256=calibration_sha,
                image_ids=b['image_ids'].tolist(),batch_source='Unchanged B0 COCO STREAM[0], filtered by original exclusions',
                preserve_loss=float(preserve.detach()),initial_equal_patch_loss_near_zero=True,
                actual_training_initialization='Original pretrained, independent P2 and P3; calibration EMA is never a training initializer',
                procedure='One P2 gradient calibration; P3 is not calibrated; lambda fixed for all12000 steps')
    write_json(root/'lambda_preserve.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--base-run',required=True)
    p.add_argument('--stage',choices=['cpu','calibrate'],required=True);p.add_argument('--checkpoint');a=p.parse_args()
    if a.stage=='cpu':units(a.run);preflight(a.run,a.base_run)
    else:calibrate(a.run,a.base_run,a.checkpoint)


if __name__=='__main__':main()
