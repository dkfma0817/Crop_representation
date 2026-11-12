"""Fixed-reference Gram anchoring with explicit letterbox area weighting.

Reference: third_party/BRIXEL/src/brixel/dinov3_main/dinov3/loss/gram_loss.py.
Area weights and zero-valid handling are this pilot's extension, not official GramLoss.
"""
import torch
import torch.nn.functional as F
from src.fgtrain.data import Data
from src.viz.c0_pca import LB


def gram_loss(student,teacher,area):
    assert student.shape==teacher.shape and area.shape==student.shape[:2]
    with torch.autocast(device_type=student.device.type,enabled=False):
        w=area.detach().float()
        assert bool(torch.isfinite(w).all()) and bool(((w>=0)&(w<=1)).all())
        valid=w>0
        # Mask before normalization: even arbitrary/nonfinite padding features cannot leak.
        s=F.normalize(torch.where(valid[:,:,None],student.float(),0.),dim=-1)
        t=F.normalize(torch.where(valid[:,:,None],teacher.detach().float(),0.),dim=-1)
        pair=w[:,:,None]*w[:,None,:]
        denominator=pair.sum((1,2))
        per=((s@s.transpose(-1,-2)-t@t.transpose(-1,-2)).square()*pair).sum((1,2))/denominator.clamp_min(1e-12)
        images=denominator>0
        # Zero-valid images are excluded from the image mean; all-zero returns graph-connected0.
        loss=per[images].mean() if bool(images.any()) else s.sum()*0.
    return loss,per,dict(valid_images=int(images.sum()),zero_valid_images=int((~images).sum()),
                         valid_pair_weight=float(denominator.sum()))


class GramData(Data):
    def __getitem__(self,item):
        b=super().__getitem__(item)
        area=[]
        for iid in b['image_ids'].tolist():
            row=self.by[int(iid)].iloc[0]
            area.append(torch.from_numpy(LB(int(row.W),int(row.H),384,16).coverage()).float())
        b['gram_area']=torch.stack(area)
        assert torch.equal(b['gram_area']>0,b['fvalid'])
        return b


def gradient_trace(crop,gram,encoder,weight):
    from src.fgtrain.train import grouped_norms
    named=[(n,p) for n,p in encoder.named_parameters() if p.requires_grad]
    ps=[p for _,p in named]
    a=torch.autograd.grad(crop,ps,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(gram,ps,retain_graph=True,allow_unused=True)
    x,y=grouped_norms(named,a),grouped_norms(named,b)
    dot=sum((g.float()*h.float()).sum() for g,h in zip(a,b) if g is not None and h is not None)
    cosine=float(dot)/(x['encoder']*y['encoder']) if x['encoder']>0 and y['encoder']>0 else None
    return dict(crop=x,gram=y,weighted_gram_norm=weight*y['encoder'],
                gram_crop_ratio=weight*y['encoder']/max(x['encoder'],1e-30),crop_gram_cosine=cosine)
