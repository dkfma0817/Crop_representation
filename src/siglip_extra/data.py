"""Unchanged COCO stream/object/nested ROIs plus COCO categories and real crop local targets.
No PACO attributes enter representation training. All experiments replay the same
first4000 steps of the original stream because the original stream has12000 rows.
"""
import os
import numpy as np,pandas as pd,torch
from PIL import Image
from src.config import COCO_TRAIN,FEAT
from src.siglip_pm.distill import TrainDS,cell_w
from src.siglip_pm.data import LB

def quadrants(boxes):
 """(...,4) pixel boxes -> (...,4 quadrants,4 coords), TL,TR,BL,BR."""
 x0,y0,x1,y1=boxes.unbind(-1);xm=(x0+x1)/2;ym=(y0+y1)/2
 return torch.stack([torch.stack(v,-1) for v in [(x0,y0,xm,ym),(xm,y0,x1,ym),(x0,ym,xm,y1),(xm,ym,x1,y1)]],-2)

class ExtraDS(TrainDS):
 def __init__(self,shared,spatial=False):
  super().__init__(shared);self.spatial=spatial
  df=pd.read_parquet(os.path.join(FEAT,'stream_rois.parquet'));df=df.set_index(['image_id','k'])
  self.categories=df.reindex(pd.MultiIndex.from_product([self.ids,range(4)],names=['image_id','k']))['cat_id'].to_numpy(dtype=np.int64).reshape(-1,4)
 def __getitem__(self,item):
  b=super().__getitem__(item);step,slots,_=item;cats=[];roiids=[];orig=[];crops=[];weights=[]
  for c,iid in slots:
   pos=self.pos[iid];boxes=[(bb,int(self.categories[pos,j])) for j,bb in enumerate(self.obj[pos])]
   boxes += [(bb,-1) for bb in self.NB[step,c] if bb[2]>bb[0]]
   if self.spatial:img=Image.open(f'{COCO_TRAIN}/{iid:012d}.jpg').convert('RGB')
   for bx,cat in boxes:
    cats.append(cat);roiids.append(iid);orig.append(bx)
    if self.spatial:
     l,t,r,bot=map(int,bx);lb=LB(r-l,bot-t,384,16)
     crops.append(lb.canvas(img.crop((l,t,r,bot))))
     regions=quadrants(torch.tensor([[lb.ox,lb.oy,lb.ox+lb.nw,lb.oy+lb.nh]],dtype=torch.float32))[0]
     w=cell_w(regions,24,16);assert np.allclose(w.sum(0).numpy(),lb.coverage(),atol=1e-6)
     weights.append(w)
  b.update(category=torch.tensor(cats),roi_image_id=torch.tensor(roiids),orig_boxes=torch.tensor(np.stack(orig),dtype=torch.float32))
  assert len(cats)==len(b['tpool'])
  if self.spatial:b.update(crops=torch.stack(crops),local_teacher_weights=torch.stack(weights))
  return b
