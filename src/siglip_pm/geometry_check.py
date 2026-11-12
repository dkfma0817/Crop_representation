"""CPU audit of real object AND nested ROI identity and both feature-grid coordinates."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from src.config import COCO_TRAIN,STREAM
from src.siglip_pm.distill import TrainDS,cell_w
from src.siglip_pm.data import LB,geometry
from src.siglip_pm.probe import roi_align

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run);ds=TrainDS(str(run/'shared'));stream=np.load(STREAM);errors=[];count=0;nested=0
 for step in [0,17,113,2999,11999]:
  slots=[(c,int(v)) for c,v in enumerate(stream[step]) if int(v) not in [388662,397819]][:2]
  b=ds[(step,slots,len(slots))];r=0
  for k,(c,iid) in enumerate(slots):
   img=Image.open(f'{COCO_TRAIN}/{iid:012d}.jpg').convert('RGB');W,H=img.size;lb=LB(W,H,384,16);pos=ds.pos[iid]
   assert torch.equal(b['full'][k],lb.canvas(img))
   boxes=[(bb,False,j) for j,bb in enumerate(ds.obj[pos])]+[(bb,True,j) for j,bb in enumerate(ds.NB[step,c]) if bb[2]>bb[0]]
   for bx,isnested,j in boxes:
    l,t,rr,bot=bx;expected=np.array([lb.ox+l*lb.nw/W,lb.oy+t*lb.nh/H,lb.ox+rr*lb.nw/W,lb.oy+bot*lb.nh/H])
    assert np.allclose(b['box_c'][r].numpy(),expected,atol=3e-5)
    assert int(b['roi_img'][r])==k and bool(b['nested'][r])==isnested
    target=ds.m['t_nested'][step,c,j] if isnested else ds.m['t_object'][pos,j]
    assert np.array_equal(b['tpool'][r].numpy(),target) and np.isfinite(target).all()
    crop,full,grid=geometry(W,H,bx,384,16);assert np.array_equal(crop.coverage(),LB(rr-l,bot-t,384,16).coverage())
    # A physical-coordinate field must be sampled at the same pixel location on BOTH maps.
    expected=(torch.tensor(grid)+1)*192
    for mult in [1,2]:
     g=24*mult;cell=16/mult;gx,gy=torch.meshgrid(torch.arange(g),torch.arange(g),indexing='xy')
     coords=torch.stack([(gx+.5)*cell,(gy+.5)*cell],-1).reshape(1,g*g,2).float()
     rb=dict(img_index=torch.tensor([0]),bil_grid=torch.tensor(grid)[None])
     sampled=roi_align(coords,mult,rb)[0];inside=(expected[:,0]>=cell/2)&(expected[:,0]<=384-cell/2)&(expected[:,1]>=cell/2)&(expected[:,1]<=384-cell/2)&torch.tensor(crop.coverage()>0)
     if inside.any():errors.append(float((sampled[inside]-expected[inside]).abs().max()))
    r+=1;count+=1;nested+=int(isnested)
  w24=cell_w(b['box_c'],24,16);w48=cell_w(b['box_c'],48,8)
  assert torch.allclose(w48.sum(1),4*w24.sum(1),rtol=2e-5,atol=1e-4)
 assert max(errors)<1e-3
 result=dict(status='passed',object_ROIs=count-nested,nested_ROIs=nested,steps=[0,17,113,2999,11999],teacher_cache_target_exact=True,roi_coordinate_max_error_px=max(errors),coarse_fine_area_consistent=True)
 json.dump(result,open(run/'geometry_check.json','w'),indent=1);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
