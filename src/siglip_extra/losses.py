"""Only the two authorized loss additions; unchanged P architecture and original losses."""
import torch
import torch.nn.functional as F
from src.siglip_pm.distill import pool,cell_w
from src.siglip_extra.data import quadrants

def negative_mask(teacher,categories,image_ids,boxes,threshold):
 t=F.normalize(teacher.detach().float(),dim=-1);sim=t@t.T;n=len(t);eye=torch.eye(n,device=t.device,dtype=torch.bool)
 # Also remove near-duplicate boxes in the same original image, regardless of their category.
 l=torch.maximum(boxes[:,None,:2],boxes[None,:,:2]);r=torch.minimum(boxes[:,None,2:],boxes[None,:,2:]);inter=(r-l).clamp_min(0).prod(-1)
 area=(boxes[:,2:]-boxes[:,:2]).prod(-1);iou=inter/(area[:,None]+area[None,:]-inter).clamp_min(1e-6)
 duplicate=(image_ids[:,None]==image_ids[None,:])&(iou>.9)
 mask=(sim<threshold)&~eye&~duplicate
 same=(categories[:,None]==categories[None,:])&(categories[:,None]>=0)&~eye
 return mask,same,sim

def contrastive(student,teacher,categories,image_ids,boxes,threshold,temperature=.07,collect=True):
 mask,same,tsim=negative_mask(teacher,categories,image_ids,boxes,threshold)
 logits=F.normalize(student.float(),dim=-1)@F.normalize(teacher.detach().float(),dim=-1).T/temperature
 n=len(logits);eye=torch.eye(n,device=logits.device,dtype=torch.bool);valid=mask.any(1)
 candidates=mask|eye;denom=logits.masked_fill(~candidates,-torch.inf).logsumexp(1)
 per=denom-logits.diagonal();loss=(per*valid.float()).sum()/valid.sum().clamp_min(1)
 if not collect:return loss,{}
 with torch.no_grad():
  rank1=logits.masked_fill(~(mask|eye),-torch.inf).argmax(1)==torch.arange(n,device=logits.device)
  sm=mask&same;sv=sm.any(1);same_rank=logits.masked_fill(~(sm|eye),-torch.inf).argmax(1)==torch.arange(n,device=logits.device)
  av=same.any(1);all_rank=logits.masked_fill(~(same|eye),-torch.inf).argmax(1)==torch.arange(n,device=logits.device)
  metrics=dict(negative_mean=float(mask.sum(1).float().mean()),negative_retained_ratio=float(mask.sum()/max(1,n*(n-1))),
               same_category_negative_mean=float(sm.sum(1).float().mean()),same_category_available_pairs=int(same.sum()),same_category_used_pairs=int(sm.sum()),
               ROIs_with_same_category_negative_ratio=float(sv.float().mean()),retrieval_rank1=float(rank1[valid].float().mean()) if valid.any() else 0.,
               same_category_retrieval_rank1=float(same_rank[sv].float().mean()) if sv.any() else 0.,same_category_all_candidate_retrieval_rank1=float(all_rank[av].float().mean()) if av.any() else None,valid_ROI_ratio=float(valid.float().mean()))
 return loss,metrics

def local_student(model,o,b):
 boxes=quadrants(b['box_c']);r=boxes.reshape(-1,4);ri=b['roi_img'].repeat_interleave(4)
 A=pool(o['coarse'][ri],cell_w(r,24,16))
 mem={k:pool(v[ri],cell_w(r,48,8) if o['grids'][k]==2 else cell_w(r,24,16)) for k,v in o['mem'].items()}
 with torch.autocast('cuda',dtype=torch.bfloat16):q=model.fuse(A,mem).float()
 # Each quadrant must span >=2 actual stride8 fine cells on BOTH axes (full ROI >=32px).
 extent=b['box_c'][:,2:]-b['box_c'][:,:2];eligible=(extent>=32).all(1)
 return q.reshape(-1,4,768),eligible

@torch.no_grad()
def local_teacher(teacher,b,chunk=96):
 out=[];whole=[]
 for lo in range(0,len(b['crops']),chunk):
  with torch.autocast('cuda',dtype=torch.bfloat16):t=teacher.forward_features(b['crops'][lo:lo+chunk]).float()
  w=b['local_teacher_weights'][lo:lo+chunk].float()
  out.append(torch.einsum('rtd,rkt->rkd',t,w)/w.sum(-1,keepdim=True).clamp_min(1e-12))
  whole.append(pool(t,w.sum(1)))
 return torch.cat(out),torch.cat(whole)

def spatial(student,teacher,eligible,collect=True):
 per=1-F.cosine_similarity(student.float(),teacher.detach().float(),dim=-1)
 loss=(per*eligible[:,None].float()).sum()/(eligible.sum()*4).clamp_min(1)
 if not collect:return loss,{}
 q=F.normalize(student.detach().float(),dim=-1);t=F.normalize(teacher.detach().float(),dim=-1)
 off=~torch.eye(4,device=q.device,dtype=torch.bool)
 teacher_region_sim=(t@t.transpose(1,2))[:,off].mean(1);student_region_sim=(q@q.transpose(1,2))[:,off].mean(1)
 return loss,dict(local_cosine=float(1-per[eligible].mean()) if eligible.any() else None,spatial_eligible_ROI_ratio=float(eligible.float().mean()),spatial_eligible_ROIs=int(eligible.sum()),
                  teacher_quadrant_similarity=float(teacher_region_sim[eligible].mean()) if eligible.any() else None,student_quadrant_similarity=float(student_region_sim[eligible].mean()) if eligible.any() else None)
