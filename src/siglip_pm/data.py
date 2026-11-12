"""Original letterbox/ROI geometry with official SigLIP2 input normalization and bilinear resize.

The geometric letterbox is retained for controlled comparison (official processor normally
stretches to square). Normalized padding is zero = RGB 0.5. No encoder attention mask:
the fixed-resolution pretrained SigLIP vision transformer has no padding mask argument.
Coverage masks are used for targets, ROI means and R2 keys, as in the original pilot.
"""
import numpy as np
import torch
from PIL import Image
from src.viz.c0_pca import LB as OriginalLB

class LB(OriginalLB):
    def canvas(self, img):
        r = np.asarray(img.convert('RGB').resize((self.nw, self.nh), resample=Image.Resampling.BILINEAR), dtype=np.float32)
        x = torch.zeros(3, self.S, self.S)
        x[:, self.oy:self.oy+self.nh, self.ox:self.ox+self.nw] = torch.from_numpy(r.copy()).permute(2,0,1) / 127.5 - 1
        return x

def geometry(W,H,ltrb,S,P):
    l,t,r,b = ltrb
    crop,full = LB(r-l,b-t,S,P),LB(W,H,S,P)
    i,j = np.divmod(np.arange(24*24),24)
    xo = l + ((j+.5)*P-crop.ox)*(r-l)/crop.nw
    yo = t + ((i+.5)*P-crop.oy)*(b-t)/crop.nh
    X,Y = full.ox+xo*full.nw/W,full.oy+yo*full.nh/H
    return crop,full,np.stack([X/(S/2)-1,Y/(S/2)-1],1).astype(np.float32)
