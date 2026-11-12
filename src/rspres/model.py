"""RSP-Residual: the RSP-Masked (pix) predictor with ONLY the output structure changed.

  RSP-Masked :  pred_n = out(out_norm(h))                                   (normalised fine feature)
  RSP-Residual: pred_n = (F384_bil - mu) / sigma + Delta_n,   Delta_n = out(out_norm(h)),   out zero-initialised (weight and bias)
                Y_hat  = sigma * pred_n + mu = F384_bil + Delta,          Delta = sigma * Delta_n   (raw DINOv3 final-token space)
F384_bil: the frozen 384 final patch map (24 x 24, after the final LayerNorm = the student context F) read bilinearly at the fine
query centres (F.grid_sample, align_corners False, border) - the same read as the 'ref_f384bil' training statistic and the 'F384-bil'
row of src.rsp.analysis. No trainable parameter on the bilinear path; mu / sigma are the RSP norm stats (frozen buffers).
Delta lives in the normalised space so the output Linear has exactly the parameterisation and gradient scale of RSP-Masked's head.
Everything else (context / evidence / z / mask query / level mask / d = 192 x 12) is the unchanged src.rsp.model.RSPPredictor.
"""
import torch
import torch.nn.functional as F

from src.rsp.common import PREFIX, S
from src.rsp.model import RSPPredictor


def f384_bil(feats, rimg, cen):
    """feats (B, 581, 768) frozen final tokens, rimg (R,), cen (R, K, 2) (y, x) canvas px -> (R, K, 768) fp32 bilinear read."""
    B = feats.shape[0]
    fm = feats[:, PREFIX:].float().transpose(1, 2).reshape(B, 768, 24, 24)
    g = (cen.float().flip(-1) / S * 2 - 1)[:, :, None]
    return F.grid_sample(fm[rimg], g, mode="bilinear", padding_mode="border", align_corners=False)[..., 0].transpose(1, 2)


class RSPResidual(RSPPredictor):
    def __init__(self, mu, sd, **kw):
        super().__init__(pix=True, **kw)
        self.register_buffer("mu", mu.float().clone(), persistent=False)
        self.register_buffer("sd", sd.float().clone(), persistent=False)

    def zero_delta(self):
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def parts(self, feats, rimg, box, ev, cen):
        """-> pred_n (R, K, 768) fp32, base raw F384_bil (R, K, 768) fp32, delta_n (R, K, 768) fp32."""
        delta_n = super().forward(feats, rimg, box, ev, cen).float()
        with torch.autocast("cuda", enabled=False):
            base = f384_bil(feats, rimg, cen)
            pred_n = (base - self.mu) / self.sd + delta_n
        return pred_n, base, delta_n

    def forward(self, feats, rimg, box, ev, cen):
        return self.parts(feats, rimg, box, ev, cen)[0]
