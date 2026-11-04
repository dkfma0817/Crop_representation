"""Reference rows for the loss-curve figure, on the same COCO dev 2,000 ROIs and with the same metrics as train_rae.evaluate:
  bil     : full-image latent bilinearly sampled onto the teacher grid (RC.bilinear) = the "Full" representation in recon space
            (the PACO "Full" row reads the ROI tokens directly and has no teacher-grid latent, so it has no recon loss of its own)
  teacher : GT crop oracle = the GT crop canvas (the pixel-loss target itself) re-encoded on the fly (full 16x16 incl. padding
            cells; the cache keeps content cells only, see hval_teacher_fullpad.py). Feature loss is measured against the cached
            teacher (content cells), so it is ~0 by construction (fp16 cache rounding only).
  CUDA_VISIBLE_DEVICES=0 python -m src.eval.refs -> results/dev_refs.json
"""
import json
import os
import sys

import torch

import src.rae.common as RC  # noqa: E402
import src.losses.pixel as PX  # noqa: E402
import src.data.dataset as DR  # noqa: E402

BETA = 0.2


@torch.no_grad()
def main():
    dev = "cuda"
    RC.set_fp32()
    rae, lp = RC.FrozenRAE(dev), PX.load_lpips(dev)
    ds = DR.CachedSet("dev", rgb=True)
    agg = {"bil": {}, "teacher": {}}
    for raw in DR.loader(ds, DR.seq_slots(ds, 16), 6):
        x_u8 = raw["rgb"]
        b = DR.to_dev(raw, dev)
        g, w = b["geom"], b["geom"]["valid_w"]
        reps = {"bil": RC.bilinear(b["full"], g), "teacher": rae.encode(x_u8)}
        for name, Z in reps.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = rae.decode(Z, chunk=64).float()
            mse, cos = PX.feat_mse_cos(Z, b["teacher"], w)
            l1, lpv = PX.pix_terms(y, b["rgb"], b["mask"], lp)
            met = PX.rgb_metrics(y.clamp(0, 1), b["rgb"], b["mask"], lp)
            vals = dict(mse=mse, cos=cos, feat_loss=mse + 1 - cos, pix_l1=l1, pix_lpips=lpv, pix_loss=l1 + BETA * lpv,
                        **{f"rgb_{k}": v for k, v in met.items()})
            for k, v in vals.items():
                agg[name].setdefault(k, []).append(v.float().cpu())
    out = {n: {k: float(torch.cat(v).mean()) for k, v in d.items()} for n, d in agg.items()}
    out["n_rois"] = int(sum(len(v) for v in agg["bil"]["mse"]))
    json.dump(out, open(os.path.join(RC.RES, "dev_refs.json"), "w"), indent=1)
    for n in ("bil", "teacher"):
        print(n, {k: round(out[n][k], 4) for k in ("feat_loss", "cos", "pix_loss", "rgb_l1", "rgb_lpips")})
    print("n_rois", out["n_rois"])


if __name__ == "__main__":
    main()
