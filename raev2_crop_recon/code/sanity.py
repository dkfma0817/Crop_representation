"""Pre-training sanity checks (no training weights are produced here that are used later).

 S1 official RAE class (stage1.rae.RAE, noise_tau=0, eval) vs our wrapper: encode (normalised latent) and decode, max |diff|
 S2 teacher recon: GT crop (256 letterbox) -> encoder -> decoder vs GT crop canvas (content pixels): L1 / LPIPS / dE / hp_corr
 S3 bilinear ROI feature (full 16x16 raw map sampled at the teacher cell centres) -> decoder, same metrics
 S4 shapes / spaces: teacher raw latent, CoordMLP output (fresh init), normalised-latent statistics of crops vs ImageNet
 S5 geometry: synthetic coordinate image (RGB encodes original x, y) -> crop canvas cell centres and full canvas at bil_grid
    must decode to the same original pixel; CoordMLP query coords P (ROI-relative crop canvas) and coords of ROI tokens
 S6 pixel-loss gradient: through the frozen decoder into Z_pred and CoordMLP params (init and after 5 Adam steps),
    decoder / LPIPS params get no grad and are unchanged
 S7 memory: CoordMLP + decoder + L1 + LPIPS forward/backward (BF16) for micro-batches 1..64 ROIs: peak GB and s/ROI
  python code/sanity.py  -> results/sanity.json, results/viz/sanity_*.png
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rae_common as RC  # noqa: E402
import rcp  # noqa: E402  (adds rae_crop_pilot/_pylib -> lpips 0.1.4)
import lpips  # noqa: E402
from models_rae import MLRDecoder  # noqa: E402
import pixel as PX  # noqa: E402

dev = "cuda"
RC.set_fp32()
OUT = {}
VIZ = os.path.join(RC.RES, "viz")
os.makedirs(VIZ, exist_ok=True)
lp = PX.load_lpips(dev)
rae = RC.FrozenRAE(dev)


def jprint(k, v):
    OUT[k] = v
    print(k, json.dumps(v, default=float), flush=True)
    json.dump(OUT, open(os.path.join(RC.RES, "sanity.json"), "w"), indent=1, default=float)


# ---------------------------------------------------------------------------------------------- data: 64 hval ROIs
rois = pd.read_parquet(os.path.join(RC.CACHE, "feat", "hval_rois.parquet"))
rng = np.random.default_rng(0)
sel = rois.iloc[np.sort(rng.choice(len(rois), 64, replace=False))]
fulls, crops, geoms, idx, rects = [], [], [], [], []
slot_of = {}
for r in sel.itertuples():
    img = Image.open(os.path.join(RC.COCO_TRAIN, f"{int(r.image_id):012d}.jpg")).convert("RGB")
    if r.image_id not in slot_of:
        slot_of[r.image_id] = len(fulls)
        fulls.append(RC.u8_canvas(img, RC.Letterbox(*img.size)))
    ltrb = (int(r.l), int(r.t), int(r.r), int(r.b))
    crops.append(RC.u8_canvas(img.crop(ltrb), RC.Letterbox(ltrb[2] - ltrb[0], ltrb[3] - ltrb[1])))
    g = RC.roi_geometry(img.size[0], img.size[1], ltrb)
    geoms.append({k: g[k] for k in RC.GEOM_KEYS})
    idx.append(slot_of[r.image_id])
    rects.append(g["crop"].rect())
fulls, crops = torch.stack(fulls), torch.stack(crops)
b = RC.to_device(RC.collate_geoms(geoms, idx), dev)
gt = crops.to(dev).float() / 255
m = RC.content_mask(rects, dev)

# ---------------------------------------------------------------------------------------------- S1
from stage1.rae import RAE  # noqa: E402  (official class)
off = RAE(encoder_name=RC.ENC_NAME, resolution=256, decoder_config_path=RC.DEC_CFG, decoder_patch_size=16,
          pretrained_decoder_path=os.path.join(RC.MODELS, "decoder.pt"), noise_tau=0.0,
          normalization_stat_path=os.path.join(RC.MODELS, "stats.pt")).to(dev).eval()
with torch.no_grad():
    x01 = crops[:16].to(dev).float() / 255
    z_off = off.encode(x01)                                     # official normalised latent (B,768,16,16)
    raw = rae.encode(crops[:16])
    z_ours = rae.normalise(raw)
    y_off = off.decode(z_off)
    y_ours = rae.decode(raw)
jprint("S1_official_vs_wrapper", dict(
    official_latent_shape=list(z_off.shape), raw_shape=list(raw.shape), decoder_out_shape=list(y_off.shape),
    encode_normalised_maxabs=float((z_off - z_ours).abs().max()), encode_normalised_meanabs=float((z_off - z_ours).abs().mean()),
    decode_maxabs=float((y_off - y_ours).abs().max()), official_training_flag=bool(off.training), noise_tau=off.noise_tau,
    official_do_normalization=bool(off.do_normalization), stats_shape=list(off.latent_mean.shape)))
del off
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------------------------- S2 / S3
with torch.no_grad():
    T = rae.encode(crops)                                        # teacher raw (64,256,768)
    Ffull = rae.encode(fulls)
    Bil = RC.bilinear(Ffull, b)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        yT = rae.decode(T, chunk=32).float()
        yB = rae.decode(Bil, chunk=32).float()
        yF = rae.decode(Ffull, chunk=32).float()
    mT = PX.rgb_metrics(yT.clamp(0, 1), gt, m, lp)
    mB = PX.rgb_metrics(yB.clamp(0, 1), gt, m, lp)
    fullgt = fulls.to(dev).float() / 255
    mF = PX.rgb_metrics(yF.clamp(0, 1), fullgt, RC.content_mask([RC.Letterbox(*Image.open(os.path.join(RC.COCO_TRAIN, f"{int(i):012d}.jpg")).size).rect()
                                                                for i in slot_of], dev), lp)
    w = b["valid_w"]
    mse_b, cos_b = PX.feat_mse_cos(Bil, T, w)
summ = lambda d: {k: float(v.mean()) for k, v in d.items()}
jprint("S2_teacher_recon_gtcrop", summ(mT))
jprint("S2b_full_image_letterbox_recon", summ(mF))
jprint("S3_bilinear_recon_gtcrop", dict(**summ(mB), feat_mse_vs_teacher=float(mse_b.mean()), feat_cos_vs_teacher=float(cos_b.mean()),
                                        svar_ratio=float(PX.svar_ratio(Bil, T, w).mean())))
PX.save_grid(os.path.join(VIZ, "sanity_teacher_bilinear.png"), [gt[:12], yT[:12].clamp(0, 1), yB[:12].clamp(0, 1)], m[:12],
             ["GT crop", "RAE teacher recon", "Bilinear recon"])

# ---------------------------------------------------------------------------------------------- S4
dec = MLRDecoder("final").to(dev)
cm = w > 0
mu_T, sd_T = T[cm].mean(0), T[cm].std(0) + 1e-6
dec.body.set_stats(mu_T, sd_T)
P = torch.as_tensor(RC.query_coords(), device=dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    Zp = dec(RC.gather_roi(Ffull, b), None, b, P).float()
Tn = rae.normalise(T)
jprint("S4_shapes_spaces", dict(
    teacher_raw=list(T.shape), full_raw=list(Ffull.shape), coordmlp_out=list(Zp.shape), roi_tokens=list(RC.gather_roi(Ffull, b).shape),
    coordmlp_out_space="raw encoder tokens (final LN w/o affine), de-normalised by CoordMLP's own per-channel teacher mean_T/std_T buffers",
    feature_loss_space="raw", decoder_input_space="raw (decoder(raw) == official RAE.decode(normalise(raw)), S1)",
    teacher_token_norm=float(T.norm(dim=-1)[cm].mean()), pred_init_token_norm=float(Zp.norm(dim=-1).mean()),
    init_pred_equals_meanT=float((Zp - mu_T).abs().max()),
    crop_normalised_latent_content_mean_abs=float(Tn.permute(0, 2, 3, 1).reshape(64, 256, 768)[cm].mean(0).abs().mean()),
    crop_normalised_latent_content_var=float(Tn.permute(0, 2, 3, 1).reshape(64, 256, 768)[cm].var(0).mean()),
    crop_normalised_latent_pad_mean_abs=float(Tn.permute(0, 2, 3, 1).reshape(64, 256, 768)[~cm].mean(0).abs().mean()) if (~cm).any() else None))

# ---------------------------------------------------------------------------------------------- S5 geometry
def coord_image(W, H):
    """RGB: R = x / W, G = y / H (256 levels each... use 16-bit via two channels is overkill; error bound reported)."""
    xs = (np.arange(W) + 0.5) / W
    ys = (np.arange(H) + 0.5) / H
    a = np.zeros((H, W, 3), np.uint8)
    a[..., 0] = np.round(xs[None, :] * 255)
    a[..., 1] = np.round(ys[:, None] * 255)
    a[..., 2] = 128
    return Image.fromarray(a)


errs_c, errs_f, errs_tok = [], [], []
for r in sel.itertuples():
    W, H = int(r.W), int(r.H)
    ltrb = (int(r.l), int(r.t), int(r.r), int(r.b))
    img = coord_image(W, H)
    g = RC.roi_geometry(W, H, ltrb)
    cr, fl = g["crop"], g["full"]
    cc = RC.u8_canvas(img.crop(ltrb), cr).float() / 255           # crop canvas
    fc = RC.u8_canvas(img, fl).float() / 255                       # full canvas
    ci, cj = np.divmod(np.arange(RC.NCELL), RC.G)
    X, Y = (cj + 0.5) * RC.PATCH, (ci + 0.5) * RC.PATCH
    xo, yo = cr.from_canvas(X, Y)
    xo, yo = xo + ltrb[0], yo + ltrb[1]                             # expected original px of each teacher cell centre
    valid = g["valid_w"] >= 0.999
    # crop canvas pixel colour at cell centre (bilinear on the canvas) -> decoded original coordinate
    grid_c = torch.tensor(np.stack([X / 128 - 1, Y / 128 - 1], -1), dtype=torch.float32).view(1, 1, -1, 2)
    vc = F.grid_sample(cc[None], grid_c, align_corners=False)[0, :, 0]
    vf = F.grid_sample(fc[None], torch.as_tensor(g["bil_grid"]).view(1, 1, -1, 2), align_corners=False)[0, :, 0]
    errs_c.append(np.abs(vc[0].numpy() * W - xo)[valid].max() / W)
    errs_f.append(np.abs(vf[0].numpy() * W - xo)[valid].max() / W)
    # ROI tokens: their crop-frame coords (CoordMLP input 'coords') must place inside tokens over the crop content
    ins = g["inside"]
    cx, cy = g["coords"][ins, 0] * RC.S, g["coords"][ins, 1] * RC.S
    hw, hh = g["coords"][ins, 2] * RC.S, g["coords"][ins, 3] * RC.S        # one full-grid cell in crop-canvas px
    out_x = (cx < cr.ox - hw) | (cx > cr.ox + cr.nw + hw)
    out_y = (cy < cr.oy - hh) | (cy > cr.oy + cr.nh + hh)
    errs_tok.append(float((out_x | out_y).mean()))
jprint("S5_geometry", dict(
    crop_canvas_cell_centre_x_err_frac_of_W_max=float(np.max(errs_c)), full_canvas_at_bil_grid_x_err_frac_of_W_max=float(np.max(errs_f)),
    quantisation_frac=1 / 255, inside_tokens_outside_crop_frac_mean=float(np.mean(errs_tok)),
    P_range=[float(P.min()), float(P.max())], P_is="teacher (crop letterbox canvas) cell centres / 16, ROI-relative",
    other_positional_inputs="u/G = full-image absolute query position; rel = token - query offsets (full-grid units); box_vector (full-grid box)",
    output_grid=[RC.G, RC.G], teacher_grid=[RC.G, RC.G]))

# ---------------------------------------------------------------------------------------------- S6 gradient
torch.manual_seed(0)
dec = MLRDecoder("final").to(dev)
dec.body.set_stats(mu_T, sd_T)
dec.train()
h_dec0, h_lp0 = RC.param_hash(rae.dec), RC.param_hash(lp)
opt = torch.optim.AdamW(dec.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
sub = list(range(8))
bs = {k: (v[sub] if torch.is_tensor(v) and v.shape[0] == 64 else v) for k, v in b.items()}
res = []
for it in range(6):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Z = dec(RC.gather_roi(Ffull, bs), None, bs, P).float()
    Z.retain_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = rae.decode(Z).float()
    l1, lpv = PX.pix_terms(y, gt[sub], m[sub], lp)
    loss = (l1 + lpv).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    names = [n for n, p in dec.named_parameters()]
    nz = [n for n, p in dec.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0]
    res.append(dict(step=it, loss=float(loss), gZ_norm=float(Z.grad.norm()), gZ_pad_cells_norm=float(Z.grad[~(bs["valid_w"] > 0)].norm()),
                    n_param_tensors=len(names), n_tensors_with_grad=len(nz),
                    last_layer_grad=float(dec.body.mlp[-1].weight.grad.norm()),
                    decoder_params_with_grad=sum(p.grad is not None for p in rae.dec.parameters()),
                    decoder_requires_grad=sum(p.requires_grad for p in rae.dec.parameters())))
    opt.step()
jprint("S6_pixel_gradient", dict(steps=res, decoder_unchanged=RC.param_hash(rae.dec) == h_dec0, lpips_unchanged=RC.param_hash(lp) == h_lp0,
                                 decoder_training_flag=bool(rae.dec.training), lpips_training_flag=bool(lp.training)))

# ---------------------------------------------------------------------------------------------- S7 memory
mem = []
for mb in (1, 2, 4, 8, 16, 32, 64):
    bi = torch.arange(mb, device=dev) % 64
    bm = {k: (v[bi] if torch.is_tensor(v) and v.shape[0] == 64 else v) for k, v in b.items()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        t0 = time.time()
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Z = dec(RC.gather_roi(Ffull, bm), None, bm, P).float()
                y = rae.decode(Z).float()
            l1, lpv = PX.pix_terms(y, gt[bi], m[bi], lp)
            (l1 + lpv).mean().backward()
            dec.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        mem.append(dict(micro_rois=mb, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, extra_peak_gb=(torch.cuda.max_memory_allocated() - base) / 2 ** 30,
                        s_per_roi=(time.time() - t0) / 3 / mb))
    except torch.cuda.OutOfMemoryError:
        mem.append(dict(micro_rois=mb, oom=True))
        torch.cuda.empty_cache()
        break
    print(mem[-1], flush=True)
jprint("S7_memory", dict(gpu=torch.cuda.get_device_name(0), total_gb=torch.cuda.get_device_properties(0).total_memory / 2 ** 30, runs=mem,
                         activation_checkpointing=False, note="process-level peak incl. frozen encoder (fp32) + decoder (fp32 weights, bf16 autocast) + LPIPS"))
print("DONE")
