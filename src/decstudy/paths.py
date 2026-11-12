"""Feature paths of the decoder study. Every path encodes each FULL image once with the frozen G1-cross EMA encoder
(except c0 = original CLIP on the GT crop, the oracle reference). Returned `dec` follows the decoder call signature
dec(full_tok, img_index, box_c, qrel, fvalid) -> (R, 576, 1024) crop-grid tokens, so the g1clip probe / inference code
paths are used unchanged.

  kind enc          : G1-cross encoder full tokens, R2 full-canvas geometry       (mode full)  [= G1-cross Encoder Only]
  kind dec_existing : G1-cross encoder + its jointly trained ROIDecoderD          (mode dec)   [= Existing G1-cross Decoder]
  kind bilinear     : G1-cross encoder + BilinearOnly (no parameters)             (mode dec)   [B0]
  kind residual     : G1-cross encoder + ResidualDecoder EMA (job.dec_ckpt)       (mode dec)   [D1 / L1 / L2]
  kind c0           : ORIGINAL CLIP on the GT C0 crop                             (mode c0)    [crop oracle]
  kind enc_any      : encoder of ANY G1 EMA checkpoint (job.ckpt), full tokens     (mode full)  [e.g. G1-bq, 2026-11-08]
"""
import torch

from src.decstudy.model import BilinearOnly, ResidualDecoder
from src.g1clip.model import ROIDecoderD
from src.maskemb import common as MC

ENC_CKPT = "output/2026-11-07/g1clip/G1cross/ema_last.pt"
MODE = {"enc": "full", "enc_any": "full", "dec_existing": "dec", "bilinear": "dec", "residual": "dec", "c0": "c0"}


def load_path(job, dev):
    kind = job["kind"]
    info = dict(kind=kind, mode=MODE[kind])
    if kind == "c0":
        vit, ei = MC.build_encoder("clip", dev)
        info.update(encoder=ei)
        dec = None
    else:
        if kind != "enc_any":
            assert job["ckpt"] == ENC_CKPT, "every G1 path starts from the same G1-cross encoder checkpoint"
        ck = torch.load(job["ckpt"], map_location="cpu", weights_only=False)
        assert ck["kind"] == "ema" and ck["cond"] == "cross" and ck["arch"] == MC.CLIP_ARCH
        vit, _ = MC.build_encoder("clip", dev)
        vit.load_state_dict(ck["enc"])
        info.update(ckpt=job["ckpt"], ckpt_sha256=MC.sha256(job["ckpt"]), step=ck["step"])
        dec = None
        if kind == "dec_existing":
            dec = ROIDecoderD(MC.S, MC.P, MC.DIM).to(dev)
            dec.load_state_dict(ck["dec"])
        elif kind == "bilinear":
            dec = BilinearOnly(MC.S).to(dev)
        elif kind == "residual":
            dk = torch.load(job["dec_ckpt"], map_location="cpu", weights_only=False)
            assert dk["kind"] == "ema" and dk["arch"] == "ResidualDecoder" and dk["enc_ckpt"] == ENC_CKPT
            dec = ResidualDecoder(MC.S, MC.P, MC.DIM).to(dev)
            dec.load_state_dict(dk["dec"])
            info.update(dec_ckpt=job["dec_ckpt"], dec_ckpt_sha256=MC.sha256(job["dec_ckpt"]), dec_cond=dk["cond"], dec_step=dk["step"])
        if dec is not None:
            dec.eval()
    vit.eval()
    for p in list(vit.parameters()) + (list(dec.parameters()) if dec is not None else []):
        p.requires_grad_(False)
    info["decoder_built"] = dec is not None
    return vit, dec, info
