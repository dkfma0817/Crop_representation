"""ImageNet-1K frozen-feature evaluation of the G1 encoders (2026-11-07): shared constants and loaders.

Encoders (identical DINOv3-B/16 lvd1689m architecture, timm, img_size 384; only the weights differ):
  original : timm pretrained weights (= PACO probe_frozen_full)
  g1pool   : output/2026-11-06/g1_cross_distill/G1pool/ema_last.pt  key 'enc'  (= PACO probe_G1pool_full encoder_ckpt)
  g1cross  : output/2026-11-06/g1_cross_distill/G1cross/ema_last.pt key 'enc'  (= PACO probe_G1cross_full encoder_ckpt)
The G1-cross ROI decoder (key 'dec') is never loaded or instantiated.
Each encoder's fp64 parameter sum must equal the PACO probe's frozen_check.json checksum (same weights as PACO).
"""
import glob
import os

import torch

ROOT = "/home/irteam/Crop_representation"
DATA = "/home/irteam/datasets/imagenet-1k"
G1 = os.path.join(ROOT, "output/2026-11-06/g1_cross_distill")
MODELS = ("original", "g1pool", "g1cross")
CKPT = {"original": None, "g1pool": os.path.join(G1, "G1pool/ema_last.pt"), "g1cross": os.path.join(G1, "G1cross/ema_last.pt")}
PACO_PROBE = {"original": "probe_frozen_full", "g1pool": "probe_G1pool_full", "g1cross": "probe_G1cross_full"}
PACO_M0 = {"original": 44.30, "g1pool": 45.87, "g1cross": 46.34}      # reference only (PACO val R2 M0, g1_cross_distill/results.json)
S, RESIZE = 384, 438                                                     # center crop 384 after shorter-side resize int(384 * 256 / 224)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def parquet_files(split):
    key = {"train": "train", "val": "validation"}[split]
    fs = sorted(glob.glob(os.path.join(DATA, "data", f"{key}-*.parquet")))
    return fs


def checksum(m):
    return float(sum(p.double().sum() for p in m.parameters()))


def load_encoder(name, dev):
    import timm
    vit = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, img_size=S, dynamic_img_size=True)
    if CKPT[name] is not None:
        ck = torch.load(CKPT[name], map_location="cpu")
        assert ck.get("kind") == "ema", ck.get("kind")
        vit.load_state_dict(ck["enc"], strict=True)
        del ck                                                           # 'dec' (G1-cross decoder) dropped with the dict
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit.eval().to(dev)
