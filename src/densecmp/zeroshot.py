"""Sanity check that this environment reproduces the official region features (NOT part of the PACO protocol).

Open-vocabulary box classification as in CLIPSelf training/zero_shot.py: dense map (official path, fp16 autocast)
-> torchvision roi_align(output 1x1, spatial_scale 1, sampling -1, aligned True) on the feature map with boxes in token
units -> L2-normalise -> argmax cosine vs the repo's text embeddings metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy
(133 COCO-panoptic classes, things first in id order). Boxes = non-crowd COCO val2017 INSTANCE boxes (80 thing classes;
COCO panoptic annotations are not on this server), 1024 letterbox (ours). Metric = Top-1 / Top-5 mean per-class accuracy.
Paper (CLIPSelf Table 2, ViT-B/16, panoptic Boxes Top-1/Top-5): EVA-CLIP 18.2 / 33.2, CLIPSelf 72.1 / 91.3.

  CUDA_VISIBLE_DEVICES=2 python -m src.densecmp.zeroshot --enc clipself --out <dir>
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

from src.densecmp import geom
from src.densecmp.encoders import PATCH, Encoder

ANN = "/home/irteam/datasets/coco/annotations/instances_val2017.json"
IMG = "/home/irteam/datasets/coco/val2017"
EMB = "/home/irteam/ext_repos/CLIPSelf/metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    enc = Encoder(a.enc).to(dev)
    coco = json.load(open(ANN))
    cats = sorted(c["id"] for c in coco["categories"])
    cid2k = {c: k for k, c in enumerate(cats)}                       # things = first 80 rows of the panoptic embedding
    emb = F.normalize(torch.from_numpy(np.load(EMB)).float(), dim=-1).to(dev)
    assert emb.shape[0] == 133
    by_img = defaultdict(list)
    for an in coco["annotations"]:
        if not an["iscrowd"] and an["bbox"][2] >= 1 and an["bbox"][3] >= 1:
            by_img[an["image_id"]].append(an)
    imgs = sorted(by_img)[: a.limit or None]
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    hits1, hits5 = defaultdict(list), defaultdict(list)
    G = a.side // PATCH
    for k in range(0, len(imgs), 8):
        ids = imgs[k:k + 8]
        xs, boxes, labels = [], [], []
        for j, i in enumerate(ids):
            im = Image.open(os.path.join(IMG, files[i])).convert("RGB")
            lb = geom.Letterbox(*im.size, a.side)
            xs.append(geom.to_tensor_canvas(im, lb, geom.OPENAI_CLIP))
            for an in by_img[i]:
                x, y, w, h = an["bbox"]
                X0, Y0 = lb.to_canvas(x, y)
                X1, Y1 = lb.to_canvas(x + w, y + h)
                boxes.append([j, X0 / PATCH, Y0 / PATCH, X1 / PATCH, Y1 / PATCH])
                labels.append(cid2k[an["category_id"]])
        f = enc(torch.stack(xs).to(dev))                                   # (B, G*G, 512)
        fm = f.transpose(1, 2).reshape(len(ids), -1, G, G)
        r = roi_align(fm, torch.tensor(boxes, dtype=torch.float32, device=dev), (1, 1), 1.0, -1, True)[..., 0, 0]
        s = F.normalize(r, dim=-1) @ emb.T
        top5 = s.topk(5, dim=-1).indices.cpu().numpy()
        for lab, t5 in zip(labels, top5):
            hits1[lab].append(t5[0] == lab)
            hits5[lab].append(lab in t5)
    res = dict(encoder=a.enc, side=a.side, n_images=len(imgs), n_boxes=sum(len(v) for v in hits1.values()), n_classes=len(hits1),
               top1_macc=100 * float(np.mean([np.mean(v) for v in hits1.values()])), top5_macc=100 * float(np.mean([np.mean(v) for v in hits5.values()])))
    json.dump(res, open(os.path.join(a.out, f"zeroshot_{a.enc}.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
