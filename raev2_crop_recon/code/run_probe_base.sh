#!/bin/bash
# RAE baselines Full / Bilinear / GT-crop: R2 + R0, reader seeds 0,1,2, test scored once from the best-val epoch.
cd /workspace/Areum/research/2026-11-04/raev2_crop_recon
export SSL_CERT_FILE=/workspace/Areum/research/envs/dinov2_ord_py310/lib/python3.10/site-packages/certifi/cacert.pem
PY=/workspace/Areum/research/envs/dinov2_ord_py310/bin/python
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 $PY code/probe_rae.py --conds full,bil,crop --seeds 0,1,2 --test --resume --out perception_cache/probe >> logs/probe_base.log 2>&1 && break
done
touch logs/probe_base.flag
