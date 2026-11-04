#!/bin/bash
# After B/C training and the baseline probe: reconstruction eval (12k only) on GPU 0.
cd /workspace/Areum/research/2026-11-04/raev2_crop_recon
export SSL_CERT_FILE=/workspace/Areum/research/envs/dinov2_ord_py310/lib/python3.10/site-packages/certifi/cacert.pem
until [ -f perception_cache/runs/B/done.json ] && [ -f perception_cache/runs/C/done.json ] && [ -f logs/probe_base.flag ]; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 /workspace/Areum/research/envs/dinov2_ord_py310/bin/python code/recon_rae.py --steps 12000 --n_per 1000 > logs/recon.log 2>&1
touch logs/recon.flag
