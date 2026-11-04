#!/bin/bash
# S8: 256 fixed dev ROIs (dev slots 0-63, 4 steps x 64 ROIs cycled), training recipe, 1,000 steps (warmup 100). Weights discarded.
cd /workspace/Areum/research/2026-11-04/raev2_crop_recon
export SSL_CERT_FILE=/workspace/Areum/research/envs/dinov2_ord_py310/lib/python3.10/site-packages/certifi/cacert.pem
PY=/workspace/Areum/research/envs/dinov2_ord_py310/bin/python
S=perception_cache/overfit/slots256.json
CUDA_VISIBLE_DEVICES=0 $PY code/train_rae.py --run OF_feat --w_feat 1 --lam 0 --beta 0.2 --micro 16 --steps 1000 --warmup 100 --eval_every 250 --save_at "" --gn_every 250 --workers 6 --slots $S --out perception_cache/overfit/feat > logs/overfit_feat.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY code/train_rae.py --run OF_pix --w_feat 0 --lam 0.1 --beta 0.2 --micro 16 --steps 1000 --warmup 100 --eval_every 250 --save_at "" --gn_every 250 --workers 6 --slots $S --out perception_cache/overfit/pix > logs/overfit_pix.log 2>&1 &
wait
touch logs/overfit.flag
