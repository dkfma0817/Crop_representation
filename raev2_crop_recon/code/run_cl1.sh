#!/bin/bash
# User request 2026-11-04 ~20:20 KST: C without LPIPS (beta 0): L = L_feat + 0.1 * L1. Everything else = C (init, stream, 12k, micro 16, seed 0).
# GPU 7 -> PACO probe R0+R2 seed 0 (test once) -> recon (C vs C_L1, separate output folder).
cd /workspace/Areum/research/2026-11-04/raev2_crop_recon
export SSL_CERT_FILE=/workspace/Areum/research/envs/dinov2_ord_py310/lib/python3.10/site-packages/certifi/cacert.pem
PY=/workspace/Areum/research/envs/dinov2_ord_py310/bin/python
OUT=perception_cache/runs/C_L1
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 $PY code/train_rae.py --run C_L1 --w_feat 1 --lam 0.1 --beta 0 --micro 16 --steps 12000 --resume --out $OUT >> logs/train_C_L1.log 2>&1
  [ -f $OUT/done.json ] && break
done
touch logs/train_C_L1.flag
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 $PY code/probe_rae.py --dec C_L1_12000=$OUT/ckpt_12000.pt --readers r0,r2 --seeds 0 --test --resume --out perception_cache/probe >> logs/probe_C_L112k.log 2>&1 && break
done
touch logs/probe_C_L112k.flag
CUDA_VISIBLE_DEVICES=7 $PY code/recon_rae.py --steps 12000 --runs A,C,C_L1 --out recon_cl1 > logs/recon_cl1.log 2>&1
touch logs/recon_cl1.flag
