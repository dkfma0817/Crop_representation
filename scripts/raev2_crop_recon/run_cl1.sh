#!/bin/bash
# User request 2026-11-04 ~20:20 KST: C without LPIPS (beta 0): L = L_feat + 0.1 * L1. Everything else = C (init, stream, 12k, micro 16, seed 0).
# GPU 7 -> PACO probe R0+R2 seed 0 (test once) -> recon (C vs C_L1, separate output folder).
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-/home/irteam/envs/dinov2_ord_py310/bin/python}
export PYTHONNOUSERSITE=1
OUT=cache/runs/C_L1
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 $PY -m src.engine.train --config configs/raev2_crop_recon/train/C_L1.yaml --out $OUT >> logs/train_C_L1.log 2>&1
  [ -f $OUT/done.json ] && break
done
touch logs/train_C_L1.flag
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 $PY -m src.eval.probe --dec C_L1_12000=$OUT/ckpt_12000.pt --readers r0,r2 --seeds 0 --test --resume --out cache/probe >> logs/probe_C_L112k.log 2>&1 && break
done
touch logs/probe_C_L112k.flag
CUDA_VISIBLE_DEVICES=7 $PY -m src.eval.recon --steps 12000 --runs A,C,C_L1 --out recon_cl1 > logs/recon_cl1.log 2>&1
touch logs/recon_cl1.flag
