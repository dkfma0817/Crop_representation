#!/bin/bash
# RAE baselines Full / Bilinear / GT-crop: R2 + R0, reader seeds 0,1,2, test scored once from the best-val epoch.
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-/home/irteam/envs/dinov2_ord_py310/bin/python}
export PYTHONNOUSERSITE=1
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 $PY -m src.eval.probe --conds full,bil,crop --seeds 0,1,2 --test --resume --out cache/probe >> logs/probe_base.log 2>&1 && break
done
touch logs/probe_base.flag
