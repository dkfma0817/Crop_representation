#!/bin/bash
# S8: 256 fixed dev ROIs (dev slots 0-63, 4 steps x 64 ROIs cycled), training recipe, 1,000 steps (warmup 100). Weights discarded.
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-/home/irteam/envs/dinov2_ord_py310/bin/python}
export PYTHONNOUSERSITE=1
S=cache/overfit/slots256.json
CUDA_VISIBLE_DEVICES=0 $PY -m src.engine.train --config configs/raev2_crop_recon/train/overfit_feat.yaml --slots $S > logs/overfit_feat.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -m src.engine.train --config configs/raev2_crop_recon/train/overfit_pix.yaml --slots $S > logs/overfit_pix.log 2>&1 &
wait
touch logs/overfit.flag
