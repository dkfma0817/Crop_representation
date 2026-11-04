#!/bin/bash
# S8: 256 fixed dev ROIs (dev slots 0-63, 4 steps x 64 ROIs cycled), training recipe, 1,000 steps (warmup 100). Weights discarded.
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-/home/irteam/envs/dinov2_ord_py310/bin/python}
export PYTHONNOUSERSITE=1
S=cache/overfit/slots256.json
CUDA_VISIBLE_DEVICES=0 $PY -m src.engine.train --run OF_feat --w_feat 1 --lam 0 --beta 0.2 --micro 16 --steps 1000 --warmup 100 --eval_every 250 --save_at "" --gn_every 250 --workers 6 --slots $S --out cache/overfit/feat > logs/overfit_feat.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -m src.engine.train --run OF_pix --w_feat 0 --lam 0.1 --beta 0.2 --micro 16 --steps 1000 --warmup 100 --eval_every 250 --save_at "" --gn_every 250 --workers 6 --slots $S --out cache/overfit/pix > logs/overfit_pix.log 2>&1 &
wait
touch logs/overfit.flag
