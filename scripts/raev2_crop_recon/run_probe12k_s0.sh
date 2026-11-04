#!/bin/bash
# Simplified pilot (user 2026-11-04 ~18:30 KST): after 12k training, PACO probe R2 + R0, seed 0 only (no seeds 1/2, no 4k; corrected ~18:35 KST).
# usage: run_probe12k_s0.sh RUN GPU W_FEAT LAM. If the (orphaned) training process dies before done.json, resume it.
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-/home/irteam/envs/dinov2_ord_py310/bin/python}
export PYTHONNOUSERSITE=1
RUN=$1; GPU=$2; WF=$3; LAM=$4; OUT=cache/runs/$RUN
for t in 1 2 3 4; do
  while pgrep -f "train_rae[.]py --run $RUN " > /dev/null && [ ! -f $OUT/done.json ]; do sleep 30; done
  [ -f $OUT/done.json ] && break
  CUDA_VISIBLE_DEVICES=$GPU $PY -m src.engine.train --run $RUN --w_feat $WF --lam $LAM --beta 0.2 --micro 16 --steps 12000 --resume --out $OUT >> logs/train_$RUN.log 2>&1
done
touch logs/train_$RUN.flag
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -m src.eval.probe --dec ${RUN}_12000=$OUT/ckpt_12000.pt --readers r0,r2 --seeds 0 --test --resume --out cache/probe >> logs/probe_${RUN}12k.log 2>&1 && break
done
touch logs/probe_${RUN}12k.flag
