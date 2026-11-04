#!/bin/bash
# usage: run_main.sh RUN GPU   (RUN = A | B | C -> configs/raev2_crop_recon/train/$RUN.yaml)   -> train (12k, resume-retry) -> PACO probes (12k: R2+R0 seeds 0,1,2 + test once; 4k: seed 0 val only)
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-/home/irteam/envs/dinov2_ord_py310/bin/python}
export PYTHONNOUSERSITE=1
RUN=$1; GPU=$2
CFG=configs/raev2_crop_recon/train/$RUN.yaml
OUT=cache/runs/$RUN
if [ "$RUN" = "B" ]; then until [ -f logs/overfit.flag ]; do sleep 10; done; fi
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -m src.engine.train --config $CFG --out $OUT >> logs/train_$RUN.log 2>&1
  [ -f $OUT/done.json ] && break
done
touch logs/train_$RUN.flag
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -m src.eval.probe --dec ${RUN}_12000=$OUT/ckpt_12000.pt --seeds 0,1,2 --test --resume --out cache/probe >> logs/probe_${RUN}12k.log 2>&1 && break
done
touch logs/probe_${RUN}12k.flag
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -m src.eval.probe --dec ${RUN}_4000=$OUT/ckpt_4000.pt --seeds 0 --resume --out cache/probe >> logs/probe_${RUN}4k.log 2>&1 && break
done
touch logs/probe_${RUN}4k.flag
