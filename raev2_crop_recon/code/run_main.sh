#!/bin/bash
# usage: run_main.sh RUN GPU W_FEAT LAM   -> train (12k, resume-retry) -> PACO probes (12k: R2+R0 seeds 0,1,2 + test once; 4k: seed 0 val only)
cd /workspace/Areum/research/2026-11-04/raev2_crop_recon
export SSL_CERT_FILE=/workspace/Areum/research/envs/dinov2_ord_py310/lib/python3.10/site-packages/certifi/cacert.pem
PY=/workspace/Areum/research/envs/dinov2_ord_py310/bin/python
RUN=$1; GPU=$2; WF=$3; LAM=$4
OUT=perception_cache/runs/$RUN
if [ "$RUN" = "B" ]; then until [ -f logs/overfit.flag ]; do sleep 10; done; fi
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU $PY code/train_rae.py --run $RUN --w_feat $WF --lam $LAM --beta 0.2 --micro 16 --steps 12000 --resume --out $OUT >> logs/train_$RUN.log 2>&1
  [ -f $OUT/done.json ] && break
done
touch logs/train_$RUN.flag
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU $PY code/probe_rae.py --dec ${RUN}_12000=$OUT/ckpt_12000.pt --seeds 0,1,2 --test --resume --out perception_cache/probe >> logs/probe_${RUN}12k.log 2>&1 && break
done
touch logs/probe_${RUN}12k.flag
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU $PY code/probe_rae.py --dec ${RUN}_4000=$OUT/ckpt_4000.pt --seeds 0 --resume --out perception_cache/probe >> logs/probe_${RUN}4k.log 2>&1 && break
done
touch logs/probe_${RUN}4k.flag
