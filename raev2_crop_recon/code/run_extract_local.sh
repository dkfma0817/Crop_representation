#!/bin/bash
# This server (rebuilt ROI lists, code/make_rois.py): stream cache sharded over $GPUS (one shard per entry, e.g. "2 3 2 3" =
# two processes per GPU; shard 0 creates the memmaps), then dev + hval, written to $FEAT_WORK (RAM) and copied to
# perception_cache/feat; cache checks on the copied files; then the shared CoordMLP init. Logs / flags in logs/.
#   GPUS="2 3 2 3" EXTRACT_WORKERS=28 EXTRACT_BS=64 nohup bash code/run_extract_local.sh > logs/run_extract_local.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/home/irteam/envs/dinov2_ord_py310/bin/python}
export PYTHONNOUSERSITE=1
export FEAT_WORK=${FEAT_WORK:-/dev/shm/raev2_feat}
FEAT=perception_cache/feat
read -r -a G <<< "${GPUS:-2 3}"
N=${#G[@]}
mkdir -p logs "$FEAT_WORK"
rm -f "$FEAT_WORK"/stream_*.npy "$FEAT"/stream_done_*.json
CUDA_VISIBLE_DEVICES=${G[0]} $PY code/extract_cache.py run stream 0 $N > logs/extract_stream0.log 2>&1 &
sleep 60
for ((k = 1; k < N; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} $PY code/extract_cache.py run stream $k $N > logs/extract_stream$k.log 2>&1 &
done
wait
CUDA_VISIBLE_DEVICES=${G[0]} $PY code/extract_cache.py run dev 0 1 > logs/extract_dev.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G[$((N - 1))]} $PY code/extract_cache.py run hval 0 1 > logs/extract_hval.log 2>&1 &
wait
echo "copy $FEAT_WORK -> $FEAT  $(date)"
cp "$FEAT_WORK"/*.npy "$FEAT"/
echo "copied $(date)"
unset FEAT_WORK                       # checks read the copied files in perception_cache/feat
for s in stream dev hval; do
  CUDA_VISIBLE_DEVICES=${G[0]} $PY code/extract_cache.py check $s > logs/check_$s.log 2>&1
done
rm -rf /dev/shm/raev2_feat
touch logs/extract.flag
CUDA_VISIBLE_DEVICES=${G[0]} $PY code/train_rae.py make_init > logs/make_init.log 2>&1
touch logs/init.flag
