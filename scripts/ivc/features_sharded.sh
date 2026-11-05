#!/usr/bin/env bash
# Sharded rerun of the streaming feature pass (the single-process pass was too slow): 8 shards over GPU 2,3, then merge.
set -u
cd "$(dirname "$0")/../.."
R=${1:-output/2026-11-05/ivc}
source ~/miniforge3/bin/activate dinov2_ord_py310
export OMP_NUM_THREADS=4 CUDA_CACHE_DISABLE=0 CUDA_CACHE_PATH=$(realpath cache)/cuda_jit
note() { printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" >> "$R/queue_status.tsv"; }
note features "sharded start (8 shards, GPU 2,3)"
for k in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$((2 + k % 2)) python -m src.ivc.features --out "$R/features" --shard $k --nshard 8 --workers 6 > "$R/logs/features_shard$k.log" 2>&1 &
done
wait
python -m src.ivc.features --out "$R/features" --merge > "$R/logs/features_merge.log" 2>&1 && note features "done (merged)" || note features "merge FAILED"
