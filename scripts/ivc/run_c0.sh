#!/usr/bin/env bash
# Re-run of the C0 probe from scratch (the first run was killed by SIGHUP after epoch 6 when the queue's tmux session closed).
set -u
cd "$(dirname "$0")/../.."
R=output/2026-11-05/ivc
source ~/miniforge3/bin/activate dinov2_ord_py310
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 CUDA_CACHE_DISABLE=0 CUDA_CACHE_PATH=$(realpath cache)/cuda_jit
note() { printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" >> "$R/queue_status.tsv"; }
note probe_c0 "restart from scratch (first run SIGHUP after epoch 6 -> $R/probe_c0_partial_sighup_ep6)"
torchrun --nproc_per_node 2 --master_port 30777 -m src.ivc.train --job configs/ivc/probe_c0.yaml --out "$R/probe_c0" \
  --exclude output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json --workers 12 > "$R/logs/probe_c0_rerun.log" 2>&1 \
  && note probe_c0 "done" || note probe_c0 "FAILED (log $R/logs/probe_c0_rerun.log)"
