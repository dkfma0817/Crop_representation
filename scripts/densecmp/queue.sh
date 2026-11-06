#!/usr/bin/env bash
# Probe queue on GPU 2,3 (DDP). Re-running skips probes with done.json. QAT is resumed manually after the test step.
set -u
source "$(dirname "$0")/common.sh"
cd $ROOT
PY=/home/irteam/miniforge3/envs/evaclip_eval/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=$ROOT/cache/cuda_jit
for spec in "eva full 1024" "clipself full 1024" "declip full 1024" "eva c0 224" "eva full 560" "clipself full 560" "declip full 560"; do
  set -- $spec
  it=p_$1_$2$3; out=$R/probes/$1_$2$3
  [ -f $out/done.json ] && { set_status $it DONE "already done"; continue; }
  rm -rf $out; set_status $it RUNNING
  $PY -m torch.distributed.run --nproc_per_node 2 --master_port $((30500 + RANDOM % 400)) -m src.densecmp.train --enc $1 --mode $2 --side $3 \
      --out $out --workers 10 > $R/logs/probe_$1_$2$3.log 2>&1
  if [ -f $out/done.json ]; then set_status $it DONE "$(tail -c 300 $out/r2/curves_dev.jsonl | tail -1 | cut -c1-60)"; else set_status $it FAILED "see logs/probe_$1_$2$3.log"; fi
done
event queue "all probes finished"
