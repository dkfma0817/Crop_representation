#!/usr/bin/env bash
# Replaces queue.sh after the user dropped the 560 secondary (2026-11-06 12:3x KST). queue.sh (pid 1574846) is SIGSTOPped so the
# running CLIPSelf probe finishes normally; this waits for it, then runs DeCLIP @1024 and the EVA crop oracle @224 on GPU 2,3.
set -u
source "$(dirname "$0")/common.sh"
cd $ROOT
PY=/home/irteam/miniforge3/envs/evaclip_eval/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=$ROOT/cache/cuda_jit
until [ -f $R/probes/clipself_full1024/done.json ] || [ -f $R/probes/clipself_full1024/failed.json ]; do sleep 30; done
sleep 20
if [ -f $R/probes/clipself_full1024/done.json ]; then set_status p_clipself_full1024 DONE; else set_status p_clipself_full1024 FAILED "see log"; fi
for spec in "declip full 1024" "eva c0 224"; do
  set -- $spec
  it=p_$1_$2$3; out=$R/probes/$1_$2$3
  [ -f $out/done.json ] && { set_status $it DONE "already done"; continue; }
  rm -rf $out; set_status $it RUNNING
  $PY -m torch.distributed.run --nproc_per_node 2 --master_port $((30500 + RANDOM % 400)) -m src.densecmp.train --enc $1 --mode $2 --side $3 \
      --out $out --workers 10 > $R/logs/probe_$1_$2$3.log 2>&1
  if [ -f $out/done.json ]; then set_status $it DONE; else set_status $it FAILED "see logs/probe_$1_$2$3.log"; fi
done
event queue "queue2 finished"
