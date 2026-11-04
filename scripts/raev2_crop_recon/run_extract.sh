#!/bin/bash
# stream cache over GPUs 0,1,7 (shard 0 creates the memmaps), then hval on GPU 0; flags in logs/
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-/home/irteam/envs/dinov2_ord_py310/bin/python}
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=0 $PY -m src.data.extract_cache run stream 0 3 > logs/extract_stream0.log 2>&1 &
sleep 60
CUDA_VISIBLE_DEVICES=1 $PY -m src.data.extract_cache run stream 1 3 > logs/extract_stream1.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY -m src.data.extract_cache run stream 2 3 > logs/extract_stream2.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 $PY -m src.data.extract_cache run hval 0 1 > logs/extract_hval.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY -m src.data.extract_cache check stream > logs/check_stream.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY -m src.data.extract_cache check hval > logs/check_hval.log 2>&1
touch logs/extract.flag
