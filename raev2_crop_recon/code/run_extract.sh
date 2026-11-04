#!/bin/bash
# stream cache over GPUs 0,1,7 (shard 0 creates the memmaps), then hval on GPU 0; flags in logs/
cd /workspace/Areum/research/2026-11-04/raev2_crop_recon
export SSL_CERT_FILE=/workspace/Areum/research/envs/dinov2_ord_py310/lib/python3.10/site-packages/certifi/cacert.pem
PY=/workspace/Areum/research/envs/dinov2_ord_py310/bin/python
CUDA_VISIBLE_DEVICES=0 $PY code/extract_cache.py run stream 0 3 > logs/extract_stream0.log 2>&1 &
sleep 60
CUDA_VISIBLE_DEVICES=1 $PY code/extract_cache.py run stream 1 3 > logs/extract_stream1.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY code/extract_cache.py run stream 2 3 > logs/extract_stream2.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 $PY code/extract_cache.py run hval 0 1 > logs/extract_hval.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY code/extract_cache.py check stream > logs/check_stream.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY code/extract_cache.py check hval > logs/check_hval.log 2>&1
touch logs/extract.flag
