#!/bin/bash
# After B/C training and the baseline probe: reconstruction eval (12k only) on GPU 0.
cd "$(dirname "$0")/../.."   # repo root
until [ -f cache/runs/B/done.json ] && [ -f cache/runs/C/done.json ] && [ -f logs/probe_base.flag ]; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 $PY -m src.eval.recon --steps 12000 --n_per 1000 > logs/recon.log 2>&1
touch logs/recon.flag
