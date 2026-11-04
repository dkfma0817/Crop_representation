#!/bin/bash
# After A training ends: wait for the baseline probe's next per-epoch state save on GPU 7, stop it, resume it on GPU 0
# (frees GPU 7 for C; probe resumes from the saved epoch, nothing recomputed).
cd "$(dirname "$0")/../.."   # repo root
ST=cache/probe/state_full_bil_crop_s012.pt
until [ -f logs/train_A.flag ]; do sleep 20; done
m0=$(stat -c %Y $ST 2>/dev/null || echo 0)
until [ "$(stat -c %Y $ST 2>/dev/null || echo 0)" != "$m0" ]; do sleep 10; done
sleep 20
pkill -f "run_probe_base[.]sh"
pkill -f "probe_rae[.]py --conds full,bil,crop"
sleep 15
echo "moved at $(date -u)" >> logs/probe_base.log
sed -i 's/CUDA_VISIBLE_DEVICES=7 \$PY code\/probe_rae.py --conds full,bil,crop/CUDA_VISIBLE_DEVICES=0 $PY code\/probe_rae.py --conds full,bil,crop/' code/run_probe_base.sh
setsid nohup code/run_probe_base.sh > logs/run_probe_base.out 2>&1 < /dev/null &
