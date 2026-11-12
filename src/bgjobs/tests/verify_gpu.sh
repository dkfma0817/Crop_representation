#!/usr/bin/env bash
# Real-GPU verification: A on GPU 0, B on GPU 1 (short), resume continuity (B straight vs stop + resume), SIGTERM safe stop.
set -u
cd "$(dirname "$0")/../../.."
P=/home/irteam/envs/dinov2_ord_py310/bin/python
V=output/2026-11-09/bgjobs_verify
rm -rf "$V"; mkdir -p "$V"
BASE=configs/bgjobs/base.yaml
COM=(--config "$BASE" --managed --set log.every=10 --set ckpt.save_every=30 --set log.grad_check_every=100000 --bg-class crop_aware_pretrain_bg)
T=output/2026-11-09/bgjobs/shared/teacher_logits/done.json
while [ ! -f "$T" ]; do read -t 5 < /dev/zero; done
echo "teacher ready $(date +%T)"
run() { d=$1; g=$2; shift 2; mkdir -p "$V/$d"; "$P" -u -m src.bgjobs.train --gpu "$g" "${COM[@]}" --set run_dir="$V/$d" "$@" > "$V/$d.$RANDOM.log" 2>&1; echo "$d rc=$? $(date +%T)"; }
run A0 0 --stop-after-steps 30
run B60 1 --set loss.lambda_rank=auto --stop-after-steps 60
run B30 1 --set loss.lambda_rank=auto --stop-after-steps 30
run B30 1 --set loss.lambda_rank=auto --stop-after-steps 30
mkdir -p "$V/S"; "$P" -u -m src.bgjobs.train --gpu 1 "${COM[@]}" --set run_dir="$V/S" --set loss.lambda_rank=auto --stop-after-steps 100000 > "$V/S.log" 2>&1 &
PID=$!
until [ -f "$V/S/logs/heartbeat.json" ] && [ "$(python3 -c "import json;print(json.load(open('$V/S/logs/heartbeat.json'))['step'])")" -ge 40 ]; do read -t 2 < /dev/zero; done
kill -TERM $PID; wait $PID; echo "SIGTERM exit $?"
cat "$V/S/control/last_train_exit.json" | tr -d '\n'; echo; ls "$V/S/checkpoints"
echo FINISHED
