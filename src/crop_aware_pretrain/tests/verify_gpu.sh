#!/usr/bin/env bash
# Real-GPU verification on two idle GPUs G0 G1 (default 2 3): resume continuity, world-size invariance + checkpoint compatibility, lock, busy guard, SIGTERM.
set -u
cd "$(dirname "$0")/../../.."
G0=${1:-2}; G1=${2:-3}
P=${PY:-/home/irteam/envs/dinov2_ord_py310/bin/python}
V=output/2026-11-09/dinov3_crop_aware_pretrain_verify
rm -rf "$V"; mkdir -p "$V"
DATA=$PWD/output/2026-11-09/dinov3_crop_aware_pretrain/data
COMMON=(--set log.every=5 --set ckpt.save_every=1000 --set log.diag_every=100000 --set log.grad_check_every=100000)
mkd() { mkdir -p "$V/$1"; ln -sfn "$DATA" "$V/$1/data"; }
run1() { d=$1; g=$2; shift 2; "$P" -u -m src.crop_aware_pretrain.train --managed --gpu "$g" "${COMMON[@]}" --set run_dir="$V/$d" "$@" > "$V/$d.$RANDOM.log" 2>&1; }
run2() {  # DDP world 2 on G0,G1 (same environment as supervisor.launch_train)
  d=$1; shift
  export CUDA_VISIBLE_DEVICES=$G0,$G1 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + RANDOM % 20000)) WORLD_SIZE=2
  for r in 0 1; do
    RANK=$r LOCAL_RANK=$r "$P" -u -m src.crop_aware_pretrain.train --managed "${COMMON[@]}" --set run_dir="$V/$d" "$@" > "$V/$d.rank$r.log" 2>&1 &
    pid[$r]=$!
  done
  wait "${pid[0]}"; rc0=$?; wait "${pid[1]}"; rc1=$?; unset CUDA_VISIBLE_DEVICES MASTER_ADDR MASTER_PORT WORLD_SIZE
  echo "run2 $d rc=$rc0,$rc1"
}
echo "A: world 1 continuous 0 -> 120"; mkd A; run1 A "$G0" --stop-after-steps 120
echo "B: world 1 stop at 60 (exact stop-after) then resume to 120"; mkd B; run1 B "$G0" --stop-after-steps 60; run1 B "$G0" --resume auto --stop-after-steps 60
echo "C: world 2 continuous 0 -> 120"; mkd C; run2 C --stop-after-steps 120
echo "D: world 2 to 60, resume world 1 to 120 (2 -> 1 ckpt)"; mkd D; run2 D --stop-after-steps 60; run1 D "$G0" --resume auto --stop-after-steps 60
echo "E: world 1 to 60, resume world 2 to 120 (1 -> 2 ckpt)"; mkd E; run1 E "$G0" --stop-after-steps 60; run2 E --resume auto --stop-after-steps 60
echo "guard/lock tests"; mkd L
"$P" -u -m src.crop_aware_pretrain.train --managed --gpu "$G0" "${COMMON[@]}" --set run_dir="$V/L" --stop-after-steps 100000 > "$V/L.log" 2>&1 &
LP=$!
until grep -q '^{"time"' "$V/L.log" 2>/dev/null || ! kill -0 $LP 2>/dev/null; do read -t 2 < /dev/zero; done
sleep_a=0; until [ -f "$V/L/logs/heartbeat.json" ] && [ "$sleep_a" -ge 1 ]; do read -t 5 < /dev/zero; sleep_a=$((sleep_a+1)); done
"$P" -u -m src.crop_aware_pretrain.train --gpu "$G0" "${COMMON[@]}" --set run_dir="$V/L2" --stop-after-steps 1 > "$V/L2.log" 2>&1; echo "busy-guard exit code (expect 76): $?"
"$P" -u -m src.crop_aware_pretrain.train --managed --gpu "$G0" "${COMMON[@]}" --set run_dir="$V/L" --stop-after-steps 1 > "$V/L3.log" 2>&1; echo "duplicate-run lock exit code (expect 75): $?"
step_before=$("$P" -c "import json;print(json.load(open('$V/L/logs/heartbeat.json'))['step'])")
echo "SIGTERM at step ~$step_before"; kill -TERM $LP; wait $LP; echo "SIGTERM exit code (expect 0): $?"
cat "$V/L/control/last_train_exit.json"; echo; ls "$V/L/checkpoints"
"$P" -c "
import json;e=json.load(open('$V/L/control/last_train_exit.json'))
assert e['reason'].startswith('signal_'),e; print('stopped by signal at step',e['step'],'-> checkpoint written')"
"$P" -u -m src.crop_aware_pretrain.train --managed --gpu "$G0" "${COMMON[@]}" --set run_dir="$V/L" --resume auto --stop-after-steps 10 > "$V/L4.log" 2>&1; echo "resume after signal rc=$?"
grep -h '"start_step"' "$V/L4.log" | tail -1 | cut -c1-300
echo FINISHED
