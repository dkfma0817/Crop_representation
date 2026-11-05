#!/usr/bin/env bash
# Shared-feature readout queue v2 (faster): independent jobs run CONCURRENTLY on GPU 2+3 (each job = its own 2-GPU DDP
# process, own init, same data order / protocol as v1). Dependencies:
#   ctrl_c0, ctrl_f672, E09, E11            start immediately (ctrl_f336 already done in v1; skipped if done.json)
#   D0                                      after ctrl_f336 + ctrl_c0      -> slot1 (E05 | E08) starts
#   E12                                     after ctrl_f672 (teacher)
#   report                                  after everything
# Speed-ups that do not change the protocol: CUDA JIT cache (NCCL init 170 s -> 1 s), uint8 canvas cache for F336/F672
# jobs (same PIL output, GPU normalisation + geometry), no per-step host syncs, one fused gradient all-reduce.
#   tmux new -d -s readout_q2 "bash scripts/readout_queue/queue2.sh output/2026-11-04/readout_queue"
set -u
cd "$(dirname "$0")/../.."
RUN=${1:?usage: queue2.sh <run dir>}
MAXJ=${MAXJ:-4}
CACHE=${CACHE:-/dev/shm/readout_cache}
mkdir -p "$RUN/logs"
export READOUT_RUN=$(realpath "$RUN")
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=4
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=$(realpath cache)/cuda_jit
source ~/miniforge3/bin/activate dinov2_ord_py310
STATUS="$RUN/queue_status.tsv"
EXCL="$RUN/data_check/train_exclude_image_ids.json"
J=configs/readout_queue/jobs
echo $$ > "$RUN/queue2.pid"
note() { printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" | tee -a "$STATUS"; }
note queue2 "start pid=$$ maxj=$MAXJ cache=$CACHE"

# ---- gates: data check + canvas cache
python -c "import json,sys; sys.exit(0 if json.load(open('$RUN/data_check/report.json'))['ok'] else 1)" || { note queue2 "ABORT: data_check not ok"; exit 2; }
if [ ! -f "$CACHE/done.json" ]; then
  note cache "build $CACHE"
  python -m src.readout.fastdata build --out "$CACHE" --workers 64 > "$RUN/logs/cache_build.log" 2>&1 || { note cache "FAILED -> jobs fall back to on-the-fly PIL"; CACHE=""; }
fi

declare -A PID
port=29900
launch() {   # launch <job> [cache dir or empty]
  local name=$1 cache=${2:-} out="$RUN/$1"
  if [ -f "$out/done.json" ]; then note "$name" "skip (done.json exists)"; return 0; fi
  [ -f "$out/failed.json" ] && mv "$out/failed.json" "$out/failed_prev_$(date +%s).json"
  if [ -d "$out" ] && [ ! -f "$out/done.json" ]; then mv "$out" "${out}_partial_$(date +%s)"; fi   # never mix two runs in one folder
  port=$((port + 7))
  local extra="" workers=6
  [ -n "$cache" ] && extra="--cache $cache"
  [ "$name" = "ctrl_c0" ] && workers=16
  note "$name" "start (port $port, ${cache:+cache}${cache:-on-the-fly})"
  ( torchrun --nproc_per_node 2 --master_port $port -m src.readout.train --job "$J/$name.yaml" --out "$out" \
      --exclude "$EXCL" --workers $workers $extra > "$RUN/logs/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$out/done.json" ]; then note "$name" "done"; else note "$name" "FAILED rc=$rc (log: $RUN/logs/$name.log)"; fi
    if grep -q "train / val image overlap" "$RUN/logs/$name.log" 2>/dev/null; then note queue2 "ABORT: train/val overlap"; touch "$RUN/ABORT"; fi
  ) &
  PID[$name]=$!
}
running() { local n=0; for k in "${!PID[@]}"; do kill -0 "${PID[$k]}" 2>/dev/null && n=$((n + 1)); done; echo $n; }
wait_slot() { while [ "$(running)" -ge "$MAXJ" ]; do sleep 10; done; }
finished() { [ -n "${PID[$1]:-}" ] && ! kill -0 "${PID[$1]}" 2>/dev/null; }
isdone() { [ -f "$RUN/$1/done.json" ]; }
abort_check() { if [ -f "$RUN/ABORT" ]; then note queue2 "aborting: killing running jobs"; for k in "${!PID[@]}"; do pkill -P "${PID[$k]}" 2>/dev/null; done; exit 3; fi; }

isdone ctrl_f336 || { wait_slot; launch ctrl_f336 "$CACHE"; }
wait_slot; launch ctrl_c0 ""
wait_slot; launch ctrl_f672 "$CACHE"
wait_slot; launch E11 "$CACHE"
wait_slot; launch E09 "$CACHE"

D0_DONE=0; E12_DONE=0
while :; do
  abort_check
  if [ $D0_DONE -eq 0 ] && { isdone ctrl_f336 || finished ctrl_f336; } && { isdone ctrl_c0 || finished ctrl_c0; }; then
    SLOT1=E08
    if isdone ctrl_c0 && isdone ctrl_f336; then
      note D0 "start"
      if python -m src.readout.d0_binding --run "$RUN" --out "$RUN/D0" --exclude "$EXCL" > "$RUN/logs/D0.log" 2>&1; then
        SLOT1=$(cat "$RUN/D0/slot1.txt"); note D0 "done -> slot1=$SLOT1"
      else note D0 "FAILED (log: $RUN/logs/D0.log) -> slot1=E08 (pre-registered default)"; fi
    else note D0 "skipped: ctrl_f336 or ctrl_c0 failed -> slot1=E08 (pre-registered default)"; fi
    echo "$SLOT1" > "$RUN/slot1_used.txt"
    wait_slot; launch "$SLOT1" "$CACHE"; D0_DONE=1
  fi
  if [ $E12_DONE -eq 0 ] && { isdone ctrl_f672 || finished ctrl_f672; }; then
    if isdone ctrl_f672; then wait_slot; launch E12 "$CACHE"; else note E12 "skipped: teacher ctrl_f672 failed"; fi
    E12_DONE=1
  fi
  if [ $D0_DONE -eq 1 ] && [ $E12_DONE -eq 1 ] && [ "$(running)" -eq 0 ]; then break; fi
  sleep 15
done

note report "start"
if python -m src.readout.report --run "$RUN" --exclude "$EXCL" > "$RUN/logs/report.log" 2>&1; then note report "done -> $RUN/REPORT.md"
else note report "FAILED (log: $RUN/logs/report.log)"; fi
note queue2 "end"
