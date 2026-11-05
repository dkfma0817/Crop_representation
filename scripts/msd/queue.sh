#!/usr/bin/env bash
# MSD pilot queue (GPU 2+3 only). Runs independently in tmux:
#   tmux new -d -s msd_q "bash scripts/msd/queue.sh output/2026-11-05/msd"
# 0. pause the user's low-bit QAT job on GPU 2+3 right after its next checkpoint is fully written
# 1. COCO decoder pre-training H0 -> H1 -> H2 -> H3 (each 2-GPU DDP, 12k steps); the PACO R2 probe of a finished
#    condition (encoder + decoder FROZEN) runs while the next condition pre-trains; references (bilinear, Full ML+R2, C0)
#    are probed under the same protocol
# 2. direct checks + report;  3. resume the QAT job (--resume, same command / env / output dir)
set -u
cd "$(dirname "$0")/../.."
RUN=${1:?usage: queue.sh <run dir>}
mkdir -p "$RUN/logs"
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=$(realpath cache)/cuda_jit
source ~/miniforge3/bin/activate dinov2_ord_py310
ST="$RUN/queue_status.tsv"
EXCL=output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json
QAT_PID=${QAT_PID:-591943}
QAT_OUT=/home/irteam/low-bit-training-v2/output/2026-11-04/scale-800m/800m-ste-code7.5e-5-compile
QAT_SAVE=/home/irteam/Crop_representation/output/2026-11-05/qat_pause
note() { printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" | tee -a "$ST"; }
echo $$ > "$RUN/queue.pid"
note queue "start pid=$$"

# ---------------------------------------------------------------- 0. pause QAT after its next checkpoint
if kill -0 "$QAT_PID" 2>/dev/null; then
  last=$(ls -d "$QAT_OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  next=$((last + 5000))
  note qat "running; waiting for checkpoint-$next (last $last) before pausing"
  until [ -f "$QAT_OUT/checkpoint-$next/trainer_state.json" ]; do
    kill -0 "$QAT_PID" 2>/dev/null || { note qat "process ended by itself"; break; }
    sleep 20
  done
  sleep 90    # let the save (and rotation of the oldest checkpoint) finish
  echo "$next" > "$QAT_SAVE/paused_at_checkpoint.txt"
  note qat "pausing (SIGTERM torchrun $QAT_PID) after checkpoint-$next"
  kill -TERM "$QAT_PID" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$QAT_PID" 2>/dev/null || break; sleep 5; done
  kill -0 "$QAT_PID" 2>/dev/null && { note qat "still alive -> SIGKILL"; pkill -KILL -P "$QAT_PID"; kill -KILL "$QAT_PID"; }
  sleep 10
  note qat "paused; GPU: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n 3,4p | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- 1. pre-training + probes
declare -A PID
port=30200
probe() {   # probe <name> [extra args]   (background)
  local name=$1; shift
  local out="$RUN/probe_$name"
  [ -f "$out/done.json" ] && { note "probe_$name" "skip (done)"; return; }
  [ -d "$out" ] && mv "$out" "${out}_partial_$(date +%s)"
  port=$((port + 7))
  note "probe_$name" "start"
  ( torchrun --nproc_per_node 2 --master_port $port -m src.readout.train --job "configs/msd/probe_$name.yaml" --out "$out" \
      --exclude "$EXCL" --workers 8 "$@" > "$RUN/logs/probe_$name.log" 2>&1 \
      && note "probe_$name" "done" || note "probe_$name" "FAILED (log: $RUN/logs/probe_$name.log)" ) &
  PID[$name]=$!
}
pretrain() {
  local c=$1 out="$RUN/$1"
  [ -f "$out/done.json" ] && { note "$c" "skip (done)"; return 0; }
  [ -d "$out" ] && [ ! -f "$out/done.json" ] && mv "$out" "${out}_partial_$(date +%s)"
  port=$((port + 7))
  note "$c" "pretrain start (COCO stream, 12k steps)"
  torchrun --nproc_per_node 2 --master_port $port -m src.msd.pretrain --cond "$c" --out "$out" --workers 10 > "$RUN/logs/$c.log" 2>&1
  if [ -f "$out/done.json" ]; then note "$c" "pretrain done ($(python -c "import json; d=json.load(open('$out/done.json')); print('selected step', d['selected']['step'], 'dev L12 %.4f' % d['selected']['dev']['L12'], 'time %ds' % d['time_s'])"))"; return 0; fi
  note "$c" "pretrain FAILED (log: $RUN/logs/$c.log)"; return 1
}

probe bil
for c in H0 H1 H2 H3; do
  if pretrain "$c"; then probe "$c"; fi
  [ "$c" = "H1" ] && probe mlr2
  [ "$c" = "H2" ] && probe c0
done
note queue "pre-training finished; waiting for probes"
wait
# ---------------------------------------------------------------- 2. direct checks + report
note direct "start"
if CUDA_VISIBLE_DEVICES=2 python -m src.msd.direct --run "$RUN" > "$RUN/logs/direct.log" 2>&1; then note direct "done"; else note direct "FAILED (log: $RUN/logs/direct.log)"; fi
note report "start"
if python -m src.msd.report --run "$RUN" > "$RUN/logs/report.log" 2>&1; then note report "done -> $RUN/REPORT.md"; else note report "FAILED (log: $RUN/logs/report.log)"; fi

# ---------------------------------------------------------------- 3. resume QAT
if [ -f "$QAT_SAVE/paused_at_checkpoint.txt" ] && [ ! -f "$QAT_SAVE/resumed.txt" ]; then
  note qat "resuming with --resume from $(ls -d "$QAT_OUT"/checkpoint-* | sort -V | tail -1)"
  ( cd "$(cat "$QAT_SAVE/cwd.txt")" && mapfile -t ENVV < "$QAT_SAVE/environ.txt" && mapfile -t CMD < "$QAT_SAVE/cmdline.txt" \
    && setsid env -i "${ENVV[@]}" "${CMD[@]}" --resume >> "$QAT_OUT/train.log" 2>&1 < /dev/null & echo $! > "$QAT_SAVE/resumed_pid.txt" )
  sleep 5
  echo "$(date '+%F %T') pid $(cat "$QAT_SAVE/resumed_pid.txt")" > "$QAT_SAVE/resumed.txt"
  note qat "resumed pid $(cat "$QAT_SAVE/resumed_pid.txt") (log appended to $QAT_OUT/train.log)"
fi
note queue "end"
