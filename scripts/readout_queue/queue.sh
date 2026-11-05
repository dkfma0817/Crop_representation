#!/usr/bin/env bash
# Shared-feature readout queue: runs independently in tmux (no Claude Code session needed).
#   tmux new -d -s readout_q "bash scripts/readout_queue/queue.sh output/<date>/readout_queue"
# Order: data_check (abort on failure) -> ctrl_f336 -> ctrl_c0 -> ctrl_f672 -> D0 -> slot1 (E05|E08) -> E09 -> E11 -> E12 -> report
# GPU 2+3 DDP, seed 0, PACO val only. A job with done.json is skipped (re-running the script resumes the queue).
# A failed job is logged (failed.json + log) and the next independent job starts; E12 needs ctrl_f672.
set -u
cd "$(dirname "$0")/../.."
RUN=${1:?usage: queue.sh <run dir>}
mkdir -p "$RUN/logs"
export READOUT_RUN=$(realpath "$RUN")
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=8
source ~/miniforge3/bin/activate dinov2_ord_py310
STATUS="$RUN/queue_status.tsv"
EXCL="$RUN/data_check/train_exclude_image_ids.json"
J=configs/readout_queue/jobs
echo $$ > "$RUN/queue.pid"
stamp() { date '+%F %T'; }
note() { printf '%s\t%s\t%s\n' "$(stamp)" "$1" "$2" | tee -a "$STATUS"; }

note queue "start pid=$$ host=$(hostname)"

# ---- 0. data check: any train/val/test overlap left after exclusion -> abort the whole queue
if [ ! -f "$RUN/data_check/report.json" ] || ! python -c "import json,sys; sys.exit(0 if json.load(open('$RUN/data_check/report.json'))['ok'] else 1)"; then
  python -m src.readout.data_check --out "$RUN/data_check" > "$RUN/logs/data_check.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then note data_check "FAILED rc=$rc -> queue aborted"; exit 2; fi
fi
note data_check "ok ($(python -c "import json; r=json.load(open('$RUN/data_check/report.json')); print('train excluded', r['n_train_excluded'], 'left', r['train_vs_eval_after_exclusion'])"))"

run_job() {   # run_job <name>
  local name=$1 out="$RUN/$1" port=$((29700 + RANDOM % 200))
  if [ -f "$out/done.json" ]; then note "$name" "skip (done.json exists)"; return 0; fi
  if [ -f "$out/failed.json" ]; then mv "$out/failed.json" "$out/failed_prev_$(date +%s).json"; fi
  note "$name" "start"
  torchrun --nproc_per_node 2 --master_port $port -m src.readout.train --job "$J/$name.yaml" --out "$out" \
    --exclude "$EXCL" --workers 12 > "$RUN/logs/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$out/done.json" ]; then note "$name" "done"; return 0; fi
  note "$name" "FAILED rc=$rc (log: $RUN/logs/$name.log)"
  # a train/val overlap assertion invalidates every comparison -> stop everything
  if grep -q "train / val image overlap" "$RUN/logs/$name.log"; then note queue "ABORT: train/val overlap"; exit 3; fi
  return 1
}

run_job ctrl_f336
run_job ctrl_c0
run_job ctrl_f672

# ---- D0 (CPU) decides slot 1
SLOT1=E08
if [ -f "$RUN/ctrl_f336/done.json" ]; then
  note D0 "start"
  if python -m src.readout.d0_binding --run "$RUN" --out "$RUN/D0" --exclude "$EXCL" > "$RUN/logs/D0.log" 2>&1; then
    SLOT1=$(cat "$RUN/D0/slot1.txt"); note D0 "done -> slot1=$SLOT1"
  else
    note D0 "FAILED (log: $RUN/logs/D0.log) -> slot1=E08 (pre-registered default)"
  fi
else
  note D0 "skipped: baseline missing -> slot1=E08 (pre-registered default)"
fi
echo "$SLOT1" > "$RUN/slot1_used.txt"

run_job "$SLOT1"
run_job E09
run_job E11
if [ -f "$RUN/ctrl_f672/done.json" ]; then run_job E12; else note E12 "skipped: teacher ctrl_f672 missing"; fi

# ---- report (CPU)
note report "start"
if python -m src.readout.report --run "$RUN" --exclude "$EXCL" > "$RUN/logs/report.log" 2>&1; then
  note report "done -> $RUN/REPORT.md"
else
  note report "FAILED (log: $RUN/logs/report.log)"
fi
note queue "end"
