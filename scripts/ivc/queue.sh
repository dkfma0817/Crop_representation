#!/usr/bin/env bash
# IVC queue (GPU 2+3 only):  tmux new -d -s ivc_q "bash scripts/ivc/queue.sh output/2026-11-05/ivc"
# 0. wait until the QAT job is paused (output/2026-11-05/ivc/qat_pause/pause.sh, after checkpoint-15000)
# 1. smoke tests  2. three probes (bil / bic / c0; R0 + R2 each, DDP 2 GPUs, concurrent) + streaming feature pass + alignment
#    shift scan  3. analysis (CPU)  4. figures  5. resume QAT (same command / env; the command already has --resume)
set -u
cd "$(dirname "$0")/../.."
RUN=${1:?usage: queue.sh <run dir>}
mkdir -p "$RUN/logs"
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=$(realpath cache)/cuda_jit
source ~/miniforge3/bin/activate dinov2_ord_py310
ST="$RUN/queue_status.tsv"
EXCL=output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json
QP="$RUN/qat_pause"
QAT_OUT=/home/irteam/low-bit-training-v2/output/2026-11-04/scale-800m/800m-ste-code7.5e-5-compile
note() { printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" | tee -a "$ST"; }
resume_qat() {
  if [ -f "$QP/paused_at_checkpoint.txt" ] && [ ! -f "$QP/resumed.txt" ]; then
    if pgrep -f "src.qat.train" > /dev/null; then note qat "a QAT process is already running; not relaunched"; return; fi
    ( cd "$(cat "$QP/cwd.txt")" && mapfile -t ENVV < "$QP/environ.txt" && mapfile -t CMD < "$QP/cmdline.txt" \
      && setsid env -i "${ENVV[@]}" "${CMD[@]}" >> "$QAT_OUT/train.log" 2>&1 < /dev/null & echo $! > "$QP/resumed_pid.txt" )
    sleep 5
    echo "$(date '+%F %T') pid $(cat "$QP/resumed_pid.txt")" > "$QP/resumed.txt"
    note qat "resumed pid $(cat "$QP/resumed_pid.txt") (cmd from $QP/cmdline.txt, includes --resume; log appended to $QAT_OUT/train.log)"
  fi
}
echo $$ > "$RUN/queue.pid"
note queue "start pid=$$"
until grep -q "paused" "$QP/log.txt" 2>/dev/null || grep -q "ended by itself" "$QP/log.txt" 2>/dev/null; do sleep 15; done
note qat "$(tail -1 "$QP/log.txt")"

# ---------------------------------------------------------------- 1. smoke
SM="$RUN/smoke"; mkdir -p "$SM"
for c in bil bic c0; do sed 's/paco_dev_images: 1000/paco_dev_images: 10/' configs/ivc/probe_$c.yaml > "$SM/probe_$c.yaml"; done
ok=1
for c in bil bic c0; do
  torchrun --nproc_per_node 2 --master_port $((30400 + RANDOM % 500)) -m src.ivc.train --job "$SM/probe_$c.yaml" --out "$SM/probe_$c" \
    --exclude "$EXCL" --limit 60 --epochs 1 --max_steps 4 --workers 4 > "$RUN/logs/smoke_$c.log" 2>&1 && [ -f "$SM/probe_$c/done.json" ] \
    && note smoke "probe_$c ok" || { note smoke "probe_$c FAILED (log $RUN/logs/smoke_$c.log)"; ok=0; }
done
CUDA_VISIBLE_DEVICES=2 python -m src.ivc.features --out "$SM/features" --limit 12 --workers 4 > "$RUN/logs/smoke_features.log" 2>&1 \
  && note smoke "features ok" || { note smoke "features FAILED"; ok=0; }
if [ $ok = 0 ]; then note queue "smoke failed -> resuming QAT and stopping"; resume_qat; note queue "end (failed)"; exit 1; fi

# ---------------------------------------------------------------- 2. probes + feature pass + shift scan
port=30600
for c in bil bic c0; do
  out="$RUN/probe_$c"
  [ -f "$out/done.json" ] && { note "probe_$c" "skip (done)"; continue; }
  [ -d "$out" ] && mv "$out" "${out}_partial_$(date +%s)"
  port=$((port + 11))
  note "probe_$c" "start"
  ( torchrun --nproc_per_node 2 --master_port $port -m src.ivc.train --job "configs/ivc/probe_$c.yaml" --out "$out" \
      --exclude "$EXCL" --workers 8 > "$RUN/logs/probe_$c.log" 2>&1 && note "probe_$c" "done" || note "probe_$c" "FAILED (log $RUN/logs/probe_$c.log)" ) &
done
( [ -f "$RUN/features/done.json" ] && note features "skip (done)" || { note features "start";
  CUDA_VISIBLE_DEVICES=3 python -m src.ivc.features --out "$RUN/features" --workers 12 > "$RUN/logs/features.log" 2>&1 && note features "done" || note features "FAILED"; } ) &
( note align "start (incl. GPU shift scan)"; CUDA_VISIBLE_DEVICES=2 python -m src.ivc.align --out "$RUN/align" > "$RUN/logs/align.log" 2>&1 && note align "done" || note align "FAILED" ) &
wait
# ---------------------------------------------------------------- 3. analysis + 4. figures (resume QAT once no GPU work is left)
( note viz "fixed ROIs start"; CUDA_VISIBLE_DEVICES=2 python -m src.ivc.viz --ann_ids "$(python -c "import json; print(','.join(map(str, json.load(open('configs/ivc/sample_rois.json'))['viz_ann_ids'])))")" \
    --tags "$(python -c "print(','.join(f'fixed{k:02d}' for k in range(12)))")" --out "$RUN/figures/rois" > "$RUN/logs/viz_fixed.log" 2>&1 && note viz "fixed ROIs done" || note viz "fixed ROIs FAILED" ) &
note analyze "start"
python -m src.ivc.analyze --run "$RUN" > "$RUN/logs/analyze.log" 2>&1 && note analyze "done -> $RUN/tables.md" || note analyze "FAILED (log $RUN/logs/analyze.log)"
for rd in r2 r0; do
  if python -m src.ivc.group_examples --run "$RUN" --reader $rd > "$RUN/logs/group_examples_$rd.txt" 2>&1; then
    CUDA_VISIBLE_DEVICES=3 python -m src.ivc.viz --ann_ids "$(sed -n 1p "$RUN/logs/group_examples_$rd.txt")" --tags "$(sed -n 2p "$RUN/logs/group_examples_$rd.txt")" \
      --out "$RUN/figures/groups_$rd" > "$RUN/logs/viz_groups_$rd.log" 2>&1 && note viz "groups $rd done" || note viz "groups $rd FAILED"
  else note viz "group examples $rd FAILED"; fi
done
wait
# ---------------------------------------------------------------- 5. resume QAT
resume_qat
note queue "end"
