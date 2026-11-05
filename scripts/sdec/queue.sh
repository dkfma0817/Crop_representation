#!/usr/bin/env bash
# S-series pilot queue (GPU 2+3 only; the QAT job is already paused):
#   COCO pre-training A0 -> A1 -> B0 -> B1 (2-GPU DDP, 12k steps, frozen encoder, H1 loss); the frozen-decoder PACO R2
#   probe of a finished condition runs while the next one pre-trains; then direct checks, report, and QAT resume.
#   tmux new -d -s sdec_q "bash scripts/sdec/queue.sh output/2026-11-05/sdec"
set -u
cd "$(dirname "$0")/../.."
RUN=${1:?usage: queue.sh <run dir>}
mkdir -p "$RUN/logs"
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=$(realpath cache)/cuda_jit
source ~/miniforge3/bin/activate dinov2_ord_py310
ST="$RUN/queue_status.tsv"
EXCL=output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json
QAT_OUT=/home/irteam/low-bit-training-v2/output/2026-11-04/scale-800m/800m-ste-code7.5e-5-compile
QAT_SAVE=/home/irteam/Crop_representation/output/2026-11-05/qat_pause
note() { printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" | tee -a "$ST"; }
echo $$ > "$RUN/queue.pid"
note queue "start pid=$$; GPU 2,3: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n 3,4p | tr '\n' ' ')"
if ps -eo cmd | grep -q "[t]orch.distributed.run --nproc_per_node 2 --master_port 30871"; then note queue "ABORT: QAT still running on GPU 2,3"; exit 2; fi
port=30400
probe() {
  local name=$1 out="$RUN/probe_$1"
  [ -f "$out/done.json" ] && { note "probe_$name" "skip (done)"; return; }
  [ -d "$out" ] && mv "$out" "${out}_partial_$(date +%s)"
  port=$((port + 7))
  note "probe_$name" "start"
  ( torchrun --nproc_per_node 2 --master_port $port -m src.readout.train --job "configs/sdec/probe_$name.yaml" --out "$out" \
      --exclude "$EXCL" --workers 8 > "$RUN/logs/probe_$name.log" 2>&1 \
      && note "probe_$name" "done" || note "probe_$name" "FAILED (log: $RUN/logs/probe_$name.log)" ) &
}
pretrain() {
  local c=$1 out="$RUN/$1"
  [ -f "$out/done.json" ] && { note "$c" "skip (done)"; return 0; }
  [ -d "$out" ] && mv "$out" "${out}_partial_$(date +%s)"
  port=$((port + 7))
  note "$c" "pretrain start (COCO stream, 12k steps)"
  torchrun --nproc_per_node 2 --master_port $port -m src.msd.pretrain --cond "$c" --out "$out" --workers 10 > "$RUN/logs/$c.log" 2>&1
  if [ -f "$out/done.json" ]; then note "$c" "pretrain done ($(python -c "import json; d=json.load(open('$out/done.json')); print('selected step', d['selected']['step'], 'dev L12 %.4f' % d['selected']['dev']['L12'], 'time %ds' % d['time_s'])"))"; return 0; fi
  note "$c" "pretrain FAILED (log: $RUN/logs/$c.log)"; return 1
}
for c in A0 A1 B0 B1; do
  if pretrain "$c"; then probe "$c"; fi
done
note queue "pre-training finished; waiting for probes"
wait
note direct "start"
if CUDA_VISIBLE_DEVICES=2 python -m src.msd.sdirect --run "$RUN" > "$RUN/logs/sdirect.log" 2>&1; then note direct "done"; else note direct "FAILED (log: $RUN/logs/sdirect.log)"; fi
note report "start"
if python -m src.msd.sreport --run "$RUN" --ref output/2026-11-05/msd > "$RUN/logs/report.log" 2>&1; then note report "done -> $RUN/REPORT.md"; else note report "FAILED (log: $RUN/logs/report.log)"; fi
# resume QAT (single instance; --resume picks the last checkpoint in its output dir)
if ! ps -eo cmd | grep -q "[t]orch.distributed.run --nproc_per_node 2 --master_port 30871"; then
  note qat "resuming with --resume from $(ls -d "$QAT_OUT"/checkpoint-* | sort -V | tail -1)"
  ( cd "$(cat "$QAT_SAVE/cwd.txt")" && mapfile -t ENVV < "$QAT_SAVE/environ.txt" && mapfile -t CMD < "$QAT_SAVE/cmdline.txt" \
    && setsid env -i "${ENVV[@]}" "${CMD[@]}" --resume >> "$QAT_OUT/train.log" 2>&1 < /dev/null & )
  sleep 15
  P=$(ps -eo pid,cmd | grep "[t]orch.distributed.run --nproc_per_node 2 --master_port 30871" | awk '{print $1}' | head -1)
  echo "$(date '+%F %T') resumed by sdec queue, torchrun pid $P" > "$QAT_SAVE/resumed.txt"
  note qat "resumed torchrun pid $P"
else
  note qat "already running; not relaunched"
fi
note queue "end"
