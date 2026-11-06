#!/usr/bin/env bash
# Spatial-variance pilot queue (sequential, GPU 2,3 only, PACO val only).
#   QAT_MODE=stop  : pause the user's QAT (only after explicit approval), run, resume QAT + verify (exit trap guarantees resume)
#   QAT_MODE=share : leave QAT running; probes share GPU 2,3 with it (slower QAT, no QAT step loss)
#   smoke_gpu -> V1 -> V2 -> V3 -> O2 (R2 probes) -> [qat_resume] -> analysis (waits for the CPU feature-stat pass)
set -u
source "$(dirname "$0")/common.sh"
cd "$ROOT"
source ~/miniforge3/bin/activate dinov2_ord_py310
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=$ROOT/cache/cuda_jit
QAT_MODE=${QAT_MODE:?set QAT_MODE=stop or share}
echo $$ > "$R/queue.pid"
finish() {
  trap - EXIT HUP TERM INT
  if [ "$QAT_MODE" = stop ] && [ -f "$QAT_SAVE/PAUSED_BY_XDEC" ] && [ -z "$(qat_pid)" ]; then
    event queue "exit trap: QAT paused by us and not running -> recover_qat.sh"
    set_status qat_resume RUNNING "from exit trap"
    if bash "$R/scripts/recover_qat.sh"; then set_status qat_resume DONE "see qat_resume_status.md"; else set_status qat_resume FAILED "retry: bash $R/scripts/recover_qat.sh"; fi
  fi
  event queue "end"
}
trap finish EXIT
trap 'event queue "signal received"; exit 130' HUP TERM INT
event queue "start pid $$ (QAT_MODE=$QAT_MODE)"
step() {
  local it=$1; shift
  [ "$(get_status $it)" = DONE ] && { event "$it" "skip (DONE)"; return 0; }
  set_status "$it" RUNNING
  if "$@"; then set_status "$it" DONE; return 0; else set_status "$it" FAILED "exit $?"; return 1; fi
}
if [ "$QAT_MODE" = stop ]; then
  step qat_pause bash "$R/scripts/pause_qat.sh" || { event queue "ABORT: QAT pause failed"; exit 1; }
  for i in $(seq 1 24); do [ -z "$(gpu23_apps)" ] && break; sleep 5; done
  [ -z "$(gpu23_apps)" ] || { event queue "ABORT: GPU 2,3 busy"; exit 1; }
else
  set_status qat_pause SKIPPED "QAT_MODE=share: QAT keeps running on GPU 2,3"
  set_status qat_resume SKIPPED "QAT never stopped"
fi
event queue "GPU 2,3: $(gpu23_mem)"
gpu_smoke() {
  local S=$R/smoke_gpu; mkdir -p $S
  for c in V2 O2; do
    sed "s/paco_dev_images: 1000/paco_dev_images: 10/" $R/configs/probe_$c.yaml > $S/probe_$c.yaml
    rm -rf $S/probe_$c
    crops=0; [ $c = O2 ] && crops=1
    SVAMP_CROPS=$crops torchrun --nproc_per_node 2 --master_port 31290 -m src.svamp.probe --job $S/probe_$c.yaml --out $S/probe_$c \
      --exclude $EXCL --limit 60 --epochs 1 --max_steps 4 --workers 4 >> $R/logs/smoke_gpu.log 2>&1 || return 1
    [ -f $S/probe_$c/done.json ] || return 1
  done
}
step smoke_gpu gpu_smoke || { event queue "ABORT: GPU smoke failed (logs/smoke_gpu.log)"; exit 1; }
for c in V1 V2 V3 O2; do step "$c" bash "$R/scripts/run_one.sh" "$c"; done
if [ "$QAT_MODE" = stop ]; then
  for i in $(seq 1 24); do [ -z "$(ps -eo args | grep -v grep | grep -E ' -m src\.svamp\.probe')" ] && [ -z "$(gpu23_apps)" ] && break; sleep 5; done
  set_status qat_resume RUNNING
  if bash "$R/scripts/resume_qat.sh"; then set_status qat_resume DONE "see qat_resume_status.md"; else set_status qat_resume FAILED "retry: bash $R/scripts/recover_qat.sh"; fi
fi
export CUDA_VISIBLE_DEVICES=
for i in $(seq 1 360); do [ -f "$R/transform_checks.json" ] && break; sleep 10; done
step analysis bash -c "python -m src.svamp.analyze --run $R > $R/logs/analysis.log 2>&1"
