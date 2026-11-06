#!/usr/bin/env bash
# Whole pilot, sequential on GPU 2,3 (never two experiments at once):
#   smoke_cpu -> qat_pause -> smoke_gpu -> E1..E7 (train + probe each) -> B0_direct probe -> R00 (train + probe)
#   -> recon (GPU feature reconstruction metrics) -> [our GPU processes gone] -> qat_resume (+verification)
#   -> analysis (CPU) -> report.  Re-running skips DONE items. Whatever happens, the EXIT/HUP/TERM trap resumes QAT.
set -u
source "$(dirname "$0")/common.sh"
cd "$ROOT"
source ~/miniforge3/bin/activate dinov2_ord_py310
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=$ROOT/cache/cuda_jit
echo $$ > "$R/queue.pid"
finish() {
  trap - EXIT HUP TERM INT
  if [ -f "$QAT_SAVE/PAUSED_BY_XDEC" ] && [ -z "$(qat_pid)" ]; then
    event queue "exit trap: QAT paused by us and not running -> recover_qat.sh"
    set_status qat_resume RUNNING "from exit trap"
    if bash "$R/scripts/recover_qat.sh"; then set_status qat_resume DONE "see qat_resume_status.md"; else set_status qat_resume FAILED "retry: bash $R/scripts/recover_qat.sh"; fi
  fi
  event queue "end"
}
trap finish EXIT
trap 'event queue "signal received"; exit 130' HUP TERM INT
event queue "start pid $$"
step() {  # step <item> <command...>
  local it=$1; shift
  [ "$(get_status $it)" = DONE ] && { event "$it" "skip (DONE)"; return 0; }
  set_status "$it" RUNNING
  if "$@"; then set_status "$it" DONE; return 0; else set_status "$it" FAILED "exit $?"; return 1; fi
}
# ---------------------------------------------------------------- checks that do not need the GPU (QAT still running)
step smoke_cpu bash -c "[ -f $R/smoke_cpu/smoke.json ] || CUDA_VISIBLE_DEVICES= python -m src.xdec.smoke --out $R/smoke_cpu > $R/logs/smoke_cpu.log 2>&1" \
  || { event queue "ABORT: CPU smoke failed (QAT untouched)"; exit 1; }
df -h "$ROOT" | tail -1 > "$R/logs/disk_at_start.txt"
# ---------------------------------------------------------------- stop QAT, take GPU 2,3
step qat_pause bash "$R/scripts/pause_qat.sh" || { event queue "ABORT: QAT pause failed"; exit 1; }
for i in $(seq 1 24); do [ -z "$(gpu23_apps)" ] && break; sleep 5; done
[ -z "$(gpu23_apps)" ] || { event queue "ABORT: GPU 2,3 busy: $(gpu23_apps | tr '\n' ';')"; exit 1; }
event queue "GPU 2,3 free: $(gpu23_mem)"
gpu_smoke() {
  local S=$R/smoke_gpu; mkdir -p $S
  CUDA_VISIBLE_DEVICES=2 python -m src.xdec.smoke --gpu --steps 20 --out $S > $R/logs/smoke_gpu.log 2>&1 || return 1
  for c in E1 E2 E7 R00; do
    rm -rf $S/train_$c
    torchrun --nproc_per_node 2 --master_port 31090 -m src.xdec.pretrain --cond $c --out $S/train_$c --steps 30 --save_at 15,30 \
      --eval_every 15 --dev_max 16 --workers 4 >> $R/logs/smoke_gpu.log 2>&1 || return 1
    [ -f $S/train_$c/done.json ] || return 1
  done
  # resume path: rerun E2 smoke with 45 steps -> must resume from ckpt_30
  torchrun --nproc_per_node 2 --master_port 31090 -m src.xdec.pretrain --cond E2 --out $S/train_E2 --steps 45 --save_at 15,30,45 \
    --eval_every 15 --dev_max 16 --workers 4 >> $R/logs/smoke_gpu.log 2>&1 || return 1
  ls $S/train_E2/config_resume_30.json > /dev/null || return 1
  sed "s|checkpoints/E2/selected.pt|smoke_gpu/train_E2/selected.pt|; s/paco_dev_images: 1000/paco_dev_images: 10/" $R/configs/probe_E2.yaml > $S/probe_E2.yaml
  sed "s/paco_dev_images: 1000/paco_dev_images: 10/" $R/configs/probe_B0_direct.yaml > $S/probe_B0.yaml
  rm -rf $S/probe_E2 $S/probe_B0
  torchrun --nproc_per_node 2 --master_port 31092 -m src.xdec.probe --job $S/probe_E2.yaml --out $S/probe_E2 --exclude $EXCL \
    --limit 60 --epochs 1 --max_steps 4 --workers 4 >> $R/logs/smoke_gpu.log 2>&1 || return 1
  torchrun --nproc_per_node 2 --master_port 31094 -m src.xdec.probe --job $S/probe_B0.yaml --out $S/probe_B0 --exclude $EXCL \
    --limit 60 --epochs 1 --max_steps 4 --workers 4 >> $R/logs/smoke_gpu.log 2>&1 || return 1
  [ -f $S/probe_E2/done.json ] && [ -f $S/probe_B0/done.json ]
}
step smoke_gpu gpu_smoke || { event queue "ABORT: GPU smoke failed (log logs/smoke_gpu.log)"; exit 1; }
# ---------------------------------------------------------------- experiments (one at a time; a failure does not stop the others)
for c in E1 E2 E3 E4 E5 E6 E7 B0_direct R00; do
  step "$c" bash "$R/scripts/run_one.sh" "$c"
done
step recon bash -c "CUDA_VISIBLE_DEVICES=2 python -m src.xdec.recon --run $R > $R/logs/recon.log 2>&1"
# ---------------------------------------------------------------- give the GPUs back
for i in $(seq 1 24); do [ -z "$(ps -eo args | grep -v grep | grep -E ' -m src\.xdec\.')" ] && [ -z "$(gpu23_apps)" ] && break; sleep 5; done
event queue "our GPU processes: $(ps -eo pid,args | grep -v grep | grep -cE ' -m src\.xdec\.') left; GPU 2,3: $(gpu23_mem)"
set_status qat_resume RUNNING
if bash "$R/scripts/resume_qat.sh"; then set_status qat_resume DONE "see qat_resume_status.md"; else set_status qat_resume FAILED "retry: bash $R/scripts/recover_qat.sh"; fi
# ---------------------------------------------------------------- CPU analysis + report (QAT already back on GPU 2,3)
export CUDA_VISIBLE_DEVICES=
step analysis bash -c "python -m src.xdec.analyze --run $R > $R/logs/analysis.log 2>&1"
step report bash -c "python -m src.xdec.report --run $R > $R/logs/report.log 2>&1"
