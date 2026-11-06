#!/usr/bin/env bash
# E6 multi-layer memory queue (sequential, GPU 2,3 only, PACO val only).
#   QAT_MODE=stop (only with the user's approval for THIS experiment) | share
#   smoke_gpu (30-step DDP train A/B + tiny probe B) -> train A -> train B -> probe A -> probe B (R0 + R2) -> recon (GPU 2)
#   -> [qat_resume + verify] -> analysis (CPU)
set -u
source "$(dirname "$0")/common.sh"
cd "$ROOT"
source ~/miniforge3/bin/activate dinov2_ord_py310
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=$ROOT/cache/cuda_jit
QAT_MODE=${QAT_MODE:?stop or share}
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
  set_status qat_pause SKIPPED "QAT_MODE=share"; set_status qat_resume SKIPPED "QAT never stopped"
fi
event queue "GPU 2,3: $(gpu23_mem)"
gpu_smoke() {
  local S=$R/smoke_gpu; rm -rf $S; mkdir -p $S
  for c in A B; do
    torchrun --nproc_per_node 2 --master_port 31690 -m src.e6ml.pretrain --cond $c --out $S/train_$c --steps 30 \
      --save_at 15,30 --eval_every 15 --dev_max 16 --workers 4 >> $R/logs/smoke_gpu.log 2>&1 || return 1
    [ -f $S/train_$c/done.json ] || return 1
  done
  sed "s|checkpoints/B/selected.pt|smoke_gpu/train_B/selected.pt|; s/paco_dev_images: 1000/paco_dev_images: 10/" $R/configs/probe_B.yaml > $S/probe.yaml
  torchrun --nproc_per_node 2 --master_port 31692 -m src.e6ml.probe --job $S/probe.yaml --out $S/probe --exclude $EXCL \
    --limit 60 --epochs 1 --max_steps 4 --workers 4 >> $R/logs/smoke_gpu.log 2>&1 || return 1
  [ -f $S/probe/done.json ]
}
step smoke_gpu gpu_smoke || { event queue "ABORT: GPU smoke failed (logs/smoke_gpu.log)"; exit 1; }
train() {
  local c=$1 out=$R/checkpoints/$1
  [ -f $out/done.json ] && return 0
  torchrun --nproc_per_node 2 --master_port $((31700 + ${#c} + $(printf '%d' "'$c"))) -m src.e6ml.pretrain --cond $c --out $out --workers 10 >> $R/logs/train_$c.log 2>&1
  [ -f $out/done.json ] || return 1
  event train_$c "$(python -c "import json;d=json.load(open('$out/done.json'));s=d['selected'];print('selected step', s['step'], 'dev L_old %.4f tok cos %.4f time %ds' % (s['dev']['loss'], s['dev']['cos'], d['time_s']))")"
}
probe() {
  local c=$1 out=$R/results/probe_$1
  [ -f $out/done.json ] && return 0
  [ -d $out ] && mv $out ${out}_partial_$(date +%s)
  torchrun --nproc_per_node 2 --master_port $((31800 + $(printf '%d' "'$c"))) -m src.e6ml.probe --job $R/configs/probe_$c.yaml --out $out --exclude $EXCL --workers 12 > $R/logs/probe_$c.log 2>&1
  [ -f $out/done.json ]
}
step train_A train A
step train_B train B
step probe_A probe A
step probe_B probe B
step recon bash -c "CUDA_VISIBLE_DEVICES=2 python -m src.e6ml.recon --run $R > $R/logs/recon.log 2>&1"
if [ "$QAT_MODE" = stop ]; then
  for i in $(seq 1 24); do [ -z "$(ps -eo args | grep -v grep | grep -E ' -m src\.e6ml\.')" ] && [ -z "$(gpu23_apps)" ] && break; sleep 5; done
  set_status qat_resume RUNNING
  if bash "$R/scripts/resume_qat.sh"; then set_status qat_resume DONE "see qat_resume_status.md"; else set_status qat_resume FAILED "retry: bash $R/scripts/recover_qat.sh"; fi
fi
export CUDA_VISIBLE_DEVICES=
step analysis bash -c "python -m src.e6ml.analyze --run $R > $R/logs/analysis.log 2>&1"
