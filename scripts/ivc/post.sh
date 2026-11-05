#!/usr/bin/env bash
# After probe_c0: analysis (CPU) -> G1-G4 example figures (GPU 2/3) -> resume the user's QAT on GPU 2,3.
set -u
cd "$(dirname "$0")/../.."
R=output/2026-11-05/ivc
QP=$R/qat_pause
QAT_OUT=/home/irteam/low-bit-training-v2/output/2026-11-04/scale-800m/800m-ste-code7.5e-5-compile
source ~/miniforge3/bin/activate dinov2_ord_py310
export OMP_NUM_THREADS=8 CUDA_CACHE_DISABLE=0 CUDA_CACHE_PATH=$(realpath cache)/cuda_jit
note() { printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" >> "$R/queue_status.tsv"; }
until grep -q "probe_c0	done\|probe_c0	FAILED" "$R/queue_status.tsv"; do sleep 15; done
note analyze "start"
python -m src.ivc.analyze --run "$R" > "$R/logs/analyze.log" 2>&1 && note analyze "done -> $R/tables.md" || note analyze "FAILED (log $R/logs/analyze.log)"
for rd in r2 r0; do
  if python -m src.ivc.group_examples --run "$R" --reader $rd > "$R/logs/group_examples_$rd.txt" 2>&1; then
    CUDA_VISIBLE_DEVICES=2 python -m src.ivc.viz --ann_ids "$(sed -n 1p "$R/logs/group_examples_$rd.txt")" --tags "$(sed -n 2p "$R/logs/group_examples_$rd.txt")" \
      --out "$R/figures/groups_$rd" > "$R/logs/viz_groups_$rd.log" 2>&1 && note viz "groups $rd done" || note viz "groups $rd FAILED"
  else note viz "group examples $rd FAILED"; fi
done
# resume QAT (GPU 2,3 job only: identified by its master port 30871; the GPU 0,1 job is a different run)
if [ ! -f "$QP/resumed.txt" ]; then
  if pgrep -f "master_port 30871" > /dev/null; then note qat "GPU 2,3 QAT already running; not relaunched"
  else
    ( cd "$(cat "$QP/cwd.txt")" && mapfile -t ENVV < "$QP/environ.txt" && mapfile -t CMD < "$QP/cmdline.txt" \
      && setsid env -i "${ENVV[@]}" "${CMD[@]}" >> "$QAT_OUT/train.log" 2>&1 < /dev/null & echo $! > "$QP/resumed_pid.txt" )
    sleep 5
    echo "$(date '+%F %T') pid $(cat "$QP/resumed_pid.txt")" > "$QP/resumed.txt"
    note qat "resumed pid $(cat "$QP/resumed_pid.txt") on GPU 2,3 (--resume from checkpoint-10000; log $QAT_OUT/train.log)"
  fi
fi
note post "end"
