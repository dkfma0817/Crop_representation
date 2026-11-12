#!/bin/bash
# Direct dense-teacher PACO evaluation (no feature-model training; only the existing R2 reader is trained per condition).
# GPU 1 only (CUDA_VISIBLE_DEVICES=1). bgjobs B: timed reservation stops it; the EXIT trap releases GPU 1 (also on failure) so B resumes.
# Restart-safe: every stage is skipped when its output exists.
# Conditions: DTM8 / DTF8 / DTM4 / DTF4 = p_T 8 / 4 x masked / full (unrestricted single-ROI). Two R2 probes share GPU 1:
#   stream 1: DTF4 -> DTM8     stream 2: DTM4 -> DTF8
# Then: epoch selection on official Val ((Obj+Part)/2; F384 / T768 / C0 epochs must equal their frozen values) -> freeze ->
#       Test inference once per condition -> release GPU 1 -> official Test + boot packs -> bootstrap B=1000 -> size AP (CPU).
set -u
cd /home/irteam/Crop_representation
R=$PWD/output/2026-11-12/dense_teacher_eval; E=$R/eval
PY=/home/irteam/envs/dinov2_ord_py310/bin/python; EPY=/home/irteam/envs/paco_eval/bin/python; OFF="/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:."
EXCL=output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json
FA=$PWD/output/2026-11-11/fine_aware_shared_s0/eval; TS=$PWD/output/2026-11-11/token_split_x2_s0/eval
C0V=$PWD/output/2026-11-08/hcf_pilot/eval_val/c0_r2; C0B=$PWD/output/2026-11-09/g1_upscaled_teacher_s0/eval/eval_boot/C0_r2
C0P=$PWD/output/2026-11-08/hcf_pilot/preds/c0_r2; C0T=$PWD/output/2026-11-08/hcf_pilot/eval_test/c0_r2/test
export CUDA_VISIBLE_DEVICES=1
log() { echo "$(date '+%F %T') $*" | tee -a $R/logs/queue.log; }
release() { bash scripts/bgjobs/ctl.sh release --gpus 1 >/dev/null 2>&1; log "GPU 1 released (bgjobs B resumes)"; }
trap 'release' EXIT
bash scripts/bgjobs/ctl.sh reserve --gpus 1 --hours 14 --note "dense teacher eval (scripts/dtev/queue.sh)" >/dev/null
log "QUEUE START pid $$ (GPU 1 reserved)"
mkdir -p $E/eval_val $E/eval_boot $E/eval_test $E/preds $E/verify $E/size $R/logs
if [ ! -d $R/source_snapshot ]; then
  mkdir -p $R/source_snapshot; cp -r src/dtev scripts/dtev $R/source_snapshot/
  { echo "git HEAD: $(git rev-parse HEAD)"; echo "branch: $(git rev-parse --abbrev-ref HEAD)"; echo "src/dtev, scripts/dtev: new, uncommitted (copied here)";
    echo "reused unchanged: src/rsp/{common,data}.py, src/tsplit/{train_bb,official_infer,features}.py, src/pacoattr/official_eval.py, src/siglip_pm/select_epoch.py, src/g1up/support.py";
    git status --porcelain src/tsplit src/pacoattr src/siglip_pm src/g1up src/rsp src/dtev scripts/dtev; } > $R/source_snapshot/GIT.txt
fi
# baselines (same R2 pipeline / official evaluator) -> symlinks to the real dirs
for n in F384_r2 T768_r2; do ln -sfn "$(readlink -f $FA/eval_val/$n)" $E/eval_val/$n; ln -sfn "$(readlink -f $FA/eval_boot/$n)" $E/eval_boot/$n; done
ln -sfn $C0V $E/eval_val/C0_r2; ln -sfn $C0B $E/eval_boot/C0_r2

cond() {  # N : R2 probe -> Val inference (10 epochs) -> official Val (CPU, background)
  N=$1; D=$R/runs/$N
  if [ ! -f $D/probe/done.json ]; then
    log "$N R2 probe start"
    OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.dtev.probe train_bb --job $D/probe/job.yaml --out $D/probe --exclude $EXCL --workers 8 > $D/logs/probe.log 2>&1 || { log "$N PROBE_FAILED"; return 1; }
    log "$N R2 probe done"
  fi
  if [ ! -f $E/verify/${N}_val.json ]; then
    log "$N Val inference"
    OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.dtev.probe official_infer --root $E --probe $D/probe --name $N --split val > $D/logs/val_infer.log 2>&1 || { log "$N VAL_INFER_FAILED"; return 1; }
  fi
  for e in $(seq 1 10); do [ -f $E/eval_val/${N}_r2/ep$e/official_results.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.pacoattr.official_eval --split val --pred $E/preds/${N}_r2/ep$e/probs_val.npz --out $E/eval_val/${N}_r2/ep$e > $D/logs/official_val_ep$e.log 2>&1 & done; wait
  log "$N official Val done"
}

( cond DTF4 && cond DTM8 ) & S1=$!
( cond DTM4 && cond DTF8 ) & S2=$!
wait $S1; A=$?; wait $S2; B=$?
[ $A -eq 0 ] && [ $B -eq 0 ] || { log "VAL stage failed (stream1 $A, stream2 $B)"; exit 1; }

if [ ! -f $E/selection_frozen.json ]; then
  $PY -m src.siglip_pm.select_epoch $E > $R/logs/select.log 2>&1
  python3 - <<P || { log "SELECTION CHECK FAILED"; exit 1; }
import json
s = json.load(open("$E/selection.json"))
f = json.load(open("$FA/selection_frozen.json")); g = json.load(open("$PWD/output/2026-11-09/g1_upscaled_teacher_s0/eval/selection_frozen.json"))
for n in ("F384_r2", "T768_r2"):
    assert s[n]["epoch"] == f[n]["epoch"], n
assert s["C0_r2"]["epoch"] == g["C0_r2"]["epoch"], "C0_r2"
P
  cp $E/selection.json $E/selection_frozen.json
  sha256sum $E/selection_frozen.json > $E/selection_frozen.sha256
fi
log "selection frozen: $(python3 -c "import json;s=json.load(open('$E/selection_frozen.json'));print({k:(v['epoch'],round((v['val_obj']+v['val_part'])/2,2)) for k,v in s.items()})")"

for N in DTM8 DTF8 DTM4 DTF4; do
  D=$R/runs/$N
  [ -f $E/verify/${N}_test.json ] || { log "$N Test inference (once)"; OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.dtev.probe official_infer --root $E --probe $D/probe --name $N --split test > $D/logs/test_infer.log 2>&1 || { log "$N TEST_INFER_FAILED"; exit 1; }; }
done
[ -f $R/cost.json ] || { log "cost"; OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.dtev.cost --out $R/cost.json > $R/logs/cost.log 2>&1 || log "COST_FAILED"; }
release
trap - EXIT
for N in DTM8 DTF8 DTM4 DTF4; do
  D=$R/runs/$N; EP=$(python3 -c "import json;print(json.load(open('$E/selection_frozen.json'))['${N}_r2']['epoch'])")
  [ -f $E/eval_test/${N}_r2/official_results.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.pacoattr.official_eval --split test --pred $E/preds/${N}_r2/ep$EP/probs_test.npz --out $E/eval_test/${N}_r2 > $D/logs/official_test.log 2>&1 &
  for sp in val test; do [ -f $E/eval_boot/${N}_r2/$sp/official_results.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.pacoattr.official_eval --split $sp --pred $E/preds/${N}_r2/ep$EP/probs_$sp.npz --out $E/eval_boot/${N}_r2/$sp --boot > $D/logs/bootpack_$sp.log 2>&1 & done
done
wait
sha256sum -c $E/selection_frozen.sha256 >> $R/logs/queue.log 2>&1
log "TEST_DONE"
for sp in val test; do
  [ -f $E/boot_$sp.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.g1up.support boot --root $E --names DTM8_r2,DTF8_r2,DTM4_r2,DTF4_r2,F384_r2,T768_r2,C0_r2 --split $sp --out $E/boot_$sp.json > $R/logs/boot_$sp.log 2>&1 &
done
for n in DTM8_r2 DTF8_r2 DTM4_r2 DTF4_r2 F384_r2 T768_r2 C0_r2; do
  ep=$(python3 -c "import json;print(json.load(open('$E/selection_frozen.json'))['$n']['epoch'])")
  case $n in F384_r2|T768_r2) pred=$TS/preds/$n/ep$ep/probs_test.npz;; C0_r2) pred=$C0P/ep$ep/probs_test.npz;; *) pred=$E/preds/$n/ep$ep/probs_test.npz;; esac
  [ -f $E/size/${n}_test.csv ] || { [ -f "$pred" ] && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.g1up.support size --split test --pred $pred --model $n --out $E/size/${n}_test.csv > $R/logs/size_$n.log 2>&1 & }
done
wait
log "ALL_DONE"
