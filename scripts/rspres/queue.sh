#!/bin/bash
# RSP-Residual (single condition) + its Step-0 control. GPU 0 only (CUDA_VISIBLE_DEVICES=0; GPU 1 runs the dense-teacher queue, untouched).
# bgjobs A: timed reservation stops it; the EXIT trap releases GPU 0 so A resumes. Restart-safe (stages skip when outputs exist;
# training resumes from recovery.pt). Existing RSP / F384 / T768 results are only symlinked, never written.
# Order: [train RSP-Residual -> R2 probe -> Val inference -> official Val] || [Step-0 R2 probe -> Val inference -> official Val]
#        -> epoch selection on official Val (F384 / T768 / RSPPIX epochs must equal their frozen values) -> freeze
#        -> Test inference once per method -> feature analysis -> release GPU 0 -> official Test + boot -> bootstrap -> size AP (CPU).
set -u
cd /home/irteam/Crop_representation
D=$PWD/output/2026-11-12; RUN=$D/rsp_residual_s0; S0=$D/rsp_residual_step0_s0; C=$D/rsp_residual_comparison; E=$C/eval
SHARED=$D/rsp_masked_comparison/shared; RC_=$D/rsp_masked_comparison/eval; FA=$PWD/output/2026-11-11/fine_aware_shared_s0/eval; TS=$PWD/output/2026-11-11/token_split_x2_s0/eval
PY=/home/irteam/envs/dinov2_ord_py310/bin/python; EPY=/home/irteam/envs/paco_eval/bin/python; OFF="/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:."
EXCL=output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json
export CUDA_VISIBLE_DEVICES=0
mkdir -p $E/eval_val $E/eval_boot $E/eval_test $E/preds $E/verify $E/size $C/logs $RUN/logs $S0/logs $RUN/probe $S0/probe
log() { echo "$(date '+%F %T') $*" | tee -a $C/logs/queue.log; }
release() { bash scripts/bgjobs/ctl.sh release --gpus 0 >/dev/null 2>&1; log "GPU 0 released (bgjobs A resumes)"; }
trap 'release' EXIT
bash scripts/bgjobs/ctl.sh reserve --gpus 0 --hours 10 --note "RSP-Residual (scripts/rspres/queue.sh)" >/dev/null
log "QUEUE START pid $$ (GPU 0 reserved)"
[ -f $RUN/checks/unit_tests.json ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$RUN/checks/unit_tests.json'))['all_pass'] else 1)" || { log "CHECKS NOT PASSED"; exit 1; }
if [ ! -d $RUN/source_snapshot ]; then
  mkdir -p $RUN/source_snapshot; cp -r src/rspres scripts/rspres $RUN/source_snapshot/
  diff src/rsp/train.py src/rspres/train.py > $RUN/train_py.diff
  { echo "git HEAD: $(git rev-parse HEAD)"; echo "src/rspres, scripts/rspres: new, uncommitted (copied here)"; echo "src/rspres/train.py = src/rsp/train.py + train_py.diff (output structure only)";
    git status --porcelain src/rsp src/rspres src/tsplit scripts/rspres; } > $RUN/source_snapshot/GIT.txt
fi
for n in F384_r2 T768_r2; do ln -sfn "$(readlink -f $FA/eval_val/$n)" $E/eval_val/$n; ln -sfn "$(readlink -f $FA/eval_boot/$n)" $E/eval_boot/$n; done
ln -sfn "$(readlink -f $RC_/eval_val/RSPPIX_r2)" $E/eval_val/RSPPIX_r2; ln -sfn "$(readlink -f $RC_/eval_boot/RSPPIX_r2)" $E/eval_boot/RSPPIX_r2
printf "name: RSPRES\nbackbone: dinov3\nmode: RSPRES\nsplit_ckpt: $RUN/train/model.pt\nseed: 0\nepochs: 10\npaco_dev_images: 1000\nreaders:\n- name: r2\n  kind: r2\n  layers:\n  - 12\n" > $RUN/probe/job.yaml
printf "name: RSPRES0\nbackbone: dinov3\nmode: RSPRES0\nsplit_ckpt: $S0/model/model.pt\nseed: 0\nepochs: 10\npaco_dev_images: 1000\nreaders:\n- name: r2\n  kind: r2\n  layers:\n  - 12\n" > $S0/probe/job.yaml

train() {
  [ -f $RUN/train/done.json ] && return 0
  for t in 1 2 3; do
    log "RSPRES train attempt $t"
    OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rspres.train --shared $SHARED --out $RUN/train --workers 12 >> $RUN/logs/train.log 2>&1
    [ -f $RUN/train/done.json ] && { log "RSPRES train done"; return 0; }
    log "RSPRES train exited without done.json; resuming from recovery.pt"; sleep 20
  done
  return 1
}
evalval() {  # N R
  N=$1; R=$2
  if [ ! -f $R/probe/done.json ]; then
    log "$N R2 probe start"
    OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rspres.probe train_bb --job $R/probe/job.yaml --out $R/probe --exclude $EXCL --workers 8 > $R/logs/probe.log 2>&1 || { log "$N PROBE_FAILED"; return 1; }
  fi
  if [ ! -f $E/verify/${N}_val.json ]; then
    log "$N Val inference"
    OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rspres.probe official_infer --root $E --probe $R/probe --name $N --split val > $R/logs/val_infer.log 2>&1 || { log "$N VAL_INFER_FAILED"; return 1; }
  fi
  for e in $(seq 1 10); do [ -f $E/eval_val/${N}_r2/ep$e/official_results.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.pacoattr.official_eval --split val --pred $E/preds/${N}_r2/ep$e/probs_val.npz --out $E/eval_val/${N}_r2/ep$e > $R/logs/official_val_ep$e.log 2>&1 & done; wait
  log "$N official Val done"
}
( train && evalval RSPRES $RUN ) & PA=$!
evalval RSPRES0 $S0 & PB=$!
wait $PA; SA=$?; wait $PB; SB=$?
[ $SA -eq 0 ] && [ $SB -eq 0 ] || { log "VAL stage failed (RSPRES $SA, RSPRES0 $SB)"; exit 1; }

if [ ! -f $E/selection_frozen.json ]; then
  $PY -m src.siglip_pm.select_epoch $E > $C/logs/select.log 2>&1
  python3 - <<P || { log "SELECTION CHECK FAILED"; exit 1; }
import json
s = json.load(open("$E/selection.json")); f = json.load(open("$FA/selection_frozen.json")); r = json.load(open("$RC_/selection_frozen.json"))
for n in ("F384_r2", "T768_r2"):
    assert s[n]["epoch"] == f[n]["epoch"], n
assert s["RSPPIX_r2"]["epoch"] == r["RSPPIX_r2"]["epoch"], "RSPPIX_r2"
P
  cp $E/selection.json $E/selection_frozen.json; sha256sum $E/selection_frozen.json > $E/selection_frozen.sha256
fi
log "selection frozen: $(python3 -c "import json;s=json.load(open('$E/selection_frozen.json'));print({k:(v['epoch'],round((v['val_obj']+v['val_part'])/2,2)) for k,v in s.items()})")"
for X in RSPRES:$RUN RSPRES0:$S0; do
  N=${X%%:*}; R=${X#*:}
  [ -f $E/verify/${N}_test.json ] || { log "$N Test inference (once)"; OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rspres.probe official_infer --root $E --probe $R/probe --name $N --split test > $R/logs/test_infer.log 2>&1 || { log "$N TEST_INFER_FAILED"; exit 1; }; }
done
for X in RSPRES:$RUN RSPRES0:$S0; do
  N=${X%%:*}; R=${X#*:}; EP=$(python3 -c "import json;print(json.load(open('$E/selection_frozen.json'))['${N}_r2']['epoch'])")
  [ -f $E/eval_test/${N}_r2/official_results.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.pacoattr.official_eval --split test --pred $E/preds/${N}_r2/ep$EP/probs_test.npz --out $E/eval_test/${N}_r2 > $R/logs/official_test.log 2>&1 &
  for sp in val test; do [ -f $E/eval_boot/${N}_r2/$sp/official_results.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.pacoattr.official_eval --split $sp --pred $E/preds/${N}_r2/ep$EP/probs_$sp.npz --out $E/eval_boot/${N}_r2/$sp --boot > $R/logs/bootpack_$sp.log 2>&1 & done
done
[ -f $C/feature_analysis.json ] || { log "feature analysis"; OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rspres.analysis --res $RUN/train/model.pt --rsp $D/rsp_masked_pix_s0/train/model.pt --out $C/feature_analysis.json > $C/logs/analysis.log 2>&1 || log "ANALYSIS_FAILED"; }
release
trap - EXIT
wait
sha256sum -c $E/selection_frozen.sha256 >> $C/logs/queue.log 2>&1
log "TEST_DONE"
for sp in val test; do
  [ -f $E/boot_$sp.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.g1up.support boot --root $E --names RSPRES_r2,RSPRES0_r2,RSPPIX_r2,F384_r2,T768_r2 --split $sp --out $E/boot_$sp.json > $C/logs/boot_$sp.log 2>&1 &
done
for n in RSPRES_r2 RSPRES0_r2 RSPPIX_r2 F384_r2 T768_r2; do
  ep=$(python3 -c "import json;print(json.load(open('$E/selection_frozen.json'))['$n']['epoch'])")
  case $n in F384_r2|T768_r2) pred=$TS/preds/$n/ep$ep/probs_test.npz;; RSPPIX_r2) pred=$RC_/preds/$n/ep$ep/probs_test.npz;; *) pred=$E/preds/$n/ep$ep/probs_test.npz;; esac
  [ -f $E/size/${n}_test.csv ] || { [ -f "$pred" ] && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.g1up.support size --split test --pred $pred --model $n --out $E/size/${n}_test.csv > $C/logs/size_$n.log 2>&1 & }
done
wait
log "ALL_DONE"
