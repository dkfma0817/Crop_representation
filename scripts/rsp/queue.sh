#!/bin/bash
# RSP-Masked / RSP-Masked-noPix GPU-1 queue (restart-safe: every stage is skipped when its done-file exists; training resumes from
# recovery.pt). GPU 1 only (CUDA_VISIBLE_DEVICES=1); GPU 0 (P1/P2) is never touched.
# bgjobs B on GPU 1: a timed reservation stops it before the first GPU stage; the EXIT trap releases it (also on failure), so B
# resumes automatically ~30 s after this queue leaves GPU 1. CPU-only stages (official evaluator, bootstrap, size AP) run after release.
#
# Order (fixed before any result):
#   A train (pix) -> [A: R2 probe -> Val inference (10 ep) -> official Val]  ||  B train (noPix) -> B: R2 probe -> Val inference -> official Val
#   -> epoch selection on official Val ((Obj+Part)/2, earliest on ties; existing F384/F768/FA-A/FA-B epochs must be unchanged) -> freeze
#   -> Test inference once per RSP method -> feature analysis + efficiency (GPU) -> release GPU 1
#   -> official Test + boot packs + paired bootstrap B=1000 + ROI-size AP (CPU)
set -u
cd /home/irteam/Crop_representation
D=$PWD/output/2026-11-12; C=$D/rsp_masked_comparison; E=$C/eval
declare -A RUN=([PIX]=$D/rsp_masked_pix_s0 [NOPIX]=$D/rsp_masked_nopix_s0)
declare -A PIXF=([PIX]=1 [NOPIX]=0)
FA=$PWD/output/2026-11-11/fine_aware_shared_s0/eval
PY=/home/irteam/envs/dinov2_ord_py310/bin/python; EPY=/home/irteam/envs/paco_eval/bin/python; OFF="/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:."
EXCL=output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json
export CUDA_VISIBLE_DEVICES=1
log() { echo "$(date '+%F %T') $*" | tee -a $C/logs/queue.log; }
release() { bash scripts/bgjobs/ctl.sh release --gpus 1 >/dev/null 2>&1; log "GPU 1 released (bgjobs B resumes)"; }
trap 'release' EXIT
bash scripts/bgjobs/ctl.sh reserve --gpus 1 --hours 14 --note "RSP queue (scripts/rsp/queue.sh)" >/dev/null
log "QUEUE START pid $$ (GPU 1 reserved)"
mkdir -p $E/eval_val $E/eval_boot $E/eval_test $E/preds $E/verify $E/size $C/logs
for R in ${RUN[@]}; do
  mkdir -p $R/logs
  if [ ! -d $R/source_snapshot ]; then   # provenance: git commit + working-tree status + exact code used
    mkdir -p $R/source_snapshot; cp -r src/rsp scripts/rsp $R/source_snapshot/
    { echo "git HEAD: $(git rev-parse HEAD)"; echo "branch: $(git rev-parse --abbrev-ref HEAD)"; echo "src/rsp, scripts/rsp: new, uncommitted (copied here)";
      echo "reused unchanged: src/tsplit/{train_bb,official_infer,features}.py, src/pacoattr/official_eval.py, src/siglip_pm/select_epoch.py, src/g1up/support.py";
      git status --porcelain src/tsplit src/pacoattr src/siglip_pm src/g1up src/rsp scripts/rsp; } > $R/source_snapshot/GIT.txt
  fi
done
# existing baselines, same R2 pipeline (src.tsplit.train_bb / official_infer / official evaluator) -> symlinks to the real dirs
for n in F384_r2 T768_r2 FAA_r2 FAB_r2; do
  ln -sfn "$(readlink -f $FA/eval_val/$n)" $E/eval_val/$n; ln -sfn "$(readlink -f $FA/eval_boot/$n)" $E/eval_boot/$n
done

train() {  # X
  X=$1; R=${RUN[$X]}
  [ -f $R/train/done.json ] && return 0
  for t in 1 2 3; do
    log "$X train attempt $t"
    OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rsp.train --pix ${PIXF[$X]} --shared $C/shared --out $R/train --workers 12 >> $R/logs/train.log 2>&1
    [ -f $R/train/done.json ] && { log "$X train done"; return 0; }
    log "$X train exited without done.json (see $R/train/failed.json / logs/train.log); resuming from recovery.pt"
    sleep 20
  done
  return 1
}

evalval() {  # X : R2 probe -> Val inference -> official Val (10 epochs, CPU)
  X=$1; R=${RUN[$X]}; N=RSP${X}
  mkdir -p $R/probe
  printf "name: $N\nbackbone: dinov3\nmode: RSP\nsplit_ckpt: $R/train/model.pt\nseed: 0\nepochs: 10\npaco_dev_images: 1000\nreaders:\n- name: r2\n  kind: r2\n  layers:\n  - 12\n" > $R/probe/job.yaml
  if [ ! -f $R/probe/done.json ]; then
    log "$X R2 probe start"
    OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rsp.probe train_bb --job $R/probe/job.yaml --out $R/probe --exclude $EXCL --workers 8 > $R/logs/probe.log 2>&1 || { log "$X PROBE_FAILED"; return 1; }
  fi
  if [ ! -f $E/verify/${N}_val.json ]; then
    log "$X Val inference"
    OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rsp.probe official_infer --root $E --probe $R/probe --name $N --split val > $R/logs/val_infer.log 2>&1 || { log "$X VAL_INFER_FAILED"; return 1; }
  fi
  for e in $(seq 1 10); do [ -f $E/eval_val/${N}_r2/ep$e/official_results.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.pacoattr.official_eval --split val --pred $E/preds/${N}_r2/ep$e/probs_val.npz --out $E/eval_val/${N}_r2/ep$e > $R/logs/official_val_ep$e.log 2>&1 & done; wait
  log "$X official Val done"
}

train PIX || { log "PIX TRAIN_FAILED"; exit 1; }
evalval PIX & PA=$!
train NOPIX || { log "NOPIX TRAIN_FAILED"; wait $PA; exit 1; }
evalval NOPIX & PB=$!
wait $PA; SA=$?; wait $PB; SB=$?
[ $SA -eq 0 ] && [ $SB -eq 0 ] || { log "VAL stage failed (A $SA, B $SB)"; exit 1; }

if [ ! -f $E/selection_frozen.json ]; then
  $PY -m src.siglip_pm.select_epoch $E > $C/logs/select.log 2>&1
  python3 - <<P || { log "SELECTION CHECK FAILED"; exit 1; }
import json
s = json.load(open("$E/selection.json")); f = json.load(open("$FA/selection_frozen.json"))
for n in ("F384_r2", "T768_r2", "FAA_r2", "FAB_r2"):
    assert s[n]["epoch"] == f[n]["epoch"], n
P
  cp $E/selection.json $E/selection_frozen.json
  sha256sum $E/selection_frozen.json > $E/selection_frozen.sha256
fi
log "selection frozen: $(python3 -c "import json;s=json.load(open('$E/selection_frozen.json'));print({k:(v['epoch'],round((v['val_obj']+v['val_part'])/2,2)) for k,v in s.items()})")"

for X in PIX NOPIX; do
  R=${RUN[$X]}; N=RSP${X}
  [ -f $E/verify/${N}_test.json ] || { log "$X Test inference (once)"; OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rsp.probe official_infer --root $E --probe $R/probe --name $N --split test > $R/logs/test_infer.log 2>&1 || { log "$X TEST_INFER_FAILED"; exit 1; }; }
done
# official Test AP first (CPU, starts now; report AP before bootstrap)
for X in PIX NOPIX; do
  N=RSP${X}; R=${RUN[$X]}; EP=$(python3 -c "import json;print(json.load(open('$E/selection_frozen.json'))['${N}_r2']['epoch'])")
  [ -f $E/eval_test/${N}_r2/official_results.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.pacoattr.official_eval --split test --pred $E/preds/${N}_r2/ep$EP/probs_test.npz --out $E/eval_test/${N}_r2 > $R/logs/official_test.log 2>&1 &
  for sp in val test; do [ -f $E/eval_boot/${N}_r2/$sp/official_results.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.pacoattr.official_eval --split $sp --pred $E/preds/${N}_r2/ep$EP/probs_$sp.npz --out $E/eval_boot/${N}_r2/$sp --boot > $R/logs/bootpack_$sp.log 2>&1 & done
done
# GPU-side analysis while the CPU evaluator runs
[ -f $C/feature_analysis.json ] || { log "feature analysis"; OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rsp.analysis --pix ${RUN[PIX]}/train/model.pt --nopix ${RUN[NOPIX]}/train/model.pt --out $C/feature_analysis.json > $C/logs/analysis.log 2>&1 || log "ANALYSIS_FAILED"; }
[ -f $C/efficiency.json ] || { log "efficiency"; OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u -m src.rsp.efficiency --pix ${RUN[PIX]}/train/model.pt --nopix ${RUN[NOPIX]}/train/model.pt --out $C/efficiency.json > $C/logs/efficiency.log 2>&1 || log "EFFICIENCY_FAILED"; }
release
trap - EXIT
wait
sha256sum -c $E/selection_frozen.sha256 >> $C/logs/queue.log 2>&1
log "TEST_DONE"
for sp in val test; do
  [ -f $E/boot_$sp.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.g1up.support boot --root $E --names RSPPIX_r2,RSPNOPIX_r2,F384_r2,T768_r2,FAA_r2,FAB_r2 --split $sp --out $E/boot_$sp.json > $C/logs/boot_$sp.log 2>&1 &
done
for n in RSPPIX_r2 RSPNOPIX_r2 F384_r2 T768_r2; do
  ep=$(python3 -c "import json;print(json.load(open('$E/selection_frozen.json'))['$n']['epoch'])")
  pred=$E/preds/$n/ep$ep/probs_test.npz; [ -f $pred ] || pred=$PWD/output/2026-11-11/token_split_x2_s0/eval/preds/$n/ep$ep/probs_test.npz
  [ -f $E/size/${n}_test.csv ] || { [ -f "$pred" ] && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$OFF $EPY -u -m src.g1up.support size --split test --pred $pred --model $n --out $E/size/${n}_test.csv > $C/logs/size_$n.log 2>&1 & }
done
wait
log "ALL_DONE"
