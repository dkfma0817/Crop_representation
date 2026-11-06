#!/usr/bin/env bash
# Runs the one-time test scoring (GPU 2) for every probe as soon as it has done.json; exits when all 7 are scored.
set -u
source "$(dirname "$0")/common.sh"
cd $ROOT
PY=/home/irteam/miniforge3/envs/evaclip_eval/bin/python
ALL="eva_full1024 clipself_full1024 declip_full1024 eva_c0224"
while true; do
  left=0
  for p in $ALL; do
    d=$R/probes/$p
    [ -f $d/test_done.json ] && continue
    left=1
    [ -f $d/done.json ] || continue
    event test "scoring $p"
    CUDA_VISIBLE_DEVICES=2 $PY -m src.densecmp.test_eval --probe $d --workers 8 > $R/logs/test_$p.log 2>&1 \
      && event test "$p: $(grep max_abs_diff $R/logs/test_$p.log | tail -1)" || { event test "$p FAILED"; touch $d/test_failed; }
  done
  [ $left = 0 ] && { set_status test DONE "all probes scored"; break; }
  ls $R/probes/*/test_failed >/dev/null 2>&1 && { set_status test FAILED "see logs/test_*.log"; break; }
  sleep 60
done
