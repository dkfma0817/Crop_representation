#!/usr/bin/env bash
# run_one.sh <cond>   cond in E1..E7, R00 (decoder train + PACO probe) or B0_direct (probe only). Resumable:
# finished stages (done.json) are skipped; an interrupted decoder run resumes from its newest 3k/6k/9k checkpoint;
# an interrupted probe is moved aside (<dir>_partial_<time>) and restarted.
set -u
source "$(dirname "$0")/common.sh"
cd "$ROOT"
c=${1:?cond}
idx=$(echo "E1 E2 E3 E4 E5 E6 E7 R00 B0_direct" | tr ' ' '\n' | grep -nx "$c" | cut -d: -f1)
port=$((31100 + 10 * idx))
if [ "$c" != B0_direct ]; then
  out=$R/checkpoints/$c
  if [ ! -f "$out/done.json" ]; then
    event "$c" "decoder training start (port $port)"
    torchrun --nproc_per_node 2 --master_port $port -m src.xdec.pretrain --cond "$c" --out "$out" --workers 10 >> "$R/logs/train_$c.log" 2>&1
    [ -f "$out/done.json" ] || { event "$c" "decoder training FAILED (log logs/train_$c.log)"; exit 1; }
  fi
  event "$c" "decoder done: $(python -c "import json;d=json.load(open('$out/done.json'));print('selected step',d['selected']['step'],'dev loss %.4f cos %.4f'%(d['selected']['dev']['loss'],d['selected']['dev']['cos']),'time %ds'%d['time_s'])")"
fi
pout=$R/results/probe_$c
if [ ! -f "$pout/done.json" ]; then
  [ -d "$pout" ] && mv "$pout" "${pout}_partial_$(date +%s)"
  event "$c" "PACO probe start (R0 + R2)"
  torchrun --nproc_per_node 2 --master_port $((port + 1)) -m src.xdec.probe --job "$R/configs/probe_$c.yaml" --out "$pout" \
    --exclude "$EXCL" --workers 12 > "$R/logs/probe_$c.log" 2>&1
  [ -f "$pout/done.json" ] || { event "$c" "probe FAILED (log logs/probe_$c.log)"; exit 1; }
fi
event "$c" "probe done"
