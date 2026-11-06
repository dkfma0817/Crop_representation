#!/usr/bin/env bash
# run_one.sh <V1|V2|V3|O2> : PACO R2 probe of the fixed transform (resumable: done.json -> skip; partial -> moved aside)
set -u
source "$(dirname "$0")/common.sh"
cd "$ROOT"
c=${1:?cond}
idx=$(echo "V1 V2 V3 O2" | tr ' ' '\n' | grep -nx "$c" | cut -d: -f1)
port=$((31300 + 10 * idx))
pout=$R/results/probe_$c
[ -f "$pout/done.json" ] && { event "$c" "skip (done)"; exit 0; }
[ -d "$pout" ] && mv "$pout" "${pout}_partial_$(date +%s)"
crops=0; [ "$c" = O2 ] && crops=1
event "$c" "PACO R2 probe start (port $port, crops=$crops)"
SVAMP_CROPS=$crops torchrun --nproc_per_node 2 --master_port $port -m src.svamp.probe --job "$R/configs/probe_$c.yaml" --out "$pout" \
  --exclude "$EXCL" --workers 12 > "$R/logs/probe_$c.log" 2>&1
[ -f "$pout/done.json" ] || { event "$c" "probe FAILED (log logs/probe_$c.log)"; exit 1; }
event "$c" "probe done"
