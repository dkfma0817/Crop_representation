#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
awk -F'\t' '{printf "%-11s %-8s %-20s %s\n", $1, $2, $3, $4}' "$R/queue_status.tsv"
echo; echo "last events:"; tail -8 "$R/logs/events.tsv" 2>/dev/null
echo; echo "tmux: $(tmux ls 2>/dev/null | grep -E 'svamp_q|qat_800m' | tr '\n' ' ')"
echo "QAT torchrun: $(qat_pid || true)  step: $(qat_step)  latest ckpt: $(qat_last_ckpt)"
echo "GPU 2,3: $(gpu23_mem)"
