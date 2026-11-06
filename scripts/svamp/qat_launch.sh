#!/usr/bin/env bash
# Runs INSIDE tmux session qat_800m: the original QAT command (saved cmdline incl. --resume) with the saved environment.
source "$(dirname "$0")/common.sh"
cd "$(cat "$QAT_SAVE/cwd.txt")" || exit 1
mapfile -t ENVV < "$QAT_SAVE/environ.txt"
mapfile -t CMD < "$QAT_SAVE/cmdline.txt"
printf '\n===== resumed by svamp queue %s (tmux %s) =====\n' "$(date '+%F %T')" "$QAT_TMUX" >> "$QAT_OUT/train.log"
exec env -i "${ENVV[@]}" "${CMD[@]}" >> "$QAT_OUT/train.log" 2>&1 < /dev/null
