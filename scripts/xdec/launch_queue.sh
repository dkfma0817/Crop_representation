#!/usr/bin/env bash
# Start the whole pilot in tmux session xdec_q (re-running resumes: DONE items are skipped).
source "$(dirname "$0")/common.sh"
tmux has-session -t xdec_q 2>/dev/null && { echo "xdec_q already running"; exit 1; }
tmux new -d -s xdec_q "bash $R/scripts/run_all.sh >> $R/logs/queue.log 2>&1"
echo "started tmux xdec_q; status: bash $R/scripts/status.sh"
