#!/usr/bin/env bash
# bash launch_queue.sh stop|share
source "$(dirname "$0")/common.sh"
m=${1:?stop or share}
tmux has-session -t svamp_q 2>/dev/null && { echo "svamp_q already running"; exit 1; }
tmux new -d -s svamp_q "QAT_MODE=$m MAXWAIT=0 bash $R/scripts/run_all.sh >> $R/logs/queue.log 2>&1"
echo "started tmux svamp_q (QAT_MODE=$m)"
