#!/usr/bin/env bash
# bash launch_queue.sh stop|share
source "$(dirname "$0")/common.sh"
m=${1:?stop or share}
tmux has-session -t e6dir_q 2>/dev/null && { echo "e6dir_q already running"; exit 1; }
tmux new -d -s e6dir_q "QAT_MODE=$m MAXWAIT=0 bash $R/scripts/run_all.sh >> $R/logs/queue.log 2>&1"
echo "started tmux e6dir_q (QAT_MODE=$m)"
