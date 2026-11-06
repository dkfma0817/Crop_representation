#!/usr/bin/env bash
# Manual / automatic recovery after an abnormal queue exit (or reboot): stop OUR leftover GPU jobs (src.xdec.* only),
# then resume QAT exactly as resume_qat.sh. Idempotent: does nothing if QAT is already running.
#   bash output/2026-11-05/xdec/scripts/recover_qat.sh
set -u
source "$(dirname "$0")/common.sh"
for p in $(ps -eo pid,args | grep -v grep | grep -E "python.* -m src\.xdec\.|torch.distributed.run.* -m src\.xdec\." | awk '{print $1}'); do
  event recover "killing leftover xdec process $p"; kill -TERM "$p" 2>/dev/null
done
sleep 20
for p in $(ps -eo pid,args | grep -v grep | grep -E "python.* -m src\.xdec\.|torch.distributed.run.* -m src\.xdec\." | awk '{print $1}'); do kill -KILL "$p" 2>/dev/null; done
if [ -n "$(qat_pid)" ]; then event recover "QAT already running; nothing to do"; exit 0; fi
bash "$R/scripts/resume_qat.sh"
