#!/usr/bin/env bash
# Restart the background supervisor if it is not running (checks every 30 seconds). Survives terminal close; it cannot survive a full environment restart, which also kills this script.
cd "$(dirname "$0")/../.."
exec 9> output/2026-11-09/bgjobs/control/keeper.lock
flock -n 9 || exit 0
while true; do
  if [ ! -f output/2026-11-09/bgjobs/control/SHUTDOWN ]; then
    pgrep -f "python -u -m src.bgjobs.supervisor" > /dev/null || bash scripts/bgjobs/ctl.sh start --trust-idle 9>&- >> output/2026-11-09/bgjobs/control/keeper.log 2>&1
  fi
  sleep 30
done
