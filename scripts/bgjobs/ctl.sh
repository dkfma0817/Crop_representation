#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
exec /home/irteam/envs/dinov2_ord_py310/bin/python -m src.bgjobs.ctl "$@"
