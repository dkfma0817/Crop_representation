#!/usr/bin/env bash
# Control wrapper: bash scripts/crop_aware_pretrain/ctl.sh start|stop|status|pause|resume|reserve --gpus 2,3 [--hours 4]|release --gpus all|clear-failure|override-stop-rule
cd "$(dirname "$0")/../.."
exec /home/irteam/envs/dinov2_ord_py310/bin/python -m src.crop_aware_pretrain.ctl "$@"
