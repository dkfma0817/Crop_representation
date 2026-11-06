#!/usr/bin/env bash
# Record the QAT state and stop it (GPU 2,3). Waits for the NEXT 5k checkpoint if it is due within MAXWAIT seconds
# (default 5400), otherwise stops at once (unsaved steps since the last checkpoint are lost and re-trained on resume).
# Only the identified QAT process tree (port 30871, cwd low-bit-training-v2, CUDA_VISIBLE_DEVICES=2,3) is signalled.
set -u
source "$(dirname "$0")/common.sh"
MAXWAIT=${MAXWAIT:-5400}
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null); [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }   # zombie = exited
mkdir -p "$QAT_SAVE"
P=$(qat_pid)
last=$(qat_last_ckpt)
if [ -z "$P" ]; then
  left=$(qat_any | tr '\n' ' ')
  echo "{\"time\": \"$(ts)\", \"running\": false, \"leftover_pids\": \"$left\", \"latest_checkpoint\": $last}" > "$QAT_SAVE/paused.json"
  event qat "not running at pause time (latest checkpoint-$last; leftover: '${left}')"
  [ -n "$left" ] && { event qat "ABORT: QAT processes without torchrun main exist"; exit 2; }
  touch "$QAT_SAVE/PAUSED_BY_XDEC"; exit 0
fi
[ "$(readlink /proc/$P/cwd)" = /home/irteam/low-bit-training-v2 ] || { event qat "ABORT: pid $P cwd mismatch"; exit 2; }
tr '\0' '\n' < /proc/$P/cmdline > "$QAT_SAVE/cmdline_at_pause.txt"
tr '\0' '\n' < /proc/$P/environ > "$QAT_SAVE/environ_at_pause.txt"
cmp -s "$QAT_SAVE/cmdline_at_pause.txt" "$QAT_SAVE/cmdline.txt" || cp "$QAT_SAVE/cmdline_at_pause.txt" "$QAT_SAVE/cmdline.txt"
cmp -s "$QAT_SAVE/environ_at_pause.txt" "$QAT_SAVE/environ.txt" || cp "$QAT_SAVE/environ_at_pause.txt" "$QAT_SAVE/environ.txt"
ps -eo pid,ppid,pgid,sid,args | grep -v grep | grep "src.qat.train" | grep -- "--output-dir $QAT_REL " | grep -v "master_port 30870" | cut -c1-160 > "$QAT_SAVE/process_tree_at_pause.txt"
cur=$(qat_step); next=$((last + 5000)); eta=$(( (next - cur) * 155 / 100 ))
event qat "running torchrun $P, step $cur, latest checkpoint-$last, next checkpoint-$next ETA ${eta}s (max wait ${MAXWAIT}s)"
waited=no
if [ "$eta" -le "$MAXWAIT" ]; then
  waited=yes
  until [ -f "$QAT_OUT/checkpoint-$next/trainer_state.json" ] && [ -f "$QAT_OUT/checkpoint-$next/optimizer.pt" ] && [ -f "$QAT_OUT/checkpoint-$next/scheduler.pt" ]; do
    alive "$P" || { event qat "process ended by itself while waiting"; break; }
    sleep 15
  done
  # wait until every checkpoint file has stopped growing (save + rotation finished)
  prev=""; for i in $(seq 1 40); do s=$(du -sb "$QAT_OUT/checkpoint-$next" 2>/dev/null | cut -f1); [ -n "$s" ] && [ "$s" = "$prev" ] && break; prev=$s; sleep 15; done
  sleep 30
fi
cur=$(qat_step)
event qat "stopping torchrun $P (SIGTERM) at step $cur"
kill -TERM "$P" 2>/dev/null
for i in $(seq 1 36); do alive "$P" || break; sleep 5; done
if alive "$P" || [ -n "$(qat_any)" ]; then
  event qat "still alive after 180 s -> SIGKILL on the QAT process groups only"
  for p in $(qat_any); do pg=$(ps -o pgid= -p $p | tr -d ' '); [ -n "$pg" ] && kill -KILL -- -"$pg" 2>/dev/null; kill -KILL "$p" 2>/dev/null; done
fi
for i in $(seq 1 24); do [ -z "$(qat_any)" ] && [ -z "$(gpu23_apps)" ] && break; sleep 5; done
left=$(qat_any | tr '\n' ' '); apps=$(gpu23_apps | tr '\n' ';')
final=$(qat_last_ckpt)
ok=$(python3 - "$QAT_OUT/checkpoint-$final" <<'PY'
import json, os, sys
d = sys.argv[1]; need = ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state_0.pth", "rng_state_1.pth",
                          "qat-rank-state-00-of-02.pt", "qat-rank-state-01-of-02.pt"]
miss = [f for f in need if not os.path.isfile(os.path.join(d, f)) or os.path.getsize(os.path.join(d, f)) == 0]
st = json.load(open(os.path.join(d, "trainer_state.json")))
with open(os.path.join(d, "model.safetensors"), "rb") as f:
    n = int.from_bytes(f.read(8), "little"); json.loads(f.read(n))
print(json.dumps(dict(missing=miss, global_step=st["global_step"], max_steps=st["max_steps"], safetensors_header_ok=True,
                      sizes={f: os.path.getsize(os.path.join(d, f)) for f in need if os.path.isfile(os.path.join(d, f))})))
PY
)
cat > "$QAT_SAVE/paused.json" <<J
{"time": "$(ts)", "running": true, "torchrun_pid": $P, "waited_for_next_checkpoint": "$waited", "step_at_stop": ${cur:-null},
 "checkpoint_for_resume": "$QAT_OUT/checkpoint-$final", "checkpoint_step": $final, "unsaved_steps_lost": $(( ${cur:-$final} - final )),
 "leftover_qat_pids": "$left", "gpu23_apps_after": "$apps", "checkpoint_check": $ok}
J
touch "$QAT_SAVE/PAUSED_BY_XDEC"
event qat "paused: step_at_stop $cur, resume checkpoint-$final, lost $(( ${cur:-$final} - final )) steps; GPU2,3: $(gpu23_mem)"
[ -z "$left" ] || exit 3
