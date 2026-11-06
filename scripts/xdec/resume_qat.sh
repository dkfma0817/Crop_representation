#!/usr/bin/env bash
# Resume the QAT job from its latest checkpoint (HF get_last_checkpoint with --resume) on GPU 2,3 in tmux qat_800m,
# then VERIFY: checkpoint loaded, step increases past the checkpoint, loss logged, GPU memory used, tmux alive.
set -u
source "$(dirname "$0")/common.sh"
if [ -n "$(qat_pid)" ]; then event qat_resume "QAT already running (torchrun $(qat_pid)); not relaunched"; exit 0; fi
[ -n "$(qat_any)" ] && { event qat_resume "FAILED: stray QAT processes $(qat_any | tr '\n' ' ')"; exit 2; }
grep -qx -- "--resume" "$QAT_SAVE/cmdline.txt" || { event qat_resume "FAILED: saved cmdline lacks --resume"; exit 2; }
for i in $(seq 1 60); do [ -z "$(gpu23_apps)" ] && break; sleep 5; done
[ -n "$(gpu23_apps)" ] && { event qat_resume "FAILED: GPU 2,3 still used by: $(gpu23_apps | tr '\n' ';')"; exit 2; }
ck=$(qat_last_ckpt)
[ -f "$QAT_OUT/checkpoint-$ck/trainer_state.json" ] || { event qat_resume "FAILED: checkpoint-$ck incomplete"; exit 2; }
off=$(stat -c %s "$QAT_OUT/train.log")
st=FAILED; first=""; now=""; loss=""
for attempt in 1 2 3; do
  tmux kill-session -t "$QAT_TMUX" 2>/dev/null
  # pane keeps running after the job ends (exit code + pane text kept for diagnosis)
  tmux new -d -s "$QAT_TMUX" "bash $R/scripts/qat_launch.sh; echo \"EXIT=\$? \$(date '+%F %T')\" >> $R/logs/qat_launch_exit.txt; sleep 86400"
  t_launch=$(ts)
  event qat_resume "attempt $attempt: launched in tmux $QAT_TMUX from checkpoint-$ck (log append offset $off)"
  for i in $(seq 1 120); do
    sleep 15
    new=$(tail -c +$((off + 1)) "$QAT_OUT/train.log" 2>/dev/null)
    now=$(echo "$new" | grep -o '[0-9]*/305176' | tail -1 | cut -d/ -f1)
    [ -z "$first" ] && [ -n "$now" ] && first=$now
    loss=$(echo "$new" | grep -o "{'loss': '[0-9.]*'" | tail -1)
    if [ -n "$now" ] && [ "$now" -gt $((ck + 5)) ] && [ -n "$loss" ] && [ -n "$(qat_pid)" ]; then st=OK; break; fi
    [ -z "$(qat_pid)" ] && [ $i -gt 8 ] && break
  done
  [ "$st" = OK ] && break
  tmux capture-pane -p -t "$QAT_TMUX" > "$R/logs/qat_pane_attempt$attempt.txt" 2>/dev/null
  event qat_resume "attempt $attempt not verified (pid '$(qat_pid)', step '${now:-}'); pane -> logs/qat_pane_attempt$attempt.txt"
  [ -n "$(qat_pid)" ] && break          # alive but slow: do not start a second copy
  for i in $(seq 1 24); do [ -z "$(qat_any)" ] && [ -z "$(gpu23_apps)" ] && break; sleep 5; done
done
P=$(qat_pid)
new=$(tail -c +$((off + 1)) "$QAT_OUT/train.log")
loaded=$(echo "$new" | grep -m3 -iE "continuing training from checkpoint|resum|Loading model from|loaded" | cut -c1-200 | tr '\n' ' ')
errs=$(echo "$new" | grep -m3 -iE "Traceback|Error" | cut -c1-200 | tr '\n' ' ')
cat > "$R/qat_resume_status.md" <<M
# QAT resume status

| item | value |
|---|---|
| verification | **$st** |
| pause record | \`qat_pause/paused.json\` |
| resume checkpoint | \`$QAT_OUT/checkpoint-$ck\` (HF Trainer restores model, optimizer.pt, scheduler.pt, rng_state_*, qat-rank-state-*) |
| launch time | $t_launch |
| verified at | $(ts) |
| first observed step after launch | ${first:-n/a} |
| step at verification | ${now:-n/a} (must be > $ck) |
| last logged loss line | ${loss:-n/a} |
| torchrun pid | ${P:-not running} |
| tmux session | $QAT_TMUX ($(tmux has-session -t $QAT_TMUX 2>/dev/null && echo alive || echo absent)) |
| log | \`$QAT_OUT/train.log\` (appended from byte $off) |
| command | saved cmdline (\`qat_pause/cmdline.txt\`, contains --resume), env \`qat_pause/environ.txt\` (CUDA_VISIBLE_DEVICES=2,3), cwd $(cat $QAT_SAVE/cwd.txt) |
| GPU 2,3 (idx, MiB, util) | $(gpu23_mem) |
| resume log lines | ${loaded:-none matched} |
| errors in new log | ${errs:-none} |
M
if [ "$st" = OK ]; then
  echo "$(ts) torchrun $P checkpoint-$ck step $now" > "$QAT_SAVE/resumed.txt"; rm -f "$QAT_SAVE/PAUSED_BY_XDEC"
  event qat_resume "OK: step $now > $ck, $loss, GPU $(gpu23_mem)"; exit 0
fi
event qat_resume "FAILED verification (step ${now:-none}, pid ${P:-none}); see qat_resume_status.md; retry: bash $R/scripts/recover_qat.sh"
exit 1
