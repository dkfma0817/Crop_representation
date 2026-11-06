# QAT resume status

| item | value |
|---|---|
| verification | **OK** |
| pause record | `qat_pause/paused.json` |
| resume checkpoint | `/home/irteam/low-bit-training-v2/output/2026-11-04/scale-800m/800m-ste-code7.5e-5-compile/checkpoint-30000` (HF Trainer restores model, optimizer.pt, scheduler.pt, rng_state_*, qat-rank-state-*) |
| launch time | 2026-11-06 06:59:48 |
| verified at | 2026-11-06 07:02:34 |
| first observed step after launch | 0 |
| step at verification | 30057 (must be > 30000) |
| last logged loss line | {'loss': '3.208' |
| torchrun pid | 1979377 |
| tmux session | qat_800m (alive) |
| log | `/home/irteam/low-bit-training-v2/output/2026-11-04/scale-800m/800m-ste-code7.5e-5-compile/train.log` (appended from byte 5410308) |
| command | saved cmdline (`qat_pause/cmdline.txt`, contains --resume), env `qat_pause/environ.txt` (CUDA_VISIBLE_DEVICES=2,3), cwd /home/irteam/low-bit-training-v2 |
| GPU 2,3 (idx, MiB, util) | 2, 90433 MiB, 100 % 3, 80245 MiB, 100 %  |
| resume log lines | ===== resumed by e6_multilayer_memory queue 2026-11-06 06:59:48 (tmux qat_800m) ===== Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as 07:00:16 [rank0] Base model loaded: LlamaForCausalLM (scratch)  |
| errors in new log | none |
