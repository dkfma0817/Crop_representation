"""Stand-in for train.py in the supervisor unit test: same contract (run lock, heartbeat, last_train_exit.json, SIGTERM = clean stop)."""
import fcntl, json, os, signal, sys, time
import yaml

cfg = yaml.safe_load(open(sys.argv[sys.argv.index("--config") + 1]))
run = cfg["run_dir"]
ctl, logs = f"{run}/control", f"{run}/logs"
rank = int(os.environ.get("RANK", 0))
stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
crash = cfg.get("test_crash", False)
if rank == 0:
    lock = open(f"{ctl}/train.lock", "w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(75)
step_file = f"{run}/fake_step"
step = int(open(step_file).read()) if os.path.exists(step_file) else 0
world = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
with open(f"{run}/launches.txt", "a") as f:
    f.write(f"{os.environ['CROP_AWARE_LAUNCH_ID']} rank{rank} gpus={os.environ['CUDA_VISIBLE_DEVICES']} step={step}\n")
if crash and step == 0 and os.path.exists(f"{run}/crash_now"):
    print("simulated crash"); sys.exit(1)
done_at = cfg.get("test_done_at")
while True:
    step += 1
    if rank == 0:
        open(step_file, "w").write(str(step))
        json.dump(dict(step=step, time=time.time(), world=world), open(f"{logs}/heartbeat.json", "w"))
    if done_at and step >= done_at:
        reason = "completed"
        break
    if stop["flag"]:
        reason = "signal_15"
        break
    time.sleep(0.2)
if rank == 0:
    json.dump(dict(reason=reason, step=step, world=world, launch_id=os.environ["CROP_AWARE_LAUNCH_ID"]), open(f"{ctl}/last_train_exit.json", "w"))
    if reason == "completed":
        json.dump(dict(status="done"), open(f"{run}/done.json", "w"))
