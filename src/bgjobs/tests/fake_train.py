"""Stand-in for src.bgjobs.train in the supervisor test: same contract (lock, heartbeat, last_train_exit.json, SIGTERM = clean stop)."""
import fcntl, json, os, signal, sys, time
a = sys.argv[1:]
run = next(s.split("=", 1)[1] for s in a if s.startswith("run_dir="))
sets = dict(s.split("=", 1) for s in a if "=" in s and not s.startswith("run_dir="))
ctl, logs = f"{run}/control", f"{run}/logs"
stop = {"f": False}
signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("f", True))
lock = open(f"{ctl}/train.lock", "w")
try:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(75)
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
open(f"{run}/launches.txt", "a").write(f"{os.environ['BGJOB_LAUNCH_ID']} gpu={gpu}\n")
if os.path.exists(f"{run}/crash"):
    sys.exit(1)
step_f = f"{run}/fake_step"
step = int(open(step_f).read()) if os.path.exists(step_f) else 0
done_at = int(sets.get("test.done_at", 0))
pend_at = int(sets.get("test.pending_at", 0))
reason = None
while True:
    step += 1
    open(step_f, "w").write(str(step))
    json.dump(dict(step=step, time=time.time()), open(f"{logs}/heartbeat.json", "w"))
    if pend_at and step == pend_at:
        d = f"{run}/evals/step_00020000"; os.makedirs(d, exist_ok=True); open(f"{d}/PENDING", "w").write("x")
    if done_at and step >= done_at:
        reason = "completed"; break
    if stop["f"]:
        reason = "signal_15"; break
    time.sleep(0.05)
json.dump(dict(reason=reason, step=step, launch_id=os.environ["BGJOB_LAUNCH_ID"]), open(f"{ctl}/last_train_exit.json", "w"))
if reason == "completed":
    json.dump(dict(status="done"), open(f"{run}/done.json", "w"))
