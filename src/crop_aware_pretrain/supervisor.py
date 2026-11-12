"""GPU-idle supervisor for dinov3_crop_aware_pretrain: one student on 0/1/2 idle GPUs, chosen from live nvidia-smi state.

  python -m src.crop_aware_pretrain.ctl start          (runs this in tmux session crop_aware_sup)

Rules: never signals anything but its own processes (identified by the session id of the children it started). A GPU may be used only if it has no external compute
process, no external memory above mem_idle, no utilization above util_idle when none of our processes are on it, is not reserved (ctl reserve), and has looked like that for
idle_stable_s. An external process appearing on a GPU we use -> safe checkpoint and restart without that GPU. World size changes only by stopping and restarting (never in-process).
The global batch and LR schedule are unchanged (micro batch x world x accum = global batch, config.world_setup), so 1 GPU and 2 GPUs share the same checkpoints.
PACO evals run on one of the GPUs that training already holds or a free one (they share the GPU with our own training; they never wait on, or preempt, an external job).
With eval.stop_rule.hold_at_gate true, training stops at the gate step until the PACO decision exists; false (default): training continues and the halt is applied when the decision arrives. Failed launches back off and halt after max_consecutive_failures; done / stop-rule halt are not restarted.
"""
import argparse
import fcntl
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.crop_aware_pretrain import ckpt as CK       # noqa: E402
from src.crop_aware_pretrain import config as CF     # noqa: E402
from src.crop_aware_pretrain import gpu as GP        # noqa: E402

CLEAN = ("completed", "stop_after_steps")


def alive(pid, popen=None):
    if popen is not None:
        return popen.poll() is None
    try:
        s = open(f"/proc/{pid}/stat").read()
    except OSError:
        return False
    return s[s.rindex(")") + 2:].split()[0] != "Z"


class Job:
    def __init__(self, kind, gpus, pids, launch_id, popens=None, step=None, started=None):
        self.kind, self.gpus, self.pids, self.launch_id, self.popens = kind, list(gpus), list(pids), launch_id, popens
        self.step, self.started = step, started or time.time()
        self.stop_at, self.stop_reason, self.kill_at = None, None, None
        self.sids = [GP.proc_sid(p) for p in pids]

    def running(self):
        return [i for i, p in enumerate(self.pids) if alive(p, self.popens[i] if self.popens else None)]


class Supervisor:
    def __init__(self, cfg, config_path, overrides):
        self.cfg, self.config_path, self.overrides = cfg, config_path, overrides
        self.D = CF.dirs(cfg)
        for k in self.D.values():
            os.makedirs(k, exist_ok=True)
        self.sc = cfg["supervisor"]
        self.ctl = self.D["control"]
        self.watch = [int(x) for x in self.sc["watch_gpus"]]
        self.idle_since = {g: None for g in self.watch}
        self.ext_since = {g: None for g in self.watch}
        self.train, self.eval = None, None
        self.next_train_at = 0.0
        self.shutdown = False
        self.trust_idle = bool(cfg["supervisor"].get("trust_idle_at_start"))
        self.lock = open(os.path.join(self.ctl, "supervisor.lock"), "w")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another supervisor is already running for this experiment", flush=True)
            sys.exit(75)
        self.lock.write(str(os.getpid()))
        self.lock.flush()
        self.ev = open(os.path.join(self.D["logs"], "supervisor_events.jsonl"), "a")
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGINT, self._sig)
        self.event("supervisor_start", pid=os.getpid(), watch=self.watch)
        self.adopt()

    def _sig(self, *_):
        self.shutdown = True

    def event(self, event_name, **kw):
        rec = dict(time=time.strftime("%F %T"), event=event_name, **kw)
        self.ev.write(json.dumps(rec, default=str) + "\n")
        self.ev.flush()
        print(json.dumps(rec, default=str), flush=True)

    # ---------------------------------------------------------------- persistent state
    def path(self, name):
        return os.path.join(self.ctl, name)

    def has(self, name):
        return os.path.exists(self.path(name))

    def done(self):
        return os.path.exists(os.path.join(self.D["run"], "done.json"))

    def halted(self):
        return self.has("HALT_stop_rule.json") or self.has("HALT_failure.json")

    def stop_decision(self):
        f = self.path("stop_decision.json")
        if os.path.exists(f) and not self.has("override_stop_rule") and json.load(open(f)).get("decision") == "halt" and not self.has("HALT_stop_rule.json"):
            CK.atomic_json(json.load(open(f)), self.path("HALT_stop_rule.json"))
            self.event("halt_stop_rule", decision=json.load(open(f)))

    def reservations(self, now):
        out = {}
        for f in glob.glob(os.path.join(self.ctl, "reserve_gpu_*.json")):
            r = json.load(open(f))
            if r.get("until") and r["until"] < now:
                os.remove(f)
                continue
            out[int(r["gpu"])] = r
        return out

    def _failures(self):
        f = self.path("failures.json")
        return json.load(open(f)) if os.path.exists(f) else dict(count=0, last_step=-1)

    def record_failure(self, kind, detail):
        s = self._failures()
        step = self.current_step()
        if step - s.get("last_step", -1) >= int(self.sc["progress_reset_steps"]) and s.get("last_step", -1) >= 0:
            s["count"] = 0
        s["count"] += 1
        s["last_step"] = step
        s["last"] = dict(kind=kind, detail=detail, time=time.strftime("%F %T"))
        CK.atomic_json(s, self.path("failures.json"))
        self.event("failure", kind=kind, count=s["count"], step=step, detail=detail)
        self.next_train_at = time.time() + float(self.sc["failure_backoff_s"])
        if s["count"] >= int(self.sc["max_consecutive_failures"]):
            CK.atomic_json(s, self.path("HALT_failure.json"))
            self.event("halt_failure", count=s["count"])

    def current_step(self):
        f = os.path.join(self.D["logs"], "heartbeat.json")
        if os.path.exists(f):
            try:
                return int(json.load(open(f))["step"])
            except Exception:  # noqa: BLE001
                return 0
        return 0

    # ---------------------------------------------------------------- launching
    def _save_active(self):
        rec = {}
        for nm, j in (("train", self.train), ("eval", self.eval)):
            if j:
                rec[nm] = dict(gpus=j.gpus, pids=j.pids, sids=j.sids, launch_id=j.launch_id, step=j.step, started=j.started, kind=j.kind)
        CK.atomic_json(rec, self.path("active.json"))

    def adopt(self):
        f = self.path("active.json")
        if not os.path.exists(f):
            return
        rec = json.load(open(f))
        for nm in ("train", "eval"):
            r = rec.get(nm)
            if r and any(alive(p) and GP.proc_sid(p) in r["sids"] for p in r["pids"]):
                j = Job(r["kind"], r["gpus"], r["pids"], r["launch_id"], None, r.get("step"), r["started"])
                j.sids = r["sids"]
                setattr(self, nm, j)
                self.event("adopted_running_job", kind=nm, gpus=j.gpus, pids=j.pids)
        for d in glob.glob(os.path.join(self.D["evals"], "step_*")):
            run = os.path.join(d, "RUNNING")
            if os.path.exists(run) and not (self.eval and os.path.basename(d) == f"step_{self.eval.step:08d}"):
                os.replace(run, os.path.join(d, "PENDING"))

    def _spawn(self, cmd, env, log_path):
        logf = open(log_path, "a")
        return subprocess.Popen(cmd, cwd=CF.ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)

    def launch_train(self, gpus, now):
        world = len(gpus)
        lid = f"L{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        cmd = self.cfg["supervisor"].get("test_train_cmd") or [sys.executable, "-u", "-m", "src.crop_aware_pretrain.train", "--managed", "--resume", "auto"]
        gate = int(self.cfg["eval"]["stop_rule"]["gate_step"])
        if self.cfg["eval"]["stop_rule"].get("hold_at_gate") and not os.path.exists(self.path("stop_decision.json")) and not self.has("override_stop_rule") and self.current_step() < gate and "test_train_cmd" not in self.cfg["supervisor"]:
            cmd = cmd + ["--stop-at-step", str(gate)]
        if self.config_path:
            cmd = cmd + ["--config", self.config_path]
        cmd = cmd + [x for ov in self.overrides for x in ("--set", ov)]
        base = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), CROP_AWARE_LAUNCH_ID=lid, PYTHONUNBUFFERED="1",
                    NCCL_IB_DISABLE="1", CUDA_CACHE_DISABLE="0", CUDA_CACHE_MAXSIZE="4294967296", CUDA_CACHE_PATH=os.path.join(CF.ROOT, "cache", "cuda_jit"))
        port = None
        if world > 1:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        popens = []
        for r in range(world):
            env = dict(base)
            if world > 1:
                env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(r), LOCAL_RANK=str(r))
            popens.append(self._spawn(cmd, env, os.path.join(self.D["logs"], f"train_{lid}_rank{r}.log")))
        self.train = Job("train", gpus, [p.pid for p in popens], lid, popens)
        self._save_active()
        self.event("train_launch", gpus=gpus, world=world, launch_id=lid, pids=self.train.pids, step=self.current_step())

    def launch_eval(self, step, gpu):
        d = os.path.join(self.D["evals"], f"step_{step:08d}")
        os.replace(os.path.join(d, "PENDING"), os.path.join(d, "RUNNING"))
        lid = f"E{step}_{time.strftime('%Y%m%d_%H%M%S')}"
        cmd = self.cfg["supervisor"].get("test_eval_cmd") or [sys.executable, "-u", "-m", "src.crop_aware_pretrain.evals", "run", "--step", str(step)]
        if self.config_path:
            cmd = cmd + ["--config", self.config_path]
        cmd = cmd + [x for ov in self.overrides for x in ("--set", ov)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CROP_AWARE_LAUNCH_ID=lid, CROP_AWARE_EVAL_STEP=str(step), PYTHONUNBUFFERED="1",
                   CUDA_CACHE_DISABLE="0", CUDA_CACHE_MAXSIZE="4294967296", CUDA_CACHE_PATH=os.path.join(CF.ROOT, "cache", "cuda_jit"))
        p = self._spawn(cmd, env, os.path.join(d, "eval.log"))
        self.eval = Job("eval", [gpu], [p.pid], lid, [p], step)
        self._save_active()
        self.event("eval_launch", step=step, gpu=gpu, launch_id=lid)

    def pending_evals(self):
        return sorted(int(os.path.basename(os.path.dirname(f)).split("_")[1]) for f in glob.glob(os.path.join(self.D["evals"], "step_*", "PENDING")))

    # ---------------------------------------------------------------- stopping / finishing
    def request_stop(self, j, reason):
        if j.stop_at is not None:
            return
        j.stop_at, j.stop_reason = time.time(), reason
        j.kill_at = time.time() + float(self.sc["stop_timeout_s"])
        for i in j.running():
            try:
                os.kill(j.pids[i], signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.event("stop_request", kind=j.kind, reason=reason, gpus=j.gpus, launch_id=j.launch_id)

    def _finish_train(self, j):
        self.train = None
        self._save_active()
        exit_f = self.path("last_train_exit.json")
        r = json.load(open(exit_f)) if os.path.exists(exit_f) else {}
        codes = [p.returncode for p in j.popens] if j.popens else None
        if codes and 75 in codes:                # the run lock was held: another copy exists; do not treat as a crash, do not retry immediately
            self.next_train_at = time.time() + 300
            self.event("train_lock_held", codes=codes, launch_id=j.launch_id)
            return
        clean = r.get("launch_id") == j.launch_id and (r.get("reason") in CLEAN or str(r.get("reason", "")).startswith("signal_"))
        self.event("train_exit", launch_id=j.launch_id, clean=clean, reason=r.get("reason"), step=r.get("step"), codes=codes, requested=j.stop_reason)
        if not clean and j.stop_reason != "hang":
            self.record_failure("train", dict(codes=codes, exit=r))
        elif j.stop_reason == "hang":
            self.record_failure("train_hang", dict(codes=codes))
        else:
            if r.get("reason") == "completed":
                self.event("training_completed", step=r.get("step"))

    def _finish_eval(self, j):
        self.eval = None
        self._save_active()
        d = os.path.join(self.D["evals"], f"step_{j.step:08d}")
        run, pend = os.path.join(d, "RUNNING"), os.path.join(d, "PENDING")
        code = j.popens[0].returncode if j.popens else None
        ok = code == 0 or (code is None and os.path.exists(os.path.join(d, "probe_ok.json")))
        if ok:
            os.replace(run, os.path.join(d, "DONE"))
            self.event("eval_done", step=j.step, gpu=j.gpus)
            self.stop_decision()
            return
        if j.stop_reason:                                 # preempted: back to the queue, not a failure
            os.replace(run, pend)
            self.event("eval_preempted", step=j.step, reason=j.stop_reason)
            return
        f = os.path.join(d, "attempts")
        n = int(open(f).read()) + 1 if os.path.exists(f) else 1
        open(f, "w").write(str(n))
        if n >= int(self.cfg["eval"]["max_attempts"]):
            os.replace(run, os.path.join(d, "FAILED"))
            self.event("eval_failed", step=j.step, attempts=n, code=code)
            self.record_failure("eval", dict(step=j.step, attempts=n, code=code))
            CK.atomic_json(dict(reason=f"eval at step {j.step} failed {n} times; training halted because its result cannot be judged"), self.path("HALT_failure.json"))
        else:
            os.replace(run, pend)
            self.event("eval_retry", step=j.step, attempts=n, code=code)

    def check_jobs(self, now):
        for j in (self.train, self.eval):
            if j is None:
                continue
            run = j.running()
            if not run:
                (self._finish_train if j.kind == "train" else self._finish_eval)(j)
                continue
            if len(run) < len(j.pids) and j.stop_at is None and now - j.started > float(self.sc["rank_grace_s"]):
                self.request_stop(j, "rank_exit")
            if j.kill_at and now > j.kill_at:
                for i in run:
                    try:
                        os.kill(j.pids[i], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self.event("force_kill_own_job", kind=j.kind, pids=[j.pids[i] for i in run], launch_id=j.launch_id)
                j.kill_at = now + 30
            if j.kind == "train" and j.stop_at is None:
                f = os.path.join(self.D["logs"], "heartbeat.json")
                if os.path.exists(f) and now - j.started > float(self.sc["hang_timeout_s"]):
                    age = now - float(json.load(open(f))["time"])
                    if age > float(self.sc["hang_timeout_s"]):
                        self.request_stop(j, "hang")

    # ---------------------------------------------------------------- main decision
    def tick(self):
        now = time.time()
        st = GP.query()
        own = {s for j in (self.train, self.eval) if j for s in j.sids}
        res = self.reservations(now)
        info = {g: GP.describe(g, st, int(self.sc["mem_idle_mib"]), int(self.sc["util_idle_pct"]), own) for g in self.watch}
        for g in self.watch:
            if info[g]["busy"] or g in res:
                self.idle_since[g] = None
            elif self.idle_since[g] is None:
                self.idle_since[g] = now - float(self.sc["idle_stable_s"]) if self.trust_idle else now
        self.trust_idle = False
        grace = float(self.sc.get("external_grace_s", 0))
        for g in self.watch:                                     # how long has something external been on this GPU?
            self.ext_since[g] = (self.ext_since[g] or now) if info[g]["busy"] else None
        usable = {g: not info[g]["busy"] and g not in res for g in self.watch}
        # a job we already run yields only to an external job that stays longer than the grace period (a reservation applies at once)
        yields = bool(self.sc.get("yield_to_external", True))
        must_yield = {g: g in res or (yields and info[g]["busy"] and now - self.ext_since[g] >= grace) for g in self.watch}
        idle_s = {g: (now - self.idle_since[g]) if self.idle_since[g] is not None else 0 for g in self.watch}
        stable = {g: usable[g] and idle_s[g] >= float(self.sc["idle_stable_s"]) for g in self.watch}

        self.check_jobs(now)
        self.stop_decision()
        paused = self.has("PAUSE")
        shutting = self.has("SHUTDOWN") or self.shutdown
        blocked = self.done() or self.halted() or self.has("HALT_stop_rule.json")
        pend = [] if self.halted() else self.pending_evals()
        tr, ev = self.train, self.eval

        # ---- stops
        for j in (tr, ev):
            if j is None or j.stop_at is not None:
                continue
            bad = [g for g in j.gpus if must_yield[g]]
            if shutting:
                self.request_stop(j, "shutdown")
            elif paused:
                self.request_stop(j, "pause")
            elif blocked and j.kind == "train":
                self.request_stop(j, "halted_or_done")
            elif bad:
                self.request_stop(j, "gpu_needed:" + ",".join(f"{g}({info[g]['why'] or 'reserved'})" for g in bad))

        tr, ev = self.train, self.eval
        # ---- start eval (one GPU, oldest pending first)
        busy_train = set(tr.gpus) if tr and tr.stop_at is None else set()
        if pend and ev is None and not paused and not shutting and (tr is None or tr.stop_at is None):
            free = [g for g in self.watch if stable[g] and g not in busy_train] or [g for g in (tr.gpus if tr else []) if usable[g]]
            if free:
                self.launch_eval(pend[0], free[0])
        tr, ev = self.train, self.eval

        # ---- start / scale training
        if not paused and not shutting and not blocked:
            eval_g = set(ev.gpus) if ev else set()
            free = [g for g in self.watch if stable[g] and g not in eval_g]
            gate = int(self.cfg["eval"]["stop_rule"]["gate_step"])
            hold = self.cfg["eval"]["stop_rule"].get("hold_at_gate") and self.current_step() >= gate and not self.has("stop_decision.json") and not self.has("override_stop_rule") and "test_train_cmd" not in self.cfg["supervisor"]
            free = [g for g in self.watch if stable[g]]            # evals share a GPU with training
            if tr is None and now >= self.next_train_at and free and not hold:
                self.launch_train(free[: int(self.sc["max_gpus"])], now)
            elif tr is not None and tr.stop_at is None and len(tr.gpus) < int(self.sc["max_gpus"]) and len(free) > len(tr.gpus):
                extra = [g for g in free if g not in tr.gpus and idle_s[g] >= float(self.sc["scale_up_stable_s"])]
                if extra and now - tr.started >= float(self.sc["min_run_before_rescale_s"]):
                    self.request_stop(tr, "scale_up")

        tr = self.train
        if not paused and not shutting and not blocked and tr is not None and tr.stop_at is None and now - tr.started >= float(self.sc["min_run_before_rescale_s"]):
            eval_free = [g for g in self.watch if stable[g] and g not in tr.gpus]    # a GPU to ourselves beats one shared with an external job
            if eval_free and any(info[g]["busy"] for g in tr.gpus):
                self.request_stop(tr, "move_to_free_gpu")

        CK.atomic_json(dict(time=time.strftime("%F %T"), paused=paused, shutdown=shutting, done=self.done(), halted=self.halted(), failures=self._failures(),
                            step=self.current_step(), reservations=sorted(res), pending_evals=pend,
                            train=dict(gpus=self.train.gpus, launch_id=self.train.launch_id, pids=self.train.pids, stopping=self.train.stop_reason) if self.train else None,
                            eval=dict(gpus=self.eval.gpus, step=self.eval.step, stopping=self.eval.stop_reason) if self.eval else None, gpus=info, idle_seconds=idle_s),
                       self.path("status.json"))
        return shutting and self.train is None and self.eval is None

    def run(self):
        errors = 0
        while True:
            try:
                if self.tick():
                    self.event("supervisor_exit", reason="shutdown")
                    return 0
                errors = 0
            except Exception as e:  # noqa: BLE001  a transient nvidia-smi error must not kill the supervisor
                errors += 1
                self.event("tick_error", error=repr(e), consecutive=errors)
                if self.shutdown and self.train is None and self.eval is None:
                    return 1
                if errors >= 5 and not self.has("HALT_failure.json"):     # a logic error must not silently keep launching jobs
                    CK.atomic_json(dict(reason=f"{errors} consecutive supervisor errors", last=repr(e)), self.path("HALT_failure.json"))
                    self.event("halt_failure", reason="supervisor tick errors")
            time.sleep(float(self.sc["poll_s"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = CF.load(a.config, a.set)
    if os.path.exists(os.path.join(CF.dirs(cfg)["control"], "SHUTDOWN")):
        os.remove(os.path.join(CF.dirs(cfg)["control"], "SHUTDOWN"))
    sys.exit(Supervisor(cfg, a.config, a.set).run())


if __name__ == "__main__":
    main()
