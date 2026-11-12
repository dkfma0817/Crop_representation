"""One-GPU-one-job background supervisor: GPU 0 and GPU 1 are independent slots, each with its own ordered job list (configs/bgjobs/queue.yaml). No DDP.
A slot key like "0b" is an additional job slot on the same physical GPU (user 2026-11-09: keep the GPU busy with queued research work).

  python -m src.bgjobs.ctl start

A slot starts its current job when its GPU has been idle for idle_stable_s (no external compute process, no external memory, no utilization without our processes, not reserved, not paused), and resumes it from
its last checkpoint after any stop. It only ever signals processes it started (session id). An external job that stays longer than external_grace_s on a GPU makes that GPU's job checkpoint and stop;
reservations and `ctl pause` apply at once. Pauses expire independently per GPU after 60 seconds of physical GPU idleness. Reservations in the crop-aware control directory are honoured too, because existing queues request GPUs through `src.crop_aware_pretrain.ctl reserve`.
Evals (official AP, R2 retrained) run one checkpoint at a time per job by default. Queue entries can opt into parallel evaluation of distinct checkpoints. Job statuses: PENDING, RUNNING, PAUSED, COMPLETED, FAILED, STOPPED_NO_GAIN.
Entries with repeat: true enqueue the next numbered round after training completes. Each round starts in a fresh run directory on the same GPU.
"""
import argparse
import fcntl
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.crop_aware_pretrain import ckpt as CK        # noqa: E402
from src.crop_aware_pretrain import config as CF      # noqa: E402
from src.crop_aware_pretrain import gpu as GP         # noqa: E402

BASE = os.path.join(_ROOT, "configs", "bgjobs", "base.yaml")
QUEUE = os.path.join(_ROOT, "configs", "bgjobs", "queue.yaml")
ROOT_DIR = os.path.join(_ROOT, "output", "2026-11-09", "bgjobs")
CROP_CTL = os.path.join(_ROOT, "output", "2026-11-09", "dinov3_crop_aware_pretrain", "control")
SC_DEFAULT = dict(poll_s=10, idle_stable_s=30, external_grace_s=60, auto_release_idle_s=75, pause_auto_release_idle_s=60, mem_idle_mib=2000, util_idle_pct=10, stop_timeout_s=900, hang_timeout_s=1800, max_failures=3, failure_backoff_s=300, progress_reset_steps=100, yield_to_external=True, tag="crop_aware_pretrain_bg",
                  coexist_patterns=["src.crop_aware_pretrain.train", "src.crop_aware_pretrain.evals", "dinov3_crop_aware_pretrain/evals",
                                    "/fgtrain_s0"])  # user: run existing A/B alongside this pilot to use spare GPU capacity
TERMINAL = ("COMPLETED", "STOPPED_NO_GAIN", "FAILED")


def alive(p):
    if p is None:
        return False
    if isinstance(p, subprocess.Popen):
        return p.poll() is None
    try:
        s = open(f"/proc/{p}/stat").read()
    except OSError:
        return False
    return s[s.rindex(")") + 2:].split()[0] != "Z"


class Slot:
    def __init__(self, key):
        self.key, self.gpu = key, int(str(key).rstrip("b"))       # key "0b" = a second job slot on physical GPU 0
        self.train, self.eval = None, None
        self.idle_since, self.ext_since, self.res_idle_since = None, None, None
        self.next_train_at = 0.0


class Sup:
    def __init__(self, root=ROOT_DIR, queue=QUEUE, base=BASE, test=None, trust_idle=False):
        self.root, self.queue_path, self.base, self.test = root, queue, base, test or {}
        self.trust = trust_idle or bool(self.test.get("trust"))   # first tick only: a GPU that is idle now does not wait idle_stable_s (for a start explicitly requested on a free GPU)
        self.ctl = os.path.join(root, "control")
        os.makedirs(self.ctl, exist_ok=True)
        self.crop_ctl = self.test.get("crop_ctl", CROP_CTL)
        self.sc = dict(SC_DEFAULT, **self.test.get("sc", {}))
        self.lock = open(os.path.join(self.ctl, "supervisor.lock"), "a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another background supervisor is already running", flush=True)
            sys.exit(75)
        self.lock.seek(0)
        self.lock.truncate()
        self.lock.write(str(os.getpid()))
        self.lock.flush()
        self.ev = open(os.path.join(self.ctl, "events.jsonl"), "a")
        self.shutdown = False
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGINT, self._sig)
        self.slots = {}
        self.pause_idle = {}
        self.released_pauses = {}
        self.load_queue()
        self.adopt()
        self.event("supervisor_start", pid=os.getpid(), gpus=sorted(str(k) for k in self.slots))

    def _sig(self, *_):
        self.shutdown = True

    def event(self, name, **kw):
        rec = dict(time=time.strftime("%F %T"), event=name, **kw)
        self.ev.write(json.dumps(rec, default=str) + "\n")
        self.ev.flush()
        print(json.dumps(rec, default=str), flush=True)

    def load_queue(self):
        with open(self.queue_path + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.q = yaml.safe_load(open(self.queue_path))
            repeated = []
            for gpu, names in self.q["gpus"].items():
                for name in list(names):
                    job = self.q["jobs"][name]
                    if not job.get("repeat") or not os.path.exists(os.path.join(self.jobdir(name), "done.json")):
                        continue
                    stem, round_number = name.rsplit("_r", 1)
                    successor = f"{stem}_r{int(round_number) + 1}"
                    if successor in self.q["jobs"]:
                        continue
                    self.q["jobs"][successor] = dict(job, after=name)
                    job["repeat"] = False
                    names.append(successor)
                    repeated.append(dict(gpu=gpu, previous=name, job=successor))
            if repeated:
                temporary = f"{self.queue_path}.tmp.{os.getpid()}"
                with open(temporary, "w") as stream:
                    yaml.safe_dump(self.q, stream, sort_keys=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.queue_path)
        for entry in repeated:
            self.event("repeat_queued", **entry)
        for g in self.q["gpus"]:
            k = g if isinstance(g, int) else (int(g) if str(g).isdigit() else str(g))
            self.slots.setdefault(k, Slot(k))

    def jobdir(self, name):
        return os.path.join(self.root, name)

    def job_cfg(self, name):
        ov = [f"run_dir={self.jobdir(name)}"] + list(self.q["jobs"][name].get("set", []))
        return ov

    def prep(self, name):
        d = self.jobdir(name)
        for s in ("checkpoints", "logs", "control", "data", "evals"):
            os.makedirs(os.path.join(d, s), exist_ok=True)

    def status(self, name):
        d = self.jobdir(name)
        if os.path.exists(os.path.join(d, "STOPPED_NO_GAIN.json")):
            return "STOPPED_NO_GAIN"
        if os.path.exists(os.path.join(d, "FAILED.json")):
            return "FAILED"
        if os.path.exists(os.path.join(d, "done.json")) and not self.pending(name) and not self.has_running_eval(name):
            return "COMPLETED"
        return "ACTIVE"

    def pending(self, name):
        return sorted(int(os.path.basename(os.path.dirname(f)).split("_")[1]) for f in glob.glob(os.path.join(self.jobdir(name), "evals", "step_*", "PENDING")))

    def has_running_eval(self, name):
        return bool(glob.glob(os.path.join(self.jobdir(name), "evals", "step_*", "RUNNING")))

    def own(self, g):
        return self.q["gpus"].get(g, []) or self.q["gpus"].get(str(g), []) or []

    def spill(self, g):
        m = self.q.get("spill") or {}
        return m.get(g, []) or m.get(str(g), []) or []

    def trained(self, name):
        return os.path.exists(os.path.join(self.jobdir(name), "done.json")) or self.status(name) in TERMINAL

    def spill_ready(self, g):
        """A slot takes spill jobs once every job in its own list has finished training (user 2026-11-09: a GPU must not sit idle while queued work waits elsewhere)."""
        return all(self.trained(n) for n in self.own(g))

    def claimed_elsewhere(self, name, g):
        return any(k != g and name in self.spill(k) and self.spill_ready(k) for k in self.slots)

    def ready(self, name):
        """A job with `after: X` waits until X has finished training."""
        dep = (self.q["jobs"].get(name) or {}).get("after")
        return not dep or self.trained(dep)

    def current_job(self, g):
        for name in self.own(g):
            if self.ready(name) and not self.trained(name) and not self.claimed_elsewhere(name, g):
                return name
        if self.spill_ready(g):
            for name in self.spill(g):
                if self.ready(name) and not self.trained(name):
                    return name
        return None

    def eval_job(self, g):
        for name in list(self.own(g)) + list(self.spill(g)):
            if self.status(name) in TERMINAL or not self.pending(name):
                continue
            running = [sl.eval for sl in self.slots.values() if sl.eval and sl.eval["job"] == name]
            if running:
                if not self.q["jobs"][name].get("parallel_eval", False):
                    continue
                if any(job["gpu"] == self._gpu(g) and not self.eval_is_scoring(job) for job in running):
                    continue
            return name
        return None

    def eval_is_scoring(self, job):
        # Official inference writes this marker just before CPU scoring. Ignore
        # markers from earlier attempts that still need to train a new reader.
        marker = os.path.join(self.jobdir(job["job"]), "evals", f"step_{job['step']:08d}", "official", "infer_ok.json")
        try:
            return os.stat(marker).st_mtime >= job["started"]
        except FileNotFoundError:
            return False

    def pause_flags(self, g):
        flags = []
        for name in (f"PAUSE_gpu{g}", "PAUSE_all"):
            try:
                stat = os.stat(os.path.join(self.ctl, name))
            except FileNotFoundError:
                continue
            flags.append([name, stat.st_ino, stat.st_mtime_ns, stat.st_size])
        return flags

    def paused(self, g):
        flags = self.pause_flags(g)
        return bool(flags) and flags != self.released_pauses.get(g)

    def release_idle_pauses(self, state, own, now):
        for gpu in state["gpus"]:
            g = gpu["index"]
            flags = self.pause_flags(g)
            if not flags or flags == self.released_pauses.get(g):
                self.pause_idle.pop(g, None)
                continue
            info = GP.describe(g, state, int(self.sc["mem_idle_mib"]), int(self.sc["util_idle_pct"]), own)
            idle = (
                not info["own_pids"] and not info["ext_pids"]
                and info["mem_used"] <= int(self.sc["mem_idle_mib"])
                and info["util"] <= int(self.sc["util_idle_pct"])
            )
            if not idle:
                self.pause_idle.pop(g, None)
                continue
            previous_flags, since = self.pause_idle.get(g, (flags, now))
            if previous_flags != flags:
                since = now
            self.pause_idle[g] = (flags, since)
            if now - since >= float(self.sc["pause_auto_release_idle_s"]):
                self.released_pauses[g] = flags
                self.pause_idle.pop(g, None)
                for slot in self.slots.values():
                    if slot.gpu == g:
                        slot.idle_since = now - float(self.sc["idle_stable_s"])
                self._save()
                self.event("pause_auto_released", gpu=g, idle_s=round(now - since), flags=[flag[0] for flag in self.released_pauses[g]])

    def reserved(self, g, now):
        out = None
        for d in (self.ctl, self.crop_ctl):
            f = os.path.join(d, f"reserve_gpu_{g}.json")
            if os.path.exists(f):
                r = json.load(open(f))
                if r.get("until") and r["until"] < now:
                    if d == self.ctl:
                        os.remove(f)
                    continue
                out = r
        return out

    def _sids(self, state=None):
        out = {s for sl in self.slots.values() for j in (sl.train, sl.eval) if j for s in j["sids"]}
        for a in (state or {}).get("apps", []):                  # the crop-aware run counts as ours, so a job may start next to it; every other process stays external
            try:
                c = open(f"/proc/{a['pid']}/cmdline").read().replace("\0", " ")
            except OSError:
                continue
            if any(p in c for p in self.sc["coexist_patterns"]):
                sid = GP.proc_sid(a["pid"])
                if sid is not None:
                    out.add(sid)
        return out

    def _save(self):
        rec = {}
        for g, sl in self.slots.items():
            rec[str(g)] = dict(train={k: v for k, v in (sl.train or {}).items() if k != "popen"} or None, eval={k: v for k, v in (sl.eval or {}).items() if k != "popen"} or None, released_pause=self.released_pauses.get(sl.gpu))
        CK.atomic_json(rec, os.path.join(self.ctl, "active.json"))

    def adopt(self):
        f = os.path.join(self.ctl, "active.json")
        if os.path.exists(f):
            rec = json.load(open(f))
            for g, r in rec.items():
                g = int(g) if str(g).isdigit() else g
                if g not in self.slots:
                    continue
                if r.get("released_pause"):
                    self.released_pauses[self.slots[g].gpu] = r["released_pause"]
                for kind in ("train", "eval"):
                    j = r.get(kind)
                    if j and any(alive(p) and GP.proc_sid(p) in j["sids"] for p in j["pids"]):
                        setattr(self.slots[g], kind, dict(j, popen=None))
                        self.event("adopted", gpu=g, kind=kind, job=j["job"])
        for g, sl in self.slots.items():
            for n in self.q["gpus"].get(g, []) or self.q["gpus"].get(str(g), []) or []:
                for d in glob.glob(os.path.join(self.jobdir(n), "evals", "step_*")):
                    run = os.path.join(d, "RUNNING")
                    if os.path.exists(run) and not any(
                            other.eval and other.eval["job"] == n and
                            other.eval["step"] == int(os.path.basename(d).split("_")[1])
                            for other in self.slots.values()):
                        os.replace(run, os.path.join(d, "PENDING"))

    # ------------------------------------------------------------- launch
    def _gpu(self, key):
        return int(str(key).rstrip("b"))

    def _env(self, g):
        return dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1", CUDA_CACHE_DISABLE="0", CUDA_CACHE_MAXSIZE="4294967296", CUDA_CACHE_PATH=os.path.join(_ROOT, "cache", "cuda_jit"))

    def cpu_command(self, g, command):
        policy = self.q.get("cpu_affinity", {})
        cpus = policy.get(g, policy.get(str(g), []))
        if cpus:
            return ["taskset", "--cpu-list", ",".join(map(str, cpus))] + command
        return command

    def apply_cpu_affinity(self):
        policy = self.q.get("cpu_affinity", {})
        background = set(self.q.get("cpu_background_affinity", []))
        if not policy and not background:
            return
        sessions = {}
        for slot in self.slots.values():
            cpus = set(policy.get(slot.gpu, policy.get(str(slot.gpu), [])))
            if cpus:
                for job in (slot.train, slot.eval):
                    if job:
                        sessions.update({sid: cpus for sid in job["sids"]})
        changed = 0
        for name in os.listdir("/proc"):
            if not name.isdecimal():
                continue
            root = f"/proc/{name}"
            try:
                cpus = sessions.get(GP.proc_sid(int(name)))
                if cpus is None and background:
                    with open(os.path.join(root, "cmdline"), "rb") as stream:
                        args = stream.read().split(b"\0")
                    cpu_only = b"src.pacoattr.official_eval" in args or (b"src.g1up.support" in args and b"boot" in args)
                    if cpu_only and os.path.realpath(os.path.join(root, "cwd")) == _ROOT:
                        cpus = background
                if cpus is None:
                    continue
                for tid in os.listdir(os.path.join(root, "task")):
                    thread = int(tid)
                    if os.sched_getaffinity(thread) != cpus:
                        os.sched_setaffinity(thread, cpus)
                        changed += 1
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        if changed:
            self.event("cpu_affinity_applied", threads=changed)

    def _sets(self, name):
        return [x for ov in self.job_cfg(name) for x in ("--set", ov)]

    def delete_dep_checkpoints(self, name):
        """`delete_after_checkpoints: true`: remove the resume checkpoints of the finished `after` job (user 2026-11-10). evals/ (results and exported EMA encoders) is kept."""
        spec = self.q["jobs"].get(name) or {}
        dep = spec.get("after")
        if not (dep and spec.get("delete_after_checkpoints") and os.path.exists(os.path.join(self.jobdir(dep), "done.json"))):
            return
        gone = sorted(glob.glob(os.path.join(self.jobdir(dep), "checkpoints", "step_*")))
        for d in gone:
            shutil.rmtree(d, ignore_errors=True)
        if gone:
            self.event("deleted_checkpoints", job=dep, before_starting=name, removed=[os.path.basename(d) for d in gone])

    def launch_train(self, g, name):
        self.delete_dep_checkpoints(name)
        self.prep(name)
        lid = f"L{time.strftime('%Y%m%d_%H%M%S')}"
        cmd = self.test.get("train_cmd", [sys.executable, "-u", "-m", "src.bgjobs.train", "--managed", "--resume", "auto", "--bg-class", self.sc["tag"]]) + ["--config", self.base] + self._sets(name)
        cmd = self.cpu_command(self._gpu(g), cmd)
        env = dict(self._env(self._gpu(g)), BGJOB_LAUNCH_ID=lid)
        log = open(os.path.join(self.jobdir(name), "logs", f"train_{lid}.log"), "a")
        p = subprocess.Popen(cmd, cwd=_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.slots[g].train = dict(job=name, gpu=self._gpu(g), launch_id=lid, pids=[p.pid], sids=[p.pid], started=time.time(), popen=p, step=None, stop_reason=None, stop_at=None, kill_at=None)
        self._save()
        self.event("train_launch", gpu=g, job=name, launch_id=lid, pid=p.pid, step=self.step_of(name))

    def launch_eval(self, g, name, step):
        d = os.path.join(self.jobdir(name), "evals", f"step_{step:08d}")
        os.replace(os.path.join(d, "PENDING"), os.path.join(d, "RUNNING"))
        lid = f"E{step}_{time.strftime('%Y%m%d_%H%M%S')}"
        cmd = self.test.get("eval_cmd", [sys.executable, "-u", "-m", "src.crop_aware_pretrain.evals", "run", "--step", str(step), "--config", self.base]) + self._sets(name)
        cmd = self.cpu_command(self._gpu(g), cmd)
        log = open(os.path.join(d, "eval.log"), "a")
        p = subprocess.Popen(cmd, cwd=_ROOT, env=dict(self._env(self._gpu(g)), BGJOB_EVAL_STEP=str(step)), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.slots[g].eval = dict(job=name, gpu=self._gpu(g), launch_id=lid, pids=[p.pid], sids=[p.pid], started=time.time(), popen=p, step=step, stop_reason=None, stop_at=None, kill_at=None)
        self._save()
        self.event("eval_launch", gpu=g, job=name, step=step)

    def step_of(self, name):
        f = os.path.join(self.jobdir(name), "logs", "heartbeat.json")
        try:
            return int(json.load(open(f))["step"])
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------- stop / finish
    def request_stop(self, j, reason):
        if j["stop_at"] is not None:
            return
        j["stop_at"], j["stop_reason"], j["kill_at"] = time.time(), reason, time.time() + float(self.sc["stop_timeout_s"])
        for p in j["pids"]:
            try:
                os.kill(p, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.event("stop_request", job=j["job"], gpu=j["gpu"], reason=reason)

    def running(self, j):
        return any(alive(j["popen"]) if j.get("popen") else alive(p) for p in j["pids"])

    def _fail(self, name, kind, detail):
        d = self.jobdir(name)
        f = os.path.join(d, "control", "failures.json")
        s = json.load(open(f)) if os.path.exists(f) else dict(count=0, last_step=-1)
        step = self.step_of(name)
        if step - s["last_step"] >= int(self.sc["progress_reset_steps"]) and s["last_step"] >= 0:
            s["count"] = 0
        s["count"] += 1
        s["last_step"] = step
        s["last"] = dict(kind=kind, detail=detail, time=time.strftime("%F %T"))
        CK.atomic_json(s, f)
        self.event("failure", job=name, kind=kind, count=s["count"], step=step, detail=detail)
        if s["count"] >= int(self.sc["max_failures"]):
            CK.atomic_json(dict(reason=f"{s['count']} consecutive {kind} failures", last=s["last"], time=time.strftime("%F %T")), os.path.join(d, "FAILED.json"))
            self.event("job_failed", job=name)

    def finish_train(self, sl, j):
        sl.train = None
        self._save()
        name = j["job"]
        f = os.path.join(self.jobdir(name), "control", "last_train_exit.json")
        r = json.load(open(f)) if os.path.exists(f) else {}
        code = j["popen"].returncode if j.get("popen") else None
        if code == 75:
            sl.next_train_at = time.time() + 300
            self.event("train_lock_held", job=name)
            return
        clean = r.get("launch_id") == j["launch_id"] and (r.get("reason") in ("completed", "stop_after_steps") or str(r.get("reason", "")).startswith("signal_"))
        self.event("train_exit", job=name, gpu=sl.gpu, clean=clean, reason=r.get("reason"), step=r.get("step"), requested=j["stop_reason"], code=code)
        if not clean:
            self._fail(name, "train", dict(code=code, exit=r))
            sl.next_train_at = time.time() + float(self.sc["failure_backoff_s"])

    def finish_eval(self, sl, j):
        sl.eval = None
        self._save()
        name, step = j["job"], j["step"]
        d = os.path.join(self.jobdir(name), "evals", f"step_{step:08d}")
        run = os.path.join(d, "RUNNING")
        code = j["popen"].returncode if j.get("popen") else None
        ok = code == 0 or (code is None and os.path.exists(os.path.join(d, "official", "DONE.json")))
        if ok:
            os.replace(run, os.path.join(d, "DONE"))
            self.event("eval_done", job=name, step=step)
            self.decide(name)
        elif j["stop_reason"]:
            os.replace(run, os.path.join(d, "PENDING"))
            self.event("eval_preempted", job=name, step=step, reason=j["stop_reason"])
        else:
            a = os.path.join(d, "attempts")
            n = int(open(a).read()) + 1 if os.path.exists(a) else 1
            open(a, "w").write(str(n))
            if n >= 2:
                os.replace(run, os.path.join(d, "FAILED"))
                self._fail(name, "eval", dict(step=step, code=code))
            else:
                os.replace(run, os.path.join(d, "PENDING"))
            self.event("eval_retry", job=name, step=step, attempts=n, code=code)

    def overall(self, root):
        f = os.path.join(root, "DONE.json")
        if not os.path.exists(f):
            return None
        s = json.load(open(f))
        return (s["val_obj"] + s["val_part"]) / 2

    def decide(self, name):
        """Stop rule on official Val Overall vs Original DINOv3 (fixed before results): best gain up to the gate < min_best_gain, or after the gate no eval improves on the best by min_improve for patience_evals evals."""
        cfg = CF.load(self.base, self.job_cfg(name))
        rule = cfg["eval"]["stop_rule"]
        base = self.overall(cfg["eval"]["baseline_step0_official"])
        vals = {}
        for s in cfg["eval"]["steps"]:
            v = self.overall(os.path.join(self.jobdir(name), "evals", f"step_{int(s):08d}", "official"))
            if v is not None:
                vals[int(s)] = v
        if base is None or not vals:
            return
        gate = int(rule["gate_step"])
        reasons = []
        steps = sorted(vals)
        if gate in vals:
            up = [vals[s] - base for s in steps if s <= gate]
            if max(up) < float(rule["min_best_gain"]):
                reasons.append(f"best gain up to step {gate} is {max(up):+.2f} < {rule['min_best_gain']}")
            after = [s for s in steps if s > gate]
            best = max(vals[s] for s in steps if s <= gate)
            for s in after:
                if vals[s] > best + float(rule["min_improve"]):
                    best = vals[s]
                    bad = 0
                else:
                    bad = locals().get("bad", 0) + 1
            if after and locals().get("bad", 0) >= int(rule["patience_evals"]):
                reasons.append(f"{rule['patience_evals']} consecutive evals after step {gate} without an improvement of {rule['min_improve']}")
        CK.atomic_json(dict(base=base, overall=vals, delta={k: v - base for k, v in vals.items()}, reasons=reasons, time=time.strftime("%F %T")), os.path.join(self.jobdir(name), "stop_decision.json"))
        if reasons:
            if rule.get("enforce", True):
                CK.atomic_json(dict(reasons=reasons, overall=vals, base=base, time=time.strftime("%F %T")), os.path.join(self.jobdir(name), "STOPPED_NO_GAIN.json"))
                self.event("stopped_no_gain", job=name, reasons=reasons)
            else:
                self.event("no_gain_recorded_only", job=name, reasons=reasons)

    def tick(self):
        now = time.time()
        self.load_queue()
        self.apply_cpu_affinity()
        st = GP.query()
        own = self._sids(st)
        shut = os.path.exists(os.path.join(self.ctl, "SHUTDOWN")) or self.shutdown
        if not shut:
            self.release_idle_pauses(st, own, now)
        for key, sl in sorted(self.slots.items(), key=lambda kv: str(kv[0])):
            g = sl.gpu
            if not any(x["index"] == g for x in st["gpus"]):
                sl.status = "GPU_NOT_VISIBLE"
                continue
            info = GP.describe(g, st, int(self.sc["mem_idle_mib"]), int(self.sc["util_idle_pct"]), own)
            res = self.reserved(g, now)
            busy = info["busy"]
            if res and res.get("until") is None and not busy and not info["own_pids"]:   # Release stale untimed reservations on idle GPUs. Timed reservations keep their expiry.
                sl.res_idle_since = sl.res_idle_since or now
                if now - sl.res_idle_since >= float(self.sc["auto_release_idle_s"]):
                    for d in (self.ctl, self.crop_ctl):
                        f = os.path.join(d, f"reserve_gpu_{g}.json")
                        if os.path.exists(f):
                            os.remove(f)
                    self.event("reservation_auto_released", gpu=g, note=res.get("note"), idle_s=round(now - sl.res_idle_since))
                    res, sl.res_idle_since = None, None
                    sl.idle_since = now - float(self.sc["idle_stable_s"])
            else:
                sl.res_idle_since = None
            sl.ext_since = (sl.ext_since or now) if busy else None
            if busy or res:
                sl.idle_since = None
            elif sl.idle_since is None:
                sl.idle_since = now - float(self.sc["idle_stable_s"]) if self.trust else now
            # finish
            for kind in ("train", "eval"):
                j = getattr(sl, kind)
                if j and not self.running(j):
                    (self.finish_train if kind == "train" else self.finish_eval)(sl, j)
            for j in (sl.train, sl.eval):
                if j and j["kill_at"] and now > j["kill_at"]:
                    for p in j["pids"]:
                        try:
                            os.kill(p, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    self.event("force_kill_own_job", job=j["job"], gpu=g)
                    j["kill_at"] = now + 30
            name = self.current_job(key)
            yield_ext = bool(self.sc["yield_to_external"]) and busy and now - sl.ext_since >= float(self.sc["external_grace_s"])
            pause = self.paused(g)
            stop_reason = "shutdown" if shut else "pause" if pause else "reserved" if res else ("gpu_needed:" + str(info["why"])) if yield_ext else None
            for j in (sl.train, sl.eval):
                if j and j["stop_at"] is None:
                    if stop_reason:
                        self.request_stop(j, stop_reason)
                    elif (j is sl.train and j["job"] != name) or self.status(j["job"]) in TERMINAL:
                        self.request_stop(j, "job_finished")
            if sl.train and sl.train["stop_at"] is None and now - sl.train["started"] > float(self.sc["hang_timeout_s"]):
                try:
                    age = now - float(json.load(open(os.path.join(self.jobdir(sl.train["job"]), "logs", "heartbeat.json")))["time"])
                except Exception:  # noqa: BLE001
                    age = 0
                if age > float(self.sc["hang_timeout_s"]):
                    self.request_stop(sl.train, "hang")
                    self._fail(sl.train["job"], "hang", dict(age=age))
            stable = not busy and not res and sl.idle_since is not None and now - sl.idle_since >= float(self.sc["idle_stable_s"])
            if stable and not pause and not shut:
                ej = None if sl.eval else self.eval_job(key)
                if ej:
                    self.launch_eval(key, ej, self.pending(ej)[0])
                if name and not sl.train and now >= sl.next_train_at and not any(x.train and x.train["job"] == name for x in self.slots.values()):
                    self.launch_train(key, name)
        self.trust = False
        self.write_status()
        return shut and all(sl.train is None and sl.eval is None for sl in self.slots.values())

    def write_status(self):
        out = {}
        for g, sl in sorted(self.slots.items(), key=lambda kv: str(kv[0])):
            jobs = []
            for n in list(self.own(g)) + [x for x in self.spill(g) if x not in self.own(g)]:
                s = self.status(n)
                if s == "ACTIVE":
                    s = "RUNNING" if sl.train and sl.train["job"] == n else "PAUSED" if self.paused(sl.gpu) else "PENDING"
                jobs.append(dict(job=n, status=s, step=self.step_of(n), pending_evals=self.pending(n)))
            out[str(g)] = dict(gpu=sl.gpu, paused=self.paused(sl.gpu), reserved=bool(self.reserved(sl.gpu, time.time())), idle_for=(time.time() - sl.idle_since) if sl.idle_since else 0, current=self.current_job(g), train=(sl.train or {}).get("job"), eval=dict(job=sl.eval["job"], step=sl.eval["step"]) if sl.eval else None, jobs=jobs)
        CK.atomic_json(dict(time=time.strftime("%F %T"), slots=out), os.path.join(self.ctl, "status.json"))

    def run(self):
        errors = 0
        while True:
            try:
                if self.tick():
                    self.event("supervisor_exit", reason="shutdown")
                    return 0
                errors = 0
            except Exception as e:  # noqa: BLE001
                errors += 1
                self.event("tick_error", error=repr(e), consecutive=errors)
                if self.shutdown and all(sl.train is None and sl.eval is None for sl in self.slots.values()):
                    return 1
            time.sleep(float(self.sc["poll_s"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-json", default="")
    ap.add_argument("--trust-idle", action="store_true")
    a = ap.parse_args()
    test = json.load(open(a.test_json)) if a.test_json else None
    ctl = os.path.join(test["root"] if test else ROOT_DIR, "control")
    if os.path.exists(os.path.join(ctl, "SHUTDOWN")):
        os.remove(os.path.join(ctl, "SHUTDOWN"))
    kw = dict(root=test["root"], queue=test["queue"], base=test["base"], test=test) if test else {}
    sys.exit(Sup(trust_idle=a.trust_idle, **kw).run())


if __name__ == "__main__":
    main()
