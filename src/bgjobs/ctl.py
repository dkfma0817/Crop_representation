"""Control of the background supervisor (src/bgjobs). Writes control files or reads status; never signals anything itself.

  bash scripts/bgjobs/ctl.sh start | stop | status
  bash scripts/bgjobs/ctl.sh pause gpu0 | gpu1 | all      (checkpoints and exits; each idle GPU automatically resumes after 60 seconds)
  bash scripts/bgjobs/ctl.sh resume gpu0 | gpu1 | all
  bash scripts/bgjobs/ctl.sh reserve --gpus 0 [--hours 4] [--note "..."]  /  release --gpus 0 | all
      (a reservation without --hours is released after 75 seconds on an idle GPU; --hours specifies a timed reservation)
  bash scripts/bgjobs/ctl.sh register --name X --gpu 0 --set a.b=c [--set ...]   (appends a job to that GPU's list)
  bash scripts/bgjobs/ctl.sh retry --name X                (clears FAILED / STOPPED_NO_GAIN so the job may run again)
"""
import argparse
import fcntl
import glob
import json
import os
import subprocess
import sys
import time

import yaml

from src.bgjobs import supervisor as SU
from src.crop_aware_pretrain import ckpt as CK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("start", "stop", "status", "pause", "resume", "reserve", "release", "register", "retry"))
    ap.add_argument("target", nargs="?", default="")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--hours", type=float, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--trust-idle", action="store_true", help="start: use GPUs that are idle right now without the idle_stable_s wait")
    a = ap.parse_args()
    root = SU.ROOT_DIR
    ctl = os.path.join(root, "control")
    os.makedirs(ctl, exist_ok=True)
    if a.cmd == "start":
        args = [sys.executable, "-u", "-m", "src.bgjobs.supervisor"] + (["--trust-idle"] if a.trust_idle else [])
        log = open(os.path.join(ctl, "supervisor.log"), "a")
        p = subprocess.Popen(args, cwd=SU._ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"started detached: pid {p.pid}, log {ctl}/supervisor.log (a second copy exits at once: lock)")
    elif a.cmd == "stop":
        open(os.path.join(ctl, "SHUTDOWN"), "w").write(time.strftime("%F %T"))
        print("SHUTDOWN requested: jobs checkpoint and stop, then the supervisor exits")
    elif a.cmd in ("pause", "resume"):
        t = a.target
        assert t in ("gpu0", "gpu1", "all"), "target: gpu0 | gpu1 | all"
        f = os.path.join(ctl, f"PAUSE_{t}")
        if a.cmd == "pause":
            open(f, "w").write(time.strftime("%F %T"))
            print(f"paused {t}: the job checkpoints and exits; idle GPUs automatically resume after 60 seconds")
        else:
            for x in ([f] if t != "all" else glob.glob(os.path.join(ctl, "PAUSE_*"))):
                if os.path.exists(x):
                    os.remove(x)
            print(f"resumed {t}")
    elif a.cmd == "reserve":
        assert a.gpus, "--gpus required"
        for g in a.gpus.split(","):
            CK.atomic_json(dict(gpu=int(g), until=time.time() + a.hours * 3600 if a.hours else None, note=a.note, time=time.strftime("%F %T")), os.path.join(ctl, f"reserve_gpu_{int(g)}.json"))
        print(f"reserved GPU(s) {a.gpus}")
    elif a.cmd == "release":
        files = glob.glob(os.path.join(ctl, "reserve_gpu_*.json")) if a.gpus in ("", "all") else [os.path.join(ctl, f"reserve_gpu_{int(g)}.json") for g in a.gpus.split(",")]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        print("released")
    elif a.cmd == "register":
        assert a.name and a.gpu >= 0, "--name and --gpu required"
        with open(SU.QUEUE + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            q = yaml.safe_load(open(SU.QUEUE))
            assert a.name not in q["jobs"], "name already registered"
            q["jobs"][a.name] = {"set": a.set}
            q["gpus"].setdefault(a.gpu, []).append(a.name)
            tmp = SU.QUEUE + ".tmp"
            yaml.safe_dump(q, open(tmp, "w"), sort_keys=False)
            os.replace(tmp, SU.QUEUE)
        print(f"registered {a.name} on GPU {a.gpu}")
    elif a.cmd == "retry":
        for f in ("FAILED.json", "STOPPED_NO_GAIN.json", "control/failures.json"):
            p = os.path.join(root, a.name, f)
            if os.path.exists(p):
                os.remove(p)
        print(f"{a.name}: failure / no-gain marks cleared")
    elif a.cmd == "status":
        f = os.path.join(ctl, "status.json")
        print(json.dumps(json.load(open(f)), indent=1) if os.path.exists(f) else "no status yet (supervisor not running)")


if __name__ == "__main__":
    main()
