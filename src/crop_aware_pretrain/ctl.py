"""Operate the supervisor of dinov3_crop_aware_pretrain (all commands only write control files or read state):

  python -m src.crop_aware_pretrain.ctl start | stop | status
  python -m src.crop_aware_pretrain.ctl pause | resume
  python -m src.crop_aware_pretrain.ctl reserve --gpus 2,3 [--hours 4] [--note "..."]   (keep the supervisor off these GPUs, safe stop if it is on them)
  python -m src.crop_aware_pretrain.ctl release --gpus 2,3 | all
  python -m src.crop_aware_pretrain.ctl clear-failure | override-stop-rule
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

from src.crop_aware_pretrain import ckpt as CK
from src.crop_aware_pretrain import config as CF

TMUX = "crop_aware_sup"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("start", "stop", "status", "pause", "resume", "reserve", "release", "clear-failure", "override-stop-rule"))
    ap.add_argument("--gpus", default="")
    ap.add_argument("--hours", type=float, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = CF.load(a.config, a.set)
    D = CF.dirs(cfg)
    os.makedirs(D["control"], exist_ok=True)
    ctl = D["control"]
    if a.cmd == "start":
        args = [sys.executable, "-u", "-m", "src.crop_aware_pretrain.supervisor"] + [x for v in a.set for x in ("--set", v)] + (["--config", a.config] if a.config else [])
        if shutil.which("tmux"):
            if subprocess.call(["tmux", "has-session", "-t", TMUX], stderr=subprocess.DEVNULL) == 0:
                print(f"tmux session {TMUX} already exists: not starting a second supervisor")
                return
            cmd = f"cd {CF.ROOT} && {' '.join(args)} 2>&1 | tee -a {D['logs']}/supervisor.log; echo EXIT=$?; sleep 3600"
            subprocess.check_call(["tmux", "new", "-d", "-s", TMUX, cmd])
            print(f"started in tmux {TMUX}")
        else:                                            # no tmux on this host: detached session; the supervisor lock still prevents a second copy
            log = open(os.path.join(D["logs"], "supervisor.log"), "a")
            p = subprocess.Popen(args, cwd=CF.ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(f"started detached (no tmux here): pid {p.pid}, log {D['logs']}/supervisor.log")
    elif a.cmd == "stop":
        open(os.path.join(ctl, "SHUTDOWN"), "w").write(time.strftime("%F %T"))
        print("SHUTDOWN requested: training/eval will checkpoint and stop, then the supervisor exits")
    elif a.cmd == "pause":
        open(os.path.join(ctl, "PAUSE"), "w").write(time.strftime("%F %T"))
        print("paused: training/eval stop at the next safe point and are not restarted until `ctl resume`")
    elif a.cmd == "resume":
        for f in ("PAUSE",):
            p = os.path.join(ctl, f)
            if os.path.exists(p):
                os.remove(p)
        print("resumed")
    elif a.cmd == "reserve":
        assert a.gpus, "--gpus required"
        until = time.time() + a.hours * 3600 if a.hours else None
        for g in a.gpus.split(","):
            CK.atomic_json(dict(gpu=int(g), until=until, note=a.note, time=time.strftime("%F %T")), os.path.join(ctl, f"reserve_gpu_{int(g)}.json"))
        print(f"reserved GPU(s) {a.gpus}" + (f" for {a.hours} h" if a.hours else " until released"))
    elif a.cmd == "release":
        files = glob.glob(os.path.join(ctl, "reserve_gpu_*.json")) if a.gpus in ("", "all") else [os.path.join(ctl, f"reserve_gpu_{int(g)}.json") for g in a.gpus.split(",")]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        print("released")
    elif a.cmd == "clear-failure":
        for f in ("HALT_failure.json", "failures.json"):
            p = os.path.join(ctl, f)
            if os.path.exists(p):
                os.remove(p)
        print("failure halt cleared")
    elif a.cmd == "override-stop-rule":
        open(os.path.join(ctl, "override_stop_rule"), "w").write(time.strftime("%F %T"))
        p = os.path.join(ctl, "HALT_stop_rule.json")
        if os.path.exists(p):
            os.remove(p)
        print("stop-rule halt overridden: training may continue past the decision")
    elif a.cmd == "status":
        f = os.path.join(ctl, "status.json")
        print(json.dumps(json.load(open(f)), indent=1) if os.path.exists(f) else "no status yet")


if __name__ == "__main__":
    main()
