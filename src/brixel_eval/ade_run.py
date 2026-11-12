"""Checkpoint-pause existing background jobs, run ADE B0/B1, then restore their pause state."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from src.brixel_eval.finish import pidfd

PROJECT = Path(__file__).resolve().parents[2]
CONTROL = PROJECT / "output/2026-11-09/bgjobs/control"
BASEPY = "/home/irteam/envs/dinov2_ord_py310/bin/python"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def active():
    path = CONTROL / "active.json"
    return json.loads(path.read_text()) if path.exists() else {}


def await_paused(first):
    seen, sessions = set(), set()
    for slot in first.values():
        for kind in ("train", "eval"):
            job = slot.get(kind) or {}
            sessions.update(job.get("sids", []))
    # Old probe workers can outlive their stopped parents and retain inherited CUDA contexts.
    removed = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            stat = (item / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[1]) == 1 and int(stat[3]) in sessions and (item / "comm").read_text().strip() == "pt_data_worker":
                cmd = (item / "cmdline").read_bytes()
                if b"Crop_representation/output/2026-11-09/bgjobs/" in cmd:
                    os.kill(int(item.name), signal.SIGTERM)
                    removed.append(int(item.name))
        except (OSError, ProcessLookupError):
            pass
    # CPU-only official evaluator controllers can outlive GPU workers. Do not wait for them.
    while True:
        for item in Path("/proc").iterdir():
            if not item.name.isdigit():
                continue
            try:
                stat = (item / "stat").read_text().rsplit(")", 1)[1].split()
                if int(stat[1]) == 1 and int(stat[3]) in sessions and (item / "comm").read_text().strip() == "pt_data_worker":
                    os.kill(int(item.name), signal.SIGTERM)
                    removed.append(int(item.name))
            except OSError:
                pass
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        pids = sorted({int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()})
        if not pids:
            break
        descriptors = []
        for pid in pids:
            path = Path(f"/proc/{pid}/cmdline")
            if not path.exists():
                continue  # NVML can briefly retain an exited process; no GPU computation remains.
            cmd = path.read_bytes()
            if not cmd:
                continue  # Exited/zombie process, possibly still visible in NVML.
            if b"/bgjobs/" not in cmd and b"src.bgjobs" not in cmd:
                raise RuntimeError(f"GPU {pid} is not a controlled background job; not stopping it")
            seen.add(pid)
            try:
                descriptors.append(pidfd(pid))
            except ProcessLookupError:
                pass
        if not descriptors:
            break
        while descriptors:
            ready, _, _ = select.select(descriptors, [], [])
            for fd in ready:
                descriptors.remove(fd)
                os.close(fd)
    return dict(stopped_gpu_pids=sorted(seen), removed_orphan_workers=removed, cpu_only_evaluators_not_waited=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.out).resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    flag = CONTROL / "PAUSE_all"
    owned, payload, child = not flag.exists(), None, None

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)

    def stage(name, out, iters, val_limit=0):
        nonlocal child
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        cmd = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", "-m", "src.brixel_eval.ade",
               "--out", str(out), "--iters", str(iters), "--workers", str(args.workers), "--resume"]
        if val_limit:
            cmd += ["--val-limit", str(val_limit)]
        with (root / "logs" / f"{name}.log").open("a") as log:
            child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / "driver_state.json", dict(status="running", phase=name, pid=os.getpid(), worker_launcher_pid=child.pid, command=cmd))
            rc = child.wait()
        child = None
        print(json.dumps(dict(event=name, returncode=rc)), flush=True)
        if rc:
            raise RuntimeError(f"{name} failed: see logs/{name}.log")

    try:
        previous = active()
        write(root / "background_jobs_before.json", previous)
        if owned:
            subprocess.run([BASEPY, "-m", "src.bgjobs.ctl", "pause", "all"], cwd=PROJECT, check=True)
            payload = flag.read_bytes()
        write(root / "driver_state.json", dict(status="running", phase="await_background_checkpoints", pid=os.getpid()))
        write(root / "background_checkpoint_pause.json", await_paused(previous))
        if not (root / "preflight_ddp/done.json").exists():
            stage("preflight_ddp", root / "preflight_ddp", 8, 4)
        smoke = json.loads((root / "preflight_ddp/verification.json").read_text())
        assert smoke["status"] == "passed" and all(smoke["unchanged"].values())
        if not (root / "done.json").exists():
            stage("train", root, 40000)
        result = json.loads((root / "done.json").read_text())
        assert result["iterations"] == 40000 and result["unchanged"]
        write(root / "driver_state.json", dict(status="done", pid=os.getpid(), results=result["results"]))
        print(json.dumps(dict(event="complete", results=result["results"])), flush=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        # Remove only the flag we created; preserve later user pause/reservation requests.
        resumed = bool(owned and payload is not None and flag.exists() and flag.read_bytes() == payload)
        if resumed:
            flag.unlink()
        write(root / "background_restore.json", dict(task_pause_released=resumed,
                                                     preserved_preexisting_pause=not owned,
                                                     completed_utc=time.strftime("%FT%TZ", time.gmtime())))
        print(json.dumps(dict(event="background_pause_restored", released_own_pause=resumed)), flush=True)


if __name__ == "__main__":
    main()
