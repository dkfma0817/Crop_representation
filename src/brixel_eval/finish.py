"""Release completed GPU inference workers while adopting their unchanged CPU evaluators."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def command(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")


def pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    # This conda Python was built against older libc headers; Linux still exposes pidfd_open.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)  # __NR_pidfd_open on this Linux x86_64 host
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open")
    return fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--runner", type=int, required=True)
    ap.add_argument("--evaluators", nargs=3, required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    selection = json.loads((root / "selection.json").read_text())
    selection_sha = hashlib.sha256((root / "selection.json").read_bytes()).hexdigest()
    cpus, gpu_pids, predictions = {}, {}, {}
    for item in args.evaluators:
        name, raw = item.split("=")
        pid = int(raw)
        assert name in ("B0", "B1", "B2")
        state = json.loads((root / f"state/test_{name}.json").read_text())
        gpu_pids[name] = state["pid"]
        gpu_cmd = command(state["pid"])
        assert b"src.brixel_eval.train" in gpu_cmd and b"test" in gpu_cmd and str(root).encode() in gpu_cmd
        cpu_cmd = command(pid)
        assert b"src.pacoattr.official_eval" in cpu_cmd and b"test" in cpu_cmd
        assert str(root / f"eval_test/{name}").encode() in cpu_cmd
        cpus[pidfd(pid)] = (name, pid)
        epoch = selection[name]["epoch"]
        assert (root / f"verify/infer_{name}_test_ep{epoch}.json").is_file()
        pred = root / f"preds/{name}/ep{epoch}/probs_test.npz"
        predictions[name] = hashlib.sha256(pred.read_bytes()).hexdigest()
    assert b"src.brixel_eval.run" in command(args.runner)
    gpu_fds = [pidfd(pid) for pid in gpu_pids.values()]
    handoff = dict(status="cpu_evaluation_running", cpu_evaluators=dict(cpus.values()),
                   released_gpu_workers=gpu_pids, previous_runner=args.runner, current_pid=os.getpid(),
                   selection_sha256=selection_sha, prediction_sha256=predictions,
                   reason="User requested original long-running DINOv3 A/B training to resume while official PACO evaluates on CPU",
                   utc=time.strftime("%FT%TZ", time.gmtime()))
    write(root / "state/cpu_handoff.json", handoff)
    # Freeze the old controller so GPU-worker exits cannot interrupt the adopted CPU evaluators.
    os.kill(args.runner, signal.SIGSTOP)
    for pid in gpu_pids.values():
        os.kill(pid, signal.SIGTERM)
    while gpu_fds:
        ready, _, _ = select.select(gpu_fds, [], [])
        for fd in ready:
            gpu_fds.remove(fd)
            os.close(fd)
    os.kill(args.runner, signal.SIGKILL)
    write(root / "state/runner.json", dict(status="cpu_evaluation_running", pid=os.getpid(), gpu_released=True))
    print(json.dumps(dict(event="gpu_released", cpu_evaluators=dict(cpus.values()))), flush=True)
    # pidfd readiness is process completion notification: no sleeps, polling, or evaluator modifications.
    while cpus:
        ready, _, _ = select.select(list(cpus), [], [])
        for fd in ready:
            name, pid = cpus.pop(fd)
            os.close(fd)
            result = root / f"eval_test/{name}/official_results.json"
            assert result.is_file(), f"CPU evaluator {name}/{pid} exited without results"
            write(root / f"state/test_{name}.json", dict(status="done", cpu_evaluator_pid=pid,
                                                         gpu_worker_released_after_complete_predictions=True))
            print(json.dumps(dict(event="official_test_done", condition=name)), flush=True)
    assert hashlib.sha256((root / "selection.json").read_bytes()).hexdigest() == selection_sha
    for name, expected in predictions.items():
        epoch = selection[name]["epoch"]
        assert hashlib.sha256((root / f"preds/{name}/ep{epoch}/probs_test.npz").read_bytes()).hexdigest() == expected
    subprocess.run([sys.executable, "-m", "src.brixel_eval.verify", "--root", str(root)], check=True)
    subprocess.run([sys.executable, "-m", "src.brixel_eval.report", "--root", str(root),
                    "--metadata", "verify/implementation.json"], check=True)
    write(root / "DONE.json", dict(status="done", selected_epochs=selection, selection_sha256=selection_sha,
                                    cpu_handoff="state/cpu_handoff.json", completed_utc=time.strftime("%FT%TZ", time.gmtime())))
    write(root / "state/runner.json", dict(pid=os.getpid(), status="done"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
