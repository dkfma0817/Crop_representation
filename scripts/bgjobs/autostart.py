"""Restore the background GPU keeper and utilization log on shell startup."""
from pathlib import Path
import fcntl
import subprocess


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "output/2026-11-09/bgjobs/control"
QUERY = "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw"
GPU_UUIDS = {
    "GPU-460e460c-cd2f-a41a-51f5-a2bed0986726",
    "GPU-73e266be-dead-23b3-3fc2-78c2d593d20b",
}


def main():
    if not CONTROL.exists() or (CONTROL / "SHUTDOWN").exists():
        return
    try:
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if set(devices.split()) != GPU_UUIDS:
        return
    lock = (CONTROL / "autostart.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    with (CONTROL / "keeper.out").open("a") as log:
        subprocess.Popen(
            ["bash", str(ROOT / "scripts/bgjobs/keeper.sh")], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if QUERY.encode() in args:
            return
    with (CONTROL / "gpu_utilization_live_20261111_194253.csv").open("a") as log:
        subprocess.Popen(
            ["stdbuf", "-oL", "nvidia-smi", QUERY, "--format=csv,noheader,nounits", "-l", "2"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )


if __name__ == "__main__":
    main()
