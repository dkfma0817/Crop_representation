"""Hands GPU 0 / 1 back to the crop-aware run once the ablation queue has finished its GPU work there. Runs beside the queue; never signals it or its stages.

A GPU is finished when its Coarse-only Val inference is done and the GPU has been empty for 3 polls (20 s apart; longer than the 30 s the queue waits before starting the A0 Coarse-only probe).
Encode latency is measured here on GPU 0, before GPU 0 is released, so that it is not shared with the crop-aware run. The state file it writes makes the queue skip its own efficiency stage.
If the queue stops or a stage fails, both reservations held by this run are released so that the GPUs do not idle.

  setsid nohup python -u -m src.arch_ablate.release_watch --run output/2026-11-09/coarse_fine_ablation_s0 &
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

PY = "/home/irteam/envs/dinov2_ord_py310/bin/python"
ROOT = Path("/home/irteam/Crop_representation")
CONTROL = ROOT / "output/2026-11-09/dinov3_crop_aware_pretrain/control"
GPUS = {0: "A1", 1: "A2"}


def empty(g):
    uuid = next(line.split(",")[1].strip() for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines() if line.split(",")[0].strip() == str(g))
    procs = [line for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True).splitlines() if line.split(",")[0].strip() == uuid]
    used = int(subprocess.check_output(["nvidia-smi", f"--id={g}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
    return not procs and used < 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = ROOT / a.run
    state = run / "state"
    env = dict(os.environ, PYTHONPATH=str(ROOT))

    def log(**kw):
        kw["time"] = time.strftime("%FT%TZ", time.gmtime())
        with (run / "logs/release_watch.jsonl").open("a") as f:
            f.write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)

    def release(g):
        path = CONTROL / f"reserve_gpu_{g}.json"
        if path.exists() and run.name in json.loads(path.read_text()).get("note", ""):
            subprocess.run([PY, "-m", "src.crop_aware_pretrain.ctl", "release", "--gpus", str(g)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, check=False)
            log(gpu=g, status="released")

    idle, released = {g: 0 for g in GPUS}, set()
    log(status="started")
    while len(released) < len(GPUS):
        stopped = (run / "queue_failed.json").exists() or (run / "queue_held.json").exists() or any(state.glob("*.failed.json"))
        if stopped:
            log(status="queue stopped or a stage failed; releasing")
            for g in GPUS:
                release(g)
            return
        for g, cond in GPUS.items():
            if g in released:
                continue
            ready = (state / f"val_infer_{cond}_coarse.done.json").exists()
            idle[g] = idle[g] + 1 if ready and empty(g) else 0
            if idle[g] < 3:
                continue
            if g == 0 and not (state / "efficiency.done.json").exists():
                if not all((state / f"train_{c}.done.json").exists() for c in GPUS.values()):
                    continue
                started = time.time()
                env_gpu = dict(env, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
                with (run / "logs/efficiency.log").open("a") as out:
                    rc = subprocess.run([PY, "-u", "-m", "src.arch_ablate.efficiency", "--run", str(run), "--out", str(run / "eval/results/efficiency.json")], cwd=ROOT, env=env_gpu, stdout=out, stderr=subprocess.STDOUT).returncode
                record = dict(stage="efficiency", gpu=0, returncode=rc, seconds=time.time() - started, note="run by release_watch before GPU 0 was released")
                log(**record)
                if rc == 0:
                    (state / "efficiency.done.json").write_text(json.dumps(record, indent=1))
            release(g)
            released.add(g)
        time.sleep(20)
    log(status="done")


if __name__ == "__main__":
    main()
