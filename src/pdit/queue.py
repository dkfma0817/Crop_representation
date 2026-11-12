"""P1 (GPU 0) / P2 (GPU 1) queue: baseline conditioning diagnosis -> 12k training -> parity checks -> diagnosis -> PACO probes (R2 + Mean, Val only) -> head cost.
Finished stages are skipped. Never touches Test and never writes outside --run.

  python -m src.pdit.queue --run output/2026-11-09/pixeldit_cond_p1p2_s0
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import yaml

PY = "/home/irteam/envs/dinov2_ord_py310/bin/python"
ROOT = "/home/irteam/Crop_representation"
GPU = {"P1": 0, "P2": 1}
BASE = "output/2026-11-08/hcf_material_causal/r384_pixeldit_hybrid/P/checkpoints/recovery.pt"
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(ROOT) / a.run
    for d in ("logs", "state", "scripts", "diag"):
        (run / d).mkdir(parents=True, exist_ok=True)

    def stage(name, module, args, gpu):
        done = run / "state" / f"{name}.done.json"
        if done.exists():
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        started = time.time()
        print(json.dumps(dict(stage=name, status="start", gpu=gpu, time=started)), flush=True)
        with (run / "logs" / f"{name}.log").open("a") as log:
            rc = subprocess.run([PY, "-u", "-m", module] + list(map(str, args)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        record = dict(stage=name, gpu=gpu, returncode=rc, seconds=time.time() - started)
        (done if rc == 0 else run / "state" / f"{name}.failed.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(record), flush=True)
        if rc:
            raise RuntimeError(name)

    def both(fn):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(fn, GPU))

    try:
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).split()
        if not (run / "state/train_P1.done.json").exists():
            assert not apps, f"GPU not free: {apps}"
        stage("diag_baseline", "src.pdit.diag", ["--condition", "baseline", "--recovery", BASE, "--out", run / "diag/baseline.json"], 0)
        both(lambda c: stage(f"train_{c}", "src.pdit.train", ["--condition", c, "--out", run / c], GPU[c]))
        cfg = {c: json.loads((run / c / "config.json").read_text()) for c in GPU}
        drop = {"args", "condition", "memory_tokens", "conditioning", "head_params"}
        assert {k: v for k, v in cfg["P1"].items() if k not in drop} == {k: v for k, v in cfg["P2"].items() if k not in drop}, "P1/P2 configs differ"
        checks = {c: [json.loads(x) for x in (run / c / "logs/objective_checks.jsonl").read_text().splitlines()] for c in GPU}
        assert [r["batch_sha"] for r in checks["P1"]] == [r["batch_sha"] for r in checks["P2"]], "P1/P2 data streams differ"
        (run / "parity_checks.json").write_text(json.dumps(dict(config_equal_except=sorted(drop), batch_sha_equal_at_steps=[r["step"] for r in checks["P1"]]), indent=1))
        both(lambda c: stage(f"diag_{c}", "src.pdit.diag", ["--condition", c, "--recovery", run / c / "checkpoints/recovery.pt", "--out", run / f"diag/{c}.json"], GPU[c]))
        for c in GPU:
            job = dict(name=f"pdit_{c}", backbone="dinov3", mode="hcf_fused", encoder_ckpt=str(run / c / "checkpoints/ema_last.pt"), full_size=384, seed=0, epochs=10, paco_dev_images=1000, readers=[dict(name="r0", kind="r0"), dict(name="r2fused", kind="r2", layers=[12])])
            (run / "scripts" / f"probe_{c}.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
        both(lambda c: stage(f"probe_{c}", "src.pdit.train_bb", ["--job", run / "scripts" / f"probe_{c}.yaml", "--out", run / c / "probe", "--exclude", EXCLUDE, "--workers", 8], GPU[c]))
        for c in GPU:
            assert json.loads((run / c / "probe/frozen_check.json").read_text())["unchanged"]
        stage("efficiency", "src.pdit.efficiency", ["--out", run / "efficiency.json"], 0)
        (run / "queue.done.json").write_text(json.dumps(dict(status="done", time=time.strftime("%FT%TZ", time.gmtime())), indent=1))
    except Exception:
        (run / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == "__main__":
    main()
