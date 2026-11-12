"""H0/H1 queue on GPU 0/1: trains, runs the 10-epoch readers, ablations, efficiency (after the S1/S2 queue ends) and tables. Val only; no Test.

  python -m src.hier.queue --run <run> --stage smoke|full
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import traceback

import yaml

from src.siglip_struct.stages import Command

GPU = {"H0": 0, "H1": 1}
CONDS = ["H0", "H1"]
S1S2 = Path("output/2026-11-09/siglip2_pixeldit_s0")
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"


def gpu_busy():
    uuids = {x.split(",")[1].strip() for x in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines() if x.split(",")[0].strip() in ("0", "1")}
    return [x for x in subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True).splitlines() if x.split(",")[0].strip() in uuids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", required=True, choices=("smoke", "full"))
    a = ap.parse_args()
    run = Path(a.run).resolve()
    for d in ("logs", "state", "scripts", "results", "figures"):
        (run / d).mkdir(parents=True, exist_ok=True)
    if not (run / "budget.json").exists():
        (run / "budget.json").write_text(json.dumps(dict(deadline_unix=time.time() + 48 * 3600), indent=1))
    cmd = Command(run)

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda f: f(), jobs))

    def train_args(c):
        return ["--condition", c, "--initial", run / "initial.pt", "--out", run / c, "--steps", 12000, "--workers", 8]

    if a.stage == "smoke":
        cmd("input_audit", "src.hier.checks", ["--run", run])
        parallel([lambda c=c: cmd(f"smoke_{c}", "src.hier.train", train_args(c) + ["--smoke"], GPU[c]) for c in CONDS], 2)
        s = {c: json.loads((run / c / "smoke.json").read_text()) for c in CONDS}
        assert s["H0"]["status"] == s["H1"]["status"] == "passed"
        assert s["H0"]["full_sha"] == s["H1"]["full_sha"] and s["H0"]["batch_sha"] == s["H1"]["batch_sha"], "global input or ROI batch differs"
        drop = {"args", "condition", "source_sha256"}
        assert {k: v for k, v in s["H0"]["cfg"].items() if k not in drop} == {k: v for k, v in s["H1"]["cfg"].items() if k not in drop}
        (run / "smoke_parity.json").write_text(json.dumps(dict(full_sha_equal=True, batch_sha_equal=True, config_equal=True, n_params=s["H0"]["cfg"]["n_params"]), indent=1))
        return
    try:
        while gpu_busy():
            print(json.dumps(dict(event="waiting for GPU 0/1", busy=gpu_busy())), flush=True)
            time.sleep(30)
        (run / "GPU_start_gate.json").write_text(json.dumps(dict(passed=True, time=time.time()), indent=1))
        parallel([lambda c=c: cmd(f"train_{c}", "src.hier.train", train_args(c), GPU[c]) for c in CONDS], 2)
        cfg = {c: json.loads((run / c / "config.json").read_text()) for c in CONDS}
        drop = {"args", "condition"}
        assert {k: v for k, v in cfg["H0"].items() if k not in drop} == {k: v for k, v in cfg["H1"].items() if k not in drop}
        checks = {c: [json.loads(x) for x in (run / c / "logs/grad_checks.jsonl").read_text().splitlines()] for c in CONDS}
        assert all(p["batch_sha"] == q["batch_sha"] for p, q in zip(checks["H0"], checks["H1"])) and len(checks["H0"]) == len(checks["H1"]), "H0/H1 data streams differ"
        (run / "parity_checks.json").write_text(json.dumps(dict(config_equal_except=sorted(drop), batch_sha_equal_at_steps=[r["step"] for r in checks["H0"]]), indent=1))
        for c in CONDS:
            job = dict(name=f"hier_{c}", backbone="dinov3", mode="hcf_fused", detail=c, encoder_ckpt=str(run / c / "checkpoints/ema_last.pt"), full_size=384, seed=0, epochs=10, paco_dev_images=1000,
                       readers=[dict(name="r0", kind="r0"), dict(name="r2fused", kind="r2", layers=[12])])
            (run / "scripts" / f"probe_{c}.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
        parallel([lambda c=c: cmd(f"probe_{c}", "src.hier.train_bb", ["--job", run / "scripts" / f"probe_{c}.yaml", "--out", run / f"probe_{c}", "--exclude", EXCLUDE, "--workers", 8], GPU[c]) for c in CONDS], 2)
        parallel([lambda c=c: cmd(f"ablate_{c}", "src.hier.ablate", ["--run", run, "--cond", c], GPU[c]) for c in CONDS], 2)
        while not ((S1S2 / "done.json").exists() or (S1S2 / "queue_failed.json").exists()):   # latency is measured on a GPU without the S1/S2 readers
            time.sleep(30)
        cmd("efficiency", "src.hier.efficiency", ["--run", run], 0)
        cmd("report", "src.hier.report", ["--run", run])
        (run / "done.json").write_text(json.dumps(dict(status="done", time=time.time(), completed_utc=time.strftime("%FT%TZ", time.gmtime())), indent=1))
    except Exception:
        (run / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == "__main__":
    main()
