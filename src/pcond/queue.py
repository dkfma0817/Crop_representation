"""P1/P2 queue on GPU 0/1; completed stages are skipped. PACO Val only: this queue has no Test stage.

  python -m src.pcond.queue --run <run> --stage checks   geometry, head cost, baseline conditioning diagnosis, smokes (B must reproduce the logged first step), probe smoke
  python -m src.pcond.queue --run <run> --stage full     P1 (GPU 0) / P2 (GPU 1) training, diagnosis, R2 + Mean readers, tables
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import yaml

from src.siglip_struct.stages import Command

GPU = {"P1": 0, "P2": 1}
CONDS = ["P1", "P2"]
CAUSAL = Path("output/2026-11-08/hcf_material_causal")
BASE = CAUSAL / "r384_pixeldit_hybrid"
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
INITIAL_SHA = "ce57bb33c0382a1a5765385f94d00b6c1b26dfb9cac5d2f0c765f8be7e87d3aa"


def gpu_busy():
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines()
    ids = {x.split(",")[1].strip() for x in lines if x.split(",")[0].strip() in ("0", "1")}
    return [x for x in subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True).splitlines() if x.split(",")[0].strip() in ids]


def foreign_queues():
    """Other pilots' queues still using GPU 0/1 (their latency measurements must not overlap these trainings; GPU 2/3 work is not checked or touched)."""
    out = subprocess.run(["pgrep", "-af", "src.(hier.(val_|test_)?queue|siglip_pixel.(val_|test_)?queue)"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "pgrep" not in line and "src.pcond" not in line]


def wait_for_gpus():
    while foreign_queues() or gpu_busy():
        print(json.dumps(dict(event="waiting for GPU 0/1", queues=foreign_queues(), busy=gpu_busy(), time=time.time())), flush=True)
        time.sleep(30)


def parallel(jobs, n):
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(lambda f: f(), jobs))


def probe_job(path, name, ckpt, dev_images=1000):
    path.write_text(yaml.safe_dump(dict(name=name, backbone="dinov3", mode="hcf_fused", encoder_ckpt=str(ckpt), full_size=384, seed=0, epochs=10, paco_dev_images=dev_images, readers=[dict(name="r0", kind="r0"), dict(name="r2fused", kind="r2", layers=[12])]), sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", required=True, choices=("checks", "full"))
    a = ap.parse_args()
    run = Path(a.run).resolve()
    for d in ("logs", "state", "scripts", "results", "figures", "diag"):
        (run / d).mkdir(parents=True, exist_ok=True)
    if not (run / "budget.json").exists():
        (run / "budget.json").write_text(json.dumps(dict(deadline_unix=time.time() + 48 * 3600), indent=1))
    snapshot = run / "source_snapshot"
    if not snapshot.exists():
        shutil.copytree("src/pcond", snapshot / "src/pcond", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy("src/siglip_pixel/crop_objectives.py", snapshot / "crop_objectives.py")
    cmd = Command(run)
    initial = (CAUSAL / "initial.pt").resolve()

    def train_args(c):
        return ["--condition", c, "--initial", initial, "--out", run / c, "--steps", 12000, "--workers", 8]

    if a.stage == "checks":
        cmd("geometry", "src.pcond.geometry_check", ["--out", run / "geometry"])
        wait_for_gpus()
        cmd("efficiency", "src.pcond.efficiency", ["--out", run / "results/head_efficiency.json"], 0)
        cmd("diag_B", "src.pcond.diag", ["--condition", "B", "--ckpt", (BASE / "P/checkpoints/recovery.pt").resolve(), "--out", run / "diag/B.json"], 0)
        parallel([lambda: (cmd("smoke_B", "src.pcond.train", ["--condition", "B", "--initial", initial, "--out", run / "smoke_B", "--smoke"], 0), cmd("smoke_P1", "src.pcond.train", train_args("P1") + ["--smoke"], 0)),
                  lambda: cmd("smoke_P2", "src.pcond.train", train_args("P2") + ["--smoke"], 1)], 2)
        s = {c: json.loads((run / c / "smoke.json").read_text()) for c in CONDS}
        s["B"] = json.loads((run / "smoke_B/smoke.json").read_text())
        assert all(v["status"] == "passed" for v in s.values())
        assert s["P1"]["steps"][0]["batch_sha"] == s["P2"]["steps"][0]["batch_sha"] == s["B"]["steps"][0]["batch_sha"]
        drop = {"args", "condition", "memory_tokens", "head_params", "heads_sha256", "source_sha256"}
        assert {k: v for k, v in s["P1"]["cfg"].items() if k not in drop} == {k: v for k, v in s["P2"]["cfg"].items() if k not in drop}
        assert s["P1"]["cfg"]["initial_sha256"] == INITIAL_SHA, s["P1"]["cfg"]["initial_sha256"]
        (run / "smoke_parity.json").write_text(json.dumps(dict(batch_sha_equal=True, features_rounded_sha={c: s[c]["steps"][0]["features_rounded_sha"] for c in s}, initial_sha256=INITIAL_SHA, B_reproduces_baseline_step1=s["B"]["baseline_step1"], head_params={c: s[c]["head_params"] for c in s}), indent=1))
        probe_job(run / "scripts/probe_smoke.yaml", "pcond_smoke", (BASE / "P/checkpoints/ema_last.pt").resolve(), dev_images=8)  # --limit 64 leaves too few images for the real 1000-image dev split
        cmd("smoke_probe", "src.pcond.train_bb", ["--job", run / "scripts/probe_smoke.yaml", "--out", run / "smoke_probe", "--exclude", EXCLUDE, "--workers", 4, "--limit", 64, "--epochs", 1, "--max_steps", 5], 0)
        return
    try:
        wait_for_gpus()
        (run / "GPU_start_gate.json").write_text(json.dumps(dict(passed=True, time=time.time()), indent=1))
        parallel([lambda c=c: cmd(f"train_{c}", "src.pcond.train", train_args(c), GPU[c]) for c in CONDS], 2)
        cfg = {c: json.loads((run / c / "config.json").read_text()) for c in CONDS}
        drop = {"args", "condition", "memory_tokens", "head_params", "heads_sha256", "source_sha256"}
        assert {k: v for k, v in cfg["P1"].items() if k not in drop} == {k: v for k, v in cfg["P2"].items() if k not in drop}
        checks = {c: [json.loads(x) for x in (run / c / "logs/objective_checks.jsonl").read_text().splitlines()] for c in CONDS}
        assert all(p["batch_sha"] == q["batch_sha"] and p["features_rounded_sha"] == q["features_rounded_sha"] for p, q in zip(checks["P1"][:1], checks["P2"][:1])), "initial ROI features differ"
        assert [p["batch_sha"] for p in checks["P1"]] == [q["batch_sha"] for q in checks["P2"]], "data streams differ"
        (run / "parity_checks.json").write_text(json.dumps(dict(config_equal_except=sorted(drop), batch_sha_equal_at_steps=[r["step"] for r in checks["P1"]], initial_features_equal_step1=True), indent=1))
        parallel([lambda c=c: cmd(f"diag_{c}", "src.pcond.diag", ["--condition", c, "--ckpt", run / c / "checkpoints/recovery.pt", "--out", run / f"diag/{c}.json"], GPU[c]) for c in CONDS], 2)
        for c in CONDS:
            probe_job(run / "scripts" / f"probe_{c}.yaml", f"pcond_{c}", run / c / "checkpoints/ema_last.pt")
        parallel([lambda c=c: cmd(f"probe_{c}", "src.pcond.train_bb", ["--job", run / "scripts" / f"probe_{c}.yaml", "--out", run / f"probe_{c}", "--exclude", EXCLUDE, "--workers", 8], GPU[c]) for c in CONDS], 2)
        for c in CONDS:
            assert json.loads((run / f"probe_{c}" / "frozen_check.json").read_text())["unchanged"]
        cmd("report", "src.pcond.report", ["--run", run])
        (run / "done.json").write_text(json.dumps(dict(status="done", completed_utc=time.strftime("%FT%TZ", time.gmtime()), time=time.time()), indent=1))
    except Exception:
        (run / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == "__main__":
    main()
