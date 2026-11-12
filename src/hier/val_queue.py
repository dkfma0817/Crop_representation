"""H0/H1 on PACO official Val only (GT box + GT category oracle attribute AP). Never starts a Test stage; project M2 is not reported.

Waits for the two trainings under the paused queue and stops that queue, then: 10-epoch readers -> Val inference (all epochs) -> official Val ->
epoch selection (max (object + part)/2, ties earlier) -> ablations -> paired bootstrap -> ROI sizes -> efficiency (after the S1/S2 Val queue) -> tables.
Results go to <run>/val_eval.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import yaml

from src.siglip_pm.supervise import EPY, ROOT
from src.siglip_struct.stages import Command

GPU = {"H0": 0, "H1": 1}
CONDS = ["H0", "H1"]
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
S1S2 = Path("output/2026-11-09/siglip2_pixeldit_s0")
OVAL = Path("output/2026-11-09/hcf_official_val")
PILOT = Path("output/2026-11-08/hcf_pilot")
OFFICIAL_WORKERS = 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--paused-queue", type=int, required=True)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    root = run / "val_eval"
    for d in ("logs", "state", "results", "scripts"):
        (root / d).mkdir(parents=True, exist_ok=True)
    if not (root / "budget.json").exists():
        (root / "budget.json").write_text(json.dumps(dict(deadline_unix=time.time() + 48 * 3600), indent=1))
    cmd = Command(root)

    def official(name, module, args):
        """The Command stage contract with more concurrent CPU jobs than its four slots."""
        done = root / "state" / f"{name}.done.json"
        if done.exists():
            return
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONPATH=f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}")
        started = time.time()
        with (root / "logs" / f"{name}.log").open("a") as log:
            rc = subprocess.run([EPY, "-u", "-m", module] + list(map(str, args)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (done if rc == 0 else root / "state" / f"{name}.failed.json").write_text(json.dumps(dict(stage=name, returncode=rc, seconds=time.time() - started), indent=1))
        if rc:
            raise RuntimeError(name)

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda f: f(), jobs))

    def wait(check, what):
        while not check():
            print(json.dumps(dict(event="waiting", what=what, time=time.time())), flush=True)
            time.sleep(30)

    try:
        wait(lambda: all((run / c / "done.json").exists() or (run / c / "failed.json").exists() for c in CONDS), "H0/H1 training")
        assert all((run / c / "done.json").exists() for c in CONDS), "training failed"
        marker = root / "old_queue_stopped.json"
        if not marker.exists():
            path = Path(f"/proc/{a.paused_queue}/cmdline")
            cmdline = path.read_bytes().decode().replace("\0", " ") if path.exists() else ""
            if "src.hier.queue" in cmdline:
                os.kill(a.paused_queue, signal.SIGKILL)
            marker.write_text(json.dumps(dict(pid=a.paused_queue, had_cmdline=cmdline, stopped_unix=time.time(), reason="Pilot protocol changed to official Val"), indent=1))
        cfg = {c: json.loads((run / c / "config.json").read_text()) for c in CONDS}
        drop = {"args", "condition"}
        assert {k: v for k, v in cfg["H0"].items() if k not in drop} == {k: v for k, v in cfg["H1"].items() if k not in drop}
        checks = {c: [json.loads(x) for x in (run / c / "logs/grad_checks.jsonl").read_text().splitlines()] for c in CONDS}
        assert all(p["batch_sha"] == q["batch_sha"] for p, q in zip(checks["H0"], checks["H1"])) and len(checks["H0"]) == len(checks["H1"]), "H0/H1 data streams differ"
        (root / "parity_checks.json").write_text(json.dumps(dict(config_equal_except=sorted(drop), batch_sha_equal_at_steps=[r["step"] for r in checks["H0"]]), indent=1))
        for c in CONDS:
            job = dict(name=f"hier_{c}", backbone="dinov3", mode="hcf_fused", detail=c, encoder_ckpt=str(run / c / "checkpoints/ema_last.pt"), full_size=384, seed=0, epochs=10, paco_dev_images=1000,
                       readers=[dict(name="r0", kind="r0"), dict(name="r2fused", kind="r2", layers=[12])])
            (root / "scripts" / f"probe_{c}.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
        parallel([lambda c=c: cmd(f"probe_{c}", "src.hier.train_bb", ["--job", root / "scripts" / f"probe_{c}.yaml", "--out", run / f"probe_{c}", "--exclude", EXCLUDE, "--workers", 8], GPU[c]) for c in CONDS], 2)
        for c in CONDS:
            assert json.loads((run / f"probe_{c}" / "frozen_check.json").read_text())["unchanged"]

        def infer(c, gpu):
            cmd(f"val_infer_{c}", "src.hier.val_infer", ["--root", root, "--run", run, "--cond", c, "--workers", 8], gpu)

        parallel([lambda: infer("H0", 0), lambda: infer("H1", 1)], 2)
        parallel([lambda: infer("P384", 0), lambda: infer("C0", 1)], 2)
        names = [f"{c}_{k}" for c in CONDS for k in ("r2", "linear")] + ["P384_linear", "C0_linear"]
        parallel([lambda n=n, e=e: official(f"official_{n}_{e}", "src.pacoattr.official_eval", ["--split", "val", "--pred", root / f"preds/{n}/ep{e}/probs_val.npz", "--out", root / f"eval_val/{n}/ep{e}"]) for n in names for e in range(1, 11)], OFFICIAL_WORKERS)
        wait(lambda: all((OVAL / f"eval_val/r384_cosine/ep{e}/official_results.json").exists() for e in range(1, 11)), "P384 R2 official Val")
        for name, target in (("P384_r2", OVAL / "eval_val/r384_cosine"), ("C0_r2", PILOT / "eval_val/c0_r2")):
            link = root / "eval_val" / name
            if not link.exists():
                link.symlink_to(target.resolve(), target_is_directory=True)
        cmd("selection", "src.siglip_pm.select_epoch", [root])
        selection = json.loads((root / "selection.json").read_text())
        parallel([lambda c=c: cmd(f"ablate_{c}", "src.hier.val_ablate", ["--root", root, "--run", run, "--cond", c, "--workers", 8], GPU[c]) for c in CONDS], 2)

        def pred(name):
            e = selection[name]["epoch"]
            return {"P384_r2": OVAL / f"preds/r384_cosine/ep{e}/probs_val.npz", "C0_r2": PILOT / f"preds/c0_r2/ep{e}/probs_val.npz"}.get(name, root / f"preds/{name}/ep{e}/probs_val.npz")

        keys = ("r2", "linear")
        models = ("P384", "H0", "H1", "C0")
        boots = [(f"{m}_{k}", pred(f"{m}_{k}")) for k in keys for m in models] + [(f"{c}_{k}_{mode}", root / f"ablation/{c}/{mode}_{k}/probs_val.npz") for c in CONDS for k in keys for mode in ("zero", "shuffle")]
        parallel([lambda n=n, p=p: official(f"boot_{n}", "src.pacoattr.official_eval", ["--split", "val", "--pred", p, "--out", root / f"eval_boot/{n}/val", "--boot"]) for n, p in boots], OFFICIAL_WORKERS)
        for key in keys:
            official(f"bootstrap_{key}", "src.hier.val_bootstrap", ["--root", root, "--key", key])
        parallel([lambda k=k, m=m: official(f"size_{k}_{m}", "src.siglip_pixel.val_size", ["--pred", pred(f"{m}_{k}"), "--model", m, "--out", root / f"results/roi_size_{k}_{m}.csv"]) for k in keys for m in models], OFFICIAL_WORKERS)
        wait(lambda: (S1S2 / "done.json").exists() or (S1S2 / "queue_failed.json").exists(), "S1/S2 Val queue (for a clean latency measurement)")
        cmd("efficiency", "src.hier.efficiency", ["--run", run], 0)
        cmd("report", "src.hier.val_report", ["--run", run])
        result = dict(status="done", protocol="PACO official Val GT-ROI oracle attribute AP; epoch by official Val; no Test", completed_utc=time.strftime("%FT%TZ", time.gmtime()), time=time.time())
        (root / "done.json").write_text(json.dumps(result, indent=1))
        (run / "done.json").write_text(json.dumps(result, indent=1))
    except Exception:
        (run / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == "__main__":
    main()
