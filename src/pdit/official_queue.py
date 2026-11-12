"""P1 / P2 re-evaluation on PACO official Val (GT box + GT category oracle attribute AP). Results go to <run>/official_val. No training, no Test.

Inference (all 10 reader epochs) -> official evaluator -> epoch = max (object + part) / 2 on official Val, ties earlier (the rule of hcf_official_val and hier) -> paired bootstrap -> ROI size -> tables.
Existing official Val outputs of the baselines are linked, never rewritten.

  python -m src.pdit.official_queue --run output/2026-11-09/pixeldit_cond_p1p2_s0
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

PY, EPY = "/home/irteam/envs/dinov2_ord_py310/bin/python", "/home/irteam/envs/paco_eval/bin/python"
ROOT = "/home/irteam/Crop_representation"
OVAL = Path(ROOT) / "output/2026-11-09/hcf_official_val"
PILOT = Path(ROOT) / "output/2026-11-08/hcf_pilot"
HIER = Path(ROOT) / "output/2026-11-09/hier_h0h1_s0/val_eval"
MODELS = ("Initial", "Cosine", "Hybrid", "P1", "P2", "C0")
NEW = ["P1_r2", "P1_linear", "P2_r2", "P2_linear", "Initial_linear", "Hybrid_linear"]
LINKED = {"Initial_r2": (OVAL, "r384_initial"), "Cosine_r2": (OVAL, "r384_cosine"), "Hybrid_r2": (OVAL, "r384_pixeldit_hybrid"), "C0_r2": (PILOT, "c0_r2"), "Cosine_linear": (HIER, "P384_linear"), "C0_linear": (HIER, "C0_linear")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(ROOT) / a.run
    root = run / "official_val"
    for d in ("logs", "state", "results", "eval_val"):
        (root / d).mkdir(parents=True, exist_ok=True)

    def stage(name, module, args, gpu=None, official=False):
        done = root / "state" / f"{name}.done.json"
        if done.exists():
            return
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        if official:
            env["PYTHONPATH"] = f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}"
        started = time.time()
        with (root / "logs" / f"{name}.log").open("a") as log:
            rc = subprocess.run([EPY if official else PY, "-u", "-m", module] + list(map(str, args)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        record = dict(stage=name, gpu=gpu, returncode=rc, seconds=time.time() - started)
        (done if rc == 0 else root / "state" / f"{name}.failed.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(record), flush=True)
        if rc:
            raise RuntimeError(name)

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda f: f(), jobs))

    def pred(name, epoch):
        if name in LINKED:
            base, old = LINKED[name]
            return base / "preds" / old / f"ep{epoch}/probs_val.npz"
        return root / "preds" / name / f"ep{epoch}/probs_val.npz"

    try:
        for name, (base, old) in LINKED.items():
            link = root / "eval_val" / name
            if not link.exists():
                assert (base / "eval_val" / old / "ep10/official_results.json").exists(), name
                link.symlink_to(base / "eval_val" / old, target_is_directory=True)
        parallel([lambda c=c, g=g: stage(f"infer_{c}", "src.pdit.official_infer", ["--root", root, "--run", run, "--cond", c], g) for c, g in (("P1", 0), ("P2", 1))], 2)
        parallel([lambda c=c, g=g: stage(f"infer_{c}", "src.pdit.official_infer", ["--root", root, "--run", run, "--cond", c], g) for c, g in (("Initial", 0), ("Hybrid", 1))], 2)
        parallel([lambda n=n, e=e: stage(f"official_{n}_{e}", "src.pacoattr.official_eval", ["--split", "val", "--pred", pred(n, e), "--out", root / f"eval_val/{n}/ep{e}"], official=True) for n in NEW for e in range(1, 11)], 14)
        stage("selection", "src.siglip_pm.select_epoch", [root])
        selection = json.loads((root / "selection.json").read_text())
        names = [f"{m}_{k}" for m in MODELS for k in ("r2", "linear")]
        assert sorted(selection) == sorted(names), sorted(selection)
        parallel([lambda n=n: stage(f"boot_{n}", "src.pacoattr.official_eval", ["--split", "val", "--pred", pred(n, selection[n]["epoch"]), "--out", root / f"eval_boot/{n}/val", "--boot"], official=True) for n in names], 12)
        for key in ("r2", "linear"):
            stage(f"bootstrap_{key}", "src.pdit.official_boot", ["--root", root, "--key", key], official=True)
        parallel([lambda n=n: stage(f"size_{n}", "src.siglip_pixel.val_size", ["--pred", pred(n, selection[n]["epoch"]), "--model", n, "--out", root / f"results/roi_size_{n}.csv"], official=True) for n in names], 12)
        stage("report", "src.pdit.official_report", ["--run", run])
        (root / "done.json").write_text(json.dumps(dict(status="done", protocol="PACO official Val GT-ROI oracle attribute AP; epoch by official Val; no Test", completed_utc=time.strftime("%FT%TZ", time.gmtime())), indent=1))
    except Exception:
        (root / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == "__main__":
    main()
