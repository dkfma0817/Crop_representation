"""Official Test for H0/H1 and the DINOv3 baselines at the epochs fixed on official Val. Starts only after the Val queue is done; results in <run>/test_eval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback

from src.siglip_struct.stages import Command

GPU = {"H0": 0, "H1": 1, "P384": 0, "C0": 1}
MODELS = ("P384", "H0", "H1", "C0")
PAIRS = "H0-P384,H1-H0,H1-P384,C0-H1,C0-H0,C0-P384"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    root = run / "test_eval"
    for d in ("logs", "state", "results", "mean"):
        (root / d).mkdir(parents=True, exist_ok=True)
    if not (root / "budget.json").exists():
        (root / "budget.json").write_text(json.dumps(dict(deadline_unix=time.time() + 48 * 3600), indent=1))
    cmd = Command(root)

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda f: f(), jobs))

    try:
        while not ((run / "val_eval/done.json").exists() or (run / "queue_failed.json").exists()):
            time.sleep(30)
        assert (run / "val_eval/done.json").exists(), "the Val queue failed; Test not started"
        selection_path = run / "val_eval/selection.json"
        selection = json.loads(selection_path.read_text())
        (root / "fixed_epochs.json").write_text(json.dumps(selection, indent=1))

        def infer(c):
            cmd(f"test_infer_{c}", "src.hier.test_infer", ["--root", root, "--run", run, "--selection", selection_path, "--cond", c, "--workers", 8], GPU[c])

        parallel([lambda: infer("H0"), lambda: infer("H1")], 2)
        parallel([lambda: infer("P384"), lambda: infer("C0")], 2)

        def official(m, key):
            r = root if key == "r2" else root / "mean"
            e = selection[f"{m}_{key}"]["epoch"]
            cmd(f"official_{m}_{key}", "src.pacoattr.official_eval", ["--split", "test", "--pred", root / f"preds/{m}_{key}/ep{e}/probs_test.npz", "--out", r / f"eval_test/{m}_{key}/test", "--boot"], official=True)

        parallel([lambda m=m, k=k: official(m, k) for m in MODELS for k in ("r2", "linear")], 4)
        for key, r in (("r2", root), ("linear", root / "mean")):
            cmd(f"bootstrap_{key}", "src.hier.test_bootstrap", ["--root", r, "--key", key, "--models", ",".join(MODELS)], official=True)
        cmd("report", "src.hier.test_report", ["--root", root, "--mean-root", root / "mean", "--models", ",".join(MODELS), "--pairs", PAIRS, "--out", root / "results"])
        (root / "done.json").write_text(json.dumps(dict(status="done", completed_utc=time.strftime("%FT%TZ", time.gmtime()), time=time.time()), indent=1))
    except Exception:
        (root / "failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == "__main__":
    main()
