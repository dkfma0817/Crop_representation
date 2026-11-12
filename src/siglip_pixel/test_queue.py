"""Official Test for S1/S2 at the epochs fixed on official Val. Starts only after the Val queue is done; R2 Test of S1/S2 (epoch 5) and all baselines are reused."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback

from src.siglip_struct.stages import Command

GPU = {"S1": 0, "S2": 1}
CONDS = ("S1", "S2")
MODELS = ("P8", "S1", "S2", "B0", "C0")
OLD = {"P8": "P", "B0": "B0", "C0": "C0"}
PAIRS = "S1-P8,S2-P8,S2-S1,C0-P8,C0-S1,C0-S2,S1-B0,S2-B0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    run, base = Path(a.run).resolve(), Path(a.base).resolve()
    root = run / "test_final"
    mean_root = run / "mean"
    for d in ("logs", "state", "results"):
        (root / d).mkdir(parents=True, exist_ok=True)
    if not (root / "budget.json").exists():
        (root / "budget.json").write_text(json.dumps(dict(deadline_unix=time.time() + 48 * 3600), indent=1))
    cmd = Command(root)

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(lambda f: f(), jobs))

    try:
        while not ((run / "val_pilot/done.json").exists() or (run / "queue_failed.json").exists()):
            time.sleep(30)
        assert (run / "val_pilot/done.json").exists(), "the Val queue failed; Test not started"
        r2_sel, mean_sel = json.loads((run / "selection.json").read_text()), json.loads((mean_root / "selection.json").read_text())
        for c in CONDS:
            assert r2_sel[f"{c}_r2"]["epoch"] == 5 and (run / f"eval_test/{c}_r2/test/boot_pack.pkl").exists(), c
        parallel([lambda c=c: cmd(f"mean_infer_{c}_test", "src.siglip_pixel.readout", ["mean_infer", "--run", mean_root, "--job", run / f"scripts/mean_{c}.yaml", "--split", "test", "--select", mean_root / "selection.json", "--workers", 8], GPU[c]) for c in CONDS], 2)

        def official(c):
            e = mean_sel[f"{c}_linear"]["epoch"]
            cmd(f"mean_official_{c}_test", "src.pacoattr.official_eval", ["--split", "test", "--pred", mean_root / f"preds/{c}_linear/ep{e}/probs_test.npz", "--out", mean_root / f"eval_test/{c}_linear/test", "--boot"], official=True)

        parallel([lambda c=c: official(c) for c in CONDS], 2)
        for key, r, b in (("r2", run, base), ("linear", mean_root, base / "mean")):
            for new, old in OLD.items():
                target = r / f"eval_test/{new}_{key}"
                if not target.exists():
                    target.symlink_to((b / f"eval_test/{old}_{key}").resolve(), target_is_directory=True)
            cmd(f"bootstrap_{key}", "src.hier.test_bootstrap", ["--root", r, "--key", key, "--models", ",".join(MODELS)], official=True)
        cmd("report", "src.hier.test_report", ["--root", run, "--mean-root", mean_root, "--models", ",".join(MODELS), "--pairs", PAIRS, "--out", root / "results"])
        (root / "done.json").write_text(json.dumps(dict(status="done", completed_utc=time.strftime("%FT%TZ", time.gmtime()), time=time.time()), indent=1))
    except Exception:
        (root / "failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc(), time=time.time()), indent=1))
        raise


if __name__ == "__main__":
    main()
