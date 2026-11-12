"""Run only B0/B1/B2, official Val epoch selection, then fixed-epoch official Test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from src.brixel_eval.train import dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(exist_ok=True)
    dump(root / "state/runner.json", dict(pid=os.getpid(), started_utc=time.strftime("%FT%TZ", time.gmtime()), status="running"))

    def stage(condition, mode, gpu):
        out = root / f"logs/{mode}_{condition}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        cmd = [sys.executable, "-u", "-m", "src.brixel_eval.train", "--root", str(root),
               "--condition", condition, "--stage", mode]
        start = time.time()
        with out.open("a") as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            dump(root / f"state/{mode}_{condition}.json", dict(pid=p.pid, command=cmd, gpu=gpu, status="running"))
            rc = p.wait()
        dump(root / f"state/{mode}_{condition}.json", dict(pid=p.pid, command=cmd, gpu=gpu,
                                                           returncode=rc, seconds=time.time() - start,
                                                           status="done" if rc == 0 else "failed"))
        print(json.dumps(dict(stage=mode, condition=condition, returncode=rc, seconds=time.time() - start)), flush=True)
        if rc:
            raise RuntimeError(f"{mode} {condition} failed; see {out}")

    try:
        # All three extractors remain frozen; no existing processes are managed by this runner.
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(stage, c, "train", g) for c, g in (("B0", 0), ("B1", 1), ("B2", 0))]
            for f in futures:
                f.result()
        initial = [json.loads((root / f"readers/{c}/config.json").read_text()) for c in ("B0", "B1", "B2")]
        keys = ("seed", "reader_init_seed", "reader_initial_hash", "epochs", "steps_per_epoch", "n_train_images",
                "n_train_rois", "heldout_train_image_ids", "exclude", "lr", "head_lr", "weight_decay", "betas",
                "warmup_epochs", "schedule", "grad_clip", "loss", "hflip", "roi_chunk", "encoder_chunk", "reader", "selection", "preprocessing")
        assert all(all(v[k] == initial[0][k] for k in keys) for v in initial)
        audits = [json.loads((root / f"verify/checkpoints_{c}.json").read_text()) for c in ("B0", "B1", "B2")]
        assert len({v["backbone_sha256"] for v in audits}) == 1
        dump(root / "verify/matched_conditions.json", dict(reader_settings_identical=True, keys=list(keys),
                                                           backbone_sha256=audits[0]["backbone_sha256"]))
        subprocess.run([sys.executable, "-m", "src.siglip_pm.select_epoch", str(root)], check=True)
        selection_hash = hashlib.sha256((root / "selection.json").read_bytes()).hexdigest()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(stage, c, "test", g) for c, g in (("B0", 0), ("B1", 1), ("B2", 0))]
            for f in futures:
                f.result()
        assert hashlib.sha256((root / "selection.json").read_bytes()).hexdigest() == selection_hash
        originals = json.loads((root / "verify/original_files.json").read_text())["source_sha256"]
        changed = [p for p, sha in originals.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha]
        dump(root / "verify/original_files_unchanged.json", dict(n_files=len(originals), changed=changed))
        assert not changed, changed
        subprocess.run([sys.executable, "-m", "src.brixel_eval.report", "--root", str(root),
                        "--metadata", "verify/matched_conditions.json", "--metadata", "verify/original_files_unchanged.json"], check=True)
        dump(root / "DONE.json", dict(status="done", selected_epochs=json.loads((root / "selection.json").read_text()),
                                      selection_sha256=selection_hash, completed_utc=time.strftime("%FT%TZ", time.gmtime())))
        dump(root / "state/runner.json", dict(pid=os.getpid(), status="done"))
    except BaseException:
        dump(root / "FAILED.json", dict(status="failed", traceback=traceback.format_exc()))
        dump(root / "state/runner.json", dict(pid=os.getpid(), status="failed"))
        raise


if __name__ == "__main__":
    main()
