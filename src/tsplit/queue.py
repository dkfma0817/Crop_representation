"""Server-side queue for the token-splitting experiment (restart safe: finished stages are skipped; flock against a second copy).

Core (Val -> frozen epoch; Test by hand, see below): F384, T768, B0 start at once; S1 / S2 / S3 after the decoder training (src.tsplit.train) is done.
Each: R2 probe (src.tsplit.train_bb, 10 epochs, PACO dev 1,000, seed 0) -> official-manifest Val inference (10 epochs) -> official Val AP
per epoch (paco_eval env, CPU). Epoch = max (Object + Part) / 2 on official Val (src.siglip_pm.select_epoch), frozen in
eval/selection_frozen.json BEFORE any Test inference; Test inference + official Test AP for the frozen epoch only.
Diagnostics (Val only, separate root eval_diag/): tap L0/1/3/6/9 (layer-L 384 tokens read directly) and rerun L0/3/6/9
(layer-L tokens x2 -> remaining frozen blocks at 48 x 48); they start after the core probes and never touch the core selection.
GPU slots: at most 3 GPU stages per GPU at once (GPU 0 and 1, both granted by the user).

  setsid nohup python -m src.tsplit.queue --run output/2026-11-11/token_split_x2_s0 &
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import yaml

PY, EPY = "/home/irteam/envs/dinov2_ord_py310/bin/python", "/home/irteam/envs/paco_eval/bin/python"
ROOT = Path("/home/irteam/Crop_representation")
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
CORE = ("F384", "T768", "B0", "S1", "S2", "S3")
DIAG = [("tapL%d" % L, "tap", L) for L in (0, 1, 3, 6, 9)] + [("rerunL%d" % L, "rerun", L) for L in (0, 3, 6, 9)]


class Queue:
    def __init__(self, run):
        self.run = run
        for d in ("logs", "state", "probes", "eval", "eval_diag", "control"):
            (run / d).mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.slots = {0: threading.Semaphore(3), 1: threading.Semaphore(3)}
        self.load = {0: 0, 1: 0}
        self.cpu = threading.Semaphore(12)

    def event(self, **kw):
        kw["time"] = time.strftime("%FT%TZ", time.gmtime())
        with self.lock, (self.run / "logs/events.jsonl").open("a") as f:
            f.write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)

    def done(self, name):
        return (self.run / "state" / f"{name}.done.json").exists()

    @staticmethod
    def running(token):
        """True while some process (e.g. an orphan of a previous queue) has `token` in its command line."""
        for d in Path("/proc").iterdir():
            if d.name.isdigit() and d.name != str(os.getpid()):
                try:
                    if token in (d / "cmdline").read_bytes().decode(errors="ignore").replace("\0", " "):
                        return True
                except OSError:
                    pass
        return False

    def adopt(self, name, token, success):
        """Stage left by a previous queue copy: wait for its process, then mark done if its output exists."""
        done = self.run / "state" / f"{name}.done.json"
        if done.exists() or token is None:
            return done.exists()
        if self.running(token):
            self.event(stage=name, status="adopting_running_process")
            while self.running(token):
                time.sleep(20)
        if success is not None and Path(success).exists():
            done.write_text(json.dumps(dict(stage=name, adopted=True, success=str(success)), indent=1))
            (self.run / "state" / f"{name}.failed.json").unlink(missing_ok=True)
            self.event(stage=name, status="adopted_done")
            return True
        return False

    def stage(self, name, module, args, gpu=None, official=False, token=None, success=None):
        done = self.run / "state" / f"{name}.done.json"
        if done.exists() or self.adopt(name, token, success):
            return
        if (self.run / "control/STOP").exists():
            raise RuntimeError("STOP file")
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                   CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        env["PYTHONPATH"] = f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}" if official else str(ROOT)
        self.event(stage=name, status="start", gpu=gpu)
        t = time.time()
        with (self.run / "logs" / f"{name}.log").open("a") as log:
            rc = subprocess.run([EPY if official else PY, "-u", "-m", module] + [str(x) for x in args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        rec = dict(stage=name, gpu=gpu, returncode=rc, seconds=round(time.time() - t))
        (done if rc == 0 else self.run / "state" / f"{name}.failed.json").write_text(json.dumps(rec, indent=1))
        self.event(**rec)
        if rc:
            raise RuntimeError(name)

    def gpu_stage(self, name, module, args, prefer=None, token=None, success=None):
        if self.done(name) or self.adopt(name, token, success):
            return
        while True:
            order = sorted((0, 1), key=lambda g: (g != prefer, self.load[g])) if prefer is not None else sorted((0, 1), key=lambda g: self.load[g])
            for g in order:
                if self.slots[g].acquire(blocking=False):
                    with self.lock:
                        self.load[g] += 1
                    try:
                        return self.stage(name, module, args, g, token=token, success=success)
                    finally:
                        with self.lock:
                            self.load[g] -= 1
                        self.slots[g].release()
            time.sleep(10)

    def official(self, tag, pred, out, split):
        if self.adopt(tag, f"--out {out}", Path(out) / "official_results.json"):
            return
        with self.cpu:
            self.stage(tag, "src.pacoattr.official_eval", ["--split", split, "--pred", pred, "--out", out], official=True)

    def condition(self, name, mode, layer=None, prefer=None, diag=False):
        probe = self.run / "probes" / name
        root = self.run / ("eval_diag" if diag else "eval")
        job = dict(name=name, backbone="dinov3", mode=mode, seed=0, epochs=10, paco_dev_images=1000, readers=[dict(name="r2", kind="r2", layers=[12])])
        if mode in ("S1", "S2", "S3"):
            job["split_ckpt"] = str(self.run / "train" / f"{mode}.pt")
        if layer is not None:
            job["layer"] = layer
        probe.mkdir(parents=True, exist_ok=True)
        self.adopt(f"probe_{name}", f"--out {probe} ", probe / "done.json")
        if not self.done(f"probe_{name}") and (probe / "r2").exists() and not (probe / "done.json").exists():
            probe.rename(probe.with_name(f"{name}.partial_{int(time.time())}"))
            probe.mkdir(parents=True)
        (probe / "job.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
        self.gpu_stage(f"probe_{name}", "src.tsplit.train_bb", ["--job", probe / "job.yaml", "--out", probe, "--exclude", EXCLUDE, "--workers", 8], prefer, token=f"--out {probe} ", success=probe / "done.json")
        self.gpu_stage(f"val_infer_{name}", "src.tsplit.official_infer", ["--root", root, "--probe", probe, "--name", name, "--split", "val"], prefer,
                      token=f"--probe {probe} --name {name} --split val", success=root / "verify" / f"{name}_val.json")
        with ThreadPoolExecutor(10) as pool:
            list(pool.map(lambda e: self.official(f"official_val_{name}_ep{e}", root / f"preds/{name}_r2/ep{e}/probs_val.npz", root / f"eval_val/{name}_r2/ep{e}", "val"), range(1, 11)))

    def safe(self, f, *a, **k):
        try:
            f(*a, **k)
        except Exception:
            self.event(status="thread_failed", traceback=traceback.format_exc()[-2000:])
            raise

    def wait_train(self):
        while not (self.run / "train/done.json").exists():
            if (self.run / "train/failed.json").exists():
                raise RuntimeError("decoder training failed")
            time.sleep(30)

    def main(self):
        threads = []

        def core(name, prefer):
            if name.startswith("S"):
                self.wait_train()
            self.condition(name, name, prefer=prefer)

        for name, prefer in (("F384", 1), ("T768", 1), ("B0", 1), ("S1", 0), ("S2", 0), ("S3", 1)):
            t = threading.Thread(target=self.safe, args=(core, name, prefer))
            t.start()
            threads.append(t)

        def diag(name, mode, layer):
            self.wait_train()
            while not all(self.done(f"official_val_{c}_ep{e}") for c in CORE for e in range(1, 11)):
                time.sleep(30)
            self.condition(name, mode, layer=layer, diag=True)

        dthreads = [threading.Thread(target=self.safe, args=(diag, n, m, L)) for n, m, L in DIAG]
        for t in dthreads:
            t.start()
        for t in threads:
            t.join()
        if not all(self.done(f"official_val_{c}_ep{e}") for c in CORE for e in range(1, 11)):
            raise RuntimeError("core official Val incomplete")
        root = self.run / "eval"
        self.stage("selection", "src.siglip_pm.select_epoch", [root])
        frozen = root / "selection_frozen.json"
        if not frozen.exists():
            frozen.write_text((root / "selection.json").read_text())
        self.event(status="selection_frozen", selection=json.loads(frozen.read_text()))
        # Official Test is NOT run by this queue (user 2026-11-11 17:25 KST): only if S2 or S3 beats B0 on official Val, then that
        # condition + F384 / F768 / B0, by hand with src.tsplit.official_infer --split test on the frozen epochs.
        self.stage("report_val", "src.tsplit.report", ["--run", self.run])
        for t in dthreads:
            t.join()
        self.stage("diag_selection", "src.siglip_pm.select_epoch", [self.run / "eval_diag"])
        (self.run / "state/report.done.json").unlink(missing_ok=True)
        self.stage("report", "src.tsplit.report", ["--run", self.run])
        (self.run / "queue_done.json").write_text(json.dumps(dict(status="done", time=time.strftime("%FT%TZ", time.gmtime())), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = ROOT / a.run
    lock = open(run / "queue.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("queue already running", flush=True)
        sys.exit(0)
    q = Queue(run)
    try:
        q.main()
    except Exception:
        (run / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc()), indent=1))
        raise


if __name__ == "__main__":
    main()
