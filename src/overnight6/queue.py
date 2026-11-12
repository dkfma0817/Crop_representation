"""Six pilots: one server-side process with a shared encoder queue and a separate evaluation queue. Started once with setsid; a flock prevents a second copy; finished stages are skipped on restart and training resumes from recovery.pt.

Encoder queue (per GPU, order of preference):
  GPU 0: E2, E2L, F1, S1, E1a, E1b        GPU 1: teacher cache, E1a, E1b, S1, F1, E2L, E2
Whichever GPU is free takes the first runnable experiment not taken yet; E1a / E1b are runnable only when the Crop768 cache, its verification and lambda_spatial are done. A training process and its checkpoint are complete only when done.json exists (written atomically last).

Evaluation queue (separate tasks): R2 probe (10 epochs) + Val inference on a GPU that has no runnable encoder job, then CPU official Val, epoch frozen on Val, one Test inference, CPU official Test.
R2 runs on GPU: the encoder forward needs about 370k images, and CPU does 4 images/s on 32 threads (measured). A GPU takes an evaluation task only when no encoder job is runnable for it, so evaluation never delays encoder training.

GPU gate: bgjobs A / B / A control are never waited for (they checkpoint and yield within 60 s). The finished crop-aware run's last evaluations may share the GPU (never waited for, never signalled). Any other process that is not this queue's own is waited for and never signalled.
Control: touch <run>/control/HOLD (no new task) or <run>/control/STOP (exit after the running tasks).

  setsid nohup /home/irteam/envs/dinov2_ord_py310/bin/python -u -m src.overnight6.queue --run results/overnight_six &
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import yaml

from src.overnight6.train import CONDS

PY, EPY = "/home/irteam/envs/dinov2_ord_py310/bin/python", "/home/irteam/envs/paco_eval/bin/python"
ROOT = Path("/home/irteam/Crop_representation")
CACHE = ROOT / "output/2026-11-10/overnight_six_teacher768"
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
ORDER = {0: ["E2", "E2L", "F1", "S1", "E1a", "E1b"], 1: ["cache", "E1a", "E1b", "S1", "F1", "E2L", "E2"]}
G1 = ROOT / "output/2026-11-09/g1_upscaled_teacher_s0/eval"
ABL = ROOT / "output/2026-11-09/coarse_fine_ablation_s0/eval"
OVAL = ROOT / "output/2026-11-09/hcf_official_val"
BASE = dict(A0_r2=dict(val_dir=OVAL / "eval_val/r384_cosine", test=ABL / "fast_test/A0_r2", boot=ABL / "eval_boot/A0_r2", preds=OVAL / "preds/r384_cosine", selection=json.loads((OVAL / "selection.json").read_text())["r384_cosine"]),
            G1U_r2=dict(val_dir=G1 / "eval_val/G1U_r2", test=G1 / "fast_ap/test/G1U_r2", boot=G1 / "eval_boot/G1U_r2", preds=G1 / "preds/G1U_r2", selection=json.loads((G1 / "selection.json").read_text())["G1U_r2"]))
NAMES = list(CONDS)
SPLITS = ("val", "test")


def parallel(jobs, n):
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda f: f(), jobs))


class Queue:
    TRAIN, REPORT = "src.overnight6.train", "src.overnight6.report"
    OWN_TAGS = ("results/overnight_six", "src.overnight6", "src.pc768mse", "results/pixeldit_crop768_vs_mse", "src.jointpg", "results/joint_pixeldit_g1")

    def __init__(self, run):
        self.run = run
        self.eval = run / "eval"
        for d in (run / "logs", run / "state", run / "control", run / "configs", run / "verification", self.eval / "selection", self.eval / "eval_val", self.eval / "fast_test", self.eval / "eval_boot", self.eval / "results"):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.tasks, self.claimed, self.failed, self.complete = deque(), set(), {}, set()
        self.cpu_threads, self.status = [], {}
        (run / "configs" / "experiments.json").write_text(json.dumps(CONDS, indent=1))
        for name, b in BASE.items():
            for path, target in ((self.eval / "eval_val" / name, b["val_dir"]), (self.eval / "fast_test" / name, b["test"]), (self.eval / "eval_boot" / name, b["boot"])):
                if not path.is_symlink():
                    assert target.exists(), target
                    path.symlink_to(target, target_is_directory=True)
            (self.eval / "selection" / f"{name}.json").write_text(json.dumps(b["selection"], indent=1))
        for name in NAMES:
            if self.done(f"train_{name}"):
                self.claimed.add(name)
                self.tasks.append(("probe", name))

    def event(self, **kw):
        kw["time"] = time.strftime("%FT%TZ", time.gmtime())
        with self.lock, (self.run / "logs" / "events.jsonl").open("a") as f:
            f.write(json.dumps(kw, default=str) + "\n")
        print(json.dumps(kw, default=str), flush=True)

    def done(self, name):
        return (self.run / "state" / f"{name}.done.json").exists()

    def control(self):
        while (self.run / "control/HOLD").exists():
            time.sleep(15)
        return (self.run / "control/STOP").exists()

    def stage(self, name, module, args, gpu=None, official=False, retries=0):
        done = self.run / "state" / f"{name}.done.json"
        if done.exists():
            return
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        env["PYTHONPATH"] = f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}" if official else str(ROOT)
        for attempt in range(retries + 1):
            self.event(stage=name, status="start", gpu=gpu, attempt=attempt)
            started = time.time()
            with (self.run / "logs" / f"{name}.log").open("a") as log:
                proc = subprocess.Popen([EPY if official else PY, "-u", "-m", module] + list(map(str, args)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                (self.run / "state" / f"{name}.pid").write_text(str(proc.pid))
                rc = proc.wait()
            record = dict(stage=name, gpu=gpu, returncode=rc, pid=proc.pid, attempt=attempt, seconds=round(time.time() - started, 1))
            if rc == 0:
                done.write_text(json.dumps(record, indent=1))
                self.event(**record)
                return
            (self.run / "state" / f"{name}.failed.json").write_text(json.dumps(record, indent=1))
            self.event(**record)
        raise RuntimeError(name)

    def gpu_processes(self, g):
        uuid = next(line.split(",")[1].strip() for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines() if line.split(",")[0].strip() == str(g))
        out = {}
        for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True).splitlines():
            parts = [x.strip() for x in line.split(",")]
            if parts[0] == uuid:
                try:
                    out[parts[1]] = Path(f"/proc/{parts[1]}/cmdline").read_bytes().decode(errors="ignore").replace("\0", " ")
                except OSError:
                    out[parts[1]] = ""
        return out

    def gate(self, g):
        """Wait only for processes that are neither ours nor the background jobs (which yield on their own)."""
        announced = None
        while True:
            others = {p: c for p, c in self.gpu_processes(g).items() if c and not any(tag in c for tag in ("bgjobs", "crop_aware_pretrain_bg", "src.crop_aware_pretrain.evals", "dinov3_crop_aware_pretrain") + self.OWN_TAGS)}
            if not others:
                return
            seen = sorted(c[:200] for c in others.values())
            if seen != announced:
                self.event(gpu=g, status="waiting_for_other_process", processes=seen)
                announced = seen
            time.sleep(15)

    # ------------------------------------------------------------- tasks
    def cache_ready(self):
        return self.done("lambda_spatial")

    def next_task(self, g):
        with self.lock:
            for name in ORDER[g]:
                if name in self.claimed:
                    continue
                if name == "cache":
                    self.claimed.add("cache")
                    return ("cache", "cache")
                if CONDS[name].get("teacher768") and ("cache" in self.failed or not self.cache_ready()):
                    continue
                self.claimed.add(name)
                return ("train", name)
            return self.tasks.popleft() if self.tasks else None

    def finished(self):
        with self.lock:
            if "cache" in self.failed:
                for name in NAMES:
                    if CONDS[name].get("teacher768") and name not in self.failed:
                        self.failed[name] = "teacher cache failed"
            return all(n in self.complete or n in self.failed for n in NAMES)

    def run_cache(self, g):
        out = CACHE
        self.stage("cache_alloc", "src.overnight6.teacher768", ["alloc", "--out", out])
        self.gate(g)
        self.stage("cache_build", "src.overnight6.teacher768", ["cache", "--out", out, "--batch", 96, "--workers", 32], g)
        self.stage("cache_verify", "src.overnight6.teacher768", ["verify", "--out", out, "--report", self.run / "verification/teacher768_preprocessing.json"], g)
        self.stage("lambda_spatial", "src.overnight6.train", ["--calibrate-spatial", "--shared768", out, "--out", self.run / "lambda_spatial.json"], g)

    def train_args(self, name):
        args = ["--condition", name]
        if CONDS[name].get("teacher768"):
            args += ["--shared768", CACHE]
        if CONDS[name].get("spatial"):
            args += ["--lambda-spatial-file", self.run / "lambda_spatial.json"]
        return args

    def run_train(self, name, g):
        self.gate(g)
        self.stage(f"smoke_{name}", self.TRAIN, self.train_args(name) + ["--out", self.run / name / "smoke", "--smoke"], g)
        self.stage(f"train_{name}", self.TRAIN, self.train_args(name) + ["--out", self.run / name], g, retries=1)
        assert (self.run / name / "done.json").exists(), name
        shutil.copy(self.run / name / "config.json", self.run / "configs" / f"{name}.json")
        with self.lock:
            self.tasks.append(("probe", name))

    def run_probe(self, name, g):
        ck = self.run / name / "checkpoints/ema_last.pt"
        assert (self.run / name / "done.json").exists() and ck.exists(), name
        job = self.run / "configs" / f"probe_{name}.yaml"
        job.write_text(yaml.safe_dump(dict(name=f"overnight_{name}", backbone="dinov3", mode="hcf_fused", encoder_ckpt=str(ck), full_size=384, seed=0, epochs=10, paco_dev_images=1000, readers=[dict(name="r2fused", kind="r2", layers=[12])]), sort_keys=False))
        out = self.run / name / "probe"
        if not self.done(f"probe_{name}") and out.exists() and not (out / "done.json").exists():
            out.rename(out.with_name(f"{out.name}.partial_{int(time.time())}"))
        self.gate(g)
        self.stage(f"probe_{name}", "src.g1up.train_bb", ["--job", job, "--out", out, "--exclude", EXCLUDE, "--workers", 8], g, retries=1)
        self.stage(f"val_infer_{name}", "src.overnight6.official_infer", ["--run", self.run, "--cond", name, "--split", "val"], g, retries=1)
        self.spawn(self.cpu_val, name)

    def spawn(self, f, name):
        def guarded():
            try:
                f(name)
            except Exception:
                with self.lock:
                    self.failed[name] = traceback.format_exc()[-1500:]
                self.event(name=name, status="failed", traceback=self.failed[name])
        t = threading.Thread(target=guarded)
        t.start()
        with self.lock:
            self.cpu_threads.append(t)

    def official(self, stage, pred, out, boot=False):
        split = "test" if "test" in stage else "val"
        self.stage(stage, "src.pacoattr.official_eval", ["--split", split, "--pred", pred, "--out", out] + (["--boot"] if boot else []), official=True)

    def pred(self, name, epoch, split):
        if name in BASE:
            return BASE[name]["preds"] / f"ep{epoch}/probs_{split}.npz"
        return self.eval / "preds" / name / f"ep{epoch}/probs_{split}.npz"

    def cpu_val(self, name):
        n = f"{name}_r2"
        parallel([lambda e=e: self.official(f"official_val_{n}_{e}", self.pred(n, e, "val"), self.eval / f"eval_val/{n}/ep{e}") for e in range(1, 11)], 6)
        path = self.eval / "selection" / f"{n}.json"
        if not path.exists():
            curve = {e: json.loads((self.eval / f"eval_val/{n}/ep{e}/official_results.json").read_text())["hierarchical"] for e in range(1, 11)}
            c = {e: ((h["obj-attr-AP"] + h["obj-attr-part-AP"]) / 2, h) for e, h in curve.items()}
            epoch = max(c, key=lambda e: (round(c[e][0], 10), -e))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(epoch=epoch, val_obj=c[epoch][1]["obj-attr-AP"], val_part=c[epoch][1]["obj-attr-part-AP"], rule="max (official VAL object AP + part AP)/2; earlier epoch on ties", curve={str(e): c[e][0] for e in c}), indent=1))
            tmp.replace(path)
        with self.lock:
            self.tasks.appendleft(("test", name))
        self.fast_report()

    def run_test(self, name, g):
        self.gate(g)
        self.stage(f"test_infer_{name}", "src.overnight6.official_infer", ["--run", self.run, "--cond", name, "--split", "test"], g, retries=1)
        self.spawn(self.cpu_test, name)

    def cpu_test(self, name):
        n = f"{name}_r2"
        epoch = json.loads((self.eval / "selection" / f"{n}.json").read_text())["epoch"]
        self.official(f"official_test_{n}", self.pred(n, epoch, "test"), self.eval / f"fast_test/{n}")
        with self.lock:
            self.complete.add(name)
        self.fast_report()

    def fast_report(self):
        try:
            subprocess.run([PY, "-u", "-m", self.REPORT, "--run", str(self.run), "--fast"], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES=""), stdout=open(self.run / "logs/fast_report.log", "a"), stderr=subprocess.STDOUT, check=False)
        except Exception:  # noqa: BLE001
            self.event(status="fast_report_failed", traceback=traceback.format_exc()[-500:])

    def run_task(self, task, g):
        kind, name = task
        try:
            if kind == "cache":
                self.run_cache(g)
            elif kind == "train":
                self.run_train(name, g)
            elif kind == "probe":
                self.run_probe(name, g)
            elif kind == "test":
                self.run_test(name, g)
        except Exception:
            with self.lock:
                self.failed[name] = traceback.format_exc()[-1500:]
            self.event(name=name, kind=kind, status="failed", traceback=self.failed[name])

    def worker(self, g):
        while True:
            if self.control():
                self.event(gpu=g, status="stopped_by_control")
                return
            task = self.next_task(g)
            if task is None:
                if self.finished():
                    return
                time.sleep(10)
                continue
            self.status[f"gpu{g}"] = task
            self.write_status()
            self.run_task(task, g)

    def write_status(self):
        with self.lock:
            s = dict(self.status, claimed=sorted(self.claimed), pending=[list(t) for t in self.tasks], failed=sorted(self.failed), complete=sorted(self.complete), time=time.strftime("%FT%TZ", time.gmtime()))
        (self.run / "status.json").write_text(json.dumps(s, indent=1, default=str))

    def final(self):
        names = [f"{n}_r2" for n in NAMES if n in self.complete] + list(BASE)
        new = [f"{n}_r2" for n in NAMES if n in self.complete]
        epochs = {n: json.loads((self.eval / "selection" / f"{n}.json").read_text())["epoch"] for n in names}
        for split in SPLITS:
            parallel([lambda n=n: self.official(f"boot_eval_{split}_{n}", self.pred(n, epochs[n], split), self.eval / f"eval_boot/{n}/{split}", boot=True) for n in new], 6)
            self.stage(f"bootstrap_{split}", "src.g1up.support", ["boot", "--root", self.eval, "--names", ",".join(names), "--split", split, "--out", self.eval / f"boot_{split}.json"], official=True)
            parallel([lambda n=n: self.stage(f"size_{split}_{n}", "src.g1up.support", ["size", "--split", split, "--pred", self.pred(n, epochs[n], split), "--model", n, "--out", self.eval / f"results/roi_size_{split}_{n}.csv"], official=True) for n in names], 8)
        self.stage("report", self.REPORT, ["--run", self.run])

    def main(self):
        threads = [threading.Thread(target=self.worker, args=(g,)) for g in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for t in list(self.cpu_threads):
            t.join()
        self.write_status()
        if self.control():
            return
        try:
            self.final()
            (self.run / "done.json").write_text(json.dumps(dict(status="done", completed_utc=time.strftime("%FT%TZ", time.gmtime()), complete=sorted(self.complete), failed=sorted(self.failed), protocol="PACO official Val/Test GT-box + GT-category oracle attribute AP; epoch by official Val"), indent=1))
        except Exception:
            (self.run / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc(), complete=sorted(self.complete), failed=self.failed), indent=1))
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = ROOT / a.run
    run.mkdir(parents=True, exist_ok=True)
    lock = open(run / "queue.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("queue already running", flush=True)
        sys.exit(0)
    lock.truncate(0)
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    shutil.copytree(ROOT / "src/overnight6", run / "source_snapshot/src/overnight6", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    Queue(run).main()


if __name__ == "__main__":
    main()
