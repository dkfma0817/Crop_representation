"""Server-side queue for the two 384 bottleneck diagnoses. Start once with setsid; a flock prevents a second copy; finished stages are skipped on restart; reader probes resume from resume.pt.

  D1 (GPU 1)  C1 Crop-from-384: CPU checks -> R2 probe -> official Val -> epoch frozen on Val -> Test once -> AP table -> bootstrap, ROI size, RGB tables.
  D2 (GPU 0)  A1 Fine-only: starts after the G1-Native GPU stages are finished (state/test_infer_G1N.done.json AND no G1-Native process on GPU 0; utilization is never used).

Never signals another process and never touches the crop-aware run: its train and eval processes may share a GPU (the crop-aware run's own policy), any other process is waited for. A reservation of GPU 1 for G1-Native is waited for too. D1 and D2 are independent.
Control: touch <run>/control/HOLD (no new stage starts) or <run>/control/STOP (exit).

  setsid nohup python -m src.diag384.queue --run output/2026-11-09/fine_bottleneck_diag_s0 &
"""
import argparse
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

PY, EPY = "/home/irteam/envs/dinov2_ord_py310/bin/python", "/home/irteam/envs/paco_eval/bin/python"
ROOT = Path("/home/irteam/Crop_representation")
PILOT = ROOT / "output/2026-11-08/hcf_pilot"
ABL = ROOT / "output/2026-11-09/coarse_fine_ablation_s0"
NATIVE = ROOT / "output/2026-11-09/g1_native768_s0"
CONTROL = ROOT / "output/2026-11-09/dinov3_crop_aware_pretrain/control"
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
D1_LINKS = {"B0_r2": "orig_r2", "C0_r2": "c0_r2"}
D1_EPOCH = {"B0_r2": 5, "C0_r2": 5}
D2_LINKS = ("A1_coarse", "A1_r2")


class Held(Exception):
    pass


class Queue:
    def __init__(self, run):
        self.run = run
        for d in ("logs", "state", "control", "scripts"):
            (run / d).mkdir(parents=True, exist_ok=True)
        for part in ("d1", "d2"):
            for d in ("checks", "eval/eval_val", "eval/eval_boot", "eval/results", "eval/fast_test"):
                (run / part / d).mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.status = {}

    def event(self, **kw):
        kw["time"] = time.strftime("%FT%TZ", time.gmtime())
        with self.lock, (self.run / "logs/events.jsonl").open("a") as f:
            f.write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)

    def done(self, name):
        return (self.run / "state" / f"{name}.done.json").exists()

    def control(self):
        if (self.run / "control/STOP").exists():
            raise Held("STOP file")
        while (self.run / "control/HOLD").exists():
            if (self.run / "control/STOP").exists():
                raise Held("STOP file")
            time.sleep(15)

    def stage(self, name, module, args, gpu=None, official=False):
        done = self.run / "state" / f"{name}.done.json"
        if done.exists():
            return
        self.control()
        out = Path(str(args[args.index("--out") + 1])) if "--out" in args else None
        while out is not None and self.running(module, out):               # a stage left running by an earlier queue process is adopted, never duplicated
            self.event(stage=name, status="waiting_for_previous_copy")
            time.sleep(30)
        if out is not None and ((out / "done.json").exists() or (name == "d1_rgb" and (out / "rgb_table.csv").exists())):
            done.write_text(json.dumps(dict(stage=name, gpu=gpu, returncode=0, note="finished by an earlier queue process"), indent=1))
            self.event(stage=name, status="adopted_from_previous_copy")
            return
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        env["PYTHONPATH"] = f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}" if official else str(ROOT)
        self.event(stage=name, status="start", gpu=gpu)
        started = time.time()
        with (self.run / "logs" / f"{name}.log").open("a") as log:
            rc = subprocess.run([EPY if official else PY, "-u", "-m", module] + list(map(str, args)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        record = dict(stage=name, gpu=gpu, returncode=rc, seconds=time.time() - started)
        (done if rc == 0 else self.run / "state" / f"{name}.failed.json").write_text(json.dumps(record, indent=1))
        self.event(**record)
        if rc:
            raise RuntimeError(name)

    @staticmethod
    def running(module, out):
        return any(f"-m {module}" in line and str(out) in line and "src.diag384.queue" not in line for line in subprocess.check_output(["ps", "-eo", "args"], text=True).splitlines())

    def gpu_processes(self, g):
        uuid = next(line.split(",")[1].strip() for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines() if line.split(",")[0].strip() == str(g))
        pids = [line.split(",")[1].strip() for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True).splitlines() if line.split(",")[0].strip() == uuid]
        cmds = {}
        for pid in pids:
            try:
                cmds[pid] = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore").replace("\0", " ")
            except OSError:
                cmds[pid] = ""
        return cmds

    def wait_gpu(self, g, allow_native=False):
        """Only the crop-aware run (train / its PACO eval) may share the GPU; every other compute process is waited for. Two polls 15 s apart. allow_native: G1-Native may share too (user override, control/D2_NO_WAIT)."""
        stable, announced = 0, None
        while True:
            self.control()
            cmds = self.gpu_processes(g)
            other = sorted(c[:150] for c in cmds.values() if not ("crop_aware_pretrain" in c or ("src.g1x.probe" in c and "dinov3_crop_aware_pretrain" in c) or (allow_native and "g1native" in c)))
            res = CONTROL / f"reserve_gpu_{g}.json"
            if g == 1 and res.exists() and "g1_native768" in json.loads(res.read_text()).get("note", ""):
                other.append("reserved for G1-Native")
            if not other:
                stable += 1
                if stable >= 2:
                    return
            else:
                stable = 0
                if other != announced:
                    self.event(gpu=g, status="waiting_for_gpu", processes=other)
                    announced = other
            time.sleep(15)

    def wait_g1_native(self):
        """GPU 0 is released to D2 only after the explicit completion state of the last G1-Native GPU stage and the actual process exit."""
        announced = False
        while True:
            self.control()
            finished = (NATIVE / "state/test_infer_G1N.done.json").exists()
            running = [c[:150] for c in self.gpu_processes(0).values() if "g1native" in c]
            if finished and not running:
                return
            if not announced:
                self.event(status="waiting_for_g1_native", finished_state=finished, running=running)
                announced = True
            time.sleep(30)

    def link(self, path, target):
        if not path.exists():
            assert Path(target).exists(), target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(os.path.realpath(target), target_is_directory=Path(target).is_dir())

    def probe(self, name, job, out, gpu):
        self.stage(name, "src.diag384.train_bb", ["--job", job, "--out", out, "--exclude", EXCLUDE, "--workers", 8], gpu)

    def job(self, path, name, mode, reader, ckpt=None):
        job = dict(name=name, backbone="dinov3", mode=mode, seed=0, epochs=10, paco_dev_images=1000, readers=[dict(name=reader, kind="r2", layers=[12])])
        if ckpt:
            job.update(encoder_ckpt=str(ckpt), full_size=384)
        path.write_text(yaml.safe_dump(job, sort_keys=False))

    @staticmethod
    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda f: f(), jobs))

    def official_val(self, part, root, name):
        self.parallel([lambda e=e: self.stage(f"{part}_official_val_{name}_{e}", "src.pacoattr.official_eval", ["--split", "val", "--pred", root / f"preds/{name}/ep{e}/probs_val.npz", "--out", root / f"eval_val/{name}/ep{e}"], official=True) for e in range(1, 11)], 10)

    def select(self, part, root, expected):
        self.stage(f"{part}_selection", "src.siglip_pm.select_epoch", [root])
        selection = json.loads((root / "selection.json").read_text())
        assert all(selection[n]["epoch"] == e for n, e in expected.items()), (selection, expected)
        frozen = root / "selection_frozen.json"
        if not frozen.exists():
            frozen.write_text(json.dumps(selection, indent=1))
        return json.loads(frozen.read_text())

    def d1(self):
        run, part, root = self.run, self.run / "d1", self.run / "d1/eval"
        self.stage("d1_checks", "src.diag384.checks", ["geometry", "--out", part / "checks"])
        rgb = threading.Thread(target=self.guard, args=("d1_rgb", lambda: self.stage("d1_rgb", "src.diag384.checks", ["rgb", "--out", part / "checks", "--procs", 16])))
        rgb.start()
        job = run / "scripts/probe_C1.yaml"
        self.job(job, "d1_C1", "c1", "r2")
        self.wait_gpu(1)
        self.probe("d1_probe_C1", job, part / "probe_C1", 1)
        self.stage("d1_infer_val_C1", "src.diag384.infer", ["--root", root, "--probe", part / "probe_C1/r2", "--name", "C1_r2", "--split", "val"], 1)
        self.official_val("d1", root, "C1_r2")
        for name, old in D1_LINKS.items():
            self.link(root / "eval_val" / name, PILOT / "eval_val" / old)
        selection = self.select("d1", root, D1_EPOCH)
        epoch = selection["C1_r2"]["epoch"]
        self.wait_gpu(1)
        self.stage("d1_infer_test_C1", "src.diag384.infer", ["--root", root, "--probe", part / "probe_C1/r2", "--name", "C1_r2", "--split", "test"], 1)
        self.stage("d1_fast_test_C1", "src.pacoattr.official_eval", ["--split", "test", "--pred", root / f"preds/C1_r2/ep{epoch}/probs_test.npz", "--out", root / "fast_test/C1_r2"], official=True)
        for name, old in D1_LINKS.items():
            self.link(root / "fast_test" / name, PILOT / "eval_test" / old / "test")
        self.stage("d1_fast_report", "src.diag384.report", ["--run", run, "--part", "d1", "--fast"])
        pred = {"C1_r2": root / f"preds/C1_r2/ep{epoch}", **{n: PILOT / "preds" / o / f"ep{D1_EPOCH[n]}" for n, o in D1_LINKS.items()}}
        names = ",".join(("B0_r2", "C1_r2", "C0_r2"))
        for split in ("val", "test"):
            self.parallel([lambda n=n: self.stage(f"d1_boot_eval_{split}_{n}", "src.pacoattr.official_eval", ["--split", split, "--pred", pred[n] / f"probs_{split}.npz", "--out", root / f"eval_boot/{n}/{split}", "--boot"], official=True) for n in pred], 3)
            self.stage(f"d1_bootstrap_{split}", "src.g1up.support", ["boot", "--root", root, "--names", names, "--split", split, "--out", root / f"boot_{split}.json"], official=True)
            self.parallel([lambda n=n: self.stage(f"d1_size_{split}_{n}", "src.g1up.support", ["size", "--split", split, "--pred", pred[n] / f"probs_{split}.npz", "--model", n, "--out", root / f"results/roi_size_{split}_{n}.csv"], official=True) for n in pred], 3)
        rgb.join()
        self.stage("d1_report", "src.diag384.report", ["--run", run, "--part", "d1"])

    def d2(self):
        run, part, root = self.run, self.run / "d2", self.run / "d2/eval"
        selection_old = json.loads((ABL / "eval/selection_frozen.json").read_text())
        expected = {n: selection_old[n]["epoch"] for n in D2_LINKS}
        no_wait = (run / "control/D2_NO_WAIT").exists()                    # user 2026-11-09: start D2 now, overlap with the G1-Native Test inference is acceptable
        if not no_wait:
            self.wait_g1_native()
        self.wait_gpu(0, allow_native=no_wait)
        self.stage("d2_check", "src.diag384.fine_norm", ["--out", part / "checks"], 0)
        job = run / "scripts/probe_fine.yaml"
        self.job(job, "d2_A1_fine", "hcf_fine", "r2fine", ABL / "A1/checkpoints/ema_last.pt")
        self.probe("d2_probe_fine", job, part / "probe_fine", 0)
        self.stage("d2_infer_val", "src.diag384.infer", ["--root", root, "--probe", part / "probe_fine/r2fine", "--name", "A1_fine", "--split", "val"], 0)
        self.official_val("d2", root, "A1_fine")
        for n in D2_LINKS:
            self.link(root / "eval_val" / n, ABL / "eval/eval_val" / n)
        selection = self.select("d2", root, expected)
        epoch = selection["A1_fine"]["epoch"]
        self.wait_gpu(0)
        self.stage("d2_infer_test", "src.diag384.infer", ["--root", root, "--probe", part / "probe_fine/r2fine", "--name", "A1_fine", "--split", "test"], 0)
        self.stage("d2_fast_test", "src.pacoattr.official_eval", ["--split", "test", "--pred", root / f"preds/A1_fine/ep{epoch}/probs_test.npz", "--out", root / "fast_test/A1_fine"], official=True)
        for n in D2_LINKS:
            self.link(root / "fast_test" / n, ABL / "eval/fast_test" / n)
        self.stage("d2_fast_report", "src.diag384.report", ["--run", run, "--part", "d2", "--fast"])
        for n in D2_LINKS:
            self.link(root / "eval_boot" / n, ABL / "eval/eval_boot" / n)
        for split in ("val", "test"):
            for n in D2_LINKS:
                target = root / f"results/roi_size_{split}_{n}.csv"
                if not target.exists():
                    shutil.copy(ABL / f"eval/results/roi_size_{split}_{n}.csv", target)
            pred = root / f"preds/A1_fine/ep{epoch}/probs_{split}.npz"
            self.stage(f"d2_boot_eval_{split}", "src.pacoattr.official_eval", ["--split", split, "--pred", pred, "--out", root / f"eval_boot/A1_fine/{split}", "--boot"], official=True)
            self.stage(f"d2_bootstrap_{split}", "src.g1up.support", ["boot", "--root", root, "--names", "A1_coarse,A1_fine,A1_r2", "--split", split, "--out", root / f"boot_{split}.json"], official=True)
            self.stage(f"d2_size_{split}", "src.g1up.support", ["size", "--split", split, "--pred", pred, "--model", "A1_fine", "--out", root / f"results/roi_size_{split}_A1_fine.csv"], official=True)
        self.stage("d2_report", "src.diag384.report", ["--run", run, "--part", "d2"])

    def guard(self, name, fn):
        try:
            fn()
            self.status[name] = "done"
        except Held as e:
            self.status[name] = f"held: {e}"
            self.event(part=name, status="held", reason=str(e))
        except Exception:
            self.status[name] = "failed"
            self.event(part=name, status="failed", traceback=traceback.format_exc()[-1500:])
        (self.run / "status.json").write_text(json.dumps(self.status, indent=1))

    def main(self):
        workers = [threading.Thread(target=self.guard, args=args) for args in (("d1", self.d1), ("d2", self.d2))]
        for t in workers:
            t.start()
        for t in workers:
            t.join()
        if all(self.status.get(k) == "done" for k in ("d1", "d2")):
            (self.run / "done.json").write_text(json.dumps(dict(status="done", completed_utc=time.strftime("%FT%TZ", time.gmtime()), protocol="PACO official Val/Test GT-box + GT-category oracle attribute AP; epoch by official Val"), indent=1))


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
    shutil.copytree(ROOT / "src/diag384", run / "source_snapshot/src/diag384", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    Queue(run).main()


if __name__ == "__main__":
    main()
