"""Server-side G0 -> G1 queue. Started once with setsid (survives VS Code / Codex sessions); a flock prevents a second copy; finished stages are skipped on restart and training resumes from recovery.pt.

Start condition, per GPU:
  P1/P2 GPU stages finished (official Val/Test may run alongside: CPU work and a short Test inference), and
  no compute process on the GPU and < 1 GiB used for 2 polls 15 s apart.
A crop-aware long-run process (dinov3_crop_aware_pretrain) on the GPU is stopped through its own `ctl reserve` (user's standing instruction 2026-11-09); any other process is waited for and never signalled.
This queue releases a GPU it reserved as soon as that GPU's G1 work is finished, so the crop-aware run resumes and the GPU does not idle.

Order: G0 (Full384 and Up768, frozen DINOv3, ROIAlign 24 x 24 + R2 reader, official Val) on whichever GPU is free -> decision -> lambda_grid calibration -> G1U on GPU 0, G1R on GPU 1, each independent -> official Val, epoch selection, Test once at that epoch, bootstrap, ROI size, tables.
Control: touch <run>/control/HOLD (no new stage starts) or <run>/control/STOP (exit).

  setsid nohup python -m src.g1up.queue --run output/2026-11-09/g1_upscaled_teacher_s0 &
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
P1P2 = ROOT / "output/2026-11-09/pixeldit_cond_p1p2_s0"
OVAL = ROOT / "output/2026-11-09/hcf_official_val"
PILOT = ROOT / "output/2026-11-08/hcf_pilot"
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
GPUS = {0: "G1U", 1: "G1R"}
G0 = ("Full384", "Up768")
LINKED = {"Cosine_r2": (OVAL, "r384_cosine"), "Hybrid_r2": (OVAL, "r384_pixeldit_hybrid"), "C0_r2": (PILOT, "c0_r2")}
EVAL_NAMES = ["G1U_r2", "G1R_r2", "Cosine_r2", "Hybrid_r2", "C0_r2"]


class Held(Exception):
    pass


class Queue:
    def __init__(self, run):
        self.run = run
        self.g0 = run / "g0"
        self.eval = run / "eval"
        for d in (run / "logs", run / "state", run / "control", run / "scripts", self.g0 / "eval_val", self.eval / "eval_val", self.eval / "results"):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.pending_g0 = [c for c in G0 if not self.done(f"g0_infer_{c}")]
        self.decide_started = False
        self.ctl_reserved = set()
        self.status = {}

    def event(self, **kw):
        kw["time"] = time.strftime("%FT%TZ", time.gmtime())
        with self.lock, (self.run / "logs" / "events.jsonl").open("a") as f:
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

    def running(self, module, out):
        needle = f"-m {module}"
        for line in subprocess.check_output(["ps", "-eo", "args"], text=True).splitlines():
            if needle in line and (out is None or str(out) in line) and "src.g1up.queue" not in line:
                return True
        return False

    def stage(self, name, module, args, gpu=None, official=False):
        done = self.run / "state" / f"{name}.done.json"
        if done.exists():
            return
        self.control()
        out = str(args[args.index("--out") + 1]) if "--out" in args else None
        while out is not None and self.running(module, out):
            self.event(stage=name, status="waiting_for_previous_copy")
            time.sleep(30)
        if out is not None and (Path(out) / "done.json").exists():
            done.write_text(json.dumps(dict(stage=name, gpu=gpu, returncode=0, note="finished by an earlier queue process"), indent=1))
            return
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        if official:
            env["PYTHONPATH"] = f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}"
        self.event(stage=name, status="start", gpu=gpu)
        started = time.time()
        with (self.run / "logs" / f"{name}.log").open("a") as log:
            rc = subprocess.run([EPY if official else PY, "-u", "-m", module] + list(map(str, args)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        record = dict(stage=name, gpu=gpu, returncode=rc, seconds=time.time() - started)
        (done if rc == 0 else self.run / "state" / f"{name}.failed.json").write_text(json.dumps(record, indent=1))
        self.event(**record)
        if rc:
            raise RuntimeError(name)

    def prereq(self):
        """P1/P2 only needs its GPU stages done (user 2026-11-09: P1/P2 official Val/Test may run alongside G1; they are CPU work plus a short Test inference)."""
        while True:
            self.control()
            if all((P1P2 / "official_val/state" / f"infer_{c}.done.json").exists() for c in ("P1", "P2", "Initial", "Hybrid")):
                return
            self.status["waiting"] = "P1/P2 GPU stages"
            time.sleep(30)

    def gpu_processes(self, g):
        uuid = next(line.split(",")[1].strip() for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines() if line.split(",")[0].strip() == str(g))
        pids = [line.split(",")[1].strip() for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True).splitlines() if line.split(",")[0].strip() == uuid]
        cmds = {}
        for pid in pids:
            try:
                cmds[pid] = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore").replace("\0", " ")
            except OSError:
                cmds[pid] = ""
        used = int(subprocess.check_output(["nvidia-smi", f"--id={g}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        return cmds, used

    def reserve(self, g):
        reserved = ROOT / "output/2026-11-09/dinov3_crop_aware_pretrain/control" / f"reserve_gpu_{g}.json"
        if reserved.exists():
            if self.run.name in json.loads(reserved.read_text()).get("note", ""):  # our own reservation from before a restart: still ours to release
                with self.lock:
                    self.ctl_reserved.add(g)
            return
        subprocess.run([PY, "-m", "src.crop_aware_pretrain.ctl", "reserve", "--gpus", str(g), "--note", f"G1 upscaled-teacher queue, run {self.run.name} (user: stop the crop-aware long run on GPU 0,1 when ready)"], cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
        with self.lock:
            self.ctl_reserved.add(g)

    def release(self, g):
        with self.lock:
            mine = g in self.ctl_reserved
            self.ctl_reserved.discard(g)
        if mine:
            subprocess.run([PY, "-m", "src.crop_aware_pretrain.ctl", "release", "--gpus", str(g)], cwd=ROOT, stdout=subprocess.DEVNULL, check=False)
            self.event(gpu=g, status="crop_aware_reservation_released")

    def wait_gpu(self, g):
        """No compute process and < 1 GiB for 2 polls. Only crop-aware processes are stopped, via its own control tool."""
        stable, announced = 0, None
        while True:
            self.control()
            cmds, used = self.gpu_processes(g)
            others = {p: c for p, c in cmds.items() if "crop_aware_pretrain" not in c}
            if not cmds and used < 1024:
                stable += 1
                if stable >= 2:
                    self.reserve(g)
                    return
            else:
                stable = 0
                if cmds and not others:
                    if announced != "crop_aware":
                        self.event(gpu=g, status="stopping_crop_aware", pids=list(cmds))
                        announced = "crop_aware"
                    self.reserve(g)
                elif announced != "other":
                    self.event(gpu=g, status="waiting_for_other_process", processes={p: c[:160] for p, c in others.items()})
                    announced = "other"
            time.sleep(15)

    def probe_job(self, path, name, mode, ckpt=None):
        job = dict(name=name, backbone="dinov3", mode=mode, seed=0, epochs=10, paco_dev_images=1000, readers=[dict(name="r2fused" if mode == "hcf_fused" else "r2", kind="r2", layers=[12])])
        if ckpt:
            job.update(encoder_ckpt=str(ckpt), full_size=384)
        path.write_text(yaml.safe_dump(job, sort_keys=False))

    def probe(self, name, job, out, gpu):
        if not self.done(name) and out.exists() and not (out / "done.json").exists() and not self.running("src.g1up.train_bb", str(out)):
            out.rename(out.with_name(f"{out.name}.partial_{int(time.time())}"))
        self.stage(name, "src.g1up.train_bb", ["--job", job, "--out", out, "--exclude", EXCLUDE, "--workers", 8], gpu)

    def decide(self):
        if self.done("g0_selection") and (self.g0 / "decision.json").exists():
            return
        with ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(lambda x: self.stage(f"g0_official_{x[0]}_{x[1]}", "src.pacoattr.official_eval", ["--split", "val", "--pred", self.g0 / f"preds/{x[0]}_r2/ep{x[1]}/probs_val.npz", "--out", self.g0 / f"eval_val/{x[0]}_r2/ep{x[1]}"], official=True), [(c, e) for c in G0 for e in range(1, 11)]))
        self.stage("g0_selection", "src.siglip_pm.select_epoch", [self.g0])
        self.stage("g0_decide", "src.g1up.support", ["decide", "--root", self.g0], official=True)

    def decision(self):
        while True:
            self.control()
            path = self.g0 / "decision.json"
            if path.exists():
                return json.loads(path.read_text())
            if (self.run / "state" / "g0_failed").exists():
                raise Held("G0 evaluation failed")
            time.sleep(15)

    def worker(self, g):
        cond = GPUS[g]
        try:
            self.prereq()
            while self.pending_g0:
                self.wait_gpu(g)                       # take a G0 job only once this GPU is free, so a busy GPU never holds one back
                with self.lock:
                    job = self.pending_g0.pop(0) if self.pending_g0 else None
                if job is None:
                    break
                yml = self.run / "scripts" / f"g0_{job}.yaml"
                self.probe_job(yml, f"g0_{job}", "bil" if job == "Full384" else "up768")
                self.probe(f"g0_probe_{job}", yml, self.g0 / f"probe_{job}", g)
                self.stage(f"g0_infer_{job}", "src.g1up.official_infer", ["--root", self.g0, "--run", self.run, "--cond", job, "--split", "val"], g)
            while not all(self.done(f"g0_infer_{c}") for c in G0):
                self.control()
                time.sleep(15)
            with self.lock:
                start = not self.decide_started
                self.decide_started = True
            if start:
                threading.Thread(target=self.decide_guard, daemon=True).start()
            self.wait_gpu(g)
            with self.lock:
                need_lambda = not (self.run / "lambda.json").exists() and not getattr(self, "lambda_claimed", False)
                if need_lambda:
                    self.lambda_claimed = True
            if need_lambda:
                self.stage("calibrate_lambda", "src.g1up.train", ["--calibrate", "--out", self.run / "lambda.json"], g)
            while not (self.run / "lambda.json").exists():
                self.control()
                if (self.run / "state/calibrate_lambda.failed.json").exists():
                    raise Held("lambda calibration failed")
                time.sleep(10)
            self.stage(f"smoke_{cond}", "src.g1up.train", ["--condition", cond, "--lambda-file", self.run / "lambda.json", "--out", self.run / f"smoke_{cond}", "--smoke"], g)
            if self.decision()["status"] != "PASS":
                raise Held("G0 decision is FAIL")
            self.wait_gpu(g)
            self.stage(f"train_{cond}", "src.g1up.train", ["--condition", cond, "--lambda-file", self.run / "lambda.json", "--out", self.run / cond], g)
            yml = self.run / "scripts" / f"probe_{cond}.yaml"
            self.probe_job(yml, f"g1_{cond}", "hcf_fused", self.run / cond / "checkpoints/ema_last.pt")
            self.wait_gpu(g)
            self.probe(f"probe_{cond}", yml, self.run / cond / "probe", g)
            self.stage(f"val_infer_{cond}", "src.g1up.official_infer", ["--root", self.eval, "--run", self.run, "--cond", cond, "--split", "val"], g)
            self.status[f"worker_{g}"] = "done"
        except Held as e:
            self.status[f"worker_{g}"] = f"held: {e}"
            self.event(gpu=g, status="held", reason=str(e))
        except Exception:
            self.status[f"worker_{g}"] = "failed"
            self.event(gpu=g, status="failed", traceback=traceback.format_exc()[-1500:])
        finally:
            self.release(g)

    def decide_guard(self):
        try:
            self.decide()
        except Exception:
            (self.run / "state" / "g0_failed").write_text(traceback.format_exc())
            self.event(status="g0_failed", traceback=traceback.format_exc()[-1500:])

    def pred(self, name, epoch, split):
        if name in LINKED:
            base, old = LINKED[name]
            return base / "preds" / old / f"ep{epoch}/probs_{split}.npz"
        return self.eval / "preds" / name / f"ep{epoch}/probs_{split}.npz"

    def final(self):
        root = self.eval
        for name, (base, old) in LINKED.items():
            link = root / "eval_val" / name
            if not link.exists():
                assert (base / "eval_val" / old / "ep10/official_results.json").exists(), name
                link.symlink_to(base / "eval_val" / old, target_is_directory=True)

        def parallel(jobs, n):
            with ThreadPoolExecutor(max_workers=n) as pool:
                list(pool.map(lambda f: f(), jobs))

        parallel([lambda n=n, e=e: self.stage(f"official_val_{n}_{e}", "src.pacoattr.official_eval", ["--split", "val", "--pred", self.pred(n, e, "val"), "--out", root / f"eval_val/{n}/ep{e}"], official=True) for n in ("G1U_r2", "G1R_r2") for e in range(1, 11)], 20)
        self.stage("selection", "src.siglip_pm.select_epoch", [root])
        selection = json.loads((root / "selection.json").read_text())
        assert sorted(selection) == sorted(EVAL_NAMES), sorted(selection)
        for name, (base, old) in LINKED.items():
            existing = json.loads((base / "selection.json").read_text())[old]["epoch"]
            assert selection[name]["epoch"] == existing, (name, selection[name]["epoch"], existing)
        (root / "selection_frozen.json").write_text(json.dumps(selection, indent=1))
        for c, g in (("G1U", 0), ("G1R", 1)):
            self.stage(f"test_infer_{c}", "src.g1up.official_infer", ["--root", root, "--run", self.run, "--cond", c, "--split", "test"], g)
        for split in ("val", "test"):
            parallel([lambda n=n: self.stage(f"boot_eval_{split}_{n}", "src.pacoattr.official_eval", ["--split", split, "--pred", self.pred(n, selection[n]["epoch"], split), "--out", root / f"eval_boot/{n}/{split}", "--boot"], official=True) for n in EVAL_NAMES], 10)
            self.stage(f"bootstrap_{split}", "src.g1up.support", ["boot", "--root", root, "--names", ",".join(EVAL_NAMES), "--split", split, "--out", root / f"boot_{split}.json"], official=True)
            parallel([lambda n=n: self.stage(f"size_{split}_{n}", "src.g1up.support", ["size", "--split", split, "--pred", self.pred(n, selection[n]["epoch"], split), "--model", n, "--out", root / f"results/roi_size_{split}_{n}.csv"], official=True) for n in EVAL_NAMES], 10)
        self.stage("report", "src.g1up.report", ["--run", self.run])

    def main(self):
        workers = [threading.Thread(target=self.worker, args=(g,)) for g in GPUS]
        for t in workers:
            t.start()
        for t in workers:
            t.join()
        (self.run / "status.json").write_text(json.dumps(self.status, indent=1))
        if all(v == "done" for k, v in self.status.items() if k.startswith("worker_")) and sum(k.startswith("worker_") for k in self.status) == 2:
            try:
                self.final()
                (self.run / "done.json").write_text(json.dumps(dict(status="done", completed_utc=time.strftime("%FT%TZ", time.gmtime()), protocol="PACO official Val/Test GT-box + GT-category oracle attribute AP; epoch by official Val"), indent=1))
            except Exception:
                (self.run / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc()), indent=1))
                raise
        else:
            (self.run / "queue_held.json").write_text(json.dumps(dict(status="held_or_failed", workers=self.status, time=time.time()), indent=1))


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
    shutil.copytree(ROOT / "src/g1up", run / "source_snapshot/src/g1up", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    Queue(run).main()


if __name__ == "__main__":
    main()
