"""Server-side queue for G1-Native 768. Started once with setsid; a flock prevents a second copy; finished stages are skipped on restart and training resumes from recovery.pt. Control: <run>/control/HOLD or STOP.

Order
  CPU       geometry check
  GPU 1     G0-Native768 frozen probe + Val inference (may share the GPU with the crop-aware run, which is not stopped by this queue); then CPU official Val and the G0 table. G0 is verification only: it does not gate or change G1-Native.
  GPU 0     smoke (shareable)
  GPU 0+1   needs both GPUs free of every process: throughput 1 GPU, throughput 2 GPU DDP -> ddp_decision.json (DDP only if 2-GPU efficiency >= 0.75 and step-1 losses match within 2e-3, rule fixed here)
  GPU       G1-Native 12k -> R2 probe -> Val inference -> efficiency; then CPU official Val -> epoch frozen on Val -> Test inference once -> official Test AP without bootstrap -> fast table -> bootstrap, ROI size, full tables.
Never signals another process. "Free" means no compute process and < 1 GiB for 2 polls; utilization is not used. A reservation made for this run (note contains the run name) is released as soon as that GPU's work is finished.

  setsid nohup python -m src.g1native.queue --run output/2026-11-09/g1_native768_s0 &
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
G1 = ROOT / "output/2026-11-09/g1_upscaled_teacher_s0"
CONTROL = ROOT / "output/2026-11-09/dinov3_crop_aware_pretrain/control"
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
LINKED = ("Cosine_r2", "G1U_r2", "Hybrid_r2", "C0_r2")
EVAL_NAMES = ("G1N_r2",) + LINKED
DDP_PORT = 29631


class Held(Exception):
    pass


class Queue:
    def __init__(self, run):
        self.run, self.g0, self.eval = run, run / "g0", run / "eval"
        for d in (run / "logs", run / "state", run / "control", run / "scripts", self.g0 / "eval_val", self.eval / "eval_val", self.eval / "eval_boot", self.eval / "results"):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

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

    def wait_gpu(self, g, shared=False):
        """shared: the crop-aware run may stay on the GPU (it is never stopped here). Otherwise no process and < 1 GiB for 2 polls."""
        stable, announced = 0, None
        while True:
            self.control()
            cmds, used = self.gpu_processes(g)
            other = {p: c for p, c in cmds.items() if not (shared and "crop_aware_pretrain" in c or shared and "src.g1x.probe" in c and "dinov3_crop_aware_pretrain" in c)}
            if not other and (shared or used < 1024):
                stable += 1
                if stable >= 2:
                    return
            else:
                stable = 0
                seen = sorted(c[:150] for c in other.values())
                if seen != announced:
                    self.event(gpu=g, status="waiting_for_gpu", shared=shared, processes=seen, used_mib=used)
                    announced = seen
            time.sleep(15)

    def release(self, g):
        path = CONTROL / f"reserve_gpu_{g}.json"
        if path.exists() and self.run.name in json.loads(path.read_text()).get("note", ""):
            subprocess.run([PY, "-m", "src.crop_aware_pretrain.ctl", "release", "--gpus", str(g)], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), stdout=subprocess.DEVNULL, check=False)
            self.event(gpu=g, status="crop_aware_reservation_released")

    def probe_job(self, path, name, mode, ckpt=None, reader="r2"):
        job = dict(name=name, backbone="dinov3", mode=mode, seed=0, epochs=10, paco_dev_images=1000, readers=[dict(name=reader, kind="r2", layers=[12])])
        if ckpt:
            job.update(encoder_ckpt=str(ckpt), full_size=384)
        path.write_text(yaml.safe_dump(job, sort_keys=False))

    def probe(self, name, job, out, gpu):
        if not self.done(name) and out.exists() and not (out / "done.json").exists():
            out.rename(out.with_name(f"{out.name}.partial_{int(time.time())}"))
        self.stage(name, "src.g1native.train_bb", ["--job", job, "--out", out, "--exclude", EXCLUDE, "--workers", 8], gpu)

    @staticmethod
    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda f: f(), jobs))

    def link(self, path, target):
        if not path.exists():
            assert Path(target).exists(), target
            path.symlink_to(os.path.realpath(target), target_is_directory=True)

    def run_g0(self):
        """Frozen teacher verification; the existing G0-Full384 and G0-Up768 are reused (same protocol, same reader job)."""
        try:
            if not self.done("g0_infer_Native768"):
                self.wait_gpu(1, shared=True)
            job = self.run / "scripts/g0_Native768.yaml"
            self.probe_job(job, "g0_Native768", "native768")
            self.probe("g0_probe_Native768", job, self.g0 / "probe_Native768", 1)
            self.stage("g0_infer_Native768", "src.g1native.official_infer", ["--root", self.g0, "--run", self.run, "--cond", "Native768", "--split", "val"], 1)
            for n in ("Full384_r2", "Up768_r2"):
                self.link(self.g0 / "eval_val" / n, G1 / "g0/eval_val" / n)
            self.parallel([lambda e=e: self.stage(f"g0_official_{e}", "src.pacoattr.official_eval", ["--split", "val", "--pred", self.g0 / f"preds/Native768_r2/ep{e}/probs_val.npz", "--out", self.g0 / f"eval_val/Native768_r2/ep{e}"], official=True) for e in range(1, 11)], 10)
            self.stage("g0_selection", "src.siglip_pm.select_epoch", [self.g0])
            old = json.loads((G1 / "g0/selection.json").read_text())
            new = json.loads((self.g0 / "selection.json").read_text())
            assert all(new[n]["epoch"] == old[n]["epoch"] for n in ("Full384_r2", "Up768_r2")), (new, old)
            self.stage("g0_compare", "src.g1native.report", ["--run", self.run, "--g0"])
        except Held as e:
            self.event(status="held", part="g0", reason=str(e))
        except Exception:
            self.event(status="g0_failed", traceback=traceback.format_exc()[-1500:])
            (self.run / "state/g0_failed").write_text(traceback.format_exc())

    def throughput(self, lam):
        if not self.done("throughput_1gpu"):
            self.stage("throughput_1gpu", "src.g1native.train", ["--condition", "G1N", "--lambda-file", lam, "--out", self.run / "throughput_1gpu", "--throughput", 220], 0)
        self.stage("throughput_2gpu", "torch.distributed.run", ["--nproc_per_node", 2, "--master_port", DDP_PORT, "-m", "src.g1native.train", "--condition", "G1N", "--lambda-file", lam, "--out", self.run / "throughput_2gpu", "--throughput", 220], "0,1")
        a, b = [json.loads((self.run / f"throughput_{n}gpu/throughput.json").read_text()) for n in (1, 2)]
        eff = b["steps_per_s"] / (2 * a["steps_per_s"])
        parity = {k: abs(a["step1"][k] - b["step1"][k]) for k in ("crop", "grid", "glob")}
        use = bool(eff >= .75 and max(parity.values()) < 2e-3)
        decision = dict(use_ddp=use, efficiency=eff, rule="DDP only if 2-GPU steps/s >= 0.75 x 2 x 1-GPU steps/s and step-1 global losses match within 2e-3", one_gpu=a, two_gpu=b, step1_loss_abs_diff=parity, speedup=b["steps_per_s"] / a["steps_per_s"])
        (self.run / "ddp_decision.json").write_text(json.dumps(decision, indent=1))
        self.event(status="ddp_decision", use_ddp=use, efficiency=eff, speedup=decision["speedup"])
        return use

    def main(self):
        run, lam = self.run, self.run / "lambda.json"
        if not lam.exists():
            shutil.copy(G1 / "lambda.json", lam)
        assert json.loads(lam.read_text())["lambda_grid"] == .61
        self.stage("geometry", "src.g1native.geometry_check", ["--out", run / "geometry"])
        g0 = threading.Thread(target=self.run_g0)
        g0.start()
        if not self.done("smoke_G1N"):
            self.wait_gpu(0, shared=True)
        self.stage("smoke_G1N", "src.g1native.train", ["--condition", "G1N", "--lambda-file", lam, "--out", run / "smoke_G1N", "--smoke"], 0)
        while not self.done("g0_infer_Native768") and not (run / "state/g0_failed").exists():
            self.control()
            time.sleep(15)
        decision = run / "ddp_decision.json"
        if decision.exists():                                             # training was started by hand on the user's request; wait for its state file
            while not self.done("train_G1N"):
                self.control()
                if (run / "state/train_G1N.failed.json").exists():
                    raise Held("train_G1N failed")
                time.sleep(15)
        else:
            self.wait_gpu(0)
            self.wait_gpu(1)
        use_ddp = json.loads(decision.read_text())["use_ddp"] if decision.exists() else self.throughput(lam)
        train_args = ["--condition", "G1N", "--lambda-file", lam, "--out", run / "G1N"]
        if use_ddp:
            self.stage("train_G1N", "torch.distributed.run", ["--nproc_per_node", 2, "--master_port", DDP_PORT, "-m", "src.g1native.train"] + train_args, "0,1")
        else:
            self.release(1)
            self.stage("train_G1N", "src.g1native.train", train_args, 0)
        self.release(1)
        job = run / "scripts/probe_G1N.yaml"
        self.probe_job(job, "g1_G1N", "hcf_fused", run / "G1N/checkpoints/ema_last.pt", "r2fused")
        self.wait_gpu(0)
        self.probe("probe_G1N", job, run / "G1N/probe", 0)
        self.stage("val_infer_G1N", "src.g1native.official_infer", ["--root", self.eval, "--run", run, "--cond", "G1N", "--split", "val"], 0)
        self.stage("efficiency", "src.g1native.efficiency", ["--run", run, "--out", self.eval / "results/efficiency.json"], 0)
        self.release(0)
        root = self.eval
        for n in LINKED:
            self.link(root / "eval_val" / n, G1 / "eval/eval_val" / n)
        self.parallel([lambda e=e: self.stage(f"official_val_G1N_{e}", "src.pacoattr.official_eval", ["--split", "val", "--pred", root / f"preds/G1N_r2/ep{e}/probs_val.npz", "--out", root / f"eval_val/G1N_r2/ep{e}"], official=True) for e in range(1, 11)], 10)
        self.stage("selection", "src.siglip_pm.select_epoch", [root])
        selection = json.loads((root / "selection.json").read_text())
        assert sorted(selection) == sorted(EVAL_NAMES), sorted(selection)
        old = json.loads((G1 / "eval/selection_frozen.json").read_text())
        assert all(selection[n]["epoch"] == old[n]["epoch"] for n in LINKED), (selection, old)
        frozen = root / "selection_frozen.json"
        if not frozen.exists():
            frozen.write_text(json.dumps(selection, indent=1))
        (root / "selection.json").write_text(frozen.read_text())       # the Test inference reads the frozen epochs
        selection = json.loads(frozen.read_text())
        epoch = selection["G1N_r2"]["epoch"]
        self.stage("test_infer_G1N", "src.g1native.official_infer", ["--root", root, "--run", run, "--cond", "G1N", "--split", "test"], 0)
        test = root / f"preds/G1N_r2/ep{epoch}/probs_test.npz"
        self.stage("fast_test_G1N", "src.pacoattr.official_eval", ["--split", "test", "--pred", test, "--out", root / "fast_test/G1N_r2"], official=True)
        self.stage("fast_report", "src.g1native.report", ["--run", run, "--fast"])
        for n in LINKED:
            self.link(root / "eval_boot" / n, G1 / "eval/eval_boot" / n)
        for split in ("val", "test"):
            pred = root / f"preds/G1N_r2/ep{epoch}/probs_{split}.npz"
            self.stage(f"boot_eval_{split}_G1N", "src.pacoattr.official_eval", ["--split", split, "--pred", pred, "--out", root / f"eval_boot/G1N_r2/{split}", "--boot"], official=True)
            self.stage(f"bootstrap_{split}", "src.g1up.support", ["boot", "--root", root, "--names", ",".join(EVAL_NAMES), "--split", split, "--out", root / f"boot_{split}.json"], official=True)
            self.stage(f"size_{split}_G1N", "src.g1up.support", ["size", "--split", split, "--pred", pred, "--model", "G1N_r2", "--out", root / f"results/roi_size_{split}_G1N_r2.csv"], official=True)
        g0.join()
        self.stage("report", "src.g1native.report", ["--run", run])
        (run / "done.json").write_text(json.dumps(dict(status="done", completed_utc=time.strftime("%FT%TZ", time.gmtime()), protocol="PACO official Val/Test GT-box + GT-category oracle attribute AP; epoch by official Val"), indent=1))


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
    shutil.copytree(ROOT / "src/g1native", run / "source_snapshot/src/g1native", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    q = Queue(run)
    try:
        q.main()
    except Held as e:
        q.event(status="held", reason=str(e))
        (run / "queue_held.json").write_text(json.dumps(dict(status="held", reason=str(e), time=time.time()), indent=1))
    except Exception:
        (run / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc()), indent=1))
        raise


if __name__ == "__main__":
    main()
