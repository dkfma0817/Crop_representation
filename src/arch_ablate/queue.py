"""Server-side queue for the Coarse-Fine interaction ablation. Started once with setsid; a flock prevents a second copy; finished stages are skipped on restart and training resumes from recovery.pt.

Start condition per GPU: the G1 run's last GPU stage on that GPU is done (state/test_infer_G1U on GPU 0, state/test_infer_G1R on GPU 1; G1's CPU bootstrap is not waited for)
AND no compute process on the GPU and < 1 GiB used for 2 polls 15 s apart. Utilization alone is never used. No process is ever signalled or reserved away: this queue only waits.

Order: A1 on GPU 0, A2 on GPU 1, each independent: smoke -> 12k COCO steps -> R2 probe (same job as A0) -> Coarse-only R2 probe -> Val inference. A0 Coarse-only probe on whichever GPU gets there first.
Then CPU official Val -> epoch frozen on Val -> Test inference once -> official Test AP without bootstrap -> fast table -> efficiency -> bootstrap, ROI size, full tables.
Control: touch <run>/control/HOLD (no new stage starts) or <run>/control/STOP (exit).

  setsid nohup python -m src.arch_ablate.queue --run output/2026-11-09/coarse_fine_ablation_s0 &
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
OVAL = ROOT / "output/2026-11-09/hcf_official_val"
A0_CKPT = ROOT / "output/2026-11-08/hcf_material_causal/r384_cosine/P/checkpoints/ema_last.pt"
EXCLUDE = "output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json"
GATE = {0: "test_infer_G1U", 1: "test_infer_G1R"}
NEW = ("A1_r2", "A2_r2", "A0_coarse", "A1_coarse", "A2_coarse")
EVAL_NAMES = ("A0_r2",) + NEW


class Held(Exception):
    pass


class Queue:
    def __init__(self, run, gpus):
        self.run, self.gpus = run, gpus
        self.eval = run / "eval"
        for d in (run / "logs", run / "state", run / "control", run / "scripts", self.eval / "eval_val", self.eval / "results"):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.claimed = set()
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

    def stage(self, name, module, args, gpu=None, official=False):
        done = self.run / "state" / f"{name}.done.json"
        if done.exists():
            return
        self.control()
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        if official:
            env["PYTHONPATH"] = f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}"
        else:
            env["PYTHONPATH"] = str(ROOT)
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

    def wait_gpu(self, g):
        """Wait only. No compute process and < 1 GiB for 2 polls."""
        stable, announced = 0, None
        while True:
            self.control()
            cmds, used = self.gpu_processes(g)
            if not cmds and used < 1024:
                stable += 1
                if stable >= 2:
                    return
            else:
                stable = 0
                seen = sorted(c[:160] for c in cmds.values())
                if seen != announced:
                    self.event(gpu=g, status="waiting_for_gpu", processes=seen, used_mib=used)
                    announced = seen
            time.sleep(15)

    def wait_gate(self, g):
        """The G1 run's last GPU stage on this GPU must be finished; the state file is the signal, not utilization."""
        marker, announced = G1 / "state" / f"{GATE[g]}.done.json", False
        while not marker.exists():
            self.control()
            if not announced:
                self.event(gpu=g, status="waiting_for_g1", marker=str(marker))
                announced = True
            time.sleep(15)
        self.status[f"gate_{g}"] = "g1 gpu work done"

    def probe_job(self, path, name, mode, ckpt):
        readers = [dict(name="r0", kind="r0"), dict(name="r2fused", kind="r2", layers=[12])] if mode == "hcf_fused" else [dict(name="r2coarse", kind="r2", layers=[12])]
        path.write_text(yaml.safe_dump(dict(name=name, backbone="dinov3", mode=mode, encoder_ckpt=str(ckpt), full_size=384, seed=0, epochs=10, paco_dev_images=1000, readers=readers), sort_keys=False))

    def probe(self, name, job, out, gpu):
        if not self.done(name) and out.exists() and not (out / "done.json").exists():
            out.rename(out.with_name(f"{out.name}.partial_{int(time.time())}"))
        self.stage(name, "src.arch_ablate.train_bb", ["--job", job, "--out", out, "--exclude", EXCLUDE, "--workers", 8], gpu)

    def infer(self, name, split, gpu):
        self.stage(f"{split}_infer_{name}", "src.arch_ablate.official_infer", ["--root", self.eval, "--run", self.run, "--name", name, "--split", split], gpu)

    def claim(self, key):
        with self.lock:
            if key in self.claimed:
                return False
            self.claimed.add(key)
            return True

    def check_parity(self, cond):
        a0 = json.loads((self.run / "smoke_A0/smoke.json").read_text())["steps"]
        other = json.loads((self.run / f"smoke_{cond}/smoke.json").read_text())["steps"]
        assert [s["batch_sha"] for s in a0] == [s["batch_sha"] for s in other], "A0 and " + cond + " see different batches"

    def worker(self, g, cond):
        try:
            self.wait_gate(g)
            self.wait_gpu(g)
            if self.claim("smoke_A0"):
                self.stage("smoke_A0", "src.arch_ablate.train", ["--condition", "A0", "--out", self.run / "smoke_A0", "--smoke"], g)
            while not self.done("smoke_A0"):
                self.control()
                if (self.run / "state/smoke_A0.failed.json").exists():
                    raise Held("A0 parity smoke failed")
                time.sleep(10)
            self.stage(f"smoke_{cond}", "src.arch_ablate.train", ["--condition", cond, "--out", self.run / f"smoke_{cond}", "--smoke"], g)
            self.check_parity(cond)
            self.wait_gpu(g)
            self.stage(f"train_{cond}", "src.arch_ablate.train", ["--condition", cond, "--out", self.run / cond], g)
            ck = self.run / cond / "checkpoints/ema_last.pt"
            fused, coarse = self.run / "scripts" / f"probe_{cond}.yaml", self.run / "scripts" / f"probe_coarse_{cond}.yaml"
            self.probe_job(fused, f"ablate_{cond}", "hcf_fused", ck)
            self.probe_job(coarse, f"ablate_{cond}_coarse", "hcf_coarse", ck)
            self.probe(f"probe_{cond}", fused, self.run / cond / "probe", g)
            self.infer(f"{cond}_r2", "val", g)
            self.probe(f"probe_coarse_{cond}", coarse, self.run / cond / "probe_coarse", g)
            self.infer(f"{cond}_coarse", "val", g)
            if self.claim("A0_coarse"):
                job = self.run / "scripts/probe_coarse_A0.yaml"
                self.probe_job(job, "ablate_A0_coarse", "hcf_coarse", A0_CKPT)
                (self.run / "A0").mkdir(exist_ok=True)
                self.wait_gpu(g)
                self.probe("probe_coarse_A0", job, self.run / "A0/probe_coarse", g)
                self.infer("A0_coarse", "val", g)
            while not (self.done("train_A1") and self.done("train_A2")):       # latency is measured before this GPU is handed back
                self.control()
                if any((self.run / "state" / f"train_{c}.failed.json").exists() for c in ("A1", "A2")):
                    raise Held("a training stage failed")
                time.sleep(15)
            if self.claim("efficiency"):
                self.wait_gpu(g)
                self.stage("efficiency", "src.arch_ablate.efficiency", ["--run", self.run, "--out", self.eval / "results/efficiency.json"], g)
            self.status[f"worker_{g}"] = "done"
        except Held as e:
            self.status[f"worker_{g}"] = f"held: {e}"
            self.event(gpu=g, status="held", reason=str(e))
        except Exception:
            self.status[f"worker_{g}"] = "failed"
            self.event(gpu=g, status="failed", traceback=traceback.format_exc()[-1500:])
        finally:
            self.release(g)

    def release(self, g):
        """Hand the GPU back to the crop-aware run if this ablation holds its reservation (user approved the stop on 2026-11-09)."""
        reserved = ROOT / "output/2026-11-09/dinov3_crop_aware_pretrain/control" / f"reserve_gpu_{g}.json"
        if reserved.exists() and self.run.name in json.loads(reserved.read_text()).get("note", ""):
            subprocess.run([PY, "-m", "src.crop_aware_pretrain.ctl", "release", "--gpus", str(g)], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), stdout=subprocess.DEVNULL, check=False)
            self.event(gpu=g, status="crop_aware_reservation_released")

    def pred(self, name, epoch, split):
        if name == "A0_r2":
            return OVAL / "preds" / "r384_cosine" / f"ep{epoch}/probs_{split}.npz"
        return self.eval / "preds" / name / f"ep{epoch}/probs_{split}.npz"

    def final(self):
        root = self.eval

        def parallel(jobs, n):
            with ThreadPoolExecutor(max_workers=n) as pool:
                list(pool.map(lambda f: f(), jobs))

        link = root / "eval_val" / "A0_r2"
        if not link.exists():
            assert (OVAL / "eval_val/r384_cosine/ep10/official_results.json").exists()
            link.symlink_to(OVAL / "eval_val/r384_cosine", target_is_directory=True)
        parallel([lambda n=n, e=e: self.stage(f"official_val_{n}_{e}", "src.pacoattr.official_eval", ["--split", "val", "--pred", self.pred(n, e, "val"), "--out", root / f"eval_val/{n}/ep{e}"], official=True) for n in NEW for e in range(1, 11)], 20)
        self.stage("selection", "src.siglip_pm.select_epoch", [root])
        selection = json.loads((root / "selection.json").read_text())
        assert sorted(selection) == sorted(EVAL_NAMES), sorted(selection)
        existing = json.loads((OVAL / "selection.json").read_text())["r384_cosine"]["epoch"]
        assert selection["A0_r2"]["epoch"] == existing, (selection["A0_r2"]["epoch"], existing)
        frozen = root / "selection_frozen.json"
        if not frozen.exists():
            frozen.write_text(json.dumps(selection, indent=1))
        selection = json.loads(frozen.read_text())
        for n in NEW:
            self.infer(n, "test", self.gpus[n.split("_")[0]] if n.split("_")[0] in self.gpus else 0)
        # official Test AP first, without bootstrap
        parallel([lambda n=n: self.stage(f"fast_test_{n}", "src.pacoattr.official_eval", ["--split", "test", "--pred", self.pred(n, selection[n]["epoch"], "test"), "--out", root / "fast_test" / n], official=True) for n in EVAL_NAMES], 6)
        self.stage("fast_report", "src.arch_ablate.report", ["--run", self.run, "--fast"])
        self.stage("efficiency", "src.arch_ablate.efficiency", ["--run", self.run, "--out", root / "results/efficiency.json"], 0)
        for split in ("val", "test"):
            parallel([lambda n=n: self.stage(f"boot_eval_{split}_{n}", "src.pacoattr.official_eval", ["--split", split, "--pred", self.pred(n, selection[n]["epoch"], split), "--out", root / f"eval_boot/{n}/{split}", "--boot"], official=True) for n in EVAL_NAMES], 10)
            self.stage(f"bootstrap_{split}", "src.g1up.support", ["boot", "--root", root, "--names", ",".join(EVAL_NAMES), "--split", split, "--out", root / f"boot_{split}.json"], official=True)
            parallel([lambda n=n: self.stage(f"size_{split}_{n}", "src.g1up.support", ["size", "--split", split, "--pred", self.pred(n, selection[n]["epoch"], split), "--model", n, "--out", root / f"results/roi_size_{split}_{n}.csv"], official=True) for n in EVAL_NAMES], 10)
        self.stage("report", "src.arch_ablate.report", ["--run", self.run])

    def main(self):
        workers = [threading.Thread(target=self.worker, args=(g, c)) for c, g in self.gpus.items()]
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
    shutil.copytree(ROOT / "src/arch_ablate", run / "source_snapshot/src/arch_ablate", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    Queue(run, dict(A1=0, A2=1)).main()


if __name__ == "__main__":
    main()
