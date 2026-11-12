"""J1 (GPU 0), W1 weight mixing + P-LowFlow (GPU 1), with one evaluation path for every candidate and baseline. Started once with setsid; flock; finished stages are skipped on restart; training resumes from recovery.pt.

Training threads: GPU 0 J1 smoke -> 12k; GPU 1 W1 mix (CPU, compatibility checks) -> LowFlow smoke -> 12k.
Evaluation tasks run beside training (user 2026-11-10: R2 may share a GPU with training), two at a time, GPU 1 then GPU 0:
  R2 probe (src.g1up.train_bb, the E2-L / overnight path) + official Val inference (src.overnight6.official_infer) for
    W1a025, W1a050, W1a075, W1a000 (= PixelDiT Hybrid encoder), W1a100 (= E2-L encoder), A0bb (= A0 encoder), then J1, LowFlow when trained.
  CPU official Val (10 epochs) -> epoch frozen on Val -> one Test inference -> CPU official Test.
  W1: the alpha among 0.25 / 0.5 / 0.75 with the highest official Val (Object + Part) / 2 at its own best epoch; Test only for that alpha (+ the two endpoints, as endpoint verification).
Baseline audit: the EXISTING A0 / Hybrid / E2-L R2 readers re-inferred through the same inference path at their frozen epochs, official Val + Test, compared with the stored numbers.
End: paired bootstrap (B = 1000), ROI size, report.

  setsid nohup /home/irteam/envs/dinov2_ord_py310/bin/python -u -m src.jointpg.queue --run results/joint_pixeldit_g1 &
"""
import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import yaml

import src.overnight6.queue as Q

ROOT = Q.ROOT
HCF = ROOT / "output/2026-11-08/hcf_material_causal"
OVAL = ROOT / "output/2026-11-09/hcf_official_val"
G1 = ROOT / "output/2026-11-09/g1_upscaled_teacher_s0/eval"
ABL = ROOT / "output/2026-11-09/coarse_fine_ablation_s0/eval"
SIX = ROOT / "results/overnight_six"
TRAIN = "src.jointpg.train"
ALPHAS = ("W1a025", "W1a050", "W1a075")
PROBES = ["W1a025", "W1a050", "W1a075", "W1a000", "W1a100", "A0bb"]
TESTED = ["A0bb", "W1a000", "W1a100", "J1", "LowFlow"]
AUDIT = dict(A0old=dict(ckpt=HCF / "r384_cosine/P/checkpoints/ema_last.pt", probe=HCF / "r384_cosine/probe/r2fused", epoch=7, val=OVAL / "eval_val/r384_cosine/ep7/official_results.json", test=ABL / "fast_test/A0_r2/official_results.json"),
             Hybridold=dict(ckpt=HCF / "r384_pixeldit_hybrid/P/checkpoints/ema_last.pt", probe=HCF / "r384_pixeldit_hybrid/probe/r2fused", epoch=7, val=OVAL / "eval_val/r384_pixeldit_hybrid/ep7/official_results.json", test=G1 / "eval_boot/Hybrid_r2/test/official_results.json"),
             E2Lold=dict(ckpt=SIX / "E2L/checkpoints/ema_last.pt", probe=SIX / "E2L/probe/r2fused", epoch=8, val=SIX / "eval/eval_val/E2L_r2/ep8/official_results.json", test=SIX / "eval/fast_test/E2L_r2/official_results.json"))
Q.CONDS = dict(J1="PixelDiT Hybrid + 0.20 G1U grid", LowFlow="PixelDiT Hybrid, lambda_flow 0.25", W1="weight mix of PixelDiT Hybrid and E2-L EMA encoders, alpha 0.25 / 0.5 / 0.75 (+ endpoints)", A0bb="A0 encoder, R2 retrained on the shared path")
Q.NAMES = []
Q.BASE = dict(E2L_r2=dict(val_dir=SIX / "eval/eval_val/E2L_r2", test=SIX / "eval/fast_test/E2L_r2", boot=SIX / "eval/eval_boot/E2L_r2", preds=SIX / "eval/preds/E2L_r2", selection=json.loads((SIX / "eval/selection/E2L_r2.json").read_text())))


class Queue(Q.Queue):
    REPORT = "src.jointpg.report"

    def __init__(self, run):
        super().__init__(run)
        self.gpu_tasks = deque()
        self.need = set(PROBES + ["J1", "LowFlow"])          # names whose evaluation must finish
        self.alpha_choice = None

    def push(self, task, first=False):
        with self.lock:
            (self.tasks.appendleft if first else self.tasks.append)(task)

    # ---------------------------------------------------------------- training
    def train_thread(self, g, name):
        try:
            self.gate(g)
            self.stage(f"smoke_{name}", TRAIN, ["--condition", name, "--out", self.run / name / "smoke", "--smoke"], g)
            self.stage(f"train_{name}", TRAIN, ["--condition", name, "--out", self.run / name], g, retries=1)
            assert (self.run / name / "done.json").exists()
            shutil.copy(self.run / name / "config.json", self.run / "configs" / f"{name}.json")
            self.push(("probe", name))
        except Exception:
            self.fail(name)

    def prepare(self):
        self.stage("w1_mix", TRAIN, ["--mix", "--out", self.run])
        a0 = self.run / "A0bb" / "checkpoints"
        a0.mkdir(parents=True, exist_ok=True)
        if not (a0 / "ema_last.pt").exists():
            (a0 / "ema_last.pt").symlink_to(AUDIT["A0old"]["ckpt"])
            (self.run / "A0bb" / "done.json").write_text(json.dumps(dict(status="done", kind="existing A0 r384_cosine EMA encoder (symlink); only the R2 reader is retrained"), indent=1))
        for n in PROBES:
            self.push(("probe", n))
        for n in AUDIT:
            self.push(("audit", n), first=True)

    def fail(self, name):
        with self.lock:
            self.failed[name] = traceback.format_exc()[-1500:]
        self.event(name=name, status="failed", traceback=self.failed[name])

    # ---------------------------------------------------------------- evaluation
    def run_probe(self, name, g):
        ck = self.run / name / "checkpoints/ema_last.pt"
        assert (self.run / name / "done.json").exists() and ck.exists(), name
        job = self.run / "configs" / f"probe_{name}.yaml"
        job.write_text(yaml.safe_dump(dict(name=f"jointpg_{name}", backbone="dinov3", mode="hcf_fused", encoder_ckpt=str(ck), full_size=384, seed=0, epochs=10, paco_dev_images=1000, readers=[dict(name="r2fused", kind="r2", layers=[12])]), sort_keys=False))
        out = self.run / name / "probe"
        if not self.done(f"probe_{name}") and out.exists() and not (out / "done.json").exists():
            out.rename(out.with_name(f"{out.name}.partial_{int(time.time())}"))
        self.stage(f"probe_{name}", "src.g1up.train_bb", ["--job", job, "--out", out, "--exclude", Q.EXCLUDE, "--workers", 8], g, retries=1)
        self.stage(f"val_infer_{name}", "src.overnight6.official_infer", ["--run", self.run, "--cond", name, "--split", "val"], g, retries=1)
        self.spawn(self.cpu_val, name)

    def cpu_val(self, name):
        n = f"{name}_r2"
        Q.parallel([lambda e=e: self.official(f"official_val_{n}_{e}", self.pred(n, e, "val"), self.eval / f"eval_val/{n}/ep{e}") for e in range(1, 11)], 5)
        path = self.eval / "selection" / f"{n}.json"
        if not path.exists():
            c = {e: json.loads((self.eval / f"eval_val/{n}/ep{e}/official_results.json").read_text())["hierarchical"] for e in range(1, 11)}
            score = {e: (h["obj-attr-AP"] + h["obj-attr-part-AP"]) / 2 for e, h in c.items()}
            epoch = max(score, key=lambda e: (round(score[e], 10), -e))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(epoch=epoch, val_obj=c[epoch]["obj-attr-AP"], val_part=c[epoch]["obj-attr-part-AP"], rule="max (official VAL object AP + part AP)/2; earlier epoch on ties", curve={str(e): s for e, s in score.items()}), indent=1))
            tmp.replace(path)
        if name in TESTED:
            self.push(("test", name), first=True)
        elif name in ALPHAS:
            with self.lock:
                ready = all((self.eval / "selection" / f"{a}_r2.json").exists() for a in ALPHAS) and self.alpha_choice is None
                if ready:
                    sel = {a: json.loads((self.eval / "selection" / f"{a}_r2.json").read_text()) for a in ALPHAS}
                    best = max(ALPHAS, key=lambda a: ((sel[a]["val_obj"] + sel[a]["val_part"]) / 2, -ALPHAS.index(a)))
                    self.alpha_choice = best
            if ready:
                choice = self.run / "w1_alpha_choice.json"
                if not choice.exists():
                    choice.write_text(json.dumps(dict(chosen=best, rule="highest official Val (Object + Part) / 2 at each alpha's own best epoch; ties -> smaller alpha; fixed before any W1 Test", val={a: dict(epoch=sel[a]["epoch"], overall=(sel[a]["val_obj"] + sel[a]["val_part"]) / 2) for a in ALPHAS}), indent=1))
                best = json.loads(choice.read_text())["chosen"]
                self.event(status="w1_alpha_chosen", chosen=best)
                for a in ALPHAS:
                    if a != best:
                        with self.lock:
                            self.complete.add(a)
                self.push(("test", best), first=True)
        self.fast_report()

    def run_test(self, name, g):
        self.stage(f"test_infer_{name}", "src.overnight6.official_infer", ["--run", self.run, "--cond", name, "--split", "test"], g, retries=1)
        self.spawn(self.cpu_test, name)

    def run_audit(self, name, g):
        spec = AUDIT[name]
        n = f"{name}_r2"
        (self.eval / "selection" / f"{n}.json").write_text(json.dumps(dict(epoch=spec["epoch"], rule="existing frozen epoch of this checkpoint (audit only)"), indent=1))
        self.stage(f"audit_val_infer_{name}", "src.overnight6.official_infer", ["--run", self.run, "--cond", name, "--split", "val", "--ckpt", spec["ckpt"], "--probe", spec["probe"], "--epochs", f"1,{spec['epoch']},10", "--soft-check"], g)
        self.stage(f"audit_test_infer_{name}", "src.overnight6.official_infer", ["--run", self.run, "--cond", name, "--split", "test", "--ckpt", spec["ckpt"], "--probe", spec["probe"]], g)
        self.spawn(self.cpu_audit, name)

    def cpu_audit(self, name):
        spec, n = AUDIT[name], f"{name}_r2"
        self.official(f"audit_official_val_{n}", self.pred(n, spec["epoch"], "val"), self.eval / f"eval_val/{n}/ep{spec['epoch']}")
        self.official(f"audit_official_test_{n}", self.pred(n, spec["epoch"], "test"), self.eval / f"fast_test/{n}")
        rows = {}
        for split, new, old in (("val", self.eval / f"eval_val/{n}/ep{spec['epoch']}/official_results.json", spec["val"]), ("test", self.eval / f"fast_test/{n}/official_results.json", spec["test"])):
            hn, ho = json.loads(Path(new).read_text())["hierarchical"], json.loads(Path(old).read_text())["hierarchical"]
            rows[split] = dict(new_obj=hn["obj-attr-AP"], old_obj=ho["obj-attr-AP"], new_part=hn["obj-attr-part-AP"], old_part=ho["obj-attr-part-AP"], max_abs_diff=max(abs(hn["obj-attr-AP"] - ho["obj-attr-AP"]), abs(hn["obj-attr-part-AP"] - ho["obj-attr-part-AP"]), *[abs(hn[l][k] - ho[l][k]) for l in ("obj-attrs", "obj-part-attrs") for k in hn[l]]))
        verify = json.loads((self.eval / "verify" / f"{name}_val.json").read_text())
        out = dict(name=name, ckpt=str(spec["ckpt"]), probe=str(spec["probe"]), epoch=spec["epoch"], reproduction_pearson_vs_probe_scores=verify.get("reproduction_pearson"), official=rows, stored_val=str(spec["val"]), stored_test=str(spec["test"]))
        (self.run / "verification").mkdir(exist_ok=True)
        (self.run / "verification" / f"baseline_audit_{name}.json").write_text(json.dumps(out, indent=1))
        self.event(status="audit_done", name=name, val_diff=rows["val"]["max_abs_diff"], test_diff=rows["test"]["max_abs_diff"])

    def run_task(self, task, g):
        kind, name = task
        try:
            self.gate(g)
            dict(probe=self.run_probe, test=self.run_test, audit=self.run_audit)[kind](name, g)
        except Exception:
            self.fail(name)

    def eval_worker(self, g):
        while True:
            if self.control():
                return
            with self.lock:
                task = self.tasks.popleft() if self.tasks else None
            if task is None:
                if self.finished_all():
                    return
                time.sleep(10)
                continue
            self.status[f"eval_gpu{g}"] = task
            self.write_status()
            self.run_task(task, g)

    def finished_all(self):
        with self.lock:
            trained = all((self.run / n / "done.json").exists() or n in self.failed for n in ("J1", "LowFlow"))
            return trained and not self.tasks and all(n in self.complete or n in self.failed for n in self.need) and all(not t.is_alive() for t in self.cpu_threads if t is not threading.current_thread())

    def main(self):
        self.prepare()
        trains = [threading.Thread(target=self.train_thread, args=(0, "J1")), threading.Thread(target=self.train_thread, args=(1, "LowFlow"))]
        evals = [threading.Thread(target=self.eval_worker, args=(g,)) for g in (1, 0)]
        for t in trains + evals:
            t.start()
        for t in trains + evals:
            t.join()
        for t in list(self.cpu_threads):
            t.join()
        self.write_status()
        if self.control():
            return
        try:
            self.final()
            (self.run / "done.json").write_text(json.dumps(dict(status="done", completed_utc=time.strftime("%FT%TZ", time.gmtime()), complete=sorted(self.complete), failed=sorted(self.failed), w1_choice=self.alpha_choice), indent=1))
        except Exception:
            (self.run / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc()), indent=1))
            raise

    def final(self):
        choice = json.loads((self.run / "w1_alpha_choice.json").read_text())["chosen"] if (self.run / "w1_alpha_choice.json").exists() else None
        new = [f"{n}_r2" for n in TESTED + ([choice] if choice else []) if n in self.complete]
        names = new + ["E2L_r2"]
        epochs = {n: json.loads((self.eval / "selection" / f"{n}.json").read_text())["epoch"] for n in names}
        for split in Q.SPLITS:
            Q.parallel([lambda n=n: self.official(f"boot_eval_{split}_{n}", self.pred(n, epochs[n], split), self.eval / f"eval_boot/{n}/{split}", boot=True) for n in new], 7)
            self.stage(f"bootstrap_{split}", "src.g1up.support", ["boot", "--root", self.eval, "--names", ",".join(names), "--split", split, "--out", self.eval / f"boot_{split}.json"], official=True)
            Q.parallel([lambda n=n: self.stage(f"size_{split}_{n}", "src.g1up.support", ["size", "--split", split, "--pred", self.pred(n, epochs[n], split), "--model", n, "--out", self.eval / f"results/roi_size_{split}_{n}.csv"], official=True) for n in names], 8)
        self.stage("report", self.REPORT, ["--run", self.run])


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
    for pkg in ("jointpg", "overnight6", "pdit", "g1up"):
        shutil.copytree(ROOT / "src" / pkg, run / "source_snapshot/src" / pkg, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    Queue(run).main()


if __name__ == "__main__":
    main()
