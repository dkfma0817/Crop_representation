"""PC768 (GPU 0) and M1 (GPU 1): the src.overnight6 queue (independent encoder and evaluation tasks, flock, restart-safe stages) with these two experiments and their baselines.

Before training: PC768 checks the Crop768 cache against the PixelDiT Hybrid data (src.pc768mse.train --verify-teacher); M1 fixes lambda_mse (--calibrate-mse). Then smoke -> 12k training -> R2 probe ->
official Val -> epoch frozen on Val -> one Test inference -> official Test; bootstrap and ROI size at the end. R2 encoding uses the GPU once that GPU's training has finished (CPU is about 4 img/s).

  setsid nohup /home/irteam/envs/dinov2_ord_py310/bin/python -u -m src.pc768mse.queue --run results/pixeldit_crop768_vs_mse &
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

import src.overnight6.queue as Q
from src.pc768mse.train import CONDS

ROOT = Q.ROOT
OVAL = ROOT / "output/2026-11-09/hcf_official_val"
G1 = ROOT / "output/2026-11-09/g1_upscaled_teacher_s0/eval"
SIX = ROOT / "results/overnight_six/eval"
Q.CONDS = CONDS
Q.NAMES = list(CONDS)
Q.ORDER = {0: ["PC768"], 1: ["M1"]}
Q.BASE = dict(
    Hybrid_r2=dict(val_dir=OVAL / "eval_val/r384_pixeldit_hybrid", test=G1 / "eval_boot/Hybrid_r2/test", boot=G1 / "eval_boot/Hybrid_r2", preds=OVAL / "preds/r384_pixeldit_hybrid", selection=json.loads((OVAL / "selection.json").read_text())["r384_pixeldit_hybrid"]),
    A0_r2=Q.BASE["A0_r2"],
    E1a_r2=dict(val_dir=SIX / "eval_val/E1a_r2", test=SIX / "fast_test/E1a_r2", boot=SIX / "eval_boot/E1a_r2", preds=SIX / "preds/E1a_r2", selection=json.loads((SIX / "selection/E1a_r2.json").read_text())))


class Queue(Q.Queue):
    TRAIN, REPORT = "src.pc768mse.train", "src.pc768mse.report"

    def train_args(self, name):
        return ["--condition", name] + (["--lambda-mse-file", self.run / "lambda_mse.json"] if name == "M1" else [])

    def run_train(self, name, g):
        self.gate(g)
        if name == "PC768":
            self.stage("verify_teacher", self.TRAIN, ["--verify-teacher", "--out", self.run / "verification/teacher768_vs_pixeldit_data.json"], g)
        else:
            self.stage("lambda_mse", self.TRAIN, ["--calibrate-mse", "--out", self.run / "lambda_mse.json"], g)
        super().run_train(name, g)


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
    for pkg in ("pc768mse", "overnight6", "pdit"):
        shutil.copytree(ROOT / "src" / pkg, run / "source_snapshot/src" / pkg, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    Queue(run).main()


if __name__ == "__main__":
    main()
