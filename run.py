"""런처: output 폴더를 만들고, 설정을 저장하고, 학습/평가를 실행해 로그를 output_dir/train.log 에 남긴다.

  output_dir 기본값 = output/<YYYY-MM-DD>/<run>   (run = YAML 의 run 값, 없으면 config 파일 이름)
  output_dir/config.yaml  : _base_ 를 펼친 최종 YAML (CLI 로 넘긴 덮어쓰기 인자는 cmd.txt 에)
  output_dir/cmd.txt      : 실제 실행한 명령
  output_dir/train.log    : 표준출력/에러 (foreground 는 콘솔에도 같이)

usage:
  python run.py --config configs/raev2_crop_recon/train/B.yaml                       # GPU 1개 (CUDA_VISIBLE_DEVICES 로 지정)
  CUDA_VISIBLE_DEVICES=2,3 python run.py --config configs/raev2_crop_recon/train/B.yaml --gpus 2 --bg
  python run.py --config configs/raev2_crop_recon/train/C.yaml --output_dir output/2026-11-05/C_lam0.2 --lam 0.2
나머지 인자(--steps, --lam, ...)는 src.engine.train 으로 그대로 전달된다.
"""
from __future__ import annotations

import argparse
import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import OUTPUT  # noqa: E402
from src.utils.yaml_config import load_yaml  # noqa: E402


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", "-c", required=True)
    pre.add_argument("--output_dir", default=None, help="기본: output/<YYYY-MM-DD>/<run>")
    pre.add_argument("--gpus", type=int, default=1, help=">1 이면 torchrun --nproc_per_node N (CUDA_VISIBLE_DEVICES 로 GPU 지정)")
    pre.add_argument("--bg", action="store_true", help="백그라운드(detached) 실행, 로그는 파일에만")
    pre.add_argument("--runner", default="src.engine.train", help="python -m <runner>")
    pre.add_argument("--log-name", default="train.log")
    known, rest = pre.parse_known_args()

    cfg = load_yaml(known.config)
    run = cfg.get("run") or Path(known.config).stem
    out_dir = Path(known.output_dir or os.path.join(OUTPUT, datetime.date.today().isoformat(), run))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(dict(cfg, out=str(out_dir), source_config=known.config), f, sort_keys=False, allow_unicode=True)

    if known.gpus > 1:
        cmd = [str(Path(sys.executable).with_name("torchrun")), "--standalone", "--nproc_per_node", str(known.gpus), "-m", known.runner]
    else:
        cmd = [sys.executable, "-m", known.runner]
    cmd += ["--config", known.config, "--out", str(out_dir)] + rest
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env.setdefault("OMP_NUM_THREADS", "16")              # torchrun would set 1, which starves the CPU side of the step
    (out_dir / "cmd.txt").write_text(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')} {shlex.join(cmd)}\n")
    log_path = out_dir / known.log_name
    print(f"[run] cmd : {shlex.join(cmd)}")
    print(f"[run] out : {out_dir}")
    print(f"[run] log : {log_path}")

    if known.bg:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log_path, "a"), stderr=subprocess.STDOUT, start_new_session=True)
        print(f"[run] background pid={p.pid}  (tail -f {log_path})")
        return 0
    with open(log_path, "a") as logf:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            logf.write(line)
            logf.flush()
        p.wait()
    print(f"[run] done (exit {p.returncode})  log={log_path}")
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
