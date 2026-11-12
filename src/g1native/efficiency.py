"""Encoder parameters, GMACs and encode latency for G1U and G1-Native (same architecture; latency measured on this GPU), plus peak training VRAM from the logs.

  python -m src.g1native.efficiency --run <run> --out <run>/eval/results/efficiency.json
"""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from src.hcf.efficiency import flops, timed
from src.hcf.model import load_hcf

G1 = Path("/home/irteam/Crop_representation/output/2026-11-09/g1_upscaled_teacher_s0")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = dict(gpu_processes_at_start=[x.strip() for x in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).splitlines() if x.strip()])
    x = torch.randn(1, 3, 384, 384, device=dev)
    for name, path in (("G1U", G1 / "G1U"), ("G1N", Path(a.run) / "G1N")):
        m, ck = load_hcf(path / "checkpoints/ema_last.pt", dev)
        assert ck["kind"] == "ema" and ck["step"] == 12000
        with torch.autocast("cuda", dtype=torch.bfloat16):
            macs = flops(m, x)
            ms = timed(lambda: m.forward_all(x))
        peak = max(json.loads(line)["mem_gb"] for line in (path / "logs/train.jsonl").read_text().splitlines())
        out[name] = dict(params_total=sum(p.numel() for p in m.parameters()), encode_gmacs=macs, encode_ms_median=ms, train_peak_gib=peak)
        del m
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
