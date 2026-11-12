"""Parameters, encode GMACs and encode latency for A0 / A1 / A2 (trained EMA checkpoints) on one GPU, plus peak training VRAM from the training logs.
Latency: bf16, one 384 x 384 image, torch.cuda.synchronize, 10 warm-up + 50 measured repeats, median. gpu_idle_at_start records whether another process was on the GPU.

  python -m src.arch_ablate.efficiency --run <run> --out <run>/eval/results/efficiency.json
"""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from src.arch_ablate.model import load_ablation
from src.arch_ablate.official_infer import A0_CKPT
from src.hcf.efficiency import flops, timed

A0_LOG = "output/2026-11-08/hcf_material_causal/r384_cosine/P/logs/train.jsonl"


def other_processes():
    return [x.strip() for x in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).splitlines() if x.strip()]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = dict(gpu_processes_at_start=other_processes(), note="latency is only comparable if no other process shared the GPU")
    x = torch.randn(1, 3, 384, 384, device=dev)
    for arch in ("A0", "A1", "A2"):
        path = A0_CKPT if arch == "A0" else Path(a.run) / arch / "checkpoints/ema_last.pt"
        m, ck = load_ablation(path, dev)
        assert ck.get("interaction", "A0") == arch
        n_new = sum(p.numel() for _, p in m.new_named())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            macs = flops(m, x)
            ms = timed(lambda: m.forward_all(x))
        log = Path(A0_LOG) if arch == "A0" else Path(a.run) / arch / "logs/train.jsonl"
        peak = max(json.loads(line)["mem_gb"] for line in log.read_text().splitlines())
        out[arch] = dict(params_total=sum(p.numel() for p in m.parameters()), params_coarse_vit=sum(p.numel() for p in m.vit.parameters()), params_new=n_new, params_interaction=sum(p.numel() for p in m.inter.parameters()), encode_gmacs=macs, encode_ms_median=ms, train_peak_gib=peak)
        del m
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
