"""Cost of the Detail branch (encoder MACs, parameters, latency, peak VRAM, token counts) and the per-image encode counts vs ROI count."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import src.hier.bbdata as HB
import src.readout.common as RC
from src.hcf.efficiency import flops
from src.hcf.model import load_hcf, n_params
from src.hier.model import load_hier
from src.hier.probe import encode

P384 = "output/2026-11-08/hcf_material_causal/r384_cosine/P/checkpoints/ema_last.pt"


class Wrap(nn.Module):
    def __init__(self, m, detail):
        super().__init__()
        self.m, self.detail = m, detail

    def forward_all(self, x):
        return self.m.forward_all(x, self.detail)


def timed(fn, reps=60, warm=10):
    ts = []
    for i in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if i >= warm:
            ts.append(1e3 * (time.perf_counter() - t))
    return float(np.median(ts))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run, dev = Path(a.run), torch.device("cuda")
    HB.DETAIL_COND = "H1"
    M = pd.read_parquet("output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet")
    iid = M.groupby("image_id").size().loc[lambda s: s >= 20].index[0]
    df = M[M.image_id == iid].iloc[:20].reset_index(drop=True)
    b0 = RC.to_dev(HB.BBImageDS(df, np.zeros((len(df), 59), np.int8), False, "bil", 384, 16)[[0]], dev)
    x, detail = b0["x"], b0["detail"]
    p384 = load_hcf(P384, dev)[0]
    out = {}
    for name in ("P384", "H0", "H1"):
        model = p384 if name == "P384" else load_hier(run / name / "checkpoints/ema_last.pt", dev)[0]
        call = (lambda: model.forward_all(x)) if name == "P384" else (lambda: model.forward_all(x, detail))

        def encode_once():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                call()
        target = model if name == "P384" else Wrap(model, detail)
        row = dict(gflops=flops(target, x), encode_ms_1image=timed(encode_once), parameters=n_params(model))
        batch_x = x.repeat(16, 1, 1, 1)
        batch_detail = detail.repeat(16, 1, 1, 1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.forward_all(batch_x) if name == "P384" else model.forward_all(batch_x, batch_detail)
        row["peak_vram_gib_batch16_inference"] = torch.cuda.max_memory_allocated() / 2 ** 30
        if name != "P384":
            row["detail_parameters"] = sum(p.numel() for _, p in model.detail_named())
            row["shared_tokens"] = dict(coarse=576, fine_48x48=2304, detail_80x80_before_alignment=6400, detail_48x48_stored=2304)
            row["stored_scalars"] = 24 * 24 * 768 + 48 * 48 * 384 + 48 * 48 * 192
            calls = {}
            for k in (1, 5, 10, 20):
                c = [0]
                batch = {key: (v[:k] if torch.is_tensor(v) and key in ("geo", "box4", "cov", "lv", "bil_grid", "img_index", "rows") else v) for key, v in b0.items()}
                batch["img_index"] = torch.zeros(k, dtype=torch.long, device=dev)
                encode(model, batch, ["Q"], count=c)
                calls[k] = c[0]

                def read():
                    bb = dict(batch)
                    bb["img_index"] = torch.zeros(k, dtype=torch.long, device=dev)
                    return encode(model, bb, ["Q"])
                row[f"encode_plus_read_ms_k{k}"] = timed(read, reps=25, warm=5)
            row["images_encoded_vs_roi_count"] = calls
            assert all(v == 1 for v in calls.values()), calls
        else:
            row["stored_scalars"] = 24 * 24 * 768 + 48 * 48 * 384
        out[name] = row
    (run / "results").mkdir(exist_ok=True)
    (run / "results/efficiency.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
