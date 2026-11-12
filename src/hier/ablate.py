"""Inference-only Detail ablation on PACO Val with the frozen encoder and the dev-M0-selected readers (no retraining).

  normal  : Q = base + alpha Proj(detail)
  zero    : Q = base                                  (the residual term removed)
  shuffle : normal path, content 8 x 8 Detail tiles permuted per image (fixed generator)
Changing the Detail input also changes its distribution, so a drop is not a pure effect of native pixel information.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.hier.bbdata as HB
import src.readout.common as RC
from src.hier.data import shuffle_tiles
from src.hier.model import load_hier
from src.hier.probe import encode
from src.ivc.common import EXCL, R0
from src.readout.models import Reader

MODES = ("normal", "zero", "shuffle")


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def dev_epoch(directory):
    curve = [r for r in lines(directory / "curves_dev.jsonl") if r["epoch"] >= 1]
    return max(curve, key=lambda r: (r["dev"]["M0"]["overall"], -r["epoch"]))["epoch"]


def reader(kind, path, dev):
    r = R0().to(dev) if kind == "r0" else Reader("r2", layers=(12,)).to(dev)
    r.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["reader"])
    return r.eval().requires_grad_(False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", required=True, choices=("H0", "H1"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run, dev = Path(a.run), torch.device("cuda")
    HB.DETAIL_COND = a.cond
    probe = run / f"probe_{a.cond}"
    model, _ = load_hier(run / a.cond / "checkpoints/ema_last.pt", dev)
    data = RC.load_data(EXCL)
    df, Y = data["val"]
    readers = {}
    for name, kind in (("r0", "r0"), ("r2fused", "r2")):
        readers[name] = (dev_epoch(probe / name), kind)
    readers = {name: (ep, reader(kind, probe / name / f"ep{ep}.pt", dev)) for name, (ep, kind) in readers.items()}
    ds = HB.BBImageDS(df, Y, False, "bil", 384, 16)
    idx = np.arange(len(ds))
    chunks = [[int(i) for i in ch] for ch in RC.EvalSampler(len(idx), 12, 0, 1)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
    scores = {(m, n): np.zeros((len(df), 59), np.float32) for m in MODES for n in readers}
    generator = torch.Generator(device=dev).manual_seed(1234)
    ratio_n, ratio_d = 0., 0.
    for b in dl:
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu().numpy()
        qs = {}
        for mode in MODES:
            batch = dict(b)
            model.detail_off = mode == "zero"
            if mode == "shuffle":
                batch["detail"] = shuffle_tiles(b["detail"], generator)
            f = encode(model, batch, ["Q"])
            qs[mode] = f["Q"]
            model.detail_off = False
            for n, (_, r) in readers.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    scores[(mode, n)][rows] = r({12: f["Q"]}, batch)["logits"].float().cpu().numpy()
        w = b["cov"].float()[..., None]
        ratio_n += float(((qs["normal"] - qs["zero"]).norm(dim=-1, keepdim=True) * w).sum())
        ratio_d += float((qs["zero"].norm(dim=-1, keepdim=True) * w).sum())
    out = run / "ablation" / a.cond
    out.mkdir(parents=True, exist_ok=True)
    ytr = RC.ytr_by_level(data)
    report = dict(cond=a.cond, epochs={n: e for n, (e, _) in readers.items()}, mean_detail_residual_over_base_val=ratio_n / ratio_d, shuffle_seed=1234, modes=list(MODES))
    for (mode, name), value in scores.items():
        np.save(out / f"{mode}_{name}.npy", value)
        report[f"{mode}/{name}"] = RC.m2_eval(value, df, Y, ytr)["M2"]
    for name, (ep, _) in readers.items():
        saved = np.load(probe / name / f"scores_val_ep{ep}.npy").astype(np.float32)
        report[f"normal_reproduces_probe/{name}"] = float(np.abs(scores[("normal", name)] - saved).max())
    (out / "ablation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
