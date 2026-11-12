"""Inference-only Detail ablation on the PACO official Val manifest with the frozen encoder and the official-Val-selected readers (no retraining).

  normal  : Q = base + alpha Proj(detail)     (must reproduce the selected-epoch predictions)
  zero    : Q = base                          (the residual term removed)
  shuffle : content 8 x 8 Detail tiles permuted per image, fixed generator
A changed Detail input also has a changed distribution, so a drop is not a pure effect of native pixel information.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.hier.bbdata as HB
import src.readout.common as RC
from src.hier.data import shuffle_tiles
from src.hier.model import load_hier
from src.hier.probe import encode
from src.hier.val_infer import DATA, load_reader

MODES = ("normal", "zero", "shuffle")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", required=True, choices=("H0", "H1"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root, run, dev = Path(a.root), Path(a.run), torch.device("cuda")
    HB.DETAIL_COND = a.cond
    model, _ = load_hier(run / a.cond / "checkpoints/ema_last.pt", dev)
    selection = json.loads((root / "selection.json").read_text())
    M = pd.read_parquet(os.path.join(DATA, "manifest_val.parquet"))
    specs = [("r2", "r2", "r2fused"), ("linear", "r0", "r0")]
    readers = {key: (selection[f"{a.cond}_{key}"]["epoch"], load_reader(kind, run / f"probe_{a.cond}" / sub / f"ep{selection[f'{a.cond}_{key}']['epoch']}.pt", dev)) for key, kind, sub in specs}
    ds = HB.BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=True)
    logits = {(m, k): torch.full((len(M), 59), float("nan")) for m in MODES for k in readers}
    generator = torch.Generator(device=dev).manual_seed(1234)
    top, bottom = 0., 0.
    for b in dl:
        b = RC.to_dev(b, dev)
        rows = b["rows"].cpu()
        qs = {}
        for mode in MODES:
            batch = dict(b)
            model.detail_off = mode == "zero"
            if mode == "shuffle":
                batch["detail"] = shuffle_tiles(b["detail"], generator)
            q = encode(model, batch, ["Q"])["Q"]
            qs[mode] = q
            model.detail_off = False
            for key, (_, r) in readers.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits[(mode, key)][rows] = r({12: q}, batch)["logits"].float().cpu()
        w = b["cov"].float()[..., None]
        top += float(((qs["normal"] - qs["zero"]).norm(dim=-1, keepdim=True) * w).sum())
        bottom += float((qs["zero"].norm(dim=-1, keepdim=True) * w).sum())
    report = dict(cond=a.cond, epochs={k: e for k, (e, _) in readers.items()}, detail_residual_over_base=top / bottom, shuffle_seed=1234, reproduction={})
    for (mode, key), v in logits.items():
        assert not torch.isnan(v).any(), (mode, key)
        if mode == "normal":
            saved = np.load(root / f"preds/{a.cond}_{key}/ep{readers[key][0]}/probs_val.npz")["logits"]
            report["reproduction"][key] = float(np.abs(v.numpy() - saved).max())
            assert report["reproduction"][key] < .05, report["reproduction"]
            continue
        out = root / "ablation" / a.cond / f"{mode}_{key}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "probs_val.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    (root / "ablation" / a.cond / "ablation_meta.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
