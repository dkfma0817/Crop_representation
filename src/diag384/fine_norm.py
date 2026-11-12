"""D2 implementation check and Fine contribution size, on the frozen A1 encoder. No training.

Checks (stops the queue on failure): every parameter frozen; Coarse-only, Fine-only and Fused ROI features are (R, 576, 768); Fused = Coarse + Fine-only; Fine-only equals s_d Wd(ROIAlign(Fine map)) computed by the Fine-only model path.
Size: per ROI, mean over cells inside the crop content (cov > 0) of the token norms of A, F = s_d Wd(B), Q = A + F; ratios of those means; mean cosine(A, F). Aggregated over the official Val manifest by level and Full384 size bin.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.arch_ablate.model import load_ablation
from src.diag384.bbdata import BBImageDS
from src.diag384.checks import size_bin
from src.diag384.infer import A1, DATA
from src.diag384.model import load_probe_encoder
from src.hcf.probe import encode


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N images only")
    a = ap.parse_args()
    dev, out = torch.device("cuda"), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fused, ck = load_ablation(A1, dev)
    fine, _ = load_probe_encoder(A1, dev, fine_only=True)
    assert ck["kind"] == "ema" and ck["step"] == 12000 and ck["interaction"] == "A1", (ck["kind"], ck["step"])
    assert not any(p.requires_grad for m in (fused, fine) for p in m.parameters()), "encoder not frozen"
    assert float(fused.s_d) == float(fine.s_d) and torch.equal(fused.Wd.weight, fine.Wd.weight), "fine weights differ"
    M = pd.read_parquet(f"{DATA}/manifest_val.parquet")
    if a.limit:
        M = M[M.image_id.isin(set(M.image_id.drop_duplicates().iloc[:a.limit]))].reset_index(drop=True)
    ds = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    bins = np.array([size_bin(w, h, W, H) for w, h, W, H in zip(M.bw, M.bh, M.W, M.H)])
    rows, max_err, dims = [], 0.0, None
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=8, collate_fn=RC.collate_one, pin_memory=True):
        b = RC.to_dev(b, dev)
        keep = b["img_index"]
        q = encode(fused, b, ["Q"])
        b["img_index"] = keep
        f = encode(fine, b, ["Q"])["Q"]
        A, Q = q[12], q["Q"]
        dims = (list(A.shape), list(f.shape), list(Q.shape))
        assert A.shape == f.shape == Q.shape and A.shape[1:] == (576, 768), dims
        max_err = max(max_err, float((A + f - Q).abs().max()))
        mask = (b["cov"] > 0).float()
        n = mask.sum(1).clamp_min(1)
        na, nf, nq = [(x.norm(dim=-1) * mask).sum(1) / n for x in (A, f, Q)]
        cos = (torch.nn.functional.cosine_similarity(A, f, dim=-1) * mask).sum(1) / n
        rows.append(torch.stack([b["rows"].float(), na, nf, nq, cos], 1).cpu())
    assert max_err < 1e-2, max_err
    t = torch.cat(rows).numpy()
    idx = t[:, 0].astype(int)
    df = pd.DataFrame(dict(level=M.level.values[idx], size_bin=bins[idx], norm_coarse=t[:, 1], norm_fine=t[:, 2], norm_fused=t[:, 3], cos_coarse_fine=t[:, 4]))
    table = []
    for level, g0 in df.groupby("level"):
        for name, g in list(g0.groupby("size_bin")) + [("all", g0)]:
            table.append(dict(level=level, size_bin=name, rois=len(g), norm_coarse=g.norm_coarse.mean(), norm_fine=g.norm_fine.mean(), norm_fused=g.norm_fused.mean(), fine_over_coarse=g.norm_fine.mean() / g.norm_coarse.mean(), fine_over_fused=g.norm_fine.mean() / g.norm_fused.mean(), cos_coarse_fine=g.cos_coarse_fine.mean()))
    pd.DataFrame(table).to_csv(out / "fine_size.csv", index=False)
    result = dict(status="PASS", checkpoint=A1, s_d=float(fused.s_d), max_abs_error_coarse_plus_fine_minus_fused=max_err, shapes_coarse_fine_fused=dims, frozen=True, trainable_module="R2 reader only (the probe trains the reader; the encoder parameters have requires_grad False)", rois=len(df))
    (out / "fine_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    print(pd.DataFrame(table).to_string(), flush=True)


if __name__ == "__main__":
    main()
