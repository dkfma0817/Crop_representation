"""ONE-TIME PACO test scoring of the fixed IVC checkpoints (user-approved 2026-11-05). No training, no selection.

Checkpoints: output/2026-11-05/ivc/probe_<cond>/<reader>/ep<k>.pt with k = the dev-selected epoch recorded in
analysis.json (Bilinear R0 10 / R2 6, Bicubic R0 10 / R2 6, C0 R0 10 / R2 5). Features: the exact training-time path
(dec mode: src.msd.common.Student + src.ivc.common.sample_roi on the C0 crop grid; c0 mode: PacoImageDS crops +
src.readout.common.encode_layers), no flip, readers under bf16 autocast. Before test, the same code re-scores VAL and
must reproduce the saved scores_val_ep<k>.npy (max |diff| reported, asserted < 0.05 logits).

  CUDA_VISIBLE_DEVICES=2 python -m src.ivc.test_eval --cond bil,bic --out output/2026-11-05/ivc/test
"""
import argparse
import json
import os

os.environ["READOUT_ALLOW_TEST"] = "1"          # this script only

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.msd.common as MS
import src.readout.common as RC
from src.ivc.common import EXCL, R0, sample_roi
from src.readout.models import Reader

RUN = "output/2026-11-05/ivc"


def load_readers(cond, sel, dev):
    out = {}
    for rd in ("r0", "r2"):
        ep = sel[f"{cond}/{rd}"]["epoch"]
        ck = torch.load(os.path.join(RUN, f"probe_{cond}", rd, f"ep{ep}.pt"), map_location="cpu")
        assert ck["epoch"] == ep
        m = R0() if rd == "r0" else Reader("r2", layers=(12,))
        m.load_state_dict(ck["reader"])
        out[rd] = (m.to(dev).eval(), ep)
    return out


@torch.no_grad()
def score(df, Y, conds, readers, dev, workers):
    mode = "c0" if conds == ["c0"] else "dec"
    if mode == "dec":
        ds = MS.MSDImageDS(df, Y, flip=False, want_crops=False)
        student = MS.Student(adapters=False).to(dev).eval()
    else:
        ds = RC.PacoImageDS(df, Y, flip=False, mode="c0")
        vit = RC.load_vit().to(dev)
    bs = 12 if mode == "dec" else 6
    chunks = [list(range(i, min(len(ds), i + bs))) for i in range(0, len(ds), bs)]
    dl = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=workers, collate_fn=RC.collate_one, pin_memory=True)
    S = {(c, rd): np.zeros((len(df), 59), np.float32) for c in conds for rd in ("r0", "r2")}
    seen = np.zeros(len(df), bool)
    for b in dl:
        if mode == "dec":
            b = MS.to_dev(b, dev)
            feats, _ = student(b["full"])
            b["geo"], b["box4"], b["cov"] = b["geo_c"], b["box4_c"], b["cov_c"]
            b["side"] = 336
            Zs = {c: sample_roi(feats[12], b["geom"], {"bil": "bilinear", "bic": "bicubic"}[c]).float() for c in conds}
            b["img_index"] = torch.arange(len(b["rows"]), device=dev)
        else:
            b = RC.to_dev(b, dev)
            x = b["x"]
            Zs = {"c0": torch.cat([RC.encode_layers(vit, x[i:i + 32], (12,))[12] for i in range(0, len(x), 32)])}
        rows = b["rows"].cpu().numpy()
        for c in conds:
            f = {12: Zs[c]}
            for rd, (m, _) in readers[c].items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = m(f, b)
                S[(c, rd)][rows] = o["logits"].float().cpu().numpy()
        seen[rows] = True
    assert seen.all()
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, help="bil,bic | c0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    conds = a.cond.split(",")
    sel = json.load(open(os.path.join(RUN, "analysis.json")))["performance"]
    readers = {c: load_readers(c, sel, dev) for c in conds}
    data = RC.load_data(EXCL, splits=("train", "val", "test"))
    assert not (set(data["train"][0].image_id) & set(data["test"][0].image_id)), "train / test image overlap"
    # 1. val re-score must match the saved val scores of the same epoch
    dva, Yva = data["val"]
    Sv = score(dva, Yva, conds, readers, dev, a.workers)
    chk = {}
    for (c, rd), s in Sv.items():
        ref = np.load(os.path.join(RUN, f"probe_{c}", rd, f"scores_val_ep{readers[c][rd][1]}.npy"))
        chk[f"{c}/{rd}"] = dict(epoch=readers[c][rd][1], max_abs_diff=float(np.abs(s - ref).max()), mean_abs_diff=float(np.abs(s - ref).mean()))
    print(json.dumps(chk, indent=1), flush=True)
    json.dump(chk, open(os.path.join(a.out, f"val_recheck_{'_'.join(conds)}.json"), "w"), indent=1)
    assert all(v["max_abs_diff"] < 0.05 for v in chk.values()), "val re-score does not reproduce the saved scores"
    # 2. test, once
    dte, Yte = data["test"]
    St = score(dte, Yte, conds, readers, dev, a.workers)
    for (c, rd), s in St.items():
        np.save(os.path.join(a.out, f"scores_test_{c}_{rd}.npy"), s)
    json.dump(dict(conds=conds, n_test_rows=len(dte), n_test_images=int(dte.image_id.nunique()),
                   epochs={f"{c}/{rd}": readers[c][rd][1] for c in conds for rd in ("r0", "r2")}),
              open(os.path.join(a.out, f"done_{'_'.join(conds)}.json"), "w"), indent=1)
    print("TEST SCORED", conds, len(dte), flush=True)


if __name__ == "__main__":
    main()
