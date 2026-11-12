"""VAW frozen-encoder R2 probe (2026-11-07). One condition per process, ONE GPU (world 1; the 4 ROI-balanced shards of
every global step are processed together = same global batches as the 2-GPU PACO probes).

Condition = (encoder, input mode):
  encoder : 'frozen' (original DINOv3-B/16 lvd1689m) | path to a G1 ema_last.pt (key 'enc'; decoder never loaded)
  mode    : 'full' (full image 384 letterbox, R2 reads the 576 full tokens with the GT-box geometry)
            'tight' / 'expand' (one 384 letterbox canvas per ROI; see src.vaw.data)
Tokens = forward_features final-LN patch tokens (CLS + registers dropped), bf16 autocast -> fp32 (= PACO G1 probes).
Reader = src.readout.models.Reader('r2', layers=[12]) (geo/box MLP, 2 cross blocks, mask cov > 0, LN) with the PACO
level heads replaced by ONE Linear(768 -> 620) attribute head; init torch.manual_seed(7) (= seed 0 rule) for every
condition. No object name / polygon / text is used.
Loss (pre-registered, identical for all conditions): z = logit,
  L = sum_{labeled} BCE(z, y) / #labeled  +  W_UNL * sum_{unlabeled} BCE(z, 0) / #unlabeled     (y: 1 pos, 0 neg)
  (labeled and unlabeled averaged separately over the step, then combined; W_UNL = 0.1 fixed before any run)
Protocol (= PACO probe): image packs of ~128 ROIs (C.full_epoch_batches, data seed 0), hflip p 0.5, AdamW reader lr 3e-4 /
head lr 1e-3, wd 0.05 (none on 1-D / query), betas (0.9, 0.999), 1 epoch warmup + cosine, grad clip 1.0, 10 epochs.
Selection: official VAW mAP on VAW VAL after every epoch; best epoch (ties -> earlier) is the checkpoint.
Test: run ONCE at the end with the selected checkpoint (guard file test_scored.json); probabilities saved for reports.

  CUDA_VISIBLE_DEVICES=2 python -m src.vaw.probe --enc frozen --mode full --out <run>/frozen_full
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.readout.common import C
from src.readout.models import Reader, n_params, param_groups
from src.vaw.common import VAW, VAW_DATA, attr_index, image_sizes, label_matrix, load_split, vg_meta, vg_path
from src.vaw.data import VawDS
from src.viz.c0_pca import load_encoder

W_UNL, EPOCHS, LR, HEAD_LR, WD = 0.1, 10, 3e-4, 1e-3, 0.05
sys.path.insert(0, os.path.join(VAW, "eval"))
from evaluator import Evaluator  # noqa: E402  (official, unmodified)


def official_evaluator():
    return Evaluator(*(os.path.join(VAW_DATA, f) for f in ("attribute_index.json", "attribute_types.json",
                                                           "attribute_parent_types.json", "head_tail.json")))


def official_map(prob, Y):
    ev = official_evaluator()
    _, pc = ev.evaluate(prob, Y)
    return float(pc["all"]["ap"])


def smoke_map(prob, Y):
    """smoke runs only (--limit): mean AP over attributes with >= 1 pos and >= 1 neg (tiny subsets break the official AP)."""
    from sklearn.metrics import average_precision_score
    aps = [average_precision_score(Y[Y[:, j] < 2, j], prob[Y[:, j] < 2, j]) for j in range(Y.shape[1])
           if (Y[:, j] == 1).any() and (Y[:, j] == 0).any()]
    return float(np.mean(aps)) if aps else 0.0


class Head620(nn.Module):
    def __init__(self, n=620):
        super().__init__()
        self.lin = nn.Linear(768, n)

    def forward(self, f, lv):
        return self.lin(f).float()


def build_reader(dev):
    torch.manual_seed(7)
    r = Reader("r2", layers=(12,))
    r.heads = Head620()
    return r.to(dev)


def vaw_loss(logits, Y):
    lab, unl = (Y < 2), (Y == 2)
    bce = F.binary_cross_entropy_with_logits(logits, (Y == 1).float(), reduction="none")
    nl, nu = lab.sum().clamp_min(1), unl.sum().clamp_min(1)
    ll, lu = (bce * lab).sum() / nl, (bce * unl).sum() / nu
    return ll + W_UNL * lu, ll.detach(), lu.detach()


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, help="'frozen' or a G1 ema_last.pt path")
    ap.add_argument("--mode", required=True, choices=["full", "tight", "expand"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N images per split")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke: steps per epoch")
    ap.add_argument("--cpu", action="store_true", help="smoke on CPU")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:
        run(a)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        json.dump(dict(status="failed", error=tb, time=time.strftime("%F %T")), open(os.path.join(a.out, "failed.json"), "w"), indent=1)
        raise


def run(a):
    dev = torch.device("cpu" if a.cpu else "cuda")
    t0 = time.time()
    a2i, i2a = attr_index()
    meta = vg_meta()
    data = {}
    for sp in ("train", "val", "test"):
        X = load_split(sp)
        if a.limit:
            keep = set(list(dict.fromkeys(int(x["image_id"]) for x in X))[: a.limit])
            X = [x for x in X if int(x["image_id"]) in keep]
        data[sp] = (X, label_matrix(X, a2i))
    ids = {int(x["image_id"]) for X, _ in data.values() for x in X}
    paths = {i: vg_path(meta[i]) for i in ids}
    sizes = image_sizes()                     # actual file sizes (asserted again at load time)
    WH = {i: sizes[i] for i in ids}
    # ---- encoder (frozen)
    vit = load_encoder("dinov3", dev)
    enc_info = dict(enc="frozen original DINOv3-B/16 lvd1689m (timm vit_base_patch16_dinov3.lvd1689m)")
    if a.enc != "frozen":
        ck = torch.load(a.enc, map_location="cpu")
        missing = vit.load_state_dict(ck["enc"])
        enc_info = dict(enc=os.path.abspath(a.enc), sha256=sha(a.enc), keys=sorted(ck.keys()), load=str(missing),
                        decoder_used=False)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    sum0 = float(sum(p.double().sum() for p in vit.parameters()))
    reader = build_reader(dev)
    opt = torch.optim.AdamW(param_groups(reader, LR, HEAD_LR, WD), lr=LR, betas=(0.9, 0.999))
    Xtr, Ytr = data["train"]
    tr_ds = VawDS(Xtr, Ytr, paths, WH, True, a.mode)
    sampler = RC.RankStepSampler(tr_ds.n_roi, 0, 1, seed=0)
    torch.manual_seed(0)
    dl = DataLoader(tr_ds, sampler=sampler, batch_size=1, num_workers=a.workers, pin_memory=not a.cpu,
                    collate_fn=RC.collate_one, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    steps_ep = [len(C.full_epoch_batches(tr_ds.n_roi, e, 0)) for e in range(1, a.epochs + 1)]
    if a.max_steps:
        steps_ep = [min(s, a.max_steps) for s in steps_ep]
    total, warm = sum(steps_ep), steps_ep[0]
    cfg = dict(args=vars(a), encoder=enc_info, encoder_checksum=sum0, n_train_rois=len(Xtr), n_train_images=len(tr_ds),
               n_val_rois=len(data["val"][0]), n_test_rois=len(data["test"][0]), steps_per_epoch=steps_ep, total_steps=total,
               loss=dict(w_unlabeled=W_UNL, rule="mean BCE over labeled (pos 1 / neg 0) + W_UNL * mean BCE(0) over unlabeled"),
               optim=dict(lr=LR, head_lr=HEAD_LR, wd=WD, betas=[0.9, 0.999], warmup_steps=warm, schedule="cosine", clip=1.0),
               reader_params=n_params(reader), head="Linear(768, 620)", reader_init_seed=7, data_seed=0, hflip=0.5,
               torch=torch.__version__, start=time.strftime("%F %T"))
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
    print(json.dumps(cfg, default=str), flush=True)
    ac = lambda: torch.autocast(dev.type, dtype=torch.bfloat16)  # noqa: E731

    @torch.no_grad()
    def encode(x):
        toks = []
        for i in range(0, len(x), 64):
            with ac():
                t = vit.forward_features(x[i:i + 64])
            toks.append(t[:, vit.num_prefix_tokens:].float())
        return {12: torch.cat(toks)}

    @torch.no_grad()
    def score(sp):
        X, Y = data[sp]
        ds = VawDS(X, Y, paths, WH, False, a.mode)
        bs = 16 if a.mode == "full" else 8
        chunks = [list(range(i, min(len(ds), i + bs))) for i in range(0, len(ds), bs)]
        dle = DataLoader(ds, sampler=chunks, batch_size=1, num_workers=a.workers, collate_fn=RC.collate_one, pin_memory=not a.cpu)
        out = torch.full((len(X), 620), float("nan"))
        reader.eval()
        for b in dle:
            b = RC.to_dev(b, dev)
            f = encode(b["x"])
            with ac():
                o = reader(f, b)["logits"].float()
            out[b["rows"].cpu()] = torch.sigmoid(o).cpu()
        reader.train()
        assert not torch.isnan(out).any()
        return out.numpy()

    hist, step = [], 0
    for ep in range(1, a.epochs + 1):
        sampler.set_epoch(ep)
        rl = torch.zeros(3, device=dev)
        n = 0
        te = time.time()
        for k, b in enumerate(dl):
            if a.max_steps and k >= a.max_steps:
                break
            b = RC.to_dev(b, dev)
            f = encode(b["x"])
            for g in opt.param_groups:
                g["lr"] = C.lr_at(step, g["base"], warm, total)
            with ac():
                o = reader(f, b)
            loss, ll, lu = vaw_loss(o["logits"].float(), b["Y"].long())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            rl += torch.stack([loss.detach(), ll, lu])
            n += 1
            step += 1
            if step % 200 == 0:
                if not torch.isfinite(rl).all():
                    raise FloatingPointError(f"non-finite loss before step {step}")
                print(json.dumps(dict(step=step, ep=ep, s=round(time.time() - t0), s_per_step=round((time.time() - te) / n, 3),
                                      loss=[round(v, 4) for v in (rl / n).tolist()])), flush=True)
        Pv = score("val")
        m = smoke_map(Pv, data["val"][1]) if a.limit else official_map(Pv, data["val"][1])
        rec = dict(epoch=ep, step=step, val_mAP=m, train_loss=(rl / max(n, 1)).tolist(), epoch_s=round(time.time() - te, 1),
                   elapsed_s=round(time.time() - t0, 1))
        hist.append(rec)
        with open(os.path.join(a.out, "curves.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        np.save(os.path.join(a.out, f"val_prob_ep{ep}.npy"), Pv.astype(np.float16))
        torch.save(dict(reader=reader.state_dict(), epoch=ep, val_mAP=m), os.path.join(a.out, f"reader_ep{ep}.pt"))
        print(json.dumps(rec), flush=True)
    best = max(hist, key=lambda r: (r["val_mAP"], -r["epoch"]))
    sel = dict(selected_epoch=best["epoch"], val_mAP=best["val_mAP"], rule="max official VAW val mAP (ties -> earlier)",
               all={r["epoch"]: r["val_mAP"] for r in hist})
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    # ---- frozen check
    sum1 = float(sum(p.double().sum() for p in vit.parameters()))
    json.dump(dict(checksum_before=sum0, checksum_after=sum1, unchanged=sum1 == sum0,
                   requires_grad_any=any(p.requires_grad for p in vit.parameters())),
              open(os.path.join(a.out, "frozen_check.json"), "w"), indent=1)
    assert sum1 == sum0, "encoder changed"
    # ---- TEST: once, selected epoch
    guard = os.path.join(a.out, "test_scored.json")
    assert not os.path.exists(guard), "test already scored for this condition"
    reader.load_state_dict(torch.load(os.path.join(a.out, f"reader_ep{best['epoch']}.pt"), map_location=dev)["reader"])
    Pv = score("val")
    assert abs((smoke_map if a.limit else official_map)(Pv, data["val"][1]) - best["val_mAP"]) < 2e-3, "reloaded reader does not reproduce val mAP"
    Pt = score("test")
    np.save(os.path.join(a.out, "test_prob.npy"), Pt.astype(np.float32))
    np.save(os.path.join(a.out, "val_prob_selected.npy"), Pv.astype(np.float32))
    json.dump(dict(epoch=best["epoch"], time=time.strftime("%F %T"), n=len(Pt)), open(guard, "w"), indent=1)
    json.dump(dict(status="done", time_s=round(time.time() - t0), time=time.strftime("%F %T")),
              open(os.path.join(a.out, "done.json"), "w"), indent=1)
    print("DONE", json.dumps(sel), flush=True)


if __name__ == "__main__":
    main()
