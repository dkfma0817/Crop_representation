"""Frozen-feature extraction for the three G1 encoders on ImageNet-1K (train / val parquet read directly, no re-saving).

One DataLoader pass per split shard; every batch tensor x goes through ALL THREE encoders (same input tensor by
construction). Per image: final-LayerNorm CLS token (768-d, fp16) for each encoder, CLS L2 norm, and vs the original
encoder: CLS cosine and mean-patch-token cosine (patch tokens = forward_features[:, num_prefix_tokens:], registers
excluded). Encoders: eval mode, requires_grad False, torch.inference_mode, bf16 autocast (= G1 / PACO probes).

Preprocessing (identical for all models; DINOv3/DINOv2 classification-eval rule scaled to 384): RGB, shorter side ->
438 (= int(384 * 256 / 224), bicubic), center crop 384, ImageNet mean / std. No augmentation (features cached once).

  python -m src.g1in.extract index --cache <dir>                       (row counts / offsets, allocate memmaps)
  CUDA_VISIBLE_DEVICES=2 python -m src.g1in.extract run --split train --shard 0 --nshards 2 --cache <dir>
  python -m src.g1in.extract verify --cache <dir>
"""
import argparse
import hashlib
import io
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms as T

from src.g1in.common import MEAN, MODELS, PACO_PROBE, G1, RESIZE, S, STD, checksum, load_encoder, parquet_files

D = 768
STATS = ["norm_original", "norm_g1pool", "norm_g1cross", "clscos_g1pool", "clscos_g1cross", "patchcos_g1pool", "patchcos_g1cross", "done"]
TF = T.Compose([T.Resize(RESIZE, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(S), T.ToTensor(), T.Normalize(MEAN, STD)])


def meta_path(cache):
    return os.path.join(cache, "index.json")


def index(a):
    os.makedirs(a.cache, exist_ok=True)
    meta = {}
    for split in ("train", "val"):
        fs = parquet_files(split)
        counts = [pq.ParquetFile(f).metadata.num_rows for f in fs]
        off = [0] + [int(v) for v in np.cumsum(counts)]
        meta[split] = dict(files=fs, counts=counts, offsets=off, n=int(off[-1]))
        n = off[-1]
        for m in MODELS:
            np.lib.format.open_memmap(os.path.join(a.cache, f"{split}_cls_{m}.npy"), "w+", np.float16, (n, D))
        np.lib.format.open_memmap(os.path.join(a.cache, f"{split}_labels.npy"), "w+", np.int16, (n,))[:] = -1
        np.lib.format.open_memmap(os.path.join(a.cache, f"{split}_stats.npy"), "w+", np.float32, (n, len(STATS)))
        print(split, len(fs), "files", n, "images", flush=True)
    json.dump(meta, open(meta_path(a.cache), "w"), indent=1)


class PQ(IterableDataset):
    """Yields (global index, image tensor, label, basename) for the files of this shard; worker w takes files[w::W]."""

    def __init__(self, files, offsets):
        self.items = list(zip(files, offsets))

    def __iter__(self):
        wi = get_worker_info()
        items = self.items[wi.id::wi.num_workers] if wi else self.items
        for f, off in items:
            k = off
            for rb in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["image", "label"]):
                imgs, labs = rb.column("image").to_pylist(), rb.column("label").to_pylist()
                for im, lab in zip(imgs, labs):
                    x = TF(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
                    yield k, x, int(lab), os.path.basename(im["path"] or "")
                    k += 1


def run(a):
    meta = json.load(open(meta_path(a.cache)))[a.split]
    fs, off = meta["files"][a.shard::a.nshards], meta["offsets"][:-1][a.shard::a.nshards]
    if a.limit_files:
        fs, off = fs[:a.limit_files], off[:a.limit_files]
    dev = torch.device("cuda")
    encs = {m: load_encoder(m, dev) for m in MODELS}
    ref = {m: json.load(open(os.path.join(G1, PACO_PROBE[m], "frozen_check.json")))["checksum_before"] for m in MODELS}
    ck0 = {m: checksum(e) for m, e in encs.items()}
    v = encs["original"]
    npt = v.num_prefix_tokens
    info = dict(split=a.split, shard=a.shard, nshards=a.nshards, n_files=len(fs), num_prefix_tokens=npt,
                n_reg=int(v.reg_token.shape[1]), has_cls=bool(v.cls_token is not None), timm_global_pool=v.global_pool,
                feature="forward_features(x)[:, 0] = final-LayerNorm CLS (NOT the timm head avg-pool)",
                checksum_loaded=ck0, checksum_paco=ref, checksum_match={m: abs(ck0[m] - ref[m]) < 1e-6 for m in MODELS},
                requires_grad_any={m: any(p.requires_grad for p in e.parameters()) for m, e in encs.items()},
                training_mode={m: e.training for m, e in encs.items()}, preprocess=repr(TF))
    assert all(info["checksum_match"].values()), info["checksum_match"]
    assert not any(info["requires_grad_any"].values()) and not any(info["training_mode"].values())
    cls = {m: np.load(os.path.join(a.cache, f"{a.split}_cls_{m}.npy"), mmap_mode="r+") for m in MODELS}
    lab = np.load(os.path.join(a.cache, f"{a.split}_labels.npy"), mmap_mode="r+")
    st = np.load(os.path.join(a.cache, f"{a.split}_stats.npy"), mmap_mode="r+")
    names = {}
    dl = DataLoader(PQ(fs, off), batch_size=a.batch, num_workers=a.workers, pin_memory=True, prefetch_factor=4, persistent_workers=False)
    t0, n, first = time.time(), 0, None
    with torch.inference_mode():
        for idx, x, y, nm in dl:
            x = x.to(dev, non_blocking=True)
            if first is None:
                first = hashlib.sha1(x[:8].float().cpu().numpy().tobytes()).hexdigest()
            out = {}
            for m, e in encs.items():                                  # the SAME tensor x for every encoder
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    t = e.forward_features(x)
                t = t.float()
                if n == 0:   # token layout check: timm Eva = [CLS, 4 registers, 576 patches]; its head pool ('avg') = patch mean
                    assert npt == 5 and e.cls_token is not None and e.reg_token.shape[1] == 4 and t.shape[1] == 5 + 576
                    assert torch.allclose(e.forward_head(t, pre_logits=True), t[:, npt:].mean(1), atol=1e-5), "patch slice mismatch"
                out[m] = (t[:, 0], t[:, npt:].mean(1))
            ii = idx.numpy()
            for m in MODELS:
                cls[m][ii] = out[m][0].half().cpu().numpy()
            s = [out[m][0].norm(dim=-1) for m in MODELS]
            for m in ("g1pool", "g1cross"):
                s.append(torch.nn.functional.cosine_similarity(out[m][0], out["original"][0], dim=-1))
            for m in ("g1pool", "g1cross"):
                s.append(torch.nn.functional.cosine_similarity(out[m][1], out["original"][1], dim=-1))
            s.append(torch.ones_like(s[0]))
            st[ii] = torch.stack(s, 1).cpu().numpy()
            lab[ii] = y.numpy().astype(np.int16)
            for i, s_ in zip(ii.tolist(), nm):
                names[i] = s_
            n += len(ii)
            if n // a.batch % 50 == 0:
                print(json.dumps(dict(n=n, s=round(time.time() - t0), ips=round(n / (time.time() - t0)))), flush=True)
    for m in MODELS:
        cls[m].flush()
    lab.flush()
    st.flush()
    info.update(n_images=n, time_s=round(time.time() - t0), first_batch_sha1=first,
                checksum_after={m: checksum(e) for m, e in encs.items()})
    info["unchanged"] = {m: info["checksum_after"][m] == ck0[m] for m in MODELS}
    assert all(info["unchanged"].values())
    json.dump(names, open(os.path.join(a.cache, f"{a.split}_names_shard{a.shard}.json"), "w"))
    json.dump(info, open(os.path.join(a.cache, f"{a.split}_shard{a.shard}_info.json"), "w"), indent=1)
    print("DONE", json.dumps({k: info[k] for k in ("n_images", "time_s", "checksum_match", "unchanged")}), flush=True)


def verify(a):
    meta = json.load(open(meta_path(a.cache)))
    rep = {}
    names = {}
    for split in ("train", "val"):
        n = meta[split]["n"]
        st = np.load(os.path.join(a.cache, f"{split}_stats.npy"), mmap_mode="r")
        lab = np.load(os.path.join(a.cache, f"{split}_labels.npy"), mmap_mode="r")
        nm = {}
        for k in range(8):
            p = os.path.join(a.cache, f"{split}_names_shard{k}.json")
            if os.path.exists(p):
                nm.update(json.load(open(p)))
        names[split] = set(nm.values())
        cnt = np.bincount(lab[lab >= 0].astype(np.int64), minlength=1000)
        finite = all(np.isfinite(np.load(os.path.join(a.cache, f"{split}_cls_{m}.npy"), mmap_mode="r")[::997].astype(np.float32)).all() for m in MODELS)
        rep[split] = dict(n=n, all_done=bool((st[:, -1] == 1).all()), n_done=int((st[:, -1] == 1).sum()), labels_missing=int((lab < 0).sum()),
                          n_classes=int((cnt > 0).sum()), per_class_min=int(cnt.min()), per_class_max=int(cnt.max()),
                          n_unique_names=len(names[split]), n_names=len(nm), cls_finite_sample=bool(finite))
    rep["train_val_name_overlap"] = len(names["train"] & names["val"])
    json.dump(rep, open(os.path.join(a.cache, "verify.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("index", "run", "verify"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", choices=("train", "val"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit_files", type=int, default=0)
    a = ap.parse_args()
    {"index": index, "run": run, "verify": verify}[a.cmd](a)


if __name__ == "__main__":
    main()
