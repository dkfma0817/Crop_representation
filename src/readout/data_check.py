"""Train / evaluation image-overlap check for the shared-feature readout queue (must pass before any training).

Sources (every image any queue job reads, plus the caches that came from other projects):
  used by the queue  : paco_train  (reader training; also the C0 decision teacher of E12 = crops of the same images)
  evaluation         : paco_val (method selection / verdicts), paco_test (never scored by the queue, checked anyway)
  other caches       : raev2_stream / raev2_dev / raev2_hval (cache/feat/*_rois.parquet, COCO train2017 ROIs of the
                       RAEv2 pilot; NOT read by the queue), paco_lp feature cache (cache/paco_lp/feat, rows = manifest
                       rows of each PACO split, so its images are the split's images)
Keys per image: COCO image id, md5 of the file bytes, md5 of the decoded RGB pixels, 64-bit dHash (9x8 grey).
Rule: a paco_train image is EXCLUDED from training if its id, byte md5, pixel md5 or dHash equals that of any
paco_val / paco_test image (dHash equality is only a near-duplicate flag; excluding a few train images is harmless).
Exit code 0 = training stream verified disjoint from val/test after exclusion; 2 = verification failed (queue aborts).

  python -m src.readout.data_check --out output/<date>/readout_queue/data_check
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
from PIL import Image

from src.config import CACHE, COCO_TRAIN, COCO_VAL, FEAT, PACO_LP
from src.rae.common import coco_image_path


def img_keys(path):
    b = open(path, "rb").read()
    im = Image.open(path).convert("RGB")
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).reshape(-1)
    dh = int("".join("1" if x else "0" for x in bits), 2)
    return dict(md5=hashlib.md5(b).hexdigest(), pix_md5=hashlib.md5(im.tobytes()).hexdigest(), dhash=f"{dh:016x}",
                W=im.size[0], H=im.size[1])


def _job(a):
    iid, p = a
    try:
        return iid, p, img_keys(p), None
    except Exception as e:  # noqa: BLE001
        return iid, p, None, repr(e)


def coco_path_by_id(iid):
    for root in (COCO_TRAIN, COCO_VAL):
        p = os.path.join(root, f"{int(iid):012d}.jpg")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(iid)


def paco_split_images(split):
    df = pd.read_parquet(os.path.join(PACO_LP, f"manifest_{split}.parquet"))
    g = df.groupby("image_id", sort=False).path.first()
    return {int(i): coco_image_path(p) for i, p in g.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    src = {f"paco_{sp}": paco_split_images(sp) for sp in ("train", "val", "test")}
    for nm in ("stream", "dev", "hval"):
        f = os.path.join(FEAT, f"{nm}_rois.parquet")
        if os.path.exists(f):
            ids = pd.read_parquet(f, columns=["image_id"]).image_id.unique()
            src[f"raev2_{nm}"] = {int(i): coco_path_by_id(i) for i in ids}
    # paco_lp feature cache: rows follow the PACO manifests (order_*.json n == manifest rows after degenerate drop)
    lp_note = {}
    for sp in ("train", "val", "test"):
        o = os.path.join(PACO_LP, "feat", f"order_{sp}.json")
        if os.path.exists(o):
            lp_note[sp] = json.load(open(o)).get("n")
    jobs = sorted({(i, p) for d in src.values() for i, p in d.items()})
    print(f"{len(jobs)} distinct images over {len(src)} sources", flush=True)
    keys, errors = {}, []
    with get_context("fork").Pool(a.workers) as pool:
        for k, (iid, p, kk, err) in enumerate(pool.imap_unordered(_job, jobs, chunksize=64)):
            if err:
                errors.append(dict(image_id=iid, path=p, error=err))
            else:
                keys[(iid, p)] = kk
            if (k + 1) % 20000 == 0:
                print(k + 1, round(time.time() - t0), flush=True)
    rows = []
    for nm, d in src.items():
        for iid, p in d.items():
            kk = keys.get((iid, p))
            if kk:
                rows.append(dict(source=nm, image_id=iid, path=p, **kk))
    tab = pd.DataFrame(rows)
    tab.to_parquet(os.path.join(a.out, "image_keys.parquet"), index=False)

    def overlap(s1, s2):
        A, B = tab[tab.source == s1], tab[tab.source == s2]
        r = {}
        for k in ("image_id", "md5", "pix_md5", "dhash"):
            inter = set(A[k]) & set(B[k])
            r[k] = len(inter)
        return r

    names = list(src)
    pairs = {f"{x}__{y}": overlap(x, y) for i, x in enumerate(names) for y in names[i + 1:]}
    # train exclusion: any key shared with val or test
    ev = tab[tab.source.isin(["paco_val", "paco_test"])]
    tr = tab[tab.source == "paco_train"]
    hit = np.zeros(len(tr), bool)
    reasons = {}
    for k in ("image_id", "md5", "pix_md5", "dhash"):
        m = tr[k].isin(set(ev[k])).values
        reasons[k] = int(m.sum())
        hit |= m
    excl = sorted(int(i) for i in tr.image_id.values[hit])
    json.dump(excl, open(os.path.join(a.out, "train_exclude_image_ids.json"), "w"))
    # verification after exclusion
    tr2 = tr[~tr.image_id.isin(set(excl))]
    left = {k: int(tr2[k].isin(set(ev[k])).sum()) for k in ("image_id", "md5", "pix_md5", "dhash")}
    n_train_expected = len(src["paco_train"])
    ok = (not errors) and all(v == 0 for v in left.values()) and len(tr) == n_train_expected
    rep = dict(ok=bool(ok), time_s=round(time.time() - t0, 1), n_images={k: len(v) for k, v in src.items()},
               n_hashed=int(len(tab)), read_errors=errors[:50], n_read_errors=len(errors),
               pair_overlaps=pairs, train_vs_eval_hits_by_key=reasons, n_train_excluded=len(excl),
               train_vs_eval_after_exclusion=left, paco_lp_feat_rows=lp_note,
               used_by_queue=["paco_train (training + E12 C0 teacher crops)", "paco_val (evaluation)"],
               not_read_by_queue=["paco_test", "raev2_stream", "raev2_dev", "raev2_hval", "paco_lp feature cache"],
               not_checkable="DINOv2 LVD-142M pretraining images (no list available)")
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "read_errors"}, indent=1), flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
