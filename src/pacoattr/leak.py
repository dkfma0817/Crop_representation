"""G1 distillation images (50,489 COCO train2017, identical for G1-pool / G1-cross) vs the OFFICIAL PACO-LVIS val (2,410) and
test (9,443) images. Same rule as src.g1eval.paco_dedup (which covered only the 2,051 probe-val images):
  exact : COCO image id, file md5, decoded-pixel md5, dHash (src.readout.data_check.img_keys; existing key table reused,
          keys computed with the same function for images not in it)
  near  : dHash Hamming <= 4, and 32x32 grayscale thumbnail correlation >= 0.85 (also against the horizontally flipped
          train image) over ALL official x G1 pairs -> montages for visual inspection (decisions in src.pacoattr.subsets)

  python -m src.pacoattr.leak --out output/2026-11-07/paco_oracle_attr/leak
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.g1eval.paco_dedup import train_ids
from src.msd.pretrain import KEYS
from src.readout.data_check import _job

DATA = "output/2026-11-07/paco_oracle_attr/data"
THR_CORR, THR_HAM = 0.85, 4


def thumb(p):
    im = Image.open(p).convert("L").resize((32, 32), Image.BILINEAR)
    x = np.asarray(im, np.float32).reshape(-1)
    x = x - x.mean()
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def thumb_flip(p):
    im = Image.open(p).convert("L").transpose(Image.FLIP_LEFT_RIGHT).resize((32, 32), Image.BILINEAR)
    x = np.asarray(im, np.float32).reshape(-1)
    x = x - x.mean()
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def ham(a, B):
    return np.bitwise_count(np.bitwise_xor(a[:, None], B[None, :]))


def montage(rows, fn, title):
    tiles = []
    for r in rows:
        t = Image.new("RGB", (330, 180), "white")
        for k, p in enumerate((r["paco_path"], r["g1_path"])):
            im = Image.open(p).convert("RGB")
            im.thumbnail((160, 150))
            t.paste(im, (5 + k * 165, 25))
        ImageDraw.Draw(t).text((5, 5), f"{r['split']} {r['paco_id']} / g1 {r['g1_id']} c={r['corr']:.3f} h={r['hamming']}", fill="black")
        tiles.append(t)
    cols = 4
    M = Image.new("RGB", (cols * 330, ((len(tiles) + cols - 1) // cols) * 180 + 20), "white")
    ImageDraw.Draw(M).text((5, 3), title, fill="black")
    for k, t in enumerate(tiles):
        M.paste(t, ((k % cols) * 330, 20 + (k // cols) * 180))
    M.save(fn, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev = os.path.join(a.out, "evidence")
    os.makedirs(ev, exist_ok=True)
    tr, cfg = train_ids()
    keys = pd.read_parquet(KEYS)
    g1k = keys[(keys.source == "raev2_stream") & keys.image_id.isin(tr)].drop_duplicates("image_id")
    assert len(g1k) == len(tr), (len(g1k), len(tr))
    P = {}
    for sp in ("val", "test"):
        M = pd.read_parquet(os.path.join(DATA, f"manifest_{sp}.parquet")).drop_duplicates("image_id")
        P[sp] = dict(zip(M.image_id.astype(int), M.path))
    known = keys[keys.source.isin(["paco_val", "paco_test"])].drop_duplicates("image_id").set_index("image_id")
    need = [(i, p) for sp in P for i, p in P[sp].items() if i not in known.index]
    with Pool(48) as pool:
        got = pool.map(_job, need, chunksize=16)
    assert all(e is None for *_, e in got), [e for *_, e in got if e][:3]
    new = pd.DataFrame([dict(image_id=i, path=p, **k) for i, p, k, _ in got]).set_index("image_id")
    K = pd.concat([known[["md5", "pix_md5", "dhash"]], new[["md5", "pix_md5", "dhash"]]])
    res = dict(g1_train_images=len(tr), g1_rule=f"stream rows 0..{cfg['steps'] - 1} minus {cfg['excluded']['stream']}",
               keys_reused=int(sum(i in known.index for sp in P for i in P[sp])), keys_computed=len(need), exact={}, near_candidates={})
    g1set = {k: set(g1k[k]) for k in ("md5", "pix_md5", "dhash")}
    for sp in P:
        ids = list(P[sp])
        kk = K.loc[ids]
        res["exact"][sp] = dict(images=len(ids), image_id=sorted(int(i) for i in set(ids) & tr),
                                **{k: sorted(int(i) for i in kk.index[kk[k].isin(g1set[k])]) for k in ("md5", "pix_md5", "dhash")})
    # thumbnails
    g1_ids = sorted(tr)
    from src.readout.data_check import coco_path_by_id
    g1_paths = [coco_path_by_id(i) for i in g1_ids]
    with Pool(64) as pool:
        Tg = np.stack(pool.map(thumb, g1_paths, chunksize=64))
        Tf = np.stack(pool.map(thumb_flip, g1_paths, chunksize=64))
    hg = np.array([int(h, 16) for h in g1k.set_index("image_id").loc[g1_ids, "dhash"]], dtype=np.uint64)
    cand = []
    for sp in P:
        ids = list(P[sp])
        with Pool(64) as pool:
            Tp = np.stack(pool.map(thumb, [P[sp][i] for i in ids], chunksize=32))
        hp = np.array([int(h, 16) for h in K.loc[ids, "dhash"]], dtype=np.uint64)
        for s in range(0, len(ids), 1000):
            C = np.maximum(Tp[s:s + 1000] @ Tg.T, Tp[s:s + 1000] @ Tf.T)
            Hm = ham(hp[s:s + 1000], hg)
            r, c = np.nonzero((C >= THR_CORR) | (Hm <= THR_HAM))
            for i, j in zip(r, c):
                cand.append(dict(split=sp, paco_id=int(ids[s + i]), g1_id=int(g1_ids[j]), corr=float(C[i, j]), hamming=int(Hm[i, j]),
                                 paco_path=P[sp][ids[s + i]], g1_path=g1_paths[j]))
        res["near_candidates"][sp] = sum(1 for x in cand if x["split"] == sp)
    D = pd.DataFrame(cand).sort_values(["split", "corr"], ascending=[True, False]).reset_index(drop=True)
    D.to_csv(os.path.join(ev, "near_candidates.csv"), index=False)
    for sp in P:
        Ds = D[D.split == sp].to_dict("records")
        for k in range(0, len(Ds), 40):
            montage(Ds[k:k + 40], os.path.join(ev, f"near_{sp}_{k // 40:02d}.jpg"), f"{sp}: candidates {k + 1}-{min(len(Ds), k + 40)} (left PACO, right G1)")
    json.dump(res, open(os.path.join(a.out, "leak_auto.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
