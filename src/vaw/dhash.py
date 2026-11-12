"""Step 0b: pixel-level duplicate check (complements the coco_id check in src.vaw.leak).

dHash (64 bit: 9x8 grayscale, horizontal gradient sign; PIL bilinear on the JPEG draft) of every VAW image (train/val/test)
and of every G1 distillation image (50,489 COCO train2017). Reported:
  (1) method validation: Hamming distance of VG<->COCO pairs that VG metadata says are the same image (coco_id mapped)
  (2) VAW test (and val) vs G1 train: candidates with Hamming <= T among images whose coco_id is NOT in G1 train
      (i.e. duplicates the coco_id mapping missed, incl. coco_id null)
  (3) cross-split: the coco_id-duplicate pairs (train/val/test) -> their Hamming distance; plus test vs train/val
      dHash-identical (Hamming 0) pairs with different VG ids
Montages of every candidate are written for visual inspection.

  python -m src.vaw.dhash --out output/2026-11-07/vaw_g1/leak
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from src.config import COCO_TRAIN
from src.g1eval.paco_dedup import train_ids
from src.vaw.common import vg_meta, vg_path

T = 4


def dhash(path):
    try:
        im = Image.open(path)
        im.draft("L", (64, 64))
        a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
        v = int(np.packbits(bits).view(">u8")[0])
        return v - (1 << 64) if v >= (1 << 63) else v          # stored as signed int64 (same bits)
    except Exception:
        return None


def hashes(paths, procs=64):
    with Pool(procs) as p:
        r = p.map(dhash, paths, chunksize=64)
    fail = [i for i, v in enumerate(r) if v is None]
    assert not fail, f"{len(fail)} images failed to hash, e.g. {[paths[i] for i in fail[:3]]}"
    return np.array(r, dtype=np.int64)


def ham(a, B):
    """a (n,) uint64, B (m,) uint64 -> (n, m) Hamming distance"""
    x = np.bitwise_xor(a[:, None], B[None, :])
    return np.bitwise_count(x) if hasattr(np, "bitwise_count") else np.unpackbits(x.view(np.uint8), axis=-1).reshape(*x.shape, 64).sum(-1)


def near(Aq, B, thr, chunk=2000):
    out = []
    for i in range(0, len(Aq), chunk):
        d = ham(Aq[i:i + chunk], B)
        r, c = np.nonzero(d <= thr)
        out += [(i + int(a), int(b), int(d[a, b])) for a, b in zip(r, c)]
    return out


def montage(pairs, fn, title):
    from PIL import ImageDraw
    if not pairs:
        return
    tiles = []
    for pa, pb, lab in pairs[:120]:
        t = Image.new("RGB", (330, 180), "white")
        for k, p in enumerate((pa, pb)):
            im = Image.open(p).convert("RGB")
            im.thumbnail((160, 150))
            t.paste(im, (5 + k * 165, 25))
        ImageDraw.Draw(t).text((5, 5), lab[:60], fill="black")
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
    D = pd.read_parquet(os.path.join(a.out, "instances.parquet"))
    I = D.drop_duplicates("image_id")[["image_id", "split", "coco_id", "G1 train"]].reset_index(drop=True)
    meta = vg_meta()
    I["path"] = [vg_path(meta[int(i)]) for i in I.image_id]
    g1 = sorted(train_ids()[0])
    cpath = {c: os.path.join(COCO_TRAIN, f"{c:012d}.jpg") for c in g1}
    hp = os.path.join(a.out, "dhash.npz")
    if os.path.exists(hp):
        z = np.load(hp)
        assert np.array_equal(z["vg_ids"], I.image_id.values) and np.array_equal(z["g1_ids"], np.array(g1))
        hv, hg = z["vg"], z["g1"]
    else:
        hv, hg = hashes(I.path.tolist()), hashes([cpath[c] for c in g1])
        np.savez(hp, vg_ids=I.image_id.values, vg=hv, g1_ids=np.array(g1), g1=hg)
    res = dict(threshold=T, n_vaw_images=len(hv), n_g1_images=len(hg))
    I["h"] = hv
    gi = {c: k for k, c in enumerate(g1)}
    U = lambda x: np.asarray(x, dtype=np.int64).view(np.uint64)  # noqa: E731
    # (1) validation on coco-mapped VAW images that ARE G1 train images
    m = I[I["G1 train"]]
    dv = ham(U(m.h.values), U(hg))[np.arange(len(m)), [gi[int(c)] for c in m.coco_id]]
    res["validation_mapped_pairs"] = dict(n=len(m), hamming_le_T=int((dv <= T).sum()), median=float(np.median(dv)),
                                          p95=float(np.percentile(dv, 95)), max=int(dv.max()))
    bad = m[dv > T]
    montage([(r.path, cpath[int(r.coco_id)], f"vg {r.image_id} / coco {int(r.coco_id)} d={d}") for r, d in zip(bad.itertuples(), dv[dv > T])][:40],
            os.path.join(ev, "validation_mapped_far.jpg"), "coco_id-mapped VG/COCO pairs with dHash distance > T")
    # (2) val/test images NOT mapped into G1 train vs all G1 train images
    cand = []
    for sp in ("test", "val"):
        q = I[(I.split == sp) & (~I["G1 train"])].reset_index(drop=True)
        for i, j, d in near(U(q.h.values), U(hg), T):
            cand.append(dict(split=sp, vg_id=int(q.image_id[i]), vg_coco_id=int(q.coco_id[i]), g1_coco_id=int(g1[j]), hamming=d,
                             vg_path=q.path[i], coco_path=cpath[g1[j]]))
    C = pd.DataFrame(cand)
    C.to_csv(os.path.join(ev, "unmapped_vs_g1_candidates.csv"), index=False)
    res["unmapped_vs_g1_candidates"] = {sp: int((C.split == sp).sum()) if len(C) else 0 for sp in ("test", "val")}
    if len(C):
        montage([(r.vg_path, r.coco_path, f"{r.split} vg {r.vg_id} / coco {r.g1_coco_id} d={r.hamming}") for r in C.sort_values("hamming").itertuples()],
                os.path.join(ev, "unmapped_vs_g1_candidates.jpg"), f"VAW val/test (coco_id not in G1) vs G1 train, dHash <= {T}")
    # (3) cross-split coco_id duplicates
    cc = I[I.coco_id >= 0].groupby("coco_id")
    pairs = []
    for c, g in cc:
        if g.split.nunique() > 1:
            g = g.reset_index(drop=True)
            for x in range(len(g)):
                for y in range(x + 1, len(g)):
                    if g.split[x] != g.split[y]:
                        d = int(ham(U([g.h[x]]), U([g.h[y]]))[0, 0])
                        pairs.append(dict(coco_id=int(c), a_split=g.split[x], a_vg=int(g.image_id[x]), b_split=g.split[y], b_vg=int(g.image_id[y]),
                                          hamming=d, a_path=g.path[x], b_path=g.path[y]))
    P = pd.DataFrame(pairs)
    P.to_csv(os.path.join(ev, "cross_split_cocoid_pairs.csv"), index=False)
    res["cross_split_cocoid_pairs"] = dict(n=len(P), hamming_le_T=int((P.hamming <= T).sum()) if len(P) else 0,
                                           by_splits={f"{k[0]}-{k[1]}": int(v) for k, v in P.groupby(["a_split", "b_split"]).size().items()} if len(P) else {})
    montage([(r.a_path, r.b_path, f"{r.a_split} {r.a_vg} / {r.b_split} {r.b_vg} d={r.hamming}") for r in P.itertuples()],
            os.path.join(ev, "cross_split_cocoid_pairs.jpg"), "VAW cross-split images with the same coco_id")
    # (3b) test vs train/val dHash-identical pairs (any coco status), excluding the coco_id pairs above
    te, tv = I[I.split == "test"].reset_index(drop=True), I[I.split != "test"].reset_index(drop=True)
    known = {(min(r.a_vg, r.b_vg), max(r.a_vg, r.b_vg)) for r in P.itertuples()} if len(P) else set()
    ex = [dict(test_vg=int(te.image_id[i]), other_vg=int(tv.image_id[j]), other_split=tv.split[j], hamming=d, a=te.path[i], b=tv.path[j])
          for i, j, d in near(U(te.h.values), U(tv.h.values), 0)
          if (min(int(te.image_id[i]), int(tv.image_id[j])), max(int(te.image_id[i]), int(tv.image_id[j]))) not in known]
    E = pd.DataFrame(ex)
    E.to_csv(os.path.join(ev, "test_vs_trainval_dhash0.csv"), index=False)
    res["test_vs_trainval_dhash0_extra"] = len(E)
    if len(E):
        montage([(r.a, r.b, f"test {r.test_vg} / {r.other_split} {r.other_vg}") for r in E.itertuples()],
                os.path.join(ev, "test_vs_trainval_dhash0.jpg"), "VAW test vs train/val, dHash identical (not coco_id pairs)")
    json.dump(res, open(os.path.join(a.out, "dhash.json"), "w"), indent=1, default=int)
    print(json.dumps(res, indent=1, default=int))


if __name__ == "__main__":
    main()
