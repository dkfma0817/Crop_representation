"""Visual-inspection montages: for every official val/test image with a near-duplicate candidate (src.pacoattr.leak,
excluding images that are EXACT G1 training images), its best candidate (max thumbnail corr; dHash-only hits kept too),
sorted by corr, 60 pairs per montage (left PACO, right G1).

  python -m src.pacoattr.leak_montage --out output/2026-11-07/paco_oracle_attr/leak
"""
import argparse
import json
import os

import pandas as pd
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev = os.path.join(a.out, "evidence")
    D = pd.read_csv(os.path.join(ev, "near_candidates.csv"))
    ex = json.load(open(os.path.join(a.out, "leak_auto.json")))["exact"]
    for sp in ("val", "test"):
        x = D[(D.split == sp) & ~D.paco_id.isin(set(ex[sp]["image_id"]))]
        best = x.sort_values(["corr", "hamming"], ascending=[False, True]).drop_duplicates("paco_id").reset_index(drop=True)
        best["n_cands"] = best.paco_id.map(x.groupby("paco_id").size())
        best.to_csv(os.path.join(ev, f"best_{sp}.csv"), index_label="k")
        for s in range(0, len(best), 60):
            tiles = []
            for k, r in best.iloc[s:s + 60].iterrows():
                t = Image.new("RGB", (250, 140), "white")
                for j, p in enumerate((r.paco_path, r.g1_path)):
                    im = Image.open(p).convert("RGB")
                    im.thumbnail((122, 112))
                    t.paste(im, (2 + j * 125, 24))
                ImageDraw.Draw(t).text((2, 2), f"#{k} c{r['corr']:.2f} h{r.hamming} n{r.n_cands}", fill="black")
                tiles.append(t)
            cols = 6
            M = Image.new("RGB", (cols * 250, ((len(tiles) + cols - 1) // cols) * 140), "white")
            for j, t in enumerate(tiles):
                M.paste(t, ((j % cols) * 250, (j // cols) * 140))
            M.save(os.path.join(ev, f"best_{sp}_{s // 60:02d}.jpg"), quality=88)
        print(sp, len(best))


if __name__ == "__main__":
    main()
