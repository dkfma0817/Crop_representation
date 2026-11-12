"""Pre-run checks for a full-image resolution of the EVA-family PACO probes (CPU; no training / evaluation).

  1. identical input tensors + ROI geometry for eva / clipself / declip (dataset built with the probe arguments)
  2. worked example: original size -> resize -> padding -> canvas -> ROI box -> token cells
  3. pixel-level ROI alignment: GT box painted in ORIGINAL pixels, pushed through the same to_tensor_canvas; painted extent
     vs analytic canvas box; painted cells must be cov > 0 cells
  4. small ROIs: number of cov>0 tokens and box area in token units (sum cov) per manifest D-bin; ROIs smaller than one token
  5. encoder position path at this side: pos_embed / RoPE native or interpolated; token order = row-major cells
  python -m src.densecmp.check_res --side 224 --out <dir>
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import src.readout.common as RC
from src.densecmp import geom
from src.densecmp.data import EvaPacoDS
from src.ivc.common import DBINS, EXCL
from src.readout.common import C

P = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, G = a.side, a.side // P
    os.makedirs(a.out, exist_ok=True)
    out = dict(side=S, patch=P, grid=[G, G])
    d = RC.load_data(EXCL, splits=("train", "val"))
    dva, Yva = d["val"]
    groups = C.image_groups(dva)
    gsel = sorted(int(i) for i in np.random.default_rng(0).choice(len(groups), 300, replace=False))
    # 1
    dss = {e: EvaPacoDS(dva, Yva, flip=False, mode="full", side=S) for e in ("eva", "clipself", "declip")}
    same = True
    for gi in gsel[:40]:
        bs = {e: ds[[gi]] for e, ds in dss.items()}
        for b in bs.values():
            same &= all(torch.equal(bs["eva"][k], b[k]) for k in ("x", "geo", "box4", "cov", "rows"))
    b = dss["eva"][[gsel[0]]]
    out["identical_inputs_3_models_40imgs"] = bool(same)
    out["input_shape"] = list(b["x"].shape[1:])
    out["cov_len"] = int(b["cov"].shape[1])
    # 2
    r = groups[gsel[0]][0]
    W, H = int(dva.W[r]), int(dva.H[r])
    bx = tuple(int(v) for v in dva.loc[r, ["l", "t", "r", "b"]])
    lb = geom.Letterbox(W, H, S)
    X0, Y0 = lb.to_canvas(bx[0], bx[1])
    X1, Y1 = lb.to_canvas(bx[2], bx[3])
    cov = geom.roi_geo(lb, bx, P)[2].reshape(G, G)
    ii, jj = np.nonzero(cov > 0)
    out["example"] = dict(image=dva.path[r].split("/")[-1], WH=[W, H], int_box_ltrb=bx, sx=lb.sx, sy=lb.sy, resized_wh=[lb.nw, lb.nh],
                          pad_left_top=[lb.ox, lb.oy], pad_right_bottom=[S - lb.ox - lb.nw, S - lb.oy - lb.nh],
                          box_canvas_xyxy=[round(float(v), 2) for v in (X0, Y0, X1, Y1)], cell_rows=[int(ii.min()), int(ii.max())],
                          cell_cols=[int(jj.min()), int(jj.max())], n_tokens_cov_gt0=int(len(ii)), area_tokens=round(float(cov.sum()), 3),
                          n_in_dinov2=int(dva.n_in[r]))
    # 3
    errs, mis, padmis = [], 0, 0
    for gi in gsel:
        rows = groups[gi]
        W, H = (int(v) for v in dva.loc[rows[0], ["W", "H"]])
        lb = geom.Letterbox(W, H, S)
        for r in rows[:3]:
            bx = tuple(int(v) for v in dva.loc[r, ["l", "t", "r", "b"]])
            m = np.zeros((H, W, 3), np.uint8)
            m[bx[1]:bx[3], bx[0]:bx[2]] = 255
            x = geom.to_tensor_canvas(Image.fromarray(m), lb, ((0, 0, 0), (1, 1, 1)))[0].numpy()
            ys, xs = np.nonzero(x > 0.5)
            X0, Y0 = lb.to_canvas(bx[0], bx[1])
            X1, Y1 = lb.to_canvas(bx[2], bx[3])
            if len(xs):
                errs.append(float(max(abs(xs.min() - X0), abs(xs.max() + 1 - X1), abs(ys.min() - Y0), abs(ys.max() + 1 - Y1))))
            cov = geom.roi_geo(lb, bx, P)[2].reshape(G, G)
            mis += sum(1 for (i, j) in set(zip(ys // P, xs // P)) if cov[i, j] <= 0)
            # box must lie inside the content rect (never on padding)
            padmis += int(X0 < lb.ox - 1e-6 or Y0 < lb.oy - 1e-6 or X1 > lb.ox + lb.nw + 1e-6 or Y1 > lb.oy + lb.nh + 1e-6)
    out["pixel_alignment"] = dict(n_rois_checked=len(errs), max_edge_err_px=max(errs), mean_edge_err_px=float(np.mean(errs)),
                                  painted_cells_outside_cov=mis, box_outside_content_rect=padmis,
                                  note="ROIs whose painted region vanishes after downscaling (< 0.5 coverage everywhere) are skipped for the edge error")
    # 4 (all val ROIs)
    ntok, area, nin, lev = [], [], dva.n_in.values, dva.level.values
    lbs = {}
    for r in range(len(dva)):
        W, H = int(dva.W[r]), int(dva.H[r])
        lb = lbs.setdefault((W, H), geom.Letterbox(W, H, S))
        cov = geom.box_cell_weights(lb, tuple(int(v) for v in dva.loc[r, ["l", "t", "r", "b"]]), P)
        ntok.append(int((cov > 0).sum()))
        area.append(float(cov.sum()))
    ntok, area = np.array(ntok), np.array(area)
    out["small_rois_val"] = dict(
        n_rois=len(dva), zero_token_rois=int((ntok == 0).sum()), rois_area_lt_1_token=int((area < 1).sum()),
        rois_area_lt_0p25_token=int((area < 0.25).sum()),
        by_bin={f"{lv}/{nm}": dict(n=int(((nin >= lo) & (nin <= hi) & (lev == lv)).sum()),
                                   tokens_min=int(ntok[(nin >= lo) & (nin <= hi) & (lev == lv)].min()),
                                   tokens_median=float(np.median(ntok[(nin >= lo) & (nin <= hi) & (lev == lv)])),
                                   area_tokens_median=round(float(np.median(area[(nin >= lo) & (nin <= hi) & (lev == lv)])), 3),
                                   frac_area_lt_1_token=round(float((area[(nin >= lo) & (nin <= hi) & (lev == lv)] < 1).mean()), 3))
                for lv in ("object", "part") for nm, lo, hi in DBINS},
        handling="every box overlaps >= 1 cell (cov > 0) -> R2 attention mask never empty; cov (fraction, < 1 for partial cells) is "
                 "an input feature of geo; same rule as the DINOv2 / 1024 runs (no special-case code)")
    # 5
    from src.densecmp.encoders import Encoder
    enc = Encoder("eva")
    v = enc.model.visual
    x = torch.zeros(1, 3, S, S)
    pe0 = v.patch_embed(x)
    i, j = G // 2, G // 3
    x[:, :, i * P:(i + 1) * P, j * P:(j + 1) * P] = 1.0
    changed = torch.nonzero((v.patch_embed(x) - pe0).abs().sum(-1)[0] > 0).flatten().tolist()
    pos = v.rescale_positional_embedding((G, G))
    rope = v.rope
    out["encoder"] = dict(native_image_size=list(v.patch_embed.img_size), tokens=int(pe0.shape[1]),
                          pos_embed_native_used=bool(pos is v.pos_embed), pos_embed_shape=list(pos.shape),
                          rope_ft_seq_len=int(rope.ft_seq_len), rope_native_used=bool(G * G == rope.ft_seq_len ** 2),
                          token_order_check=dict(cell=[i, j], changed=changed, expected=i * G + j))
    json.dump(out, open(os.path.join(a.out, f"check_res_{S}.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
