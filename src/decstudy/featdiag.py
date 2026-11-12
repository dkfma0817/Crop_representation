"""COCO-dev feature diagnostics after training (GPU, one device): every path on the SAME dev ROIs (468 COCO dev images
minus PACO matches, the G1-cross diagnostic set) against the same C0 crop teacher.
  (1) Table C metrics (src.decstudy.train.zstats): token / pooled cosine to the teacher, L_detail, normalised covariance
      error + covariance cosine similarity, spatial token variance ratio, ROI relation KL + Pearson r, ||Delta|| / ||B||
  (2) residual direction: cos(Delta, T - B) over valid cells (does the residual point from B toward the teacher?)
  (3) context use: the decoder KEYS see another image's tokens at every full-canvas cell that does not overlap the ROI box
      (B and in-box keys untouched) -> relative change of the decoder output (residual paths: ||Delta' - Delta|| / ||Delta||;
      existing decoder: ||Z' - Z|| / ||Z||). 0 would mean the decoder ignores out-of-box context.
  python -m src.decstudy.featdiag --out <dir>
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

import src.readout.common as RC
from src.decstudy import losses as LS
from src.decstudy.model import ResidualDecoder, bilinear_roi
from src.decstudy.train import (S, P, D, dev_diag, dev_set, load_frozen, path_bilinear, path_existing, path_residual, run_enc, set_fp32)
from src.g1clip.model import ROIDecoderD
from src.maskemb import common as MC

RR = "output/2026-11-07/decoder_study"


def residual_with_keys(dec, tok_b, tok_k, ri, box_c, qrel, fv_r, chunk=64):
    """ResidualDecoder.parts with bilinear B from tok_b (per image) and keys from tok_k (per ROI)."""
    Bf = bilinear_roi(tok_b, ri, box_c, qrel, S)
    out = []
    for i in range(0, len(ri), chunk):
        sl = slice(i, i + chunk)
        bc = box_c[sl].float()
        q = dec.ln_qin(Bf[sl] + dec.q_mlp(dec.pos_feat(bc, qrel[sl].float()))) + dec.q_grid
        kg, _ = dec.key_geo(bc)
        kv = dec.ln_k(tok_k[sl] + dec.k_mlp(kg))
        m = fv_r[sl].bool()
        for blk in dec.blocks:
            q = blk(q, kv, m)
        out.append(dec.out(dec.ln_out(q)).float())
    return Bf, torch.cat(out)


def existing_with_keys(dec, tok_k, box_c, qrel, fvalid_r, chunk=64):
    out = []
    for i in range(0, len(box_c), chunk):
        sl = slice(i, i + chunk)
        bc = box_c[sl].float()
        q = dec.queries(bc, qrel[sl].float())
        kg, _ = dec.key_geo(bc)
        kv = dec.ln_k(tok_k[sl] + dec.k_mlp(kg))
        for blk in dec.blocks:
            q = blk(q, kv, fvalid_r[sl].bool())
        out.append(dec.out(dec.ln_out(q)).float())
    return torch.cat(out)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    set_fp32()
    MC.set_clip_norm()
    dev = torch.device("cuda")
    vit, enc0, info, ck = load_frozen(dev)
    old = ROIDecoderD(S, P, D).to(dev).eval()
    old.load_state_dict(ck["dec"])
    res_decs = {}
    for n in ("D1", "L1", "L2"):
        dk = torch.load(f"{RR}/dec/{n}/ema_last.pt", map_location="cpu", weights_only=False)
        m = ResidualDecoder(S, P, D).to(dev).eval()
        m.load_state_dict(dk["dec"])
        res_decs[n] = m
    ds, chunks, n_rois = dev_set()
    paths = {"B0_bilinear": path_bilinear(), "existing_g1cross_decoder": path_existing(old)}
    paths.update({n: path_residual(m) for n, m in res_decs.items()})
    table = dev_diag(paths, vit, enc0, ds, chunks, dev)
    extra = {k: {"dir_cos": [0.0, 0], "ctx_rel_change": [0.0, 0], "ctx_tok_cos_change": [0.0, 0]} for k in list(res_decs) + ["existing_g1cross_decoder"]}
    for ch in chunks:
        b = RC.to_dev(ds[ch], dev)
        T, _ = run_enc(enc0, b["crops"])
        F_, _ = run_enc(vit, b["full"])
        ri, cc = b["roi_img"], b["ccov"]
        m = (cc > 0).float()
        _, ov = old.key_geo(b["box_c"].float())
        other = (ri + 1) % len(ch) if len(ch) > 1 else ri
        tok_k = torch.where((ov > 0)[..., None], F_[ri], F_[other])          # out-of-box keys from another image
        fv_r = b["fvalid"][ri] & b["fvalid"][other]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for n, dec in res_decs.items():
                Bf, d0 = residual_with_keys(dec, F_, F_[ri], ri, b["box_c"], b["qrel"], b["fvalid"][ri])
                _, d1 = residual_with_keys(dec, F_, tok_k, ri, b["box_c"], b["qrel"], fv_r)
                dc = (F.cosine_similarity(d0.float(), (T - Bf).float(), dim=-1) * m).sum(1) / m.sum(1)
                rel = ((d1 - d0) * m[..., None]).flatten(1).norm(dim=1) / (d0 * m[..., None]).flatten(1).norm(dim=1).clamp_min(1e-12)
                tc0, tc1 = LS.detail_terms(Bf + d0, T, cc)[0], LS.detail_terms(Bf + d1, T, cc)[0]
                for k_, v in (("dir_cos", dc), ("ctx_rel_change", rel), ("ctx_tok_cos_change", tc1 - tc0)):
                    extra[n][k_][0] += float(v.sum())
                    extra[n][k_][1] += len(v)
            z0 = existing_with_keys(old, F_[ri], b["box_c"], b["qrel"], b["fvalid"][ri])
            z1 = existing_with_keys(old, tok_k, b["box_c"], b["qrel"], fv_r)
            rel = ((z1 - z0) * m[..., None]).flatten(1).norm(dim=1) / (z0 * m[..., None]).flatten(1).norm(dim=1).clamp_min(1e-12)
            tc0, tc1 = LS.detail_terms(z0, T, cc)[0], LS.detail_terms(z1, T, cc)[0]
            for k_, v in (("ctx_rel_change", rel), ("ctx_tok_cos_change", tc1 - tc0)):
                extra["existing_g1cross_decoder"][k_][0] += float(v.sum())
                extra["existing_g1cross_decoder"][k_][1] += len(v)
    ex = {k: {kk: (s / n if n else None) for kk, (s, n) in v.items()} for k, v in extra.items()}
    out = dict(n_dev_images=sum(len(c) for c in chunks), n_dev_rois=n_rois, table=table, residual_and_context=ex, encoder=info,
               notes=dict(ctx_rel_change="relative change of the decoder output when out-of-box keys come from another image",
                          ctx_tok_cos_change="token cosine distance to the teacher after the swap minus before (>0 = the real context helped)",
                          dir_cos="cos(Delta, T - B) over valid cells"))
    json.dump(out, open(os.path.join(a.out, "featdiag.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
