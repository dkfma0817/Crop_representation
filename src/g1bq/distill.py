"""G1-bq training = src.g1clip.distill (G1-cross: losses L_roi + 1.11 L_detail + 0.5 L_global, COCO 12k stream minus PACO
val/test matches, 12,000 steps, encoder AdamW 5e-5 LLRD 0.75, decoder 3e-4, warmup 600 + cosine, EMA, final EMA ckpt,
seed-7 decoder init) with the decoder class replaced by src.g1bq.model.BQDecoder. Nothing else changes; the existing
G1-cross loss weights file is reused (no re-calibration: same loss).

  torchrun --nproc_per_node 2 -m src.g1bq.distill --cond cross --weights output/2026-11-07/g1clip/calib/weights.json --out <dir>
  python -m src.g1bq.distill --smoke --out <dir>       (CPU checks, see check_bq)
"""
import json
import os
import sys

import torch

import src.g1clip.distill as GD
from src.g1bq.model import BQDecoder

GD.ROIDecoderD = BQDecoder           # build() creates the decoder through this module-level name (under seed 7)


def check_bq(out):
    """CPU: (1) B is built from the student encoder tokens (d B / d ftok != 0, B changes when ftok changes); (2) L_detail
    gradient reaches the encoder through the QUERY path alone (keys detached) and through the key path alone;
    (3) parameter / FLOP difference vs the G1-cross decoder."""
    import numpy as np
    from src.config import STREAM
    import src.readout.common as RC
    from src.g1clip.model import ROIDecoderD
    from src.decstudy.model import bilinear_roi
    torch.set_num_threads(48)
    GD.MC.set_clip_norm()
    os.makedirs(out, exist_ok=True)
    ex, rois, ds = GD.data()
    ids = [int(i) for i in np.load(STREAM)[0] if int(i) not in set(ex["raev2_stream"])][:2]
    b = ds[ids]
    enc, enc0, dec = GD.build("cross", torch.device("cpu"))
    assert isinstance(dec, BQDecoder)
    torch.manual_seed(7)
    ref = ROIDecoderD(GD.S, GD.P, GD.D)
    sd, sr = dec.state_dict(), ref.state_dict()
    res = dict(shared_params_equal_g1cross_init=all(torch.equal(sd[k], sr[k]) for k in sr), new_params=sorted(set(sd) - set(sr)),
               n_params_bq=sum(p.numel() for p in dec.parameters()), n_params_g1cross=sum(p.numel() for p in ref.parameters()))
    with torch.no_grad():
        T, _ = GD.run_enc(enc0, b["crops"])
    ftok, _ = GD.run_enc(enc, b["full"])
    ri, cc = b["roi_img"], b["ccov"]
    Bf = bilinear_roi(ftok, ri, b["box_c"], b["qrel"], GD.S)
    gB = torch.autograd.grad(Bf.sum(), ftok, retain_graph=True)[0]
    res["dB_dftok_nonzero"] = bool(gB.abs().sum() > 0)
    # query path only: keys from detached tokens
    def detail(z):
        tok = (GD.cosd(z, T) * cc).sum(1) / cc.sum(1)
        return tok.mean() + GD.cosd(GD.pool(z, cc), GD.pool(T, cc)).mean()

    class KeyDetached(BQDecoder):
        pass
    z_q = dec_forward_split(dec, ftok, ftok.detach(), ri, b)
    gq = torch.autograd.grad(detail(z_q), ftok, retain_graph=True)[0]
    z_k = dec_forward_split(dec, ftok.detach(), ftok, ri, b)
    gk = torch.autograd.grad(detail(z_k), ftok, retain_graph=True)[0]
    res["grad_to_encoder_tokens_query_path"] = float(gq.norm())
    res["grad_to_encoder_tokens_key_path"] = float(gk.norm())
    z = dec(ftok, ri, b["box_c"], b["qrel"], b["fvalid"])
    ep = dict(enc.named_parameters())
    names = ["norm.weight", "blocks.23.mlp.fc2.weight", "blocks.0.attn.qkv.weight"]
    g = torch.autograd.grad(detail(z), [ep[n] for n in names], retain_graph=True)
    res["detail_grad_encoder_params"] = {n: float(x.norm()) for n, x in zip(names, g)}
    R, Q, Dm = z.shape
    res["extra_flops_per_step_approx"] = dict(bilinear_sampling=f"{R}x{Q}x{Dm}x4 MACs = {R * Q * Dm * 4 / 1e9:.3f} GMAC",
                                              ln_qin=f"{R * Q * Dm * 5 / 1e9:.3f} GFLOP",
                                              note="G1-cross decoder itself ~ 2 x (Q-proj + KV-proj + out + FFN) per ROI, unchanged")
    assert res["shared_params_equal_g1cross_init"] and res["dB_dftok_nonzero"] and res["grad_to_encoder_tokens_query_path"] > 0 \
        and res["grad_to_encoder_tokens_key_path"] > 0 and all(v > 0 for v in res["detail_grad_encoder_params"].values())
    json.dump(res, open(os.path.join(out, "check_bq.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("BQ CHECK OK")


def dec_forward_split(dec, tok_q, tok_k, ri, b, chunk=64):
    from src.decstudy.model import bilinear_roi
    Bf = bilinear_roi(tok_q, ri, b["box_c"], b["qrel"], dec.S, chunk)
    outs = []
    for i in range(0, len(ri), chunk):
        sl = slice(i, i + chunk)
        bc = b["box_c"][sl].float()
        q = dec.ln_qin(Bf[sl] + dec.q_mlp(dec.pos_feat(bc, b["qrel"][sl].float()))) + dec.q_grid
        kg, _ = dec.key_geo(bc)
        kv = dec.ln_k(tok_k[ri[sl]] + dec.k_mlp(kg))
        for blk in dec.blocks:
            q = blk(q, kv, b["fvalid"][ri[sl]].bool())
        outs.append(dec.out(dec.ln_out(q)))
    return torch.cat(outs)


if __name__ == "__main__":
    if "--check_bq" in sys.argv:
        check_bq(sys.argv[sys.argv.index("--out") + 1])
    else:
        GD.main()
