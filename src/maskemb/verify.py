"""Step-0 sanity checks (CPU only; no GPU, no training): checkpoints, weights, features, preprocessing, ROI geometry.
Writes <RUN>/sanity/verify.json and <RUN>/sanity/geom_*.png.

  python -m src.maskemb.verify
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.maskemb import common as MC

DATA = "output/2026-11-07/paco_oracle_attr/data"          # official val manifest (all 31,919 annotations), read-only


def tdiff(a, b):
    a, b = a.float(), b.float()
    return dict(max_abs=float((a - b).abs().max()), rel_fro=float((a - b).norm() / b.norm().clamp_min(1e-12)))


def tok_cos(a, b):
    c = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
    return dict(mean=float(c.mean()), min=float(c.min()), p01=float(c.flatten().quantile(0.01)))


def main():
    torch.set_num_threads(32)
    out = dict(time=time.strftime("%F %T"))
    os.makedirs(MC.run_path("sanity"), exist_ok=True)
    # ---------------------------------------------------------------- checkpoints
    out["sha256"] = {p: MC.sha256(p) for p in MC.SHA}
    assert all(out["sha256"][p] == MC.SHA[p] for p in MC.SHA)
    ck = MC.load_me_state()
    A = vars(ck["args"])
    out["maskembed_ckpt"] = dict(arch=ck["arch"], epoch=ck["epoch"], metric=ck["metric"], n_state=len(ck["state_dict"]),
                                 has_ema=("state_dict_ema" in ck), **{k: A[k] for k in ("student_model", "student_pretrained", "teacher_model",
                                 "teacher_pretrained", "skip_blocks", "student_head", "decoder_layers", "decoder_width", "epochs", "data_dir",
                                 "model_ema", "student_model_kwargs", "mean", "std", "img_size", "loss_function", "mask_sampling")})
    sd = ck["state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    out["maskembed_ckpt"]["prefixes"] = sorted({k.split(".")[0] for k in sd})
    out["maskembed_ckpt"]["n_encoder_keys"] = len(enc_sd)
    out["maskembed_ckpt"]["n_decoder_keys"] = sum(k.startswith("decoder.") for k in sd)
    # ---------------------------------------------------------------- models
    clip, ci = MC.build_encoder("clip")
    me, mi = MC.build_encoder("maskembed")
    clip_sc, _ = MC.build_encoder("clip", soft_cap_clip=True)
    out["clip_info"], out["maskembed_info"] = ci, mi
    arch = {}
    for nm, m in (("clip", clip), ("maskembed", me)):
        arch[nm] = dict(cls=type(m).__name__, block=type(m.blocks[0]).__name__, attn=type(m.blocks[0].attn).__name__,
                        act=type(m.blocks[0].mlp.act).__name__, depth=len(m.blocks), embed_dim=m.embed_dim, heads=m.blocks[0].attn.num_heads,
                        grid=list(m.patch_embed.grid_size), patch=list(m.patch_embed.patch_size), img_size=list(m.patch_embed.img_size),
                        num_prefix_tokens=m.num_prefix_tokens, global_pool=m.global_pool, norm=str(m.norm), fc_norm=str(m.fc_norm),
                        head=str(m.head), norm_pre=str(m.norm_pre), qkv_bias=m.blocks[0].attn.qkv.bias is not None,
                        soft_cap=getattr(m.blocks[0].attn, "max_attn_val", None),
                        n_params=sum(p.numel() for p in m.parameters()), pretrained_cfg_mean=m.pretrained_cfg.get("mean"),
                        pretrained_cfg_std=m.pretrained_cfg.get("std"), interpolation=m.pretrained_cfg.get("interpolation"))
    out["arch"] = arch
    sc, sm = clip.state_dict(), me.state_dict()
    assert list(sc) == list(sm), set(sc) ^ set(sm)
    out["same_param_names_and_shapes"] = all(sc[k].shape == sm[k].shape for k in sc)
    # MaskEmbed: every model tensor == the checkpoint's encoder tensor (no layer scale in this ckpt -> nothing absorbed)
    out["maskembed_loaded_equals_ckpt"] = all(torch.equal(sm[k], enc_sd[k]) for k in sm)
    out["maskembed_unused_ckpt_encoder_keys"] = sorted(set(enc_sd) - set(sm))
    # ---------------------------------------------------------------- CLIP == OpenAI original
    oai, osd = MC.openai_reference()
    mapped, proj = MC.oai_to_timm_keys(osd)
    assert set(mapped) == set(sc), set(mapped) ^ set(sc)
    out["clip_timm_vs_openai_pt"] = dict(max_abs_over_all_tensors=max(float((sc[k].float() - mapped[k].float()).abs().max()) for k in sc),
                                          n_tensors=len(sc), openai_dtype=str(osd["visual.conv1.weight"].dtype),
                                          proj_shape=list(proj.shape), proj_used=False)
    # ---------------------------------------------------------------- MaskEmbed vs CLIP weight change
    ch = {}
    for k in sc:
        grp = "blocks." + k.split(".")[1] if k.startswith("blocks.") else k.rsplit(".", 1)[0] if "." in k else k
        a = ch.setdefault(grp, [0.0, 0.0])
        a[0] += float((sm[k].float() - sc[k].float()).pow(2).sum())
        a[1] += float(sc[k].float().pow(2).sum())
    rel = {g: (v[0] / max(v[1], 1e-12)) ** 0.5 for g, v in ch.items()}
    out["maskembed_minus_clip_rel_change"] = rel
    out["maskembed_norm_vs_clip_ln_post"] = dict(me_w_mean=float(sm["norm.weight"].mean()), clip_w_mean=float(sc["norm.weight"].mean()),
                                                 me_b_absmean=float(sm["norm.bias"].abs().mean()), clip_b_absmean=float(sc["norm.bias"].abs().mean()))
    # init correspondence: a random-init / other-model control has rel change ~1; fine-tuned-from-CLIP must be << 1
    blk = [rel[f"blocks.{i}"] for i in range(24)]
    wcos = {k: float(torch.nn.functional.cosine_similarity(sm[k].float().flatten(), sc[k].float().flatten(), dim=0)) for k in sc
            if sc[k].ndim >= 2 and k not in ("cls_token",)}
    g = torch.Generator().manual_seed(0)
    rcos = [float(torch.nn.functional.cosine_similarity(sc[k].float().flatten(), sc[k].float().flatten()[torch.randperm(sc[k].numel(), generator=g)], dim=0))
            for k in list(wcos)[:20]]
    out["maskembed_vs_clip_weight_cos"] = dict(n_matrices=len(wcos), min=min(wcos.values()), median=float(np.median(list(wcos.values()))),
                                               max=max(wcos.values()), patch_embed=wcos["patch_embed.proj.weight"], pos_embed=wcos["pos_embed"],
                                               control_clip_vs_permuted_clip_absmax=max(abs(v) for v in rcos))
    out["init_correspondence"] = dict(patch_embed=rel["patch_embed.proj"], pos_embed=rel["pos_embed"], cls=rel["cls_token"],
                                      blocks_min=min(blk), blocks_max=max(blk), blocks_mean=float(np.mean(blk)),
                                      all_tensors_changed=all(not torch.equal(sm[k], sc[k]) for k in sc))
    # ---------------------------------------------------------------- inputs
    CP = MC.set_clip_norm()
    M = pd.read_parquet(os.path.join(DATA, "manifest_val.parquet"))
    rng = np.random.default_rng(0)
    imgs = sorted(rng.choice(M.image_id.unique(), 3, replace=False).tolist())
    rows = M[M.image_id.isin(imgs)]
    xs, meta = [], []
    for iid in imgs:
        r = rows[rows.image_id == iid].iloc[0]
        img = Image.open(r.path).convert("RGB")
        lb = CP.LB(img.size[0], img.size[1], MC.S, MC.P)
        x = lb.canvas(img)
        ref = np.asarray(img.resize((lb.nw, lb.nh), resample=Image.BICUBIC), np.float32) / 255.0
        ref = (torch.from_numpy(ref).permute(2, 0, 1) - torch.tensor(MC.CLIP_MEAN).view(3, 1, 1)) / torch.tensor(MC.CLIP_STD).view(3, 1, 1)
        content = x[:, lb.oy:lb.oy + lb.nh, lb.ox:lb.ox + lb.nw]
        pad = x.clone()
        pad[:, lb.oy:lb.oy + lb.nh, lb.ox:lb.ox + lb.nw] = 0
        meta.append(dict(image_id=int(iid), W=img.size[0], H=img.size[1], nw=lb.nw, nh=lb.nh, ox=lb.ox, oy=lb.oy,
                         content_max_abs_vs_clip_norm=float((content - ref).abs().max()), padding_max_abs=float(pad.abs().max())))
        xs.append(x)
    out["canvas_checks"] = meta
    assert all(m["content_max_abs_vs_clip_norm"] < 1e-5 and m["padding_max_abs"] == 0 for m in meta)
    x = torch.stack(xs)
    # ---------------------------------------------------------------- features
    with torch.no_grad():
        ff_c = clip.forward_features(x)
        ff_m = me.forward_features(x)
        ff_sc = clip_sc.forward_features(x)
        t_ref = MC.oai_patch_tokens(oai, x)
        t_c, t_m = MC.tokens(clip, x, amp=False), MC.tokens(me, x, amp=False)
        t_c16, t_m16 = MC.tokens(clip, x, amp=True), MC.tokens(me, x, amp=True)
        t_cf16, t_mf16 = MC.tokens(clip, x, dtype=torch.float16), MC.tokens(me, x, dtype=torch.float16)
        pooled_c = clip.forward_head(ff_c)
    out["features"] = dict(
        forward_features_shape=list(ff_c.shape), patch_tokens_shape=list(t_c.shape), maskembed_patch_tokens_shape=list(t_m.shape),
        cls_dropped=bool(torch.equal(t_c, ff_c[:, 1:].float())), cls_equals_global_pool_output=bool(torch.allclose(pooled_c, ff_c[:, 0])),
        clip_timm_vs_openai_code_fp32=dict(**tdiff(t_c, t_ref), cos=tok_cos(t_c, t_ref)),
        clip_vs_clip_in_softcap_svit_fp32=tok_cos(ff_sc[:, 1:], t_c),
        bf16_vs_fp32=dict(clip=tok_cos(t_c16, t_c), maskembed=tok_cos(t_m16, t_m)),
        fp16_vs_fp32=dict(clip=tok_cos(t_cf16, t_c), maskembed=tok_cos(t_mf16, t_m), clip_max_abs=float((t_cf16 - t_c).abs().max()),
                          maskembed_max_abs=float((t_mf16 - t_m).abs().max()), finite=bool(torch.isfinite(t_cf16).all() and torch.isfinite(t_mf16).all())),
        maskembed_vs_clip_patch_cos=tok_cos(t_m, t_c), maskembed_vs_clip_cls_cos=tok_cos(ff_m[:, :1], ff_c[:, :1]),
        token_norm=dict(clip=float(t_c.norm(dim=-1).mean()), maskembed=float(t_m.norm(dim=-1).mean())),
        max_attn_logit_clip=None)
    assert t_c.shape == (3, 576, 1024) and t_m.shape == (3, 576, 1024)
    # largest |attention logit| of the original CLIP on these images (how much the soft cap 30*tanh(a/30) can matter)
    mx = []

    def hook(mod, inp, outp):
        xx = inp[0]
        B, N, C = xx.shape
        qkv = mod.qkv(xx).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0] * mod.scale, qkv[1]
        mx.append(float((q @ k.transpose(-2, -1)).abs().max()))
    hs = [b.attn.register_forward_hook(hook) for b in clip.blocks]
    with torch.no_grad():
        clip.forward_features(x[:1])
    for h in hs:
        h.remove()
    out["features"]["max_attn_logit_clip"] = dict(per_block_max=max(mx), blocks_over_10=sum(v > 10 for v in mx), blocks_over_30=sum(v > 30 for v in mx))
    json.dump(out, open(MC.run_path("sanity", "verify.json"), "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("sha256", "same_param_names_and_shapes", "maskembed_loaded_equals_ckpt", "clip_timm_vs_openai_pt",
                                          "init_correspondence", "maskembed_vs_clip_weight_cos", "features")}, indent=1, default=str))


if __name__ == "__main__":
    main()
