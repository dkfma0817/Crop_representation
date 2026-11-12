"""MaskEmbed vs OpenAI CLIP ViT-L/14@336 on PACO (2026-11-07): shared pieces.

Encoders (both frozen, both timm VisionTransformer, 24 blocks, width 1024, 1 prefix token (CLS), 24 x 24 patch grid):
  clip      : timm `vit_large_patch14_clip_quickgelu_336.openai` (= the MaskEmbed TEACHER model name and the student's
              pretrained init, ckpt args), standard timm attention. Weights verified equal to OpenAI ViT-L-14-336px.pt.
  maskembed : `svit_large_patch14_clip_quickgelu_336` (locality_alignment.backbones: same timm ViT with StableBlock =
              attention-logit soft-capping 30*tanh(a/30), the architecture MaskEmbed was trained in), pretrained=False,
              num_classes=0, official load_checkpoint_auto(strict=False) of clip-vit-l-336.pth.tar (encoder keys only;
              the MaskEmbed decoder is never built; the encoder's training output head Linear(1024,1024) is dropped
              as in the official README / patch-seg usage).
Features (identical extraction point for both): forward_features(x) -> final block output after the model's own final
LayerNorm (`norm`; = ln_post for CLIP, the re-initialised + trained norm for MaskEmbed), CLS dropped -> (B, 576, 1024).
No projection (CLIP proj 1024->768 / MaskEmbed head not used), no extra feature normalisation. bf16 autocast -> fp32
(the existing probe protocol).
Input: the existing letterbox canvases (src.viz.c0_pca.LB at S=336, P=14: longer side -> 336, PIL bicubic, centred,
padding = 0 AFTER normalisation) with the OpenAI CLIP mean / std instead of the ImageNet ones (set_clip_norm()).
"""
import importlib
import os
import sys

import torch

RUN = "output/2026-11-07/maskembed_paco"
S, P, G, DIM = 336, 14, 24, 1024
LA_REPO = "/home/irteam/ext_repos/locality-alignment"
CKPT_ME = "/home/irteam/ext_repos/checkpoints/maskembed/clip-vit-l-336.pth.tar"
CKPT_OAI = "/home/irteam/ext_repos/checkpoints/openai_clip/ViT-L-14-336px.pt"
SHA = {CKPT_ME: "ae12a3353d1707eabf35ae2eb75392166850f01d929b3f1e346ede546b480fa3",
       CKPT_OAI: "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"}
CLIP_ARCH = "vit_large_patch14_clip_quickgelu_336.openai"
ME_ARCH = "svit_large_patch14_clip_quickgelu_336"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
ENCODERS = ("clip", "maskembed")


def set_clip_norm():
    """LB.canvas reads the module tensors src.viz.c0_pca.MEAN_T / STD_T at call time -> overwrite them IN PLACE in this
    process (DataLoader workers fork / re-import through this module). No file is modified."""
    import src.viz.c0_pca as CP
    CP.MEAN_T.copy_(torch.tensor(CLIP_MEAN).view(3, 1, 1))
    CP.STD_T.copy_(torch.tensor(CLIP_STD).view(3, 1, 1))
    return CP


def _la():
    if LA_REPO not in sys.path:
        sys.path.insert(0, LA_REPO)
    import locality_alignment  # noqa: F401  (registers svit_* with timm)
    from locality_alignment.backbones import svit
    if not getattr(svit.StableBlock, "_timm_compat", False):
        svit.StableBlock = _compat_block(svit.StableBlock)
    return importlib.import_module("locality_alignment")


def _compat_block(base):
    """The repo targets timm 1.0.8; timm 1.0.27 (this env, also used for the CLIP baseline) passes extra block_fn kwargs.
    They are dropped ONLY at their no-op values (asserted), so the StableBlock built is the one the repo defines.
    In-process name rebinding of svit.StableBlock (create_svit looks it up at call time); no repo file is modified."""
    from timm.models.vision_transformer import Attention

    class StableBlockCompat(base):
        _timm_compat = True

        def __init__(self, *a, scale_attn_norm=False, scale_mlp_norm=False, proj_bias=True, attn_layer=Attention, depth=0,
                     device=None, dtype=None, **kw):
            assert not scale_attn_norm and not scale_mlp_norm and proj_bias and attn_layer is Attention, \
                (scale_attn_norm, scale_mlp_norm, proj_bias, attn_layer)
            assert device is None and dtype is None
            super().__init__(*a, **kw)

    StableBlockCompat.__name__ = "StableBlock"
    return StableBlockCompat


def load_me_state():
    import argparse
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(CKPT_ME, map_location="cpu", weights_only=True)


def build_encoder(name, dev="cpu", soft_cap_clip=False):
    """-> (vit, info). soft_cap_clip=True: the ORIGINAL CLIP weights inside the svit architecture (verification only)."""
    import timm
    if name == "clip" and not soft_cap_clip:
        vit = timm.create_model(CLIP_ARCH, pretrained=True, num_classes=0)
        info = dict(arch=CLIP_ARCH, weights="timm hub timm/vit_large_patch14_clip_336.openai")
    elif name == "clip":
        _la()
        vit = timm.create_model("svit_large_patch14_clip_quickgelu_336.openai", pretrained=True, num_classes=0)
        info = dict(arch="svit (soft-cap) + OpenAI CLIP weights", weights="timm hub")
    elif name == "maskembed":
        la = _la()
        vit = timm.create_model(ME_ARCH, pretrained=False, num_classes=0)
        import argparse
        with torch.serialization.safe_globals([argparse.Namespace]):    # torch>=2.6 weights_only default; ckpt holds args
            inc = la.load_checkpoint_auto(vit, CKPT_ME, strict=False)
        info = dict(arch=ME_ARCH, weights=CKPT_ME, missing_keys=list(inc.missing_keys), unexpected_keys=list(inc.unexpected_keys))
        assert not inc.missing_keys and sorted(inc.unexpected_keys) == ["head.bias", "head.weight"], inc
    else:
        raise ValueError(name)
    vit = vit.to(dev).eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    assert vit.num_prefix_tokens == 1 and vit.embed_dim == DIM and tuple(vit.patch_embed.grid_size) == (G, G)
    return vit, info


def tokens(vit, x, chunk=64, amp=True, dtype=torch.bfloat16):
    """(B,3,336,336) -> (B, 576, 1024) fp32 final-norm patch tokens (CLS dropped)."""
    out = []
    for i in range(0, len(x), chunk):
        with torch.autocast(x.device.type, dtype=dtype, enabled=amp):
            t = vit.forward_features(x[i:i + chunk])
        out.append(t[:, vit.num_prefix_tokens:].float())
    return torch.cat(out)


def openai_reference():
    """OpenAI's own model code (github.com/openai/CLIP clip/model.py, imported by path) built from ViT-L-14-336px.pt."""
    spec = importlib.util.spec_from_file_location("oai_clip_model", "/home/irteam/ext_repos/openai-CLIP/clip/model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jit = torch.jit.load(CKPT_OAI, map_location="cpu")
    sd = jit.state_dict()
    model = mod.build_model(sd).float().eval()
    return model, sd


def oai_patch_tokens(model, x):
    """OpenAI VisionTransformer.forward up to the transformer, then ln_post on ALL tokens (CLS dropped), no proj."""
    v = model.visual
    z = v.conv1(x).reshape(x.shape[0], v.conv1.out_channels, -1).permute(0, 2, 1)
    z = torch.cat([v.class_embedding.to(z.dtype) + torch.zeros(z.shape[0], 1, z.shape[-1], dtype=z.dtype), z], 1)
    z = v.ln_pre(z + v.positional_embedding.to(z.dtype))
    z = v.transformer(z.permute(1, 0, 2)).permute(1, 0, 2)
    return v.ln_post(z)[:, 1:]


def oai_to_timm_keys(sd):
    """OpenAI visual.* -> timm VisionTransformer names (the conversion timm applies for the .openai weights)."""
    out = {"cls_token": sd["visual.class_embedding"][None, None], "pos_embed": sd["visual.positional_embedding"][None],
           "patch_embed.proj.weight": sd["visual.conv1.weight"], "norm_pre.weight": sd["visual.ln_pre.weight"],
           "norm_pre.bias": sd["visual.ln_pre.bias"], "norm.weight": sd["visual.ln_post.weight"], "norm.bias": sd["visual.ln_post.bias"]}
    n = len({k.split(".")[3] for k in sd if k.startswith("visual.transformer.resblocks.")})
    for i in range(n):
        a, b = f"visual.transformer.resblocks.{i}.", f"blocks.{i}."
        for s, t in (("ln_1", "norm1"), ("ln_2", "norm2"), ("mlp.c_fc", "mlp.fc1"), ("mlp.c_proj", "mlp.fc2"), ("attn.out_proj", "attn.proj")):
            out[b + t + ".weight"], out[b + t + ".bias"] = sd[a + s + ".weight"], sd[a + s + ".bias"]
        out[b + "attn.qkv.weight"], out[b + "attn.qkv.bias"] = sd[a + "attn.in_proj_weight"], sd[a + "attn.in_proj_bias"]
    return out, sd["visual.proj"]


def sha256(p, n=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def run_path(*a):
    return os.path.join(RUN, *a)
