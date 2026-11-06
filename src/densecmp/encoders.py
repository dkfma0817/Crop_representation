"""Frozen EVA02-CLIP-B/16 family encoders through the OFFICIAL repo code (one repo per process: both ship a package
named `open_clip`).

  name        repo      checkpoint                                         dense path (official)
  eva         CLIPSelf  EVA02_CLIP_B_psz16_s8B.pt (original EVA-CLIP)      CLIPSelf encode_dense ('v2'): blocks 1-11, last block
                                                                           WITHOUT attention mixing (v-proj + MLP), CLS dropped,
                                                                           final LN, head (768->512), L2-normalised
  clipself    CLIPSelf  eva_vitb16_coco_clipself_patches.pt               same path (= CLIPSelf test_eva_vitb16_macc_boxes_masks.sh)
  declip      DeCLIP    evab_dinov2B_csa_560_0.25_seg.pt (internal name   DeCLIP encode_dense mode 'qq' (= inference mode of a
                        'evab_dinov2B_qq_560_0.25', epoch 6)               qq_vfm_distill model in DeCLIP training/zero_shot.py):
                                                                           last block = softmax(q q^T) v, no residual / FFN, final
                                                                           LN, head, L2-normalised = CONTENT feature
  eva_qq      DeCLIP    original EVA-CLIP                                   DeCLIP 'qq' path on the original weights (path control)
  declip_csa  DeCLIP    epoch_6.pt of DeCLIP_EVA-B_DINOv2-B_csa_0.05_2.0   DeCLIP encode_dense mode 'csa' (its training mode)

Model construction = repo `open_clip.eva_clip.create_model('EVA02-CLIP-B-16', pretrained=<ckpt>, force_custom_clip=True)`
(the route of `create_model(..., pretrained='eva', cache_dir=<ckpt>)` used by both training/main.py). Positional embedding
(abs, bicubic) and RoPE frequencies (intp_freq, bicubic) are interpolated by the official code for inputs other than 224.
xformers is not installed: `xops.memory_efficient_attention` is replaced by an SDPA shim with the same signature/layout
(equivalence vs the repo's own non-xformers branch is checked in verify.py). Forward under FP16 autocast = the repos' default
`--precision amp` (torch.cuda.amp.autocast). bf16 was measured to corrupt EVA dense tokens at 1024 (3% of content tokens with
cos < 0.99 vs fp32, min 0.32), fp16 matches fp32 (mean cos 1.0000, p1 0.9995); see verify/precision_check.txt. Output fp32.
"""
import os
import sys
import types

import torch
import torch.nn.functional as F

CK = "/home/irteam/ext_repos/checkpoints"
REPO = {"CLIPSelf": "/home/irteam/ext_repos/CLIPSelf/src", "DeCLIP": "/home/irteam/ext_repos/DeCLIP/src"}
ENC = {
    "eva": dict(repo="CLIPSelf", ckpt="EVA02_CLIP_B_psz16_s8B.pt", path="v2"),
    "clipself": dict(repo="CLIPSelf", ckpt="eva_vitb16_coco_clipself_patches.pt", path="v2"),
    "declip": dict(repo="DeCLIP", ckpt="declip_zsss_evab_dinov2B_csa_560_0.25_seg.pt", path="qq"),
    "eva_qq": dict(repo="DeCLIP", ckpt="EVA02_CLIP_B_psz16_s8B.pt", path="qq"),
    "declip_csa": dict(repo="DeCLIP", ckpt="declip_catseg_EVA-B_DINOv2-B_csa_0.05_2.0_epoch_6.pt", path="csa"),
}
MODEL_NAME = "EVA02-CLIP-B-16"
PATCH = 16
DIM = 512


def _sdpa_xops():
    def memory_efficient_attention(q, k, v, p=0.0, scale=None, attn_bias=None):
        """xformers layout (B, N, H, C) in / out."""
        assert attn_bias is None
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=p, scale=scale)
        return o.transpose(1, 2)
    return types.SimpleNamespace(memory_efficient_attention=memory_efficient_attention)


def _allow_numpy_globals():
    """DeCLIP checkpoints store numpy scalars (optimizer / scaler state); allow-list exactly these for weights_only load
    (pickled under the numpy-1 names 'numpy.core.multiarray.scalar' / 'numpy.dtype')."""
    import numpy as np
    sc = np._core.multiarray.scalar if hasattr(np, "_core") else np.core.multiarray.scalar
    torch.serialization.add_safe_globals([(sc, "numpy.core.multiarray.scalar"), sc, np.dtype,
                                          *[type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32", "bool")]])


_allow_numpy_globals()


def import_repo(repo):
    src = REPO[repo]
    for m in [m for m in sys.modules if m == "open_clip" or m.startswith("open_clip.") or m == "training" or m.startswith("training.")]:
        mod = sys.modules[m]
        if getattr(mod, "__file__", None) and not mod.__file__.startswith(src):
            raise RuntimeError(f"another repo's open_clip already imported ({mod.__file__}); one repo per process")
    if src not in sys.path:
        sys.path.insert(0, src)
    import open_clip  # noqa: F401
    from open_clip import eva_clip
    import open_clip.eva_clip.eva_vit_model as EV
    assert EV.__file__.startswith(src), EV.__file__
    if getattr(EV, "xops", None) is None:          # repo sets xops = None when xformers is missing
        EV.xops = _sdpa_xops()
    return eva_clip, EV


def state_dict_of(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    meta = {k: ck[k] for k in ("epoch", "name") if isinstance(ck, dict) and k in ck}
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    return sd, meta


class Encoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        cfg = ENC[name]
        self.name, self.path, self.ckpt = name, cfg["path"], os.path.join(CK, cfg["ckpt"])
        eva_clip, self.EV = import_repo(cfg["repo"])
        self.repo_src = REPO[cfg["repo"]]
        self.model = eva_clip.create_model(MODEL_NAME, pretrained=self.ckpt, force_custom_clip=True, precision="fp32", device="cpu")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.load_report = self._check_load()

    @torch.no_grad()
    def _check_load(self):
        """Every visual tensor of the model must equal the checkpoint tensor (rope freqs are recomputed buffers; compared)."""
        sd, meta = state_dict_of(self.ckpt)
        ck_vis = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
        mv = self.model.visual.state_dict()
        missing = [k for k in mv if k not in ck_vis]
        unexpected = [k for k in ck_vis if k not in mv]
        mism, rope = [], {}
        for k, v in mv.items():
            if k not in ck_vis:
                continue
            c = ck_vis[k].to(v.dtype)
            if "freqs_" in k:
                rope[k] = float((v - c).abs().max()) if v.shape == c.shape else f"shape {tuple(v.shape)} vs {tuple(c.shape)}"
            elif v.shape != c.shape or not torch.equal(v, c):
                mism.append(k)
        n_params = sum(v.numel() for k, v in mv.items() if "freqs_" not in k)
        return dict(ckpt=self.ckpt, ckpt_meta={k: str(v) for k, v in meta.items()}, n_visual_tensors_model=len(mv),
                    n_visual_tensors_ckpt=len(ck_vis), missing_in_ckpt=missing, unexpected_in_ckpt=unexpected,
                    value_mismatch=mism, rope_buffer_maxdiff=rope, n_visual_params=n_params, repo=self.repo_src, path=self.path)

    def forward(self, x):
        """x (B,3,S,S) normalised -> (B, (S/16)^2, 512) float32, L2-normalised (official dense output)."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            if self.path == "v2":
                f = self.model.encode_dense(x, normalize=True, keep_shape=False)
            else:
                f = self.model.encode_dense(x, normalize=True, keep_shape=False, mode=self.path)
        f = f.float()
        assert torch.isfinite(f).all(), "non-finite encoder output"
        return f
