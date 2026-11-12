"""Frozen official DINOv3-B/BRIXEL features with an audited common backbone.

Inputs are already ImageNet-normalized RGB tensors resized to 256 x 256 by
the caller. B0 and B2 differ only in the caller's full-image/crop input; both
return final normalized patch tokens. B1 returns the official BRIXEL head.
No downloads, source patches, or CUDA-extension imports are needed for B0/B2.

The cached timm EVA checkpoint is converted back to official names. timm's
conversion discards the distilled ViT-B's zero QKV biases, the unused mask
token, and deterministic RoPE/bias-mask buffers. These are reconstructed
explicitly, rather than left randomly initialized. This recovers inference
weights, not evidence of byte identity with an unavailable original .pth.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLONE_DIR = ROOT / "third_party/BRIXEL"
DEFAULT_WEIGHTS_DIR = ROOT / "output/2026-11-10/brixel_vitb256_s0/weights"
DEFAULT_ADAPTER_WEIGHTS = DEFAULT_WEIGHTS_DIR / "brixel_vitb256.pth"
DEFAULT_BACKBONE_WEIGHTS = DEFAULT_WEIGHTS_DIR / "dinov3_vitb16_from_timm.pth"
TIMM_REPOSITORY = "models--timm--vit_base_patch16_dinov3.lvd1689m"
BACKBONE_PREFIX = "adapter.backbone."


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash every key, dtype, shape and tensor byte, including all buffers."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        header = json.dumps([key, str(value.dtype), list(value.shape)],
                            separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_tensor(left: Tensor, right: Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    # Compare bytes so signed zeros, buffers, and dtypes cannot escape auditing.
    left = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    right = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left, right)


def _load_state(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Expected a nonempty flat state dictionary: {path}")
    if any(not isinstance(k, str) or not isinstance(v, Tensor)
           for k, v in state.items()):
        raise ValueError(f"Expected tensor values and unwrapped checkpoint keys: {path}")
    return dict(state)


def find_cached_timm_weights() -> Path:
    """Resolve the local HF snapshot, preferring its current main revision."""
    hf_home = Path(os.environ.get("HF_HOME", str(
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "huggingface")))
    hub = Path(os.environ.get("HF_HUB_CACHE", os.environ.get(
        "HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))))
    repository = hub / TIMM_REPOSITORY
    main = repository / "refs/main"
    if main.is_file():
        candidate = repository / "snapshots" / main.read_text().strip() / "model.safetensors"
        if candidate.is_file():
            return candidate.absolute()
    candidates = sorted((repository / "snapshots").glob("*/model.safetensors"))
    candidates = [p for p in candidates if p.is_file()]
    if len(candidates) == 1:
        return candidates[0].absolute()
    raise FileNotFoundError(
        f"Cannot uniquely resolve cached timm DINOv3-B weights under {repository}; "
        "provide timm_weights explicitly. No network fallback is permitted.")


def _official_factory(clone_dir: Path):
    source = clone_dir / "src"
    dino_source = source / "brixel/dinov3_main"
    if not (dino_source / "dinov3/hub/backbones.py").is_file():
        raise FileNotFoundError(f"Official BRIXEL clone not found: {clone_dir}")
    # Both names occur in the official clone's imports. Refuse a previously
    # imported, unrelated installation instead of silently mixing implementations.
    for name in ("brixel", "dinov3"):
        for module_name, module in tuple(sys.modules.items()):
            if module_name == name or module_name.startswith(name + "."):
                origin = getattr(module, "__file__", None)
                if origin and not Path(origin).resolve().is_relative_to(source):
                    raise RuntimeError(f"{module_name} already imported outside {source}: {origin}")
    for directory in (source, dino_source):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    return importlib.import_module("dinov3.hub.backbones").dinov3_vitb16


def _convert_timm(state: Mapping[str, Tensor], backbone: nn.Module):
    """Invert timm/models/eva.py:checkpoint_filter_fn for distilled ViT-B."""
    converted = {}
    for key, value in state.items():
        target = key.replace("reg_token", "storage_tokens")
        target = target.replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma")
        if target in converted:
            raise ValueError(f"Ambiguous timm key conversion: {key}")
        converted[target] = value

    expected = backbone.state_dict()
    reconstructed = []
    for key, value in expected.items():
        if key in converted:
            continue
        if key == "mask_token" or key.endswith(".attn.qkv.bias"):
            converted[key] = torch.zeros_like(value)
        elif key.endswith(".attn.qkv.bias_mask"):
            mask = torch.ones_like(value)
            mask[768:1536] = 0
            converted[key] = mask
        elif key == "rope_embed.periods":
            converted[key] = value.detach().clone()  # deterministic official base=100 buffer
        else:
            raise ValueError(f"Cached timm checkpoint is missing learned tensor: {key}")
        reconstructed.append(key)
    _check_schema(converted, expected, "converted timm backbone")
    return converted, reconstructed


def _check_schema(state: Mapping[str, Tensor], expected: Mapping[str, Tensor], label: str):
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    incompatible = [k for k in state.keys() & expected.keys()
                    if state[k].shape != expected[k].shape or state[k].dtype != expected[k].dtype]
    if missing or unexpected or incompatible:
        raise ValueError(f"Invalid {label}: missing={missing}, unexpected={unexpected}, "
                         f"shape_or_dtype_mismatch={sorted(incompatible)}")


def _verified_backbone(clone_dir: Path, timm_weights: Path, backbone_weights: Path | None):
    # Factory initializes only before strict loading; no random tensor survives.
    backbone = _official_factory(clone_dir)(pretrained=False)
    converted, reconstructed = _convert_timm(_load_state(timm_weights), backbone)
    if backbone_weights is None:
        state = converted
    else:
        state = _load_state(backbone_weights)
        _check_schema(state, converted, "official backbone")
        # This also checks the reconstructed constants; do not accept nonzero
        # QKV biases or altered buffers just because learned weight shapes match.
        different = [k for k in converted if not _same_tensor(state[k], converted[k])]
        if different:
            raise ValueError(f"Backbone differs from cached timm reference: {different}")
    backbone.load_state_dict(state, strict=True)
    if state_dict_sha256(backbone.state_dict()) != state_dict_sha256(converted):
        raise RuntimeError("Strict backbone load changed the reference tensor bytes")
    audit = {
        "clone_dir": str(clone_dir),
        "timm_weights": str(timm_weights),
        "timm_file_sha256": _file_sha256(timm_weights),
        "backbone_weights": str(backbone_weights) if backbone_weights else None,
        "backbone_file_sha256": _file_sha256(backbone_weights) if backbone_weights else None,
        "backbone_tensor_count": len(state),
        "backbone_strict": True,
        "reference": "cached timm vit_base_patch16_dinov3.lvd1689m",
        "reconstructed_keys": reconstructed,
        "reconstruction_note": (
            "timm EVA drops distilled ViT-B zero QKV biases, the unused mask token, "
            "RoPE periods and QKV bias masks. These are reconstructed with zero "
            "tokens/biases and deterministic official buffers. Inference-equivalent "
            "recovery; original omitted checkpoint bytes cannot be verified."),
    }
    return backbone, audit


def prepare_backbone_weights(
    output_path: str | Path = DEFAULT_BACKBONE_WEIGHTS,
    *,
    clone_dir: str | Path = DEFAULT_CLONE_DIR,
    timm_weights: str | Path | None = None,
) -> dict:
    """Save a new official-format .pth; verify and reuse, never replace, an existing one."""
    destination = Path(output_path).expanduser().resolve()
    reference = Path(timm_weights).expanduser().absolute() if timm_weights else find_cached_timm_weights()
    backbone, audit = _verified_backbone(
        Path(clone_dir).expanduser().resolve(), reference,
        destination if destination.exists() else None)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Serialize completely before atomically exposing a new file. link()
        # refuses to overwrite an existing checkpoint, including concurrent writes.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".pth", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(backbone.state_dict(), stream)
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    audit.update(backbone_weights=str(destination),
                 backbone_file_sha256=_file_sha256(destination),
                 backbone_sha256=state_dict_sha256(backbone.state_dict()))
    return audit


def audit_and_load_adapter(model: nn.Module, checkpoint: Mapping[str, Tensor]) -> dict:
    """Reject unknown/missing adapter keys and any frozen-backbone override.

    Backbone keys may be entirely absent (official adapter-only export) or
    present, but every supplied tensor must match the independent reference.
    All adapter/head parameters AND buffers are required, with exact dtypes.
    """
    expected = model.state_dict()
    backbone = model.adapter.backbone
    before = state_dict_sha256(backbone.state_dict())
    required = {k: v for k, v in expected.items() if not k.startswith(BACKBONE_PREFIX)}
    supplied = {k: v for k, v in checkpoint.items() if not k.startswith(BACKBONE_PREFIX)}
    _check_schema(supplied, required, "adapter/head checkpoint")
    backbone_keys = [k for k in checkpoint if k.startswith(BACKBONE_PREFIX)]
    different = [k for k in backbone_keys
                 if k not in expected or not _same_tensor(checkpoint[k], expected[k])]
    if different:
        raise ValueError(f"Adapter checkpoint attempts a frozen backbone mismatch: {sorted(different)}")
    # Never pass checkpoint backbone tensors to the loader. Complete its state
    # with independently loaded reference tensors and enforce strict=True.
    merged = dict(expected)
    merged.update(supplied)
    model.load_state_dict(merged, strict=True)
    after = state_dict_sha256(backbone.state_dict())
    if before != after:
        raise RuntimeError(f"Adapter load altered backbone: {before} -> {after}")
    loaded = model.state_dict()
    if any(not _same_tensor(loaded[k], v) for k, v in supplied.items()):
        raise RuntimeError("Strict adapter load did not preserve checkpoint bytes")
    return {
        "adapter_strict": True,
        "adapter_tensor_count": len(required),
        "adapter_backbone_tensor_count": len(backbone_keys),
        "adapter_missing_keys": [],
        "adapter_unexpected_keys": [],
        "adapter_backbone_mismatches": [],
        "backbone_sha256_before": before,
        "backbone_sha256_after": after,
        "backbone_unchanged": True,
    }


class FrozenBackbone(nn.Module):
    """Official frozen B0/B1/B2 encoder; crop extraction belongs to the caller."""

    embed_dim = feature_dim = 768
    patch_size = 16
    input_size = 256

    def __init__(
        self,
        variant: str = "B0",
        *,
        clone_dir: str | Path = DEFAULT_CLONE_DIR,
        timm_weights: str | Path | None = None,
        backbone_weights: str | Path | None = None,
        adapter_weights: str | Path | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if variant not in ("B0", "B1", "B2"):
            raise ValueError(f"Expected B0, B1 or B2, got {variant!r}")
        self.variant = variant
        self.grid_size = 64 if variant == "B1" else 16
        clone = Path(clone_dir).expanduser().resolve()
        reference = Path(timm_weights).expanduser().absolute() if timm_weights else find_cached_timm_weights()
        weights = Path(backbone_weights).expanduser().resolve() if backbone_weights else None
        backbone, self.audit = _verified_backbone(clone, reference, weights)
        before = state_dict_sha256(backbone.state_dict())
        self.audit.update(variant=variant, grid_size=self.grid_size,
                          backbone_sha256_before=before, backbone_sha256_after=before,
                          backbone_unchanged=True, adapter_weights=None)
        if variant == "B1":
            adapter_path = Path(adapter_weights or DEFAULT_ADAPTER_WEIGHTS).expanduser().resolve()
            checkpoint = _load_state(adapter_path)
            # Lazy: importing this module requires the official CUDA extension.
            try:
                official = importlib.import_module("brixel.models")
            except (ImportError, RuntimeError, OSError) as exc:
                raise RuntimeError(
                    "B1 requires the official BRIXEL dependencies and matching "
                    "MultiScaleDeformableAttention extension. B0/B2 remain usable. "
                    f"Original import error: {exc}") from exc
            self.model = official.BRIXEL(official.DINOv3_Adapter(
                backbone=backbone, interaction_indexes=[3, 6, 9, 11]))
            if state_dict_sha256(backbone.state_dict()) != before:
                raise RuntimeError("Official adapter construction altered the backbone")
            self.audit.update(audit_and_load_adapter(self.model, checkpoint))
            self.audit.update(adapter_weights=str(adapter_path),
                              adapter_file_sha256=_file_sha256(adapter_path))
        else:
            self.model = backbone
        self.requires_grad_(False)
        self.eval()
        self.to(device=device)
        after = state_dict_sha256(self.backbone.state_dict())
        if before != after:
            raise RuntimeError("Backbone tensor bytes changed during model construction/device transfer")
        self.audit.update(backbone_sha256_after=after, backbone_sha256=after,
                          model_sha256=state_dict_sha256(self.model.state_dict()),
                          all_frozen=all(not p.requires_grad for p in self.parameters()),
                          eval_mode=not any(m.training for m in self.modules()))

    @property
    def backbone(self) -> nn.Module:
        return self.model.adapter.backbone if self.variant == "B1" else self.model

    def train(self, mode: bool = True):
        """Keep stochastic layers and BatchNorm frozen even under parent.train()."""
        return super().train(False)

    @torch.no_grad()
    def features(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError(f"Expected normalized RGB [B,3,256,256], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("Expected floating-point, already normalized inputs")
        if self.variant == "B1":
            output = self.model(x)
        else:
            patches = self.backbone.forward_features(x)["x_norm_patchtokens"]
            output = patches.transpose(1, 2).reshape(x.shape[0], 768, 16, 16)
        expected = (x.shape[0], 768, self.grid_size, self.grid_size)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"Official {self.variant} output {tuple(output.shape)} != {expected}")
        return output.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)
