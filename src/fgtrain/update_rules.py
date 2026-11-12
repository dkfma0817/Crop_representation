"""Whole-encoder asymmetric projection and its norm-matched control."""
import math

import torch


GP_NORM_MIN = 1e-7
GC_NORM_MIN = 1e-12
RHO_EPSILON = 1e-12
FP32_EQ_RTOL = 1e-5
BF16_SUM_RTOL = 2e-2


class GradientLayout:
    """One immutable ordering, also retaining AdamW's unused-parameter semantics."""

    def __init__(self, encoder):
        self.named = [(name, parameter) for name, parameter in encoder.named_parameters() if parameter.requires_grad]
        self.parameters = [parameter for _, parameter in self.named]
        self.sizes = [parameter.numel() for parameter in self.parameters]
        self.names = [name for name, _ in self.named]

    def pack(self, gradients):
        return torch.cat([gradient.reshape(-1) if gradient is not None else torch.zeros_like(parameter).reshape(-1)
                          for parameter, gradient in zip(self.parameters, gradients)])

    def assign(self, gradient, used):
        for parameter, part, active in zip(self.parameters, gradient.split(self.sizes), used):
            parameter.grad = part.view_as(parameter) if active else None

    def block_summary(self, crop, patch):
        parts_c, parts_p = crop.split(self.sizes), patch.split(self.sizes)
        grouped = {}
        for (name, _), c, p in zip(self.named, parts_c, parts_p):
            if name.startswith('blocks.'):
                key = 'block_' + name.split('.')[1].zfill(2)
            elif name.startswith('patch_embed.'):
                key = 'patch_embed'
            elif name.startswith('norm.'):
                key = 'final_norm'
            else:
                key = 'other'
            dot = torch.stack([torch.dot(c, p), torch.dot(c, c), torch.dot(p, p)])
            grouped[key] = grouped[key] + dot if key in grouped else dot
        result = {}
        for name, tensor in grouped.items():
            dot, c2, p2 = tensor.detach().cpu().tolist()
            denominator = math.sqrt(max(c2, 0) * max(p2, 0))
            result[name] = dict(crop_norm=math.sqrt(max(c2, 0)), patch_norm=math.sqrt(max(p2, 0)),
                                cosine=dot / denominator if denominator else None)
        return result


def separate_gradients(losses, layout, retain_graph=False):
    gradients = []
    active = [False] * len(layout.parameters)
    for index, loss in enumerate(losses):
        parts = torch.autograd.grad(loss, layout.parameters, retain_graph=index < 2 or retain_graph,
                                    create_graph=False, allow_unused=True)
        active = [old or part is not None for old, part in zip(active, parts)]
        gradients.append(layout.pack(parts))
    return gradients, active


@torch.no_grad()
def combine(crop, global_gradient, patch, preserve_weight, condition):
    """Operate on unscaled FP32 gradients from the same effective global batch."""
    assert condition in ('U1', 'U2')
    assert crop.dtype == global_gradient.dtype == patch.dtype == torch.float32
    assert crop.shape == global_gradient.shape == patch.shape
    scalars = torch.stack([torch.dot(crop, patch), torch.dot(crop, crop),
                           torch.dot(patch, patch), torch.dot(global_gradient, global_gradient)])
    if not bool(torch.isfinite(scalars).all()):
        raise FloatingPointError('Non-finite per-loss encoder gradient before projection')
    dot, c2, p2, g2 = scalars.cpu().tolist()
    crop_norm, patch_norm, global_norm = (math.sqrt(max(x, 0)) for x in (c2, p2, g2))
    valid = patch_norm > GP_NORM_MIN and crop_norm > GC_NORM_MIN
    conflict = dot < 0
    applied = conflict and valid
    coefficient = dot / p2 if applied else 0.
    projected = crop.add(patch, alpha=-coefficient) if applied else crop
    projected_norm = float(torch.linalg.vector_norm(projected)) if applied else crop_norm
    rho = projected_norm / (crop_norm + RHO_EPSILON) if applied else 1.
    effective_weight = preserve_weight - coefficient
    if condition == 'U1':
        gradient = projected + .5 * global_gradient + preserve_weight * patch
    else:
        gradient = rho * crop + .5 * global_gradient + preserve_weight * patch
    norm = float(torch.linalg.vector_norm(gradient))
    if not math.isfinite(norm):
        raise FloatingPointError('Non-finite composite encoder gradient')
    info = dict(crop_norm=crop_norm, global_norm=global_norm, patch_norm=patch_norm,
                crop_patch_dot=dot, crop_patch_cosine=dot / (crop_norm * patch_norm) if crop_norm * patch_norm else None,
                conflict=conflict, projection_applied=applied, projection_skipped_tiny=not valid,
                projected_crop_norm=projected_norm, rho=rho,
                lambda_eff=effective_weight, lambda_eff_ratio=effective_weight / preserve_weight,
                lambda_applied=effective_weight if condition == 'U1' else preserve_weight,
                composite_norm=norm, nonfinite=False)
    return gradient, projected, info
