"""Fixed partial projections built directly from the unchanged U1 projection."""
import math

import torch

from src.fgtrain.update_rules import combine as full_projection
from src.fgtrain.update_rules import RHO_EPSILON


ALPHAS = {'U3': .25, 'U4': .50}


@torch.no_grad()
def partial_combine(crop, glob, patch, weight, alpha):
    assert alpha in (0., .25, .5, 1.), 'Only the two fixed pilots and unit-test endpoints'
    full, projected, original = full_projection(crop, glob, patch, weight, 'U1')
    info = dict(original)
    applied = original['projection_applied'] and alpha > 0
    if alpha == 1:
        gradient, partial = full, projected
    else:
        # Reuse U1's whole-encoder dot/norm, numerical skips and projection.
        # Interpolation is the partial-projection formula, up to FP32 rounding.
        partial = torch.lerp(crop, projected, alpha) if applied else crop
        gradient = partial + .5 * glob + weight * patch
    after = float(torch.linalg.vector_norm(partial)) if applied else original['crop_norm']
    rho = after / (original['crop_norm'] + RHO_EPSILON) if applied else 1.
    effective = weight + alpha * (original['lambda_eff'] - weight)
    norm = float(torch.linalg.vector_norm(gradient))
    if not math.isfinite(norm):
        raise FloatingPointError('Non-finite partial composite gradient')
    info.update(alpha=alpha, projection_applied=applied, projected_crop_norm=after,
                crop_norm_before=original['crop_norm'], crop_norm_after=after, rho=rho,
                lambda_eff=effective, lambda_eff_ratio=effective / weight,
                lambda_applied=effective, composite_norm=norm)
    return gradient, partial, info


def combine(crop, glob, patch, weight, condition):
    return partial_combine(crop, glob, patch, weight, ALPHAS[condition])
