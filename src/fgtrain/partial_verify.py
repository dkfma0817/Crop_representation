"""Endpoint and numerical tests for the two fixed partial-projection pilots."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

import src.readout.common as RC
from src.config import STREAM
from src.fgtrain.common import encoder, state_sha, write_json
from src.fgtrain.patch_preserve import PatchData
from src.fgtrain.p1 import read, sha
from src.fgtrain.partial_rules import partial_combine
from src.fgtrain.update_rules import (
    GradientLayout, separate_gradients, combine as full_projection,
    GP_NORM_MIN, FP32_EQ_RTOL, BF16_SUM_RTOL,
)
from src.fgtrain.update_verify import unit_vectors, error, loss_terms


def check_vectors(crop, glob, patch, weight):
    baseline = crop + .5 * glob + weight * patch
    full, _, _ = full_projection(crop, glob, patch, weight, 'U1')
    records = {}
    for alpha in (0., .25, .5, 1.):
        result, partial, info = partial_combine(crop, glob, patch, weight, alpha)
        if alpha == 0:
            assert torch.equal(result, baseline), 'alpha0 endpoint'
        if alpha == 1:
            assert torch.equal(result, full), 'alpha1 endpoint'
        if info['projection_applied']:
            coefficient = alpha * info['crop_patch_dot'] / float(torch.dot(patch, patch))
            expected_crop = crop.add(patch, alpha=-coefficient)
        else:
            expected_crop = crop
        expected = expected_crop + .5 * glob + weight * patch
        alternate = crop + .5 * glob + info['lambda_eff'] * patch
        direct_error, adaptive_error = error(result, expected), error(result, alternate)
        assert direct_error['relative_l2'] < FP32_EQ_RTOL, direct_error
        assert adaptive_error['relative_l2'] < FP32_EQ_RTOL, adaptive_error
        assert torch.isfinite(result).all() and torch.isfinite(partial).all()
        if not info['projection_applied']:
            assert torch.equal(result, baseline) and info['rho'] == 1.
        records[str(alpha)] = dict(info=info, direct_formula_error=direct_error,
                                   adaptive_lambda_error=adaptive_error,
                                   endpoint_bit_equal=(torch.equal(result, baseline) if alpha == 0
                                                       else torch.equal(result, full) if alpha == 1 else None))
        del result, partial, expected, alternate, expected_crop
    return records


def unit_tests(root):
    torch.set_num_threads(2)
    unit_vectors(root)  # Unmodified original U1/U2 tests, including P2 sum.
    generator = torch.Generator().manual_seed(0)
    crop, glob, independent = [torch.randn(10003, generator=generator) for _ in range(3)]
    patch = independent - 1.5 * crop
    weight = .06517137116454548
    cases = {}
    for name, c, p in [('conflict', crop, patch), ('nonconflict', crop, crop),
                       ('zero_patch', crop, torch.zeros_like(crop)),
                       ('tiny_patch', crop, -crop * 1e-12),
                       ('zero_crop', torch.zeros_like(crop), patch),
                       ('tiny_crop', crop * 1e-16, patch)]:
        cases[name] = check_vectors(c, glob, p, weight)
    for bad_value in (float('nan'), float('inf')):
        bad = crop.clone()
        bad[0] = bad_value
        for alpha in (.25, .5):
            try:
                partial_combine(bad, glob, patch, weight, alpha)
            except FloatingPointError:
                continue
            raise AssertionError('Non-finite gradient was accepted')
    write_json(Path(root) / 'partial_unit_tests.json', dict(status='passed', cases=cases,
               endpoints_bit_equal=True, nonfinite_rejected=True,
               numerical_skip='Unchanged U1: gp<=1e-7 or gc<=1e-12',
               coefficient_autograd='torch.no_grad, no extra derivative',
               reuse='Unchanged update_rules.combine(U1) followed by fixed-alpha interpolation'))


def preflight(root, base, prior, p2):
    root, base, prior, p2 = map(Path, (root, base, prior, p2))
    assert read(root / 'partial_unit_tests.json')['status'] == 'passed'
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher = encoder(base / 'initial_encoder.pt')
    teacher_sha = state_sha(teacher.state_dict())
    versions = [p._version for p in teacher.parameters()]
    student = encoder(base / 'initial_encoder.pt', train=True)
    layout = GradientLayout(student)
    ordering = hashlib.sha256('\n'.join(layout.names).encode()).hexdigest()
    old = read(prior / 'preflight.json')
    assert ordering == old['parameter_order_sha256'] and old['status'] == 'passed'
    assert old['FP32_encoder_sum_unit_test']['relative_l2'] < FP32_EQ_RTOL
    dataset = PatchData(base / 'teacher')
    excluded = set(read(base / 'teacher/manifest.json')['excluded_train_images'])
    ids = [int(i) for i in np.load(STREAM)[0] if int(i) not in excluded]
    batch = RC.to_dev(dataset[ids], 'cuda')
    weight = read(p2 / 'P2/config.json')['lambda_preserve']
    records = []
    for label, path in [('Original', base / 'initial_encoder.pt'),
                        ('P2_final12000_EMA', p2 / 'P2/export/encoder.pt')]:
        state = torch.load(path, map_location='cpu', weights_only=True)
        student.load_state_dict(state, strict=True)
        del state
        existing, losses = loss_terms(batch, student, teacher)
        gradients, active = separate_gradients(losses, layout, retain_graph=True)
        reference = torch.autograd.grad(losses[0] + .5 * losses[1] + weight * losses[2],
                                        layout.parameters, allow_unused=True)
        packed = layout.pack(reference)
        sum_error = error(gradients[0] + .5 * gradients[1] + weight * gradients[2], packed)
        assert sum_error['relative_l2'] < BF16_SUM_RTOL, sum_error
        cases = check_vectors(*gradients, weight)
        if label == 'Original':
            assert cases['0.25']['info']['patch_norm'] <= GP_NORM_MIN
            assert not cases['0.25']['info']['projection_applied']
        blocks = layout.block_summary(gradients[0], gradients[2])
        for alpha in (.25, .5):
            fused, projected, info = partial_combine(*gradients, weight, alpha)
            total = layout.block_summary(fused, fused)
            assert all(total[k]['crop_norm'] > 0 for k in
                       ['patch_embed', 'final_norm'] + [f'block_{i:02d}' for i in range(12)])
            del fused, projected
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        assert [p._version for p in teacher.parameters()] == versions
        record = dict(snapshot=label, checkpoint=str(path), checkpoint_sha256=sha(path),
                      batch_index=0, image_ids=ids, BF16_sum_error=sum_error,
                      partial_cases=cases, block_gradients=blocks,
                      unused_parameters=[n for n,v in zip(layout.names, active) if not v])
        records.append(record)
        write_json(root / 'preflight_progress.json', record)
        print(f'{label}: actual batch endpoint, partial and BF16 tests passed', flush=True)
        del gradients, losses, existing, reference, packed
    assert state_sha(teacher.state_dict()) == teacher_sha
    write_json(root / 'preflight.json', dict(status='passed', records=records,
               parameter_order_sha256=ordering, parameter_count=sum(layout.sizes),
               teacher_frozen=True, teacher_state_sha256=teacher_sha,
               reused_U1_FP32_sum_reference=dict(source=str(prior / 'preflight.json'),
                   sha256=sha(prior / 'preflight.json'), result=old['FP32_encoder_sum_unit_test']),
               BF16_sum_relative_tolerance=BF16_SUM_RTOL, FP32_relative_tolerance=FP32_EQ_RTOL,
               no_optimizer_update=True, no_retraining=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--base-run')
    p.add_argument('--prior-run')
    p.add_argument('--p2-run')
    p.add_argument('--stage', choices=('unit', 'preflight'), required=True)
    args = p.parse_args()
    if args.stage == 'unit':
        unit_tests(args.run)
    else:
        preflight(args.run, args.base_run, args.prior_run, args.p2_run)


if __name__ == '__main__':
    main()
