"""Fixed numerical tests and a read-only final-P2 gradient preflight."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import src.readout.common as RC
from src.config import STREAM
from src.fgtrain.common import encoder, state_sha, write_json
from src.fgtrain.patch_preserve import PatchData, patch_loss
from src.fgtrain.train import terms
from src.fgtrain.p1 import read, sha
from src.ivc.ft import run_enc
from src.fgtrain.update_rules import (
    GradientLayout, separate_gradients, combine, GP_NORM_MIN, GC_NORM_MIN,
    RHO_EPSILON, FP32_EQ_RTOL, BF16_SUM_RTOL,
)


def error(actual, expected):
    difference = actual - expected
    scale = max(float(torch.linalg.vector_norm(expected)), 1e-30)
    return dict(relative_l2=float(torch.linalg.vector_norm(difference)) / scale,
                max_absolute=float(difference.abs().max()), reference_norm=scale)


def unit_vectors(out):
    generator = torch.Generator().manual_seed(0)
    crop, patch, glob = [torch.randn(10003, generator=generator) for _ in range(3)]
    patch -= 1.5 * crop
    weight = .06517137116454548
    u1, projected, info = combine(crop, glob, patch, weight, 'U1')
    alternate = crop + .5 * glob + info['lambda_eff'] * patch
    equivalence = error(u1, alternate)
    assert info['projection_applied'] and equivalence['relative_l2'] < FP32_EQ_RTOL
    assert abs(float(torch.dot(projected, patch))) < 2e-5 * float(crop.norm() * patch.norm())
    u2, projected2, info2 = combine(crop, glob, patch, weight, 'U2')
    ratio_error = abs(float((info2['rho'] * crop).norm()) - float(projected2.norm())) / float(projected2.norm())
    assert ratio_error < FP32_EQ_RTOL and torch.equal(projected, projected2)
    for vector in (crop, torch.zeros_like(crop), crop * 1e-12):
        baseline = crop + .5 * glob + weight * vector
        for condition in ('U1', 'U2'):
            fused, _, check = combine(crop, glob, vector, weight, condition)
            assert torch.equal(fused, baseline)
            assert not check['projection_applied'] and check['rho'] == 1
    for condition in ('U1', 'U2'):
        _, _, check = combine(torch.zeros_like(crop), glob, patch, weight, condition)
        assert not check['projection_applied'] and check['rho'] == 1
    bad = crop.clone()
    bad[0] = float('nan')
    try:
        combine(bad, glob, patch, weight, 'U1')
        raise AssertionError('Non-finite gradient was accepted')
    except FloatingPointError:
        pass
    parameter = torch.randn(3, 5, generator=generator, requires_grad=True)
    crop_loss = parameter.square().mean()
    global_loss = parameter.sin().mean()
    patch_loss_value = parameter.cos().mean()
    parts = [torch.autograd.grad(loss, parameter, retain_graph=True)[0] for loss in (crop_loss, global_loss, patch_loss_value)]
    reference = torch.autograd.grad(crop_loss + .5 * global_loss + weight * patch_loss_value, parameter)[0]
    sum_error = error(parts[0] + .5 * parts[1] + weight * parts[2], reference)
    assert sum_error['relative_l2'] < FP32_EQ_RTOL
    write_json(Path(out) / 'unit_vectors.json', dict(status='passed', U1_adaptive_equivalence=equivalence,
               U2_norm_relative_error=ratio_error, nonconflict_identity=True, zero_preservation_identity=True,
               tiny_preservation_identity=True, zero_crop_skipped=True, nonfinite_rejected=True,
               P2_FP32_synthetic_sum=sum_error,
               constants=dict(gp_norm_min=GP_NORM_MIN, gc_norm_min=GC_NORM_MIN, rho_epsilon=RHO_EPSILON,
                              fp32_relative_tolerance=FP32_EQ_RTOL, bf16_sum_relative_tolerance=BF16_SUM_RTOL)))


def loss_terms(batch, student, teacher):
    existing = terms(batch, student, None)
    with torch.no_grad():
        target, _ = run_enc(teacher, batch['full'])
    preserve, _, _ = patch_loss(existing['tok'], target, batch['patch_area'], batch['roi_union'], False)
    return existing, [existing['crop'].mean(), existing['glob'].mean(), preserve]


def actual_preflight(root, base, p2):
    root, base, p2 = map(Path, (root, base, p2))
    assert read(root / 'unit_vectors.json')['status'] == 'passed'
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = read(p2 / 'P2/config.json')
    weight = cfg['lambda_preserve']
    teacher = encoder(base / 'initial_encoder.pt')
    teacher_sha = state_sha(teacher.state_dict())
    student = encoder(base / 'initial_encoder.pt', train=True)
    layout = GradientLayout(student)
    dataset = PatchData(base / 'teacher')
    excluded = set(read(base / 'teacher/manifest.json')['excluded_train_images'])
    stream = np.load(STREAM)
    batch_indices = np.linspace(0, 11999, 8, dtype=int).tolist()
    records = []
    snapshots = [('Original', base / 'initial_encoder.pt'), ('P2_final_12000_EMA', p2 / 'P2/export/encoder.pt')]
    for snapshot, path in snapshots:
        state = torch.load(path, map_location='cpu', weights_only=True)
        student.load_state_dict(state, strict=True)
        del state
        for batch_index in (batch_indices[:1] if snapshot == 'Original' else batch_indices):
            ids = [int(i) for i in stream[batch_index] if int(i) not in excluded]
            batch = RC.to_dev(dataset[ids], 'cuda')
            student.zero_grad(set_to_none=True)
            existing, losses = loss_terms(batch, student, teacher)
            gradients, active = separate_gradients(losses, layout, retain_graph=True)
            separate_sum = gradients[0] + .5 * gradients[1] + weight * gradients[2]
            reference = torch.autograd.grad(losses[0] + .5 * losses[1] + weight * losses[2], layout.parameters,
                                            allow_unused=True)
            original = layout.pack(reference)
            p2_error = error(separate_sum, original)
            assert p2_error['relative_l2'] < BF16_SUM_RTOL, p2_error
            a, projected, stats = combine(*gradients, weight, 'U1')
            b, control_projected, stats2 = combine(*gradients, weight, 'U2')
            alternate = gradients[0] + .5 * gradients[1] + stats['lambda_eff'] * gradients[2]
            equivalence = error(a, alternate)
            assert equivalence['relative_l2'] < FP32_EQ_RTOL, equivalence
            norm_match_error = abs(float((stats2['rho'] * gradients[0]).norm()) - float(control_projected.norm()))
            assert norm_match_error < FP32_EQ_RTOL * max(stats2['crop_norm'], 1e-8)
            assert torch.equal(projected, control_projected)
            groups = layout.block_summary(gradients[0], gradients[2])
            layout.assign(a, active)
            total_groups = layout.block_summary(a, a)
            assert all(total_groups[key]['crop_norm'] > 0 for key in ['patch_embed', 'final_norm'] + [f'block_{i:02d}' for i in range(12)])
            assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            if snapshot == 'Original':
                assert stats['patch_norm'] <= GP_NORM_MIN and not stats['projection_applied'], stats
            record = dict(snapshot=snapshot, checkpoint=str(path), checkpoint_sha256=sha(path),
                          batch_index=batch_index, image_ids=ids, losses=dict(zip(['crop','global','patch'], [float(x.detach()) for x in losses])),
                          P2_BF16_sum_error=p2_error, U1_FP32_equivalence=equivalence,
                          U2_crop_norm_absolute_error=norm_match_error, U1=stats, U2=stats2,
                          blocks=groups, total_norms=total_groups, unused_parameters=[n for n,v in zip(layout.names, active) if not v],
                          all_encoder_parameters_fp32=all(p.dtype == torch.float32 for p in layout.parameters), teacher_frozen=True)
            records.append(record)
            write_json(root / 'preflight_progress.json', record)
            print(json.dumps(dict(snapshot=snapshot, batch_index=batch_index, P2_error=p2_error, **stats)), flush=True)
            del a, b, alternate, projected, control_projected, original, separate_sum, gradients, losses, existing, reference
            student.zero_grad(set_to_none=True)
    assert state_sha(teacher.state_dict()) == teacher_sha
    final = [r for r in records if r['snapshot'].startswith('P2_')]
    # Isolate BF16 backward rounding from normalization/order errors using the
    # same actual final encoder and batch, only for this read-only unit test.
    from unittest.mock import patch
    from torch.nn.attention import sdpa_kernel, SDPBackend

    def fp32_forward(model, image):
        tokens = model.forward_features(image).float()
        return tokens[:, model.num_prefix_tokens:], tokens[:, 0]

    with patch('src.fgtrain.train.run_enc', fp32_forward), torch.backends.cudnn.flags(allow_tf32=False), sdpa_kernel(SDPBackend.MATH):
        reference_terms = terms(batch, student, None)
        with torch.no_grad():
            target, _ = fp32_forward(teacher, batch['full'])
        reference_patch, _, _ = patch_loss(reference_terms['tok'], target, batch['patch_area'], batch['roi_union'], False)
        reference_losses = [reference_terms['crop'].mean(), reference_terms['glob'].mean(), reference_patch]
        fp32_parts, _ = separate_gradients(reference_losses, layout, retain_graph=True)
        full_gradient = torch.autograd.grad(reference_losses[0] + .5 * reference_losses[1] + weight * reference_losses[2],
                                            layout.parameters, allow_unused=True)
        fp32_error = error(fp32_parts[0] + .5 * fp32_parts[1] + weight * fp32_parts[2], layout.pack(full_gradient))
    assert fp32_error['relative_l2'] < FP32_EQ_RTOL, fp32_error
    print(json.dumps(dict(FP32_encoder_sum_unit_test=fp32_error)), flush=True)
    result = dict(status='passed', records=records, P2_final_batch_count=len(final),
                  P2_conflict_frequency=float(np.mean([r['U1']['conflict'] for r in final])),
                  P2_projection_frequency=float(np.mean([r['U1']['projection_applied'] for r in final])),
                  P2_midpoint_available=False,
                  midpoint_note='No P2 step6000 weights remain; only the saved final12000 EMA is used. No P2 replay.',
                  original_gp_skipped=True, parameter_count=sum(layout.sizes), parameter_order_sha256=__import__('hashlib').sha256('\n'.join(layout.names).encode()).hexdigest(),
                  FP32_encoder_sum_unit_test=fp32_error,
                  FP32_unit_TF32_disabled=True,
                  FP32_unit_attention_backend='math reference only',
                  precision='Existing P2 BF16 autocast forward, FP32 weights and unscaled per-loss gradients, no GradScaler.',
                  summation_note='BF16 reverse-mode rounding can differ between three separate VJPs and one total-loss VJP. Both formulas use identical forward tensors. Error measured at fixed2% relative-L2 tolerance; no residual correction.',
                  execution=dict(world=1, accumulation=1, duplicate_allreduce=False, batch='Unchanged global16-image stream, original exclusions',ddp='Separate independent one-GPU experiments; multi-rank projection is not enabled'),
                  teacher_frozen=True, teacher_state_sha=teacher_sha)
    write_json(root / 'preflight.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--base-run')
    parser.add_argument('--p2-run')
    parser.add_argument('--stage', choices=['unit', 'preflight'], required=True)
    args = parser.parse_args()
    if args.stage == 'unit':
        unit_vectors(args.run)
    else:
        actual_preflight(args.run, args.base_run, args.p2_run)


if __name__ == '__main__':
    main()
