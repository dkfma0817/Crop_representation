"""U1/U2 reuse P2 objectives and stream, changing only gradient combination."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.readout.common as RC
from src.config import STREAM
from src.msd.common import StreamSampler
from src.ivc.ft import enc_param_groups, ema_update
from src.fgtrain.common import encoder, state_sha, freeze_export, write_json
from src.fgtrain.audit import target_audit
from src.fgtrain.train import modules, diagnose
from src.fgtrain.patch_preserve import PatchData, drift_stats
from src.fgtrain.patch_verify import batch0
from src.fgtrain.p1 import read, sha
from src.fgtrain.update_rules import GradientLayout, separate_gradients, combine
from src.fgtrain.update_verify import loss_terms


def train(root, base, shared, condition):
    root, base, shared = map(Path, (root, base, shared))
    assert condition in ('U1', 'U2')
    assert int(os.environ.get('WORLD_SIZE', '1')) == 1
    assert not torch.distributed.is_initialized()
    assert torch.cuda.device_count() == 1
    physical = int(os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.cuda.get_device_properties(0)
    write_json(root / 'device_mapping.json', dict(physical_gpu=physical, internal_device='cuda:0',
               CUDA_VISIBLE_DEVICES=str(physical), name=device.name, uuid=str(device.uuid)))
    out = root / condition
    out.mkdir(exist_ok=True)
    if (out / 'done.json').exists():
        assert read(out / 'done.json')['steps'] == 12000
        return
    assert read(shared / 'preflight.json')['status'] == 'passed'
    protocol = read(shared / 'training_protocol.json')
    parent_config = read(Path(protocol['p2_run']) / 'P2/config.json')
    weight = parent_config['lambda_preserve']
    total = parent_config['steps']
    torch.set_num_threads(2)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    scope = read(base / 'official_split_exclusion.json')
    assert all(v == 0 for v in scope['train_official_overlap'].values())
    assert all(v == 0 for v in scope['dev_official_overlap'].values())
    write_json(out / 'target_verification.json', target_audit(base))
    enc, _, _ = modules(base / 'initial_encoder.pt', 'B0')
    initial_sha = state_sha(enc.state_dict())
    assert initial_sha == parent_config['initial_sha']
    ema = copy.deepcopy(enc).eval().requires_grad_(False)
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    teacher = encoder(base / 'initial_encoder.pt')
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    teacher_sha = state_sha(teacher.state_dict())
    assert teacher_sha == initial_sha
    teacher_versions = [p._version for p in teacher.parameters()]
    layout = GradientLayout(enc)
    opt = torch.optim.AdamW(enc_param_groups(enc, 5e-5, .75, .05), betas=(.9, .999))
    start, recovery = 0, out / 'recovery.pt'
    if recovery.exists():
        ck = torch.load(recovery, map_location='cuda', weights_only=False)
        assert ck['condition'] == condition and ck['total_steps'] == total and ck['lambda_preserve'] == weight
        enc.load_state_dict(ck['enc'], strict=True)
        ema.load_state_dict(ck['ema'], strict=True)
        opt.load_state_dict(ck['opt'])
        start = ck['step']
        torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        del ck
        for filename in ('gradient_log.jsonl', 'block_gradients.jsonl', 'train_log.jsonl'):
            path = out / filename
            if not path.exists():
                continue
            original_log = path.read_text()
            retained = []
            for line in original_log.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record['step'] <= start:
                    retained.append(line)
            recovered_log = '\n'.join(retained) + ('\n' if retained else '')
            if recovered_log != original_log:
                archive = out / 'restart_logs' / f'{time.time_ns()}_{filename}'
                archive.parent.mkdir(exist_ok=True)
                archive.write_text(original_log)
                path.write_text(recovered_log)
    cfg = dict(parent_config)
    cfg.update(condition=condition, physical_gpu=physical, source_hash=read(shared / 'source_manifest.json')['aggregate_sha256'],
               gradient_rule=protocol['gradient_rules'][condition], numerical=protocol['numerical'],
               precision=protocol['precision'], accumulation=1, grad_scaler=False, world=1,
               preflight=str(shared / 'preflight.json'), preflight_sha256=sha(shared / 'preflight.json'))
    write_json(out / 'config.json', cfg)
    write_json(out / 'initialization_verification.json', dict(initial_state_sha256=initial_sha,
               original_pretrained=True, independent_start=True, parameter_count=sum(layout.sizes),
               parameter_order=layout.names, teacher_state_sha256=teacher_sha,
               initial_weights=str(base / 'initial_encoder.pt'), original_file_sha256=sha(base / 'initial_encoder.pt')))
    ds, dev = PatchData(base / 'teacher'), PatchData(base / 'teacher', 'dev')
    excluded = read(base / 'teacher/manifest.json')['excluded_train_images']
    sampler = StreamSampler(np.load(STREAM)[:total], 0, 1, exclude=excluded, start=start)
    loader = DataLoader(ds, sampler=sampler, batch_size=None, num_workers=4, pin_memory=False,
                        prefetch_factor=2, persistent_workers=True)
    if start == 0:
        diagnose(ema, None, None, dev, 'cuda', 0, out, 'ema')

    def save(step):
        state = dict(enc=enc.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step,
                     condition=condition, total_steps=total, lambda_preserve=weight,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        temporary = out / 'recovery.tmp'
        torch.save(state, temporary)
        temporary.replace(recovery)

    started, window = time.time(), []
    with (out / 'gradient_log.jsonl').open('a', buffering=1024*1024) as log:
        for step, batch in enumerate(loader, start=start):
            batch = RC.to_dev(batch, 'cuda')
            for group in opt.param_groups:
                group['lr'] = RC.C.lr_at(step, group['base'], 600, total)
            opt.zero_grad(set_to_none=True)
            existing, losses = loss_terms(batch, enc, teacher)
            if step == 0:
                initial = json.loads((base / 'B0/per_loss_gradients.jsonl').read_text().splitlines()[0])
                match = {key:math.isclose(float(existing[key].mean().detach()), initial[key]['loss'], rel_tol=1e-5, abs_tol=1e-7)
                         for key in ('crop', 'glob')}
                write_json(out / 'initial_P2_loss_match.json', match)
                assert all(match.values()), match
            gradients, used = separate_gradients(losses, layout)
            fused, projected, info = combine(*gradients, weight, condition)
            log_step = step + 1
            if log_step % 50 == 0 or log_step == 1:
                blocks = layout.block_summary(gradients[0], gradients[2])
                if log_step in (1, 6000, 12000):
                    total_blocks = layout.block_summary(fused, fused)
                    for key in ['patch_embed', 'final_norm'] + [f'block_{i:02d}' for i in range(12)]:
                        assert total_blocks[key]['crop_norm'] > 0, key
                    blocks['composite'] = total_blocks
                with (out / 'block_gradients.jsonl').open('a') as block_log:
                    block_log.write(json.dumps(dict(step=log_step, blocks=blocks)) + '\n')
                assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
                assert [p._version for p in teacher.parameters()] == teacher_versions
            layout.assign(fused, used)
            clipped = float(torch.nn.utils.clip_grad_norm_(enc.parameters(), 1., error_if_nonfinite=True))
            opt.step()
            momentum = 1 - (1-.999) * (math.cos(math.pi * step / total) + 1) / 2
            ema_update(ema, enc, momentum)
            crop, glob, patch = [float(loss.detach()) for loss in losses]
            # Unweighted patch loss already equals valid-area mean patch drift in P2.
            row = dict(step=log_step, total_loss=crop + .5 * glob + weight * patch,
                       crop=crop, global_loss=glob, patch=patch, weighted_patch=weight * patch,
                       crop_cos=float((1 - existing['crop']).mean().detach()),
                       cls_cos=float(existing['cls_cos'].mean().detach()),
                       mean_cos=float(existing['mean_cos'].mean().detach()),
                       patch_drift=patch, clip_input_norm=clipped, clip_max_norm=1.,
                       lr=max(g['lr'] for g in opt.param_groups), image_ids=batch['image_ids'].tolist(),
                       roi_count=len(batch['roi_img']), **info)
            log.write(json.dumps(row) + '\n')
            window.append(row)
            if log_step % 50 == 0:
                log.flush()
                keys = ['total_loss','crop','global_loss','patch','crop_norm','global_norm','patch_norm',
                        'rho','lambda_eff','lambda_eff_ratio','composite_norm','crop_cos','cls_cos','mean_cos']
                progress = dict(step=log_step, seconds=round(time.time()-started),
                                **{key:float(np.mean([r[key] for r in window])) for key in keys},
                                conflict_frequency=float(np.mean([r['conflict'] for r in window])),
                                projection_frequency=float(np.mean([r['projection_applied'] for r in window])))
                write_json(out / 'progress.json', progress)
                with (out / 'train_log.jsonl').open('a') as training_log:
                    training_log.write(json.dumps(progress) + '\n')
                print(json.dumps(progress), flush=True)
                window = []
            del projected, fused, gradients, losses, existing
            if log_step % 500 == 0:
                save(log_step)
            if log_step == 6000:
                torch.save(dict(enc={k:v.detach().cpu() for k,v in ema.state_dict().items()}, step=6000, kind='ema'),
                           out / 'midpoint_ema_006000.pt')
            if log_step % 1000 == 0:
                diagnose(ema, None, None, dev, 'cuda', log_step, out, 'ema')
    save(total)
    assert state_sha(teacher.state_dict()) == teacher_sha
    sample = batch['full'][:2] if 'batch' in locals() else batch0(base, 'cuda')['full'][:2]
    exported = freeze_export(ema, out / 'export', sample, base / 'initial_encoder.pt', step=total)
    assert exported['parameter_count'] == 85641216 and exported['additional_modules'] == []
    write_json(out / 'done.json', dict(status='done', steps=total, export=exported,
               final_ema_sha=exported['state_sha256'], initial_sha=initial_sha,
               lambda_preserve=weight, teacher_frozen=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--base-run', required=True)
    parser.add_argument('--shared', required=True)
    parser.add_argument('--condition', choices=('U1', 'U2'), required=True)
    args = parser.parse_args()
    train(args.run, args.base_run, args.shared, args.condition)


if __name__ == '__main__':
    main()
