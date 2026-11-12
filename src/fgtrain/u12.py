"""Independent U1/U2 training and mandatory matched PACO Val + ADE evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import traceback

from src.config import STREAM
from src.fgtrain.common import ROOT, RUN, write_json
from src.fgtrain.pipeline import Pipeline
from src.fgtrain.p1 import P1_RUN, N1_RUN, read, sha
from src.fgtrain.audit import target_audit
from src.fgtrain.update_rules import GP_NORM_MIN, GC_NORM_MIN, RHO_EPSILON, FP32_EQ_RTOL, BF16_SUM_RTOL


SHARED = ROOT / 'output/2026-11-11/fgtrain_s0_u12'
P2 = ROOT / 'output/2026-11-11/fgtrain_p2_s0'
P23 = ROOT / 'output/2026-11-11/fgtrain_p23_s0'
REPORT = ROOT / 'results/u1_u2_gradient_conflict'
SOURCES = ['update_rules.py', 'update_verify.py', 'update_train.py', 'u12.py', 'update_report.py']


def immutable(path, data):
    if path.exists():
        assert read(path) == data, f'Fixed U-series artifact changed: {path}'
    else:
        write_json(path, data)


def prepare(root, base):
    root, base = Path(root), Path(base)
    root.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    cfg = read(P2 / 'P2/config.json')
    assert cfg['steps'] == 12000 and cfg['seed'] == 0 and cfg['world'] == 1
    assert cfg['global_images_per_step'] == 16 and cfg['rois_per_image'] == 4
    assert cfg['lambda_global'] == .5 and cfg['lambda_preserve'] == read(P23 / 'lambda_preserve.json')['lambda_preserve']
    assert cfg['initial_sha'] == read(base / 'initialisation.json')['state_sha256']
    old_source = read(P23 / 'source_manifest.json')['files']
    used_existing = ['src/fgtrain/train.py', 'src/fgtrain/patch_train.py', 'src/fgtrain/patch_preserve.py',
                     'src/fgtrain/common.py', 'src/g1eval/ade.py', 'src/fgtrain/ade.py',
                     'src/g1up/train_bb.py', 'src/fgtrain/infer.py', 'src/pacoattr/official_eval.py', 'src/ivc/ft.py']
    for path in used_existing:
        assert sha(ROOT / path) == old_source[path], ('P2 source mismatch', path)
    manifest = read(P23 / 'data_manifest.json')
    assert sha(STREAM) == manifest['stream_sha256']
    assert sha(base / 'teacher/rois.parquet') == manifest['roi_coordinates_sha256']
    assert sha(base / 'official_split_exclusion.json') == manifest['exclusions_sha256']
    actual_targets = target_audit(base)
    assert actual_targets == read(P2 / 'P2/target_verification.json')
    immutable(root / 'data_manifest.json', dict(manifest, target_verification=actual_targets))
    runs = {name: str(root.parent / f'fgtrain_{name.lower()}_s0') for name in ('U1', 'U2')}
    for name, path in runs.items():
        target = Path(path)
        (target / name).mkdir(parents=True, exist_ok=True)
        (target / 'logs').mkdir(exist_ok=True)
        link = root / name
        if link.is_symlink():
            assert link.resolve() == target / name
        elif not link.exists():
            link.symlink_to(target / name, target_is_directory=True)
        else:
            raise RuntimeError(f'Unexpected existing U-series folder: {link}')
    protocol = dict(conditions=['U1', 'U2'], p2_run=str(P2), p23_run=str(P23), base_run=str(base),
                    run_directories=runs, initial_sha=cfg['initial_sha'], independent_original_starts=True,
                    seed=0, steps=12000, world_per_run=1, accumulation=1, nominal_batch_images=16,
                    optimizer=cfg, lambda_preserve=cfg['lambda_preserve'],
                    gradient_rules=dict(U1='gc_proj +0.5gg +lambda*gp, projecting whole-encoder gc once only when dot<0 and norms valid',
                                        U2='rho*gc +0.5gg +lambda*gp; same projection computation, original crop direction'),
                    numerical=dict(gp_norm_min=GP_NORM_MIN, gc_norm_min=GC_NORM_MIN, rho_epsilon=RHO_EPSILON,
                                   fp32_relative_tolerance=FP32_EQ_RTOL, bf16_gradient_sum_tolerance=BF16_SUM_RTOL),
                    precision='P2 BF16 autocast forward; FP32 weights and unscaled FP32 loss gradients; no GradScaler',
                    projection='One global coefficient from all trainable Encoder parameter gradients in fixed named_parameters order',
                    parameter_unused='A parameter unused by all three losses keeps grad=None, preserving P2 AdamW behavior',
                    initial_preflight='Fixed actual COCO stream batches linspace(0,11999,8), same exclusions; Original and existing P2 final12000 EMA',
                    diagnostic_frequency=dict(gradient_steps=1, flush_steps=50, block_steps=50, dev_steps=1000),
                    encoder_selection='Final12000 EMA only', PACO='Same Official Val2410, bilinearROIAlign24, R2 seed0/init7,10epochs; max Object/Part mean with earlier tie',
                    ADE='Mandatory for both; unchanged src.g1eval.ade; world2/batch16/40k, frozen encoder, separate same-init Linear768→150 heads, final iteration',
                    gpu_plan=dict(U1=0, U2=1, ADE=[0,1], background='Coexist with existing managed A/B; no worker interruption'),
                    mathematical_limits='U1 equals adaptive preservation weighting; U2 matches crop gradient norm, not composite or AdamW parameter update norm',
                    PACO_Test=False, new_methods_after_results=False)
    immutable(root / 'training_protocol.json', protocol)
    files = {str((ROOT / 'src/fgtrain' / name).relative_to(ROOT)): sha(ROOT / 'src/fgtrain' / name) for name in SOURCES}
    files.update({path: sha(ROOT / path) for path in used_existing})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    immutable(root / 'source_manifest.json', dict(files=files, aggregate_sha256=digest))
    for name in SOURCES:
        destination = root / 'source_snapshot' / name
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(ROOT / 'src/fgtrain' / name, destination)
    inputs = [base / 'initial_encoder.pt', base / 'B0/export/encoder.pt', base / 'B1/export/encoder.pt',
              N1_RUN / 'N1/export/encoder.pt', P1_RUN / 'P1/export/encoder.pt',
              P2 / 'P2/recovery.pt', P2 / 'P2/export/encoder.pt', P23 / 'P3/export/encoder.pt',
              P2 / 'P2/config.json', P23 / 'lambda_preserve.json', base / 'teacher/rois.parquet',
              base / 'target_freeze.json', base / 'official_split_exclusion.json']
    inputs += [ROOT / path for path in used_existing]
    immutable(root / 'protected_inputs.json', {str(path): sha(path) for path in inputs})
    for name, path in runs.items():
        for file in ('source_manifest.json', 'data_manifest.json', 'training_protocol.json'):
            immutable(Path(path) / file, read(root / file))
    from src.fgtrain.update_report import report
    report(root)


class UpdatePipeline(Pipeline):
    def report(self):
        from src.fgtrain.update_report import report
        with self.report_lock:
            report(self.run)

    def stage(self, name, module, args, gpus='', official=False):
        for condition, path in read(self.run / 'training_protocol.json')['run_directories'].items():
            if condition in name:
                link = Path(path) / 'logs' / f'{name}.log'
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(self.run / 'logs' / f'{name}.log')
        return super().stage(name, module, args, gpus, official)

    def branch(self, condition, gpu, base):
        root = Path(read(self.run / 'training_protocol.json')['run_directories'][condition])
        args = ['--run', root, '--base-run', base, '--shared', self.run, '--condition', condition]
        self.stage('train_' + condition, 'src.fgtrain.update_train', args, str(gpu))
        done = read(root / condition / 'done.json')
        assert done['steps'] == 12000 and done['export']['strict_loading']
        self.report()
        self.probe(condition, gpu)

    def execute(self, base):
        if not (self.state / 'preflight.json').exists():
            cached = self.run / 'preflight.json'
            if cached.exists() and read(cached).get('status') == 'passed' and 'FP32_encoder_sum_unit_test' in read(cached):
                write_json(self.state / 'preflight.json', dict(reused_completed_preflight=True, source=str(cached), sha256=sha(cached)))
            else:
                self.stage('preflight', 'src.fgtrain.update_verify', ['--run', self.run, '--base-run', base,
                           '--p2-run', P2, '--stage', 'preflight'], '0')
        check = read(self.run / 'preflight.json')
        assert check['status'] == 'passed' and check['FP32_encoder_sum_unit_test']['relative_l2'] < FP32_EQ_RTOL
        self.report()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.branch, 'U1', 0, base), pool.submit(self.branch, 'U2', 1, base)]
            for future in futures:
                future.result()
        for future in self.cpu_futures:
            future.result()
        self.cpu.shutdown(wait=True)
        self.stage('ade_U1_U2', 'torch.distributed.run', ['--standalone', '--nproc_per_node', '2', '-m', '--',
                   'src.fgtrain.ade', '--run', self.run, '--models', 'U1,U2', '--out', self.run / 'ade', '--workers', '12'], '0,1')
        ade = read(self.run / 'ade/done.json')
        assert ade['iters'] == 40000 and set(ade['results']) == {'U1', 'U2'} and all(ade['unchanged'].values())
        for path, digest in read(self.run / 'protected_inputs.json').items():
            assert sha(path) == digest, ('Existing artifact changed', path)
        self.report()
        for condition, path in read(self.run / 'training_protocol.json')['run_directories'].items():
            write_json(Path(path) / 'done.json', dict(status='done', steps=12000, PACO_Val=True, ADE=True,
                       report=str(REPORT / 'report.html'), condition=condition))
        write_json(self.run / 'done.json', dict(status='done', conditions=['U1', 'U2'], PACO_Val=True, ADE_both=True,
                   PACO_Test=False, protected_inputs_unchanged=True, report=str(REPORT / 'report.html')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=str(SHARED))
    parser.add_argument('--base-run', default=str(RUN))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(args.run)
    prepare(root, args.base_run)
    if not args.execute:
        return
    assert read(root / 'unit_vectors.json')['status'] == 'passed'
    pipe = UpdatePipeline(root)
    write_json(root / 'launch.json', dict(pid=os.getpid(), python=sys.executable, command=sys.argv))

    def stop(signum, frame):
        pipe.terminate_children()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    try:
        pipe.execute(args.base_run)
    except BaseException:
        pipe.terminate_children()
        write_json(root / 'failed.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
