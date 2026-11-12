"""Two fixed partial-projection runs, matched Official Val and mandatory ADE."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import difflib
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
from src.fgtrain.audit import target_audit
from src.fgtrain.p1 import read, sha
from src.fgtrain.partial_rules import ALPHAS
from src.fgtrain.u12 import UpdatePipeline, immutable, P2


SHARED = ROOT / 'output/2026-11-11/fgtrain_s0_u34'
PRIOR = ROOT / 'output/2026-11-11/fgtrain_s0_u12'
REPORT = ROOT / 'results/u3_u4_partial_projection'
NEW_SOURCES = ['partial_rules.py', 'partial_verify.py', 'partial_train.py', 'partial_report.py', 'u34.py']


def prepare(root, base):
    root, base = Path(root).resolve(), Path(base).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    prior_sources = read(PRIOR / 'source_manifest.json')['files']
    for path, digest in prior_sources.items():
        assert sha(ROOT / path) == digest, ('Existing U1 source changed', path)
    cfg = read(P2 / 'P2/config.json')
    prior_cfg = read(PRIOR / 'U1/config.json')
    for key in ['initial_sha','steps','seed','world','lambda_global','lambda_preserve',
                'encoder_lr','llrd','wd','betas','warmup','schedule','ema_m0',
                'global_images_per_step','rois_per_image','cache']:
        assert cfg[key] == prior_cfg[key], ('P2/U1 protocol mismatch', key)
    assert cfg['lambda_preserve'] == .06517137116454548
    assert cfg['steps'] == 12000 and cfg['global_images_per_step'] == 16
    assert cfg['initial_sha'] == read(base / 'initialisation.json')['state_sha256']
    manifest = read(PRIOR / 'data_manifest.json')
    assert sha(STREAM) == manifest['stream_sha256']
    assert sha(base / 'teacher/rois.parquet') == manifest['roi_coordinates_sha256']
    assert sha(base / 'official_split_exclusion.json') == manifest['exclusions_sha256']
    assert target_audit(base) == manifest['target_verification']
    immutable(root / 'data_manifest.json', manifest)
    runs = {name:str(root.parent / f'fgtrain_{name.lower()}_s0') for name in ALPHAS}
    for name, path in runs.items():
        (Path(path) / name).mkdir(parents=True, exist_ok=True)
        (Path(path) / 'logs').mkdir(exist_ok=True)
        link = root / name
        if link.is_symlink():
            assert link.resolve() == Path(path) / name
        elif not link.exists():
            link.symlink_to(Path(path) / name, target_is_directory=True)
        else:
            raise RuntimeError(f'Unexpected existing directory {link}')
    protocol = copy.deepcopy(read(PRIOR / 'training_protocol.json'))
    protocol.update(conditions=list(ALPHAS), alphas=ALPHAS, run_directories=runs,
        prior_run=str(PRIOR), base_run=str(base), lambda_preserve=cfg['lambda_preserve'],
        gradient_rules={n:f'gc_partial=lerp(gc,U1_projected_gc,{a}); gc_partial +0.5gg +lambda*gp'
                        for n,a in ALPHAS.items()},
        initial_preflight='Unmodified U1 vector tests + partial endpoint tests; same actual fixed COCO stream batch0 at Original and existing final P2 EMA; no optimizer updates',
        gpu_plan=dict(U3=0,U4=1,ADE=[0,1],background='Coexist with existing managed A/B; no interruption or training setting changes'),
        mathematical_limits='Partial projection equals adaptive preservation weighting; one whole-encoder coefficient; AdamW momentum, weight decay and clipping affect actual updates',
        new_training_conditions_exactly=['U3','U4'], endpoints_reused=['P2','U1'],
        PACO_Test=False, sweeps=False, automatic_followup_methods=False)
    immutable(root / 'training_protocol.json', protocol)
    files = dict(prior_sources)
    for path in ['src/fgtrain/pipeline.py','src/msd/common.py','src/readout/common.py',
                 'src/readout/models.py','src/pdit/bbdata.py','src/fgtrain/data.py',
                 'src/g1up/official_infer.py','src/fgtrain/n1.py']:
        files[path]=sha(ROOT/path)
    files.update({f'src/fgtrain/{n}':sha(ROOT / 'src/fgtrain' / n) for n in NEW_SOURCES})
    digest = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    immutable(root / 'source_manifest.json',dict(files=files,aggregate_sha256=digest,
        original_U1_source_manifest_sha256=sha(PRIOR / 'source_manifest.json')))
    for path in files:
        out = root / 'source_snapshot' / path
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copy2(ROOT / path,out)
    diff = ''.join(difflib.unified_diff((ROOT/'src/fgtrain/update_train.py').read_text().splitlines(True),
              (ROOT/'src/fgtrain/partial_train.py').read_text().splitlines(True),
              fromfile='unchanged_U1_update_train.py',tofile='partial_train.py'))
    (root/'source_snapshot/partial_train_vs_U1.diff').write_text(diff)
    protected = read(PRIOR / 'protected_inputs.json')
    for path,digest in protected.items():
        assert sha(path) == digest, ('Prior protected input changed',path)
    for path in [PRIOR/'results.json',PRIOR/'training_protocol.json',PRIOR/'source_manifest.json',
                 PRIOR/'preflight.json',PRIOR/'unit_vectors.json']:
        protected[str(path)] = sha(path)
    for name in ('U1','U2'):
        for filename in ('export/encoder.pt','export/encoder_eval.pt','config.json',
                         'selection.json','gradient_log.jsonl','block_gradients.jsonl'):
            path=PRIOR/name/filename
            protected[str(path)] = sha(path)
    immutable(root/'protected_inputs.json',protected)
    for name,path in runs.items():
        for filename in ('source_manifest.json','data_manifest.json','training_protocol.json'):
            immutable(Path(path)/filename,read(root/filename))
    from src.fgtrain.partial_report import report
    report(root)


def matched_training(root):
    rows = {}
    for name in ('U3','U4'):
        rows[name] = [json.loads(x) for x in (root/name/'gradient_log.jsonl').read_text().splitlines()]
        assert len(rows[name]) == 12000 and [r['step'] for r in rows[name]] == list(range(1,12001))
        assert all(r['alpha'] == ALPHAS[name] and not r['nonfinite'] for r in rows[name])
    old = [json.loads(x) for x in (PRIOR/'U1/gradient_log.jsonl').read_text().splitlines()]
    assert len(old) == 12000
    for a,b,c in zip(rows['U3'],rows['U4'],old):
        for key in ('step','image_ids','roi_count','lr'):
            assert a[key] == b[key] == c[key], ('Matched stream mismatch',a['step'],key)
    result = dict(status='passed',steps=12000,matched_to='U1 complete original training stream',
        exact_fields=['step','image_ids','roi_count','lr'],alphas=ALPHAS,
        same_roi_coordinates_sha256=read(root/'data_manifest.json')['roi_coordinates_sha256'])
    write_json(root/'matched_training_verification.json',result)


class PartialPipeline(UpdatePipeline):
    def report(self):
        from src.fgtrain.partial_report import report
        with self.report_lock:
            report(self.run)

    def branch(self,condition,gpu,base):
        run=Path(read(self.run/'training_protocol.json')['run_directories'][condition])
        self.stage('train_'+condition,'src.fgtrain.partial_train',
            ['--run',run,'--base-run',base,'--shared',self.run,'--condition',condition],str(gpu))
        done=read(run/condition/'done.json')
        assert done['steps']==12000 and done['alpha']==ALPHAS[condition]
        assert done['export']['strict_loading'] and not done['export']['additional_modules']
        self.report()
        self.probe(condition,gpu)

    def execute(self,base):
        assert read(self.run/'partial_unit_tests.json')['status']=='passed'
        if not (self.state/'preflight.json').exists():
            path=self.run/'preflight.json'
            if path.exists() and read(path).get('status')=='passed':
                write_json(self.state/'preflight.json',dict(reused=True,source=str(path),sha256=sha(path)))
            else:
                self.stage('preflight','src.fgtrain.partial_verify',
                    ['--stage','preflight','--run',self.run,'--base-run',base,
                     '--prior-run',PRIOR,'--p2-run',P2],'0')
        assert read(self.run/'preflight.json')['status']=='passed'
        self.report()
        pool=ThreadPoolExecutor(max_workers=2)
        try:
            futures=[pool.submit(self.branch,'U3',0,base),pool.submit(self.branch,'U4',1,base)]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            self.terminate_children()
            raise
        finally:
            pool.shutdown(wait=True)
        for future in self.cpu_futures:
            future.result()
        self.cpu.shutdown(wait=True)
        matched_training(self.run)
        self.stage('ade_U3_U4','torch.distributed.run',
            ['--standalone','--nproc_per_node','2','-m','--','src.fgtrain.ade',
             '--run',self.run,'--models','U3,U4','--out',self.run/'ade','--workers','12'],'0,1')
        ade=read(self.run/'ade/done.json')
        assert ade['iters']==40000 and set(ade['results'])=={'U3','U4'}
        assert all(ade['unchanged'].values())
        for path,digest in read(self.run/'protected_inputs.json').items():
            assert sha(path)==digest,('Existing artifact changed',path)
        self.report()
        for name,path in read(self.run/'training_protocol.json')['run_directories'].items():
            write_json(Path(path)/'done.json',dict(status='done',condition=name,alpha=ALPHAS[name],
                steps=12000,PACO_Val=True,ADE=True,PACO_Test=False,report=str(REPORT/'report.html')))
        write_json(self.run/'done.json',dict(status='done',conditions=list(ALPHAS),alphas=ALPHAS,
            PACO_Val=True,ADE_both=True,PACO_Test=False,protected_inputs_unchanged=True,
            report=str(REPORT/'report.html')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default=str(SHARED))
    parser.add_argument('--base-run',default=str(RUN))
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    root=Path(args.run).resolve()
    prepare(root,args.base_run)
    if not args.execute:
        return
    pipe=PartialPipeline(root)
    write_json(root/'launch.json',dict(pid=os.getpid(),python=sys.executable,command=sys.argv))
    def stop(signum,frame):
        pipe.terminate_children()
        raise SystemExit(128+signum)
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop)
    try:
        pipe.execute(args.base_run)
    except BaseException:
        pipe.terminate_children()
        write_json(root/'failed.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
