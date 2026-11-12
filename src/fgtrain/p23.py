"""P2 GPU0 + P3 GPU1, then unchanged PACO Val and mandatory world2 ADE.

Restartable stage markers prevent repeating finished training. No Test path.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import traceback

import torch

from src.config import STREAM
from src.fgtrain.common import ROOT,RUN,state_sha,write_json
from src.fgtrain.pipeline import Pipeline
from src.fgtrain.p1 import read,sha,checked_apps,CONTROL,P1_RUN,N1_RUN
from src.fgtrain.p23_report import report
from src.fgtrain.audit import target_audit
from src.fgtrain.n1 import metrics
from src.crop_aware_pretrain import gpu as GPU
from src.brixel_eval.finish import pidfd

SHARED=ROOT/'output/2026-11-11/fgtrain_p23_s0'
PREFIX='fgtrain per-GPU priority: fgtrain_p23_s0:'
OWN_SOURCES={'patch_preserve.py','patch_verify.py','patch_train.py','patch_watch.py','p23.py','p23_report.py'}


def immutable(path,value):
    path=Path(path)
    if path.exists():assert read(path)==value, f'Previously fixed artifact changed: {path}'
    else:write_json(path,value)


def prepare(root,base):
    root,base=Path(root),Path(base);root.mkdir(parents=True,exist_ok=True)
    cfg=read(base/'B0/config.json');initial=read(base/'initialisation.json')
    assert cfg['initial_sha']==initial['state_sha256']
    assert initial['initialisation']=='Original pretrained DINOv3; neither G1-pool nor A0'
    assert cfg['steps']==12000 and cfg['seed']==0 and cfg['world']==1 and cfg['lambda_global']==.5
    assert cfg['global_images_per_step']==16 and cfg['rois_per_image']==4 and cfg['decoder_parameters']==0
    replay=P1_RUN/'calibration_B0_replay';replay_cfg=read(replay/'config.json')
    assert all(replay_cfg[k]==v for k,v in cfg.items()),'B0 replay training settings differ'
    assert replay_cfg['final_step']==6000 and replay_cfg['total_schedule_steps']==12000 and replay_cfg['calibration_replay']
    assert all(read(replay/'initial_B0_loss_match.json').values())
    assert read(replay/'target_verification.json')['sha256']==read(base/'target_freeze.json')['sha256']
    assert read(replay/'done.json')['step']==6000
    checkpoint=replay/'recovery.pt'
    ck=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    assert ck['condition']=='B0' and ck['step']==6000 and ck['total_steps']==12000 and ck['lambda_gram']==0
    audit=dict(status='passed',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
               calibration_state_sha256=state_sha(ck['ema']),checkpoint_step=6000,kind='EMA',
               initial_state_sha256=cfg['initial_sha'],B0_config_equal=True,initial_crop_global_equal=True,
               optimizer_scheduler_ema_batch_equal=True,teacher_cache_sha_equal=True,
               replay_config_sha256=sha(replay/'config.json'),
               replay_source_sha256=sha(ROOT/'src/fgtrain/p1_train.py'),
               original_exclusion_sha256=sha(base/'official_split_exclusion.json'),
               numerical_replay_note=read(P1_RUN/'calibration_replay_match.json'))
    del ck
    immutable(root/'calibration_source_audit.json',audit)
    scope=read(base/'official_split_exclusion.json')
    assert all(v==0 for v in scope['train_official_overlap'].values()) and all(v==0 for v in scope['dev_official_overlap'].values())
    target=target_audit(base)
    manifest=read(base/'teacher/manifest.json')
    immutable(root/'data_manifest.json',dict(base_manifest=manifest,source_manifest_sha256=sha(base/'teacher/manifest.json'),
              roi_coordinates_sha256=sha(base/'teacher/rois.parquet'),stream=str(STREAM),stream_sha256=sha(STREAM),
              exclusions=scope,exclusions_sha256=sha(base/'official_split_exclusion.json'),target_cache=target))
    runs={n:str(root.parent/f'fgtrain_{n.lower()}_s0') for n in ['P2','P3']}
    for name,path in runs.items():
        path=Path(path);path.mkdir(exist_ok=True);(path/name).mkdir(exist_ok=True);(path/'logs').mkdir(exist_ok=True)
        link=root/name
        if link.is_symlink():assert link.resolve()==path/name
        elif not link.exists():link.symlink_to(path/name,target_is_directory=True)
        else:raise RuntimeError(f'Unexpected existing directory {link}')
    protocol=dict(conditions=['P2','P3'],base_run=str(base),p1_run=str(P1_RUN),run_directories=runs,
        student='Original pretrained vit_base_patch16_dinov3.lvd1689m; independent initializations',
        initial_sha=cfg['initial_sha'],B0_config=cfg,steps=12000,seed=0,physical_gpu={'P2':0,'P3':1},
        process_gpu='Each process sees exactly one GPU as cuda:0; batch/settings unchanged',
        preservation='Final-LN FP32 normalized576patch cosine; frozen same Full384 Original reference',
        P2='w=1',P3='exact ROI union, raw1-0.75m; normalize per-image sum(v*w)/sum(v)=1',
        denominator='sum fractional valid coverage for both; average valid images; all-zero graph-connected0',
        lambda_preserve='One P2 calibration at verified B0 EMA6000 STREAM[0]; target gradient ratio0.5; shared unchanged for P3',
        training_encoder_selection='Final12000 EMA; no Val checkpoint selection',
        PACO='Unchanged R2 seed0/init7, bilinearROIAlign24,10epochs, OfficialVal2410; max mean Object/Part, earlier tie',
        ADE='Both mandatory: unchanged src.g1eval.ade,40k,world2/globalbatch16,seed0,frozen final-LN,linear768to150,final iteration',
        diagnostics='Step1,6000,12000 crop/preservation gradients and per-block norms; same COCO dev126 final drift',
        PACO_Test=False,additional_experiments=False,architecture_changes=False)
    immutable(root/'training_protocol.json',protocol)
    # Validate reuse against exact full-Val outputs, never Test scores.
    rows={}
    for name,folder in [('Frozen',base/'Frozen'),('B0',base/'B0'),('B1',base/'B1'),('P1',P1_RUN/'P1')]:
        config=read(folder/'probe/config.json');job=config['job']
        assert job['mode']=='bil' and job['seed']==0 and job['epochs']==10 and job['backbone']=='dinov3'
        assert job['readers']==[dict(name='r2',kind='r2',layers=[12])]
        curve={}
        for e in range(1,11):
            p=folder/'official'/f'ep{e}/official_results.json';r=read(p)
            assert r['split']=='val' and r['n_eval_images']==2410 and r['alignment_mismatch']==0
            curve[e]=metrics(p)['overall']
        best=max(curve,key=lambda e:(curve[e],-e))
        assert best==read(folder/'selection.json')['epoch']
        rows[name]=dict(epoch=best,metrics=metrics(folder/'official'/f'ep{best}/official_results.json'),probe_config_sha256=sha(folder/'probe/config.json'))
    assert abs(rows['B0']['metrics']['overall']-55.45040831830657)<1e-10
    ade_cfg=read(base/'ade/config.json');p1ade=read(P1_RUN/'ade/config.json')
    assert ade_cfg['cfg']==p1ade['cfg'] and ade_cfg['world']==p1ade['world']==2
    assert ade_cfg['cfg']['iters']==40000 and ade_cfg['cfg']['batch']==16
    immutable(root/'baseline_audit.json',dict(status='passed',PACO=rows,ADE_config=ade_cfg['cfg'],ADE_world=2,
              ADE_source_sha256=sha(ROOT/'src/g1eval/ade.py'),B0_exact_overall=rows['B0']['metrics']['overall'],test_not_consulted=True))
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}
    digest=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    immutable(root/'source_manifest.json',dict(files=sources,aggregate_sha256=digest))
    protected=[base/'initial_encoder.pt',base/'B0/export/encoder.pt',base/'B1/export/encoder.pt',
        N1_RUN/'N1/export/encoder.pt',P1_RUN/'P1/export/encoder.pt',checkpoint,
        base/'B0/config.json',base/'official_split_exclusion.json',base/'target_freeze.json',base/'teacher/rois.parquet',
        P1_RUN/'README.md',P1_RUN/'results.json',N1_RUN/'training_protocol.json',N1_RUN/'evaluation_protocol.json']
    protected += [p for p in (ROOT/'src/fgtrain').glob('*.py') if p.name not in OWN_SOURCES]
    protected += [ROOT/'src/g1eval/ade.py',ROOT/'src/g1up/train_bb.py',ROOT/'src/pacoattr/official_eval.py']
    immutable(root/'protected_inputs.json',{str(p):sha(p) for p in protected})
    # Exact source files used for these runs remain reviewable even after later work.
    import shutil
    for name in OWN_SOURCES:
        path=root/'source_snapshot/src/fgtrain'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():shutil.copy2(ROOT/'src/fgtrain'/name,path)
    for name,path in runs.items():
        for f in ['training_protocol.json','source_manifest.json','data_manifest.json']:
            immutable(Path(path)/f,read(root/f))
    report(root)


class PatchPipeline(Pipeline):
    def report(self):
        with self.report_lock:report(self.run)

    def stage(self,name,module,args,gpus='',official=False):
        # Per-run log links make the two independent experiment directories usable.
        for condition,path in read(self.run/'training_protocol.json')['run_directories'].items():
            if condition in name:
                p=Path(path)/'logs'/f'{name}.log'
                if not p.is_symlink() and not p.exists():p.symlink_to(self.run/'logs'/f'{name}.log')
        return super().stage(name,module,args,gpus,official)

    def acquire(self,gpus,label):
        # Only the existing owned A/B sessions may be checkpoint-paused.
        snapshot=read(CONTROL/'active.json')
        sessions={s for v in snapshot.values() for k in ['train','eval'] for s in (v.get(k) or {}).get('sids',[])}
        apps=checked_apps(gpus,sessions)
        for g in gpus:
            f=CONTROL/f'PAUSE_gpu{g}'
            assert not f.exists() or f.read_text().startswith(PREFIX),f'Existing pause owner: {f}'
        marker='reserve_'+label
        with self.lock:
            self.active[marker]=dict(pid=os.getpid(),gpus=','.join(map(str,gpus)),command=[],start=time.strftime('%FT%TZ',time.gmtime()))
            write_json(self.run/'active.json',self.active)
        for g in gpus:(CONTROL/f'PAUSE_gpu{g}').write_text(f'{PREFIX} ready for {label}\n')
        while True:
            current=read(CONTROL/'active.json')
            sessions|={s for v in current.values() for k in ['train','eval'] for s in (v.get(k) or {}).get('sids',[])}
            remaining=checked_apps(gpus,sessions);fds=[]
            for a in remaining:
                try:fds.append(pidfd(a['pid']))
                except ProcessLookupError:pass
            if not fds:break
            try:select.select(fds,[],[])
            finally:
                for fd in fds:os.close(fd)
        exits={}
        for v in snapshot.values():
            job=v.get('train')
            if not job or job['gpu'] not in gpus:continue
            folder=CONTROL.parent/job['job']
            # A parent may already have freed its CUDA context while finishing
            # the checkpoint/exit record. Wait for its clean process exit too.
            for pid in job['pids']:
                try:fd=pidfd(pid)
                except ProcessLookupError:continue
                try:select.select([fd],[],[])
                finally:os.close(fd)
            p=folder/'control/last_train_exit.json'
            assert p.exists(),f'No clean checkpoint exit record: {p}'
            q=read(p)
            assert q['launch_id']==job['launch_id'] and (q['reason'].startswith('signal_') or q['reason']=='completed'),q
            checkpoint=folder/'checkpoints'/f"step_{q['step']:08d}"/'state.pt'
            assert checkpoint.exists(),f'Missing paused A/B checkpoint: {checkpoint}'
            exits[job['job']]=dict(q,checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size)
        write_json(self.run/f'GPU_handoff_{label}.json',dict(physical_gpus=gpus,internal_device='cuda:0 for single GPU tasks',
                   existing_jobs=snapshot,apps_before=apps,apps_after=[],checkpoint_pause='Existing supervisor SIGTERM save-and-exit policy',
                   captured_train_exits=exits,ready=True))
        return marker

    def release(self,marker):
        with self.lock:self.active.pop(marker,None);write_json(self.run/'active.json',self.active)

    def branch(self,name,gpu,base,marker):
        root=Path(read(self.run/'training_protocol.json')['run_directories'][name])
        self.stage('train_'+name,'src.fgtrain.patch_train',['--run',root,'--base-run',base,'--shared',self.run,'--condition',name],str(gpu))
        done=read(root/name/'done.json');assert done['steps']==12000 and done['export']['strict_loading']
        self.release(marker);self.report();self.probe(name,gpu)

    def execute(self,base):
        m0=self.acquire([0],'P2')
        self.stage('calibrate_patch','src.fgtrain.patch_verify',['--run',self.run,'--base-run',base,'--stage','calibrate',
                   '--checkpoint',P1_RUN/'calibration_B0_replay/recovery.pt'],'0')
        self.report()
        m1=self.acquire([1],'P3')
        # No batch splitting, DDP or altered stream: independent one-GPU runs.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(self.branch,'P2',0,base,m0),pool.submit(self.branch,'P3',1,base,m1)]
            for f in futures:f.result()
        for f in self.cpu_futures:f.result()
        self.cpu.shutdown(wait=True)
        self.stage('patch_drift','src.fgtrain.patch_train',['--run',self.run,'--base-run',base,'--shared',self.run,
                   '--stage','drift','--p1-run',P1_RUN],'0')
        self.report()
        # Mandatory both-model ADE; each head is newly trained, identical init/batches.
        marker=self.acquire([0,1],'ADE')
        self.stage('ade_P2_P3','torch.distributed.run',['--standalone','--nproc_per_node','2','-m','--','src.fgtrain.ade',
                   '--run',self.run,'--models','P2,P3','--out',self.run/'ade','--workers',12],'0,1')
        self.release(marker)
        ade=read(self.run/'ade/done.json');assert ade['iters']==40000 and set(ade['results'])=={'P2','P3'}
        for path,expected in read(self.run/'protected_inputs.json').items():assert sha(path)==expected,('Protected artifact changed',path)
        self.report()
        for name,path in read(self.run/'training_protocol.json')['run_directories'].items():
            write_json(Path(path)/'done.json',dict(status='done',training_steps=12000,PACO_Val=True,ADE=True,PACO_Test=False,
                       encoder=str(Path(path)/name/'export/encoder.pt'),shared_report=str(self.run/'README.md')))
        write_json(self.run/'done.json',dict(status='done',conditions=['P2','P3'],PACO_Test_ran=False,ADE_both_ran=True,protected_inputs_unchanged=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(SHARED));p.add_argument('--base-run',default=str(RUN));p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=Path(a.run);prepare(root,a.base_run)
    if not a.execute:return
    assert read(root/'unit_verification.json')['status']=='passed' and read(root/'roi_union_preflight.json')['status']=='passed'
    pipe=PatchPipeline(root)
    write_json(root/'launch.json',dict(pid=os.getpid(),command=sys.argv,python=sys.executable))
    if not (root/'background_policy.json').exists():
        write_json(root/'background_policy.json',dict(enabled=True,ownership_prefix=PREFIX,
            allow_background_with_stages=['probe_P2','infer_P2','probe_P3','infer_P3','patch_drift'],
            utilization_policy='Stream15s; after startup120s, two samples below50 with free30GiB enable A/B for that training stage'))
    for module,label in [('src.fgtrain.resources','resource_manager'),('src.fgtrain.patch_watch','utilization_monitor')]:
        with (root/f'{label}.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m',module,'--run',root],cwd=ROOT,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(root/f'{label}_process.json',dict(pid=proc.pid))
    def stop(sig,frame):pipe.terminate_children();raise SystemExit(128+sig)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    try:pipe.execute(a.base_run)
    except BaseException:
        pipe.terminate_children();write_json(root/'failed.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
