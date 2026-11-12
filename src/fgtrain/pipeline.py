"""Restartable two-GPU foreground pilot; existing background work resumes on exit.

The default queue uses PACO Val only. Optional final Test reporting freezes Val selections first.
No evaluator is replaced.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import yaml
from src.fgtrain.common import RUN, ROOT, EXCLUDE, encoder, freeze_export, state_sha, write_json

EPY='/home/irteam/envs/paco_eval/bin/python'
FROZEN_PROBE=ROOT/'output/2026-11-09/g1_upscaled_teacher_s0/g0/probe_Full384'
FROZEN_EVAL=ROOT/'output/2026-11-09/g1_upscaled_teacher_s0/g0/eval_val/Full384_r2'
C0_PROBE=ROOT/'output/2026-11-06/ivc_v3/probe_c0'
C0_EVAL=ROOT/'output/2026-11-09/g1_upscaled_teacher_s0/eval/eval_val/C0_r2'
G1_CKPT=ROOT/'output/2026-11-06/g1_cross_distill/G1cross/ema_last.pt'


class Pipeline:
    def __init__(self,run):
        self.run=Path(run)
        self.state=self.run/'state';self.state.mkdir(exist_ok=True)
        (self.run/'logs').mkdir(exist_ok=True)
        self.lock=threading.Lock();self.report_lock=threading.Lock();self.cache_ready=threading.Event();self.cpu=ThreadPoolExecutor(max_workers=3);self.cpu_futures=[];self.active={};self.children={}
        self.filelock=(self.run/'pipeline.lock').open('w')
        fcntl.flock(self.filelock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.filelock.write(str(os.getpid()));self.filelock.flush()

    def event(self,**kw):
        with self.lock:
            kw['utc']=time.strftime('%FT%TZ',time.gmtime())
            with (self.run/'events.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
        print(json.dumps(kw),flush=True)

    def stage(self,name,module,args,gpus='',official=False):
        done=self.state/f'{name}.json'
        if done.exists():return
        cmd=[EPY if official else sys.executable,'-u','-m',module]+list(map(str,args))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
                 CUDA_CACHE_DISABLE='0',CUDA_CACHE_MAXSIZE='4294967296',CUDA_CACHE_PATH=str(ROOT/'cache/cuda_jit'))
        env['PYTHONPATH']=f'/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}' if official else str(ROOT)
        start=time.time()
        with (self.run/'logs'/f'{name}.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with self.lock:
                self.active[name]=dict(pid=child.pid,gpus=gpus,command=cmd,start=time.strftime('%FT%TZ',time.gmtime()))
                self.children[name]=child
                write_json(self.run/'active.json',self.active)
            self.event(stage=name,status='started',pid=child.pid,gpus=gpus)
            rc=child.wait()
        with self.lock:
            self.active.pop(name,None);self.children.pop(name,None);write_json(self.run/'active.json',self.active)
        if rc:
            self.event(stage=name,status='failed',returncode=rc)
            raise RuntimeError(f'{name}: returncode {rc}; see logs/{name}.log')
        write_json(done,dict(stage=name,command=cmd,gpus=gpus,seconds=round(time.time()-start)))
        self.event(stage=name,status='done',seconds=round(time.time()-start))

    def report(self):
        with self.report_lock:
            self.stage('report_'+str(time.time_ns()),'src.fgtrain.report',['--run',self.run])

    def train(self,name,gpu,steps=12000):
        args=['--condition',name,'--run',self.run,'--cache',self.run/'teacher','--out',self.run/name,'--steps',steps,'--workers',4]
        self.stage('train_'+name,'src.fgtrain.train',args,str(gpu))
        result=json.loads((self.run/name/'done.json').read_text())
        assert result['steps']==steps and result['export']['strict_loading']

    def cache_and_b1(self):
        self.stage('teacher_full','src.fgtrain.cache',['--root',self.run/'teacher','--initial',self.run/'initial_encoder.pt','--workers',12,'--batch',64],'1')
        import numpy as np
        assert np.load(self.run/'teacher/completed.npy').all()
        self.cache_ready.set()
        self.train('B1',1)
        self.probe('B1',1)

    def probe(self,name,gpu):
        folder=self.run/name
        cfg=dict(name='fgtrain_'+name,backbone='dinov3',mode='bil',seed=0,epochs=10,paco_dev_images=1000,
                 encoder_ckpt=str(folder/'export/encoder_eval.pt'),readers=[dict(name='r2',kind='r2',layers=[12])])
        conf=self.run/'configs'/f'probe_{name}.yaml';conf.parent.mkdir(exist_ok=True)
        conf.write_text(yaml.safe_dump(cfg,sort_keys=False))
        self.stage('probe_'+name,'src.g1up.train_bb',['--job',conf,'--out',folder/'probe','--exclude',EXCLUDE,'--workers',8],str(gpu))
        check=json.loads((folder/'probe/frozen_check.json').read_text());assert check['unchanged'] and not check['requires_grad_any']
        self.stage('infer_'+name,'src.fgtrain.infer',['--encoder',folder/'export/encoder.pt','--probe',folder/'probe','--out',folder/'preds','--workers',8],str(gpu))
        with self.lock:
            self.cpu_futures.append(self.cpu.submit(self.finish_official,name,folder/'preds'))

    def finish_official(self,name,preds):
        self.official_all(name,preds)
        self.report()

    def official_all(self,name,preds):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(self.stage,f'official_{name}_ep{e}','src.pacoattr.official_eval',
                      ['--split','val','--pred',Path(preds)/f'ep{e}/probs_val.npz','--out',self.run/name/'official'/f'ep{e}'],official=True) for e in range(1,11)]
            for future in futures:future.result()
        curve={e:json.loads((self.run/name/'official'/f'ep{e}/official_results.json').read_text())['hierarchical'] for e in range(1,11)}
        best=max(curve,key=lambda e:((curve[e]['obj-attr-AP']+curve[e]['obj-attr-part-AP'])/2,-e))
        write_json(self.run/name/'selection.json',dict(epoch=best,rule='max Official Val (Object AP+Part AP)/2 across epochs 1..10; earlier tie',
                   test_used=False,official_val_images=2410))
        self.stage('bootpack_'+name,'src.pacoattr.official_eval',['--split','val','--pred',Path(preds)/f'ep{best}/probs_val.npz',
                   '--out',self.run/'bootstrap_inputs'/name/'val','--boot'],official=True)

    def reuse_baselines(self):
        # Reuse completed 10-epoch original probes only after recipe/split/weights checks.
        import torch
        from src.g1in.common import checksum
        for name,probe,official,mode in [('Frozen',FROZEN_PROBE,FROZEN_EVAL,'bil'),('C0',C0_PROBE,C0_EVAL,'c0')]:
            cfg=json.loads((probe/'config.json').read_text());job=cfg['job']
            assert job['backbone']=='dinov3' and job['mode']==mode and job['epochs']==10 and job['seed']==0 and job['paco_dev_images']==1000
            assert cfg['n_train_rows']==200088 and cfg['n_train_images']==37472 and cfg['n_val_images']==2051
            assert all((probe/'r2'/f'ep{e}.pt').exists() for e in range(1,11))
            init=encoder(self.run/'initial_encoder.pt','cpu')
            assert abs(checksum(init)-json.loads((probe/'frozen_check.json').read_text())['checksum_before'])<1e-6
            target=self.run/name;target.mkdir(exist_ok=True)
            if not (target/'probe').exists():(target/'probe').symlink_to(probe,target_is_directory=True)
            if not (target/'official').exists():(target/'official').symlink_to(official,target_is_directory=True)
            curve={e:json.loads((official/f'ep{e}/official_results.json').read_text())['hierarchical'] for e in range(1,11)}
            assert all(json.loads((official/f'ep{e}/official_results.json').read_text())['n_eval_images']==2410 for e in range(1,11))
            best=max(curve,key=lambda e:((curve[e]['obj-attr-AP']+curve[e]['obj-attr-part-AP'])/2,-e))
            write_json(target/'selection.json',dict(epoch=best,rule='max Official Val mean Object/Part; earlier ties',reused=True,test_used=False))
            write_json(target/'reuse.json',dict(probe=str(probe),official=str(official),protocol_verified=True,mode=mode,initial_sha=state_sha(init.state_dict()),
                        warning='C0 uses per-ROI native RGB384 re-encoding and is an information oracle, not a deployable encoder-only method' if name=='C0' else None))
        # Frozen ADE baseline is an EXACT existing run of the unchanged A.main protocol (40k, world2, workers12).
        ade=ROOT/'output/2026-11-07/g1_leakcheck_ade20k/ade'
        ac=json.loads((ade/'config.json').read_text());done=json.loads((ade/'done.json').read_text())
        assert ac['cfg']['crop']==384 and ac['cfg']['iters']==40000 and ac['world']==2 and ac['cfg']['seed']==0
        assert ac['n_train']==20210 and ac['n_val']==2000 and all(done['unchanged'].values())
        write_json(self.run/'ade_baseline_reuse.json',dict(source=str(ade),config=ac,results=done['results'],checksums=done['checksum_after'],
                    note='Original and G1-cross reference mIoU reused; B0/B1 use unchanged A.main with same world2/workers12/seed/sampler/augmentation'))

    def export_g1(self):
        import torch
        folder=self.run/'G1cross/export';folder.mkdir(parents=True,exist_ok=True)
        ck=torch.load(G1_CKPT,map_location='cpu',weights_only=False)
        assert ck['kind']=='ema'
        torch.save(ck['enc'],folder/'encoder.pt')
        torch.save(dict(enc=ck['enc'],kind='encoder_only',step=12000),folder/'encoder_eval.pt')
        del ck
        write_json(self.run/'G1cross/provenance.json',dict(source=str(G1_CKPT),crop_teacher=384,training_steps=12000,initial='Original pretrained',
                    encoder_only=True,official_scope_training_overlap='Existing G1 used legacy exclusions: full official Val146/Test475 exact images were seen during unsupervised COCO training; new B methods exclude them. Reference only.',old_reader='full-canvas R2',new_reader='matched bilinear ROIAlign 24x24 R2; trained 10 epochs'))

    def b0_and_controls(self):
        self.train('B0',0)
        self.probe('B0',0)
        self.probe('G1cross',0)
        self.cache_ready.wait()
        self.train('FrozenDecoder',0)

    def bootstrap(self):
        # Existing official per-image matching + verified bootstrap; no new AP implementation.
        self.stage('paired_bootstrap','src.fgtrain.bootstrap',['--run',self.run],official=True)

    def b2_gate(self):
        import numpy as np
        b=json.loads((self.run/'paired_bootstrap.json').read_text())
        point=b['points'];boot=b['boot']
        keys=['obj_pattern_marking','part_pattern_marking']
        delta={k:point['B1'][k]-point['B0'][k] for k in ['obj','part']+keys}
        ov=.5*(np.array(boot['B1']['obj'])+np.array(boot['B1']['part'])-np.array(boot['B0']['obj'])-np.array(boot['B0']['part']))
        lo,hi=np.percentile(ov,[2.5,97.5])
        # Avoid launching pixel supervision on a weak/noisy pilot gain.
        pattern_ci={k:np.percentile(np.array(boot['B1'][k])-np.array(boot['B0'][k]),[2.5,97.5]).tolist() for k in keys}
        ep10=[]
        for name in ['B0','B1']:
            h=json.loads((self.run/name/'official/ep10/official_results.json').read_text())['hierarchical']
            ep10.append([h['obj-attrs']['pattern_marking'],h['obj-part-attrs']['pattern_marking']])
        passed=bool(lo>0 and all(delta[k]>0 for k in keys) and all(pattern_ci[k][0]>0 for k in keys) and all(np.array(ep10[1])>np.array(ep10[0])))
        write_json(self.run/'b2_gate.json',dict(run_B2=passed,delta=delta,overall_ci=[float(lo),float(hi)],pattern_ci=pattern_ci,
                   rule='Both Pattern AP deltas and their paired-image 95% CI lower bounds >0, Overall CI lower bound >0, both ep10 Pattern deltas >0; seed0 evidence only',
                   decoder_reconstruction_not_a_success_criterion=True,test_used=False))
        return passed

    def run_all(self):
        assert json.loads((self.run/'preflight_state.json').read_text())['status']=='passed'
        self.reuse_baselines();self.export_g1();self.report()
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(self.b0_and_controls),pool.submit(self.cache_and_b1)]
            from concurrent.futures import as_completed
            try:
                for f in as_completed(fs):f.result()
            except BaseException:
                self.cache_ready.set();self.terminate_children();raise
        for f in self.cpu_futures:f.result()
        self.cpu.shutdown(wait=True)
        self.bootstrap()
        if self.b2_gate():
            self.cpu=ThreadPoolExecutor(max_workers=1);self.cpu_futures=[]
            self.train('B2',0);self.probe('B2',0)
            for f in self.cpu_futures:f.result()
            self.cpu.shutdown(wait=True)
        models='B0,B1,B2' if (self.run/'B2/done.json').exists() else 'B0,B1'
        self.stage('ade','torch.distributed.run',['--standalone','--nproc_per_node','2','-m','--','src.fgtrain.ade',
                   '--run',self.run,'--models',models,'--out',self.run/'ade','--workers',12],'0,1')
        self.report()
        write_json(self.run/'done.json',dict(status='done',B2_ran='B2' in models,utc=time.strftime('%FT%TZ',time.gmtime())))

    def run_final_test(self, source, methods=('B0','B1'), report_module='src.fgtrain.report', once=False):
        """Final reporting only: reuse inference/evaluator with immutable Val selections."""
        source=Path(source)
        selections={m:json.loads((source/m/'selection.json').read_text()) for m in methods}
        artifacts={}
        for m,selection in selections.items():
            assert selection['test_used'] is False
            verification=json.loads((source/m/'export/export_verification.json').read_text())
            assert verification['strict_loading'] and verification['additional_modules']==[]
            epoch=selection['epoch']
            paths=[source/m/'export/encoder.pt',source/m/'probe/r2'/f'ep{epoch}.pt']
            artifacts[m]=dict(epoch=epoch,selection=selection,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        protocol=dict(split='test',methods=artifacts,selection_split='val',test_for_selection=False,
                      encoder_only=True,training_or_hyperparameter_changes=False)
        path=self.run/'fixed_protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol, 'Final Test selections are immutable'
        else:write_json(path,protocol)
        def evaluate(m,gpu):
            epoch=artifacts[m]['epoch'];folder=self.run/m
            self.stage('infer_test_'+m,'src.fgtrain.infer',[
                '--encoder',source/m/'export/encoder.pt','--probe',source/m/'probe',
                '--out',folder/'preds','--split','test','--epochs',epoch,'--workers',4],str(gpu))
            if once and not (self.state/f'official_test_{m}.json').exists():
                attempt=folder/'official_attempt.json'
                assert not attempt.exists(), 'Official Test was already attempted; do not automatically repeat it'
                write_json(attempt,dict(method=m,epoch=epoch,protocol=str(path),utc=time.strftime('%FT%TZ',time.gmtime())))
            self.stage('official_test_'+m,'src.pacoattr.official_eval',[
                '--split','test','--pred',folder/'preds'/f'ep{epoch}/probs_test.npz','--out',folder/'official'],official=True)
            h=json.loads((folder/'official/official_results.json').read_text())['hierarchical']
            return m,dict(overall=(h['obj-attr-AP']+h['obj-attr-part-AP'])/2,object=h['obj-attr-AP'],part=h['obj-attr-part-AP'],
                          obj_pattern=h['obj-attrs']['pattern_marking'],part_pattern=h['obj-part-attrs']['pattern_marking'],
                          obj_color=h['obj-attrs']['color'],part_color=h['obj-part-attrs']['color'],
                          obj_material=h['obj-attrs']['material'],part_material=h['obj-part-attrs']['material'],epoch=epoch)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(evaluate,m,gpu) for gpu,m in enumerate(selections)]
            results=dict(f.result() for f in fs)
        assert all(json.loads((source/m/'selection.json').read_text())==s for m,s in selections.items())
        write_json(self.run/'results.json',dict(split='test',metrics=results,selection_split='val',test_for_selection=False))
        if report_module:self.stage('report_final_test',report_module,['--run',source])
        write_json(self.run/'done.json',dict(status='done',split='test',utc=time.strftime('%FT%TZ',time.gmtime())))

    def terminate_children(self):
        for child in list(self.children.values()):
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN));p.add_argument('--final-test',action='store_true');a=p.parse_args()
    if a.final_test:
        target=Path(a.run)/'test';target.mkdir(exist_ok=True)
        pipe=Pipeline(target)
        try:pipe.run_final_test(a.run)
        except BaseException:
            pipe.terminate_children();write_json(target/'failed.json',dict(traceback=traceback.format_exc()));raise
        return
    pipe=Pipeline(a.run)
    # Idle backfill is opt-in: never hold both GPUs during CPU-only stages.
    owner=json.loads((pipe.run/'resource_ownership.json').read_text())
    flag=Path(owner['flag'])
    policy_file=pipe.run/'background_policy.json'
    backfill=policy_file.exists() and json.loads(policy_file.read_text()).get('enabled',False)
    if backfill:
        if owner['owned'] and flag.exists() and flag.read_text()==owner['payload']:
            flag.unlink()
        write_json(pipe.run/'background_pause_verification_latest.json',dict(
            mode='idle_backfill',global_pause=False,
            rule='Existing background supervisor detects foreground GPU processes and checkpoint-yields automatically'))
    else:
        if owner['owned'] and not flag.exists():flag.write_text(owner['payload'])
        from src.brixel_eval.ade_run import active, await_paused
        write_json(pipe.run/'background_pause_verification_latest.json',await_paused(active()))
    def stop(sig,frame):
        pipe.terminate_children();raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        pipe.run_all()
    except BaseException:
        pipe.terminate_children()
        write_json(pipe.run/'failed.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
    finally:
        record=pipe.run/'resource_ownership.json'
        if record.exists():
            owner=json.loads(record.read_text());flag=Path(owner['flag'])
            released=bool(owner['owned'] and flag.exists() and flag.read_text()==owner['payload'])
            if released:flag.unlink()
            write_json(pipe.run/'background_restored.json',dict(released_own_pause=released,preserved_preexisting_pause=not owner['owned']))

if __name__=='__main__':main()
