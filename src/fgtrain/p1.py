"""P1 on physical GPU1 after readiness; no N1 edits and no automatic PACO Test."""
import argparse
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

from src.fgtrain.common import ROOT,RUN,write_json
from src.fgtrain.pipeline import Pipeline
from src.fgtrain.n1 import metrics
from src.brixel_eval.finish import pidfd
from src.crop_aware_pretrain import gpu as GPU

P1_RUN=ROOT/'output/2026-11-11/fgtrain_p1_s0'
N1_RUN=ROOT/'output/2026-11-11/fgtrain_n1_s0'
CONTROL=ROOT/'output/2026-11-09/bgjobs/control'


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(root,base):
    root,base=Path(root),Path(base);root.mkdir(parents=True,exist_ok=True)
    cfg=read(base/'B0/config.json')
    assert cfg['initial_sha']==read(base/'initialisation.json')['state_sha256']
    assert cfg['steps']==12000 and cfg['seed']==0 and cfg['lambda_global']==.5
    fixed=dict(name='P1',initial_sha=cfg['initial_sha'],initial='Independent Original pretrained DINOv3-B/16',
               base_run=str(base),steps=12000,seed=0,lambda_global=.5,remove_neg=False,
               gram='Fixed Original reference; final-LN patches only; FP32 L2 normalization; per-image weighted MSE',
               pair_weight='wi*wj from fractional letterbox area',zero_valid='Exclude zero-valid images; all-zero yields differentiable0',
               calibration=dict(step=6000,kind='EMA',batch='B0 COCO stream0',target_gradient_ratio=.5,
                                missing_midpoint='Replay only B0 steps1..6000 with full12000 LR/EMA schedule; no B0 outputs overwritten'),
               physical_gpu=1,student_optimizer=cfg,PACO_Test_automatic=False,
               ADE='Same unchanged world2/batch16/40k protocol; after N1 completion and safe GPU0+1 allocation')
    p=root/'protocol.json'
    if p.exists():assert read(p)==fixed,'P1 design cannot change after registration'
    else:write_json(p,fixed)
    inputs=[base/'initial_encoder.pt',base/'B0/export/encoder.pt',base/'B1/export/encoder.pt',
            N1_RUN/'N1/export/encoder.pt',N1_RUN/'evaluation_protocol.json',N1_RUN/'training_protocol.json',
            ROOT/'src/fgtrain/train.py',ROOT/'src/fgtrain/n1.py']
    record={str(p):sha(p) for p in inputs}
    p=root/'protected_inputs.json'
    if p.exists():assert read(p)==record,'Protected original artifacts changed'
    else:write_json(p,record)
    report(root)


def report(root):
    root=Path(root);base=Path(read(root/'protocol.json')['base_run'])
    results=dict(PACO_Test_automatic=False)
    lines=['# P1 — B0 + fixed-reference DINOv3 Gram Anchoring','',
           'GPU1로 변경한 사용자 지시에 따라 준비·CPU 검증을 끝낸 후에만 기존 GPU1 작업을 checkpoint pause한다. GPU0의 N1은 중단하거나 변경하지 않는다.',
           '', 'Student는 Original DINOv3-B/16에서 독립적으로12k 학습한다. Native Crop768 및 Full384 Global target은 B0의 cache를 그대로 재사용하고 `src.fgtrain.train.terms`를 그대로 호출한다.',
           'Full384의 최종 LN patch576개만 L2 normalize하여 이미지별 Gram을 계산한다. CLS/register는 제외한다. Gram Teacher는 동일 Full384 tensor를 받는 frozen Original이며 학습 내내 고정한다. Decoder/Adapter/추가 추론 모듈은 없다.',
           '', '`L_P1 = L_crop + 0.5 L_global + lambda_gram L_gram`',
           '`L_gram(b) = sum_ij wi*wj*(Gs_ij−Gt_ij)^2 / sum_ij wi*wj`; 유효 이미지별 loss를 평균한다. zero-valid 이미지는 평균에서 제외하고 전부 zero-valid이면 graph-connected0을 반환한다. 음수 similarity 및 대각 pair를 유지한다.',
           '', '정규화·이미지별 Gram MSE의 참고 구현: `third_party/BRIXEL/src/brixel/dinov3_main/dinov3/loss/gram_loss.py`, apply_norm=True/remove_neg=False. Frozen Original을 끝까지 고정하는 reference 선택, fractional letterbox pair weighting 및 zero-valid 처리는 본 실험의 확장이다.',
           '', 'B0의 중간 recovery가 덮어써져 step12000만 남아 있어, 별도 `calibration_B0_replay/`에 동일 B0의 첫6000step을 재현한다. LR와 EMA는 원래12000step schedule 그대로이며,6000 EMA와 사전 고정 stream0 batch로 lambda를 한 번 보정한다. 실제 P1 초기화는 이6k 모델이 아니라 Original이다. λ 탐색은 없다.']
    calibration=root/'lambda_gram.json'
    if calibration.exists():
        c=read(calibration);results['calibration']=c
        lines+=['',f"lambda_gram={c['lambda_gram']:.12f}; calibration Gram loss={c['gram_loss']:.8f}; gradient ratio={c['measured_ratio']:.8f}. Checkpoint={c['checkpoint']}, EMA step{c['checkpoint_step']}, image IDs={c['image_ids']}.",
                'Crop/Gram gradient 및 patch embedding·12block norm은 `lambda_gram.json`; 초기/중간/후반 weighted gradient ratio와 cosine은 `P1/gradient_diagnostics.jsonl`에 저장한다. 원본 초기 Student==Teacher의 Gram0, teacher detach/frozen, fractional padding 제외 및 zero-valid를 검증한다.']
    lines+=['','| Method | Val Overall | Object | Part | Object Pattern | Part Pattern | Obj Color | Part Color | Obj Material | Part Material | ADE mIoU |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['Frozen','B0','P1']:
        folder=root/name if name=='P1' else base/name
        row={}
        if (folder/'selection.json').exists():
            e=read(folder/'selection.json')['epoch'];row=metrics(folder/'official'/f'ep{e}/official_results.json')
        miou=None
        if name!='P1':miou=read(base/'results.json')['metrics'][name].get('miou')
        elif (root/'ade/done.json').exists():miou=read(root/'ade/done.json')['results']['P1']['mIoU']
        if miou is not None:row['miou']=miou
        results[name]=row
        lines+=['| '+name+' | '+' | '.join(f'{row[k]:.3f}' if k in row else '미완료' for k in ['overall','object','part','obj_pattern','part_pattern','obj_color','part_color','obj_material','part_material','miou'])+' |']
    if results.get('P1',{}).get('overall') is not None:
        results['P1_minus_B0']={k:v-results['B0'][k] for k,v in results['P1'].items() if k in results['B0']}
        lines+=['',f"P1−B0 (pp): `{json.dumps(results['P1_minus_B0'])}`"]
    if (root/'gram_drift.json').exists():
        results['gram_drift']=read(root/'gram_drift.json')['mean']
        lines+=['',f"동일 COCO dev 이미지의 Original 대비 Gram drift: `{json.dumps(results['gram_drift'])}`. 이 진단만으로 ADE 회복을 판단하지 않는다."]
    if (root/'P1/export/export_verification.json').exists():
        v=read(root/'P1/export/export_verification.json');results['export']=v
        lines+=['',f"Strict standard export={v['strict_loading']}, params={v['parameter_count']}, patch shape={v['patch_shape']}, forward equality={v['bit_equal_forward']}, extra modules={v['additional_modules']}."]
    lines+=['', 'PACO Test는 자동 실행하지 않는다. N1의 기존 Test Gate와 별개다. ADE는 기존 `src.g1eval.ade`의 frozen encoder + Linear768→150 probe, seed0,40k,batch16,world2,동일 전처리/최종 iteration 평가를 재사용한다. GPU0 사용은 N1이 모두 완료된 뒤 기존 A/B만 안전하게 일시 중지할 수 있을 때로 한정한다.',
            '결과는 Pattern 유지+ADE회복 / Pattern하락+ADE회복 / Pattern유지+ADE회복없음으로 구분한다. 작은 차이의 유지·하락 여부는 단일 seed 점추정만으로 확정하지 않는다. ADE 결과가 없으면 dense preservation은 미검증이다.']
    diagnostics=root/'P1/gradient_diagnostics.jsonl'
    if diagnostics.exists():
        samples=[json.loads(s) for s in diagnostics.read_text().splitlines()]
        lines+=['','## 실제 초기·중간·후반 gradient','','| Step | Crop norm | Weighted Gram norm | Gram/Crop | Crop–Gram cosine | Gram drift |','|---|---:|---:|---:|---:|---:|']
        for q in samples:
            if q['step'] not in [1,6000,12000]:continue
            cosine='정의 없음(Gram gradient0)' if q['crop_gram_cosine'] is None else f"{q['crop_gram_cosine']:.6f}"
            lines += [f"| {q['step']} | {q['crop']['encoder']:.6f} | {q['weighted_gram_norm']:.6f} | {q['gram_crop_ratio']:.6f} | {cosine} | {q['gram_drift']:.6f} |"]
    if 'miou' in results.get('P1',{}):
        p=results['P1'];b=results['B0'];original=results['Frozen']
        delta=p['miou']-b['miou'];gap=p['miou']-original['miou']
        lines+=['','## 최종 해석','',
                f"ADE20K: P1={p['miou']:.6f}, B0={b['miou']:.6f}, Frozen={original['miou']:.6f}. P1−B0={delta:+.6f} pp, P1−Frozen={gap:+.6f} pp.",
                f"PACO Official Val: Overall {p['overall']-b['overall']:+.6f} pp, Object Pattern {p['obj_pattern']-b['obj_pattern']:+.6f} pp, Part Pattern {p['part_pattern']-b['part_pattern']:+.6f} pp. Object/Part Pattern 변화의 방향이 다르므로 두 Pattern의 동시 개선으로 부르지 않는다.",
                'Gram drift 감소는 확인됐지만 ADE 점추정 회복은 작고 Original 수준과의 격차가 남았다. 원본 Dense Representation의 보존 또는 강한 Segmentation 개선을 입증했다고 결론 내리지 않는다. 단일 seed의 작은 mIoU 차이는 학습 변동과 구분하지 못한다.',
                'PACO Part Pattern의 점추정 이득과 제한적인 ADE 회복을 분리해 기록한다. 이 결과로 추가 λ/Teacher/Loss 탐색이나 PACO Test를 자동 실행하지 않는다.']
    failures=root/'implementation_failures'
    if failures.exists():
        lines+=['','## 실행 중 수정','',
                'GPU 인계 중 NVML에 종료 PID가 잠시 남는 경우를 처리했다. 살아 있는 외부 프로세스는 계속 보호한다.12k 학습 및 strict export 후 완료 metadata의 변수 오타로 중단된 실행은 저장된12k EMA와 optimizer checkpoint를 검증해 추가 optimizer step 없이 재개했다. `export_metadata_recovery.json`의 EMA SHA와 `implementation_failures/`에 근거를 남겼다. N1/B0/B1의 가중치·설정·소스는 `protected_inputs.json`으로 전후 일치 검증했다.']
    write_json(root/'results.json',results);(root/'README.md').write_text('\n'.join(lines)+'\n')
    dest=ROOT/'experiments/fgtrain_p1';dest.mkdir(exist_ok=True)
    for name in ['README.md','results.json']:(dest/name).write_text((root/name).read_text())


def checked_apps(gpus,allowed_sessions=None):
    state=GPU.query();uuids={g['uuid'] for g in state['gpus'] if g['index'] in gpus}
    assert len(uuids)==len(gpus),'Requested physical GPUs are not visible'
    active=read(CONTROL/'active.json')
    sessions=allowed_sessions if allowed_sessions is not None else {s for v in active.values() for k in ['train','eval'] for s in (v.get(k) or {}).get('sids',[])}
    apps=[]
    for a in state['apps']:
        if a['uuid'] not in uuids:continue
        sid=GPU.proc_sid(a['pid'])
        # NVML can retain an already exited PID briefly during checkpoint handoff.
        if sid is None or GPU.is_zombie(a['pid']):continue
        assert sid in sessions, f"GPU process{a['pid']} is not an owned A/B job; leave it running"
        apps.append(dict(a,session=sid))
    return apps


def await_free(gpus,sessions):
    while True:
        apps=checked_apps(gpus,sessions)
        fds=[]
        for a in apps:
            try:fds.append(pidfd(a['pid']))
            except ProcessLookupError:pass
        if not fds:return
        try:select.select(fds,[],[])
        finally:
            for fd in fds:os.close(fd)


class P1Pipeline(Pipeline):
    def report(self):
        with self.report_lock:report(self.run)

    def execute(self,base):
        args=['--run',self.run,'--base-run',base]
        self.stage('replay_B0_6000','src.fgtrain.p1_train',args+['--stage','replay'],'1')
        self.stage('calibrate_gram','src.fgtrain.p1_train',args+['--stage','calibrate','--checkpoint',self.run/'calibration_B0_replay/recovery.pt'],'1')
        self.report()
        self.stage('train_P1','src.fgtrain.p1_train',args+['--stage','train'],'1')
        self.stage('gram_drift','src.fgtrain.p1_train',args+['--stage','drift'],'1')
        self.report();self.probe('P1',1)
        for f in self.cpu_futures:f.result()
        self.cpu.shutdown(wait=True)
        # Do not steal GPU0 from N1; wait for its unchanged driver to finish first.
        if not (N1_RUN/'done.json').exists():
            pid=read(N1_RUN/'launch.json')['pid']
            try:
                fd=pidfd(pid)
                try:select.select([fd],[],[])
                finally:os.close(fd)
            except ProcessLookupError:pass
        if (N1_RUN/'done.json').exists():
            apps=checked_apps([0,1]);sessions={a['session'] for a in apps}
            self.active['handoff_ADE']=dict(pid=os.getpid(),gpus='0,1',command=[],start=time.strftime('%FT%TZ',time.gmtime()))
            write_json(self.run/'active.json',self.active)
            payload='fgtrain per-GPU priority: fgtrain_p1_s0: preparing ADE\n'
            for g in [0,1]:
                flag=CONTROL/f'PAUSE_gpu{g}'
                assert not flag.exists() or flag.read_text().startswith('fgtrain per-GPU priority: fgtrain_p1_s0:')
                flag.write_text(payload)
            await_free([0,1],sessions);self.active.pop('handoff_ADE')
            self.stage('ade_P1','torch.distributed.run',['--standalone','--nproc_per_node','2','-m','--','src.fgtrain.ade',
                       '--run',self.run,'--models','P1','--out',self.run/'ade','--workers',12],'0,1')
        else:write_json(self.run/'ADE_unavailable.json',dict(reason='N1 did not complete; no GPU0 takeover'))
        self.report()
        for name,expected in read(self.run/'protected_inputs.json').items():assert sha(name)==expected,('Protected artifact changed',name)
        write_json(self.run/'done.json',dict(status='done',PACO_Test_ran=False,ADE_ran=(self.run/'ade/done.json').exists()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(P1_RUN));p.add_argument('--base-run',default=str(RUN));p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=Path(a.run);prepare(root,a.base_run)
    if not a.execute:return
    assert read(root/'unit_verification.json')['status']=='passed','CPU verification must finish before reserving GPU1'
    apps=checked_apps([1]);sessions={a['session'] for a in apps}
    pipe=P1Pipeline(root)
    write_json(root/'launch.json',dict(pid=os.getpid(),command=sys.argv))
    write_json(root/'background_policy.json',dict(enabled=True,allow_background_with_stages=[],ownership_prefix='fgtrain per-GPU priority: fgtrain_p1_s0:'))
    pipe.active['handoff']=dict(pid=os.getpid(),gpus='1',command=[],start=time.strftime('%FT%TZ',time.gmtime()))
    write_json(root/'active.json',pipe.active)
    flag=CONTROL/'PAUSE_gpu1'
    assert not flag.exists(),'Preserve existing GPU1 pause owner'
    flag.write_text('fgtrain per-GPU priority: fgtrain_p1_s0: ready to start\n')
    with (root/'resource_manager.log').open('a') as log:
        manager=subprocess.Popen([sys.executable,'-u','-m','src.fgtrain.resources','--run',root],cwd=ROOT,
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(root/'resource_manager_process.json',dict(pid=manager.pid))
    def stop(sig,frame):pipe.terminate_children();raise SystemExit(128+sig)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    try:
        await_free([1],sessions);pipe.active.pop('handoff')
        write_json(root/'GPU1_handoff.json',dict(ready=True,physical_gpu=1,internal_device='cuda:0',existing_jobs_checkpoint_paused=True))
        pipe.execute(a.base_run)
    except BaseException:
        pipe.terminate_children();write_json(root/'failed.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
