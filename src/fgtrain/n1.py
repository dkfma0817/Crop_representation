"""N1's predeclared Official Val gate and one final Test; reuse the B0 evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from src.fgtrain.common import ROOT, RUN, write_json
from src.fgtrain.pipeline import Pipeline

N1_RUN=ROOT/'output/2026-11-11/fgtrain_n1_s0'
KEYS=['overall','object','obj_color','obj_pattern','obj_material','obj_ref',
      'part','part_color','part_pattern','part_material','part_ref']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def metrics(path):
    h=read(path)['hierarchical'];o,p=h['obj-attr-AP'],h['obj-attr-part-AP']
    r=dict(overall=(o+p)/2,object=o,part=p)
    for prefix,key in [('obj','obj-attrs'),('part','obj-part-attrs')]:
        for label,k in [('color','color'),('pattern','pattern_marking'),('material','material'),('ref','transparency')]:
            r[prefix+'_'+label]=h[key][k]
    return r


def prepare(run,base):
    run,base=Path(run),Path(base);run.mkdir(parents=True,exist_ok=True)
    cfg=read(base/'B0/config.json');probe=read(base/'B0/probe/config.json')
    initial=read(base/'initialisation.json');selection=read(base/'B0/selection.json')
    export=read(base/'B0/export/export_verification.json')
    inf=read(base/'B0/preds/inference_verification.json')
    assert cfg['steps']==12000 and cfg['seed']==0 and cfg['lambda_global']==.5
    assert cfg['lambda_spatial']==0 and cfg['decoder_parameters']==0
    assert cfg['initial_sha']==initial['state_sha256']
    assert initial['initialisation']=='Original pretrained DINOv3; neither G1-pool nor A0'
    job=probe['job']
    assert job['backbone']=='dinov3' and job['mode']=='bil' and job['seed']==0 and job['epochs']==10
    assert job['readers']==[dict(name='r2',kind='r2',layers=[12])]
    assert probe['n_train_rows']==200088 and probe['n_train_images']==37472
    assert probe['S']==384 and probe['P']==16 and probe['world']==1
    assert export['strict_loading'] and export['parameter_count']==85641216
    assert export['additional_modules']==[] and export['bit_equal_forward']
    assert inf['counts']==dict(images=2410,encoder_canvases=2410,rois=31919)
    assert inf['standard_encoder'] and not inf['roi_rgb_reencoding']
    curve={}
    for e in range(1,11):
        path=base/'B0/official'/f'ep{e}/official_results.json'
        out=read(path)
        assert out['split']=='val' and out['n_eval_images']==2410 and out['alignment_mismatch']==0
        curve[e]=metrics(path)['overall']
    best=max(curve,key=lambda e:(curve[e],-e))
    assert best==selection['epoch']==7 and selection['test_used'] is False
    measured=metrics(base/'B0/official'/f'ep{best}/official_results.json')
    for k,v in dict(overall=55.450,object=54.511,part=56.390,obj_pattern=59.644,part_pattern=55.243).items():
        assert abs(measured[k]-v)<.000501,(k,measured[k],v)
    frozen=read(base/'B0/probe/frozen_check.json')
    assert frozen['unchanged'] and not frozen['requires_grad_any']
    scope=read(base/'official_split_exclusion.json')
    assert all(v==0 for v in scope['train_official_overlap'].values())
    assert all(v==0 for v in scope['dev_official_overlap'].values())
    sources=[base/'initial_encoder.pt',base/'B0/export/encoder.pt',base/'B0/probe/r2/ep7.pt',
             base/'B0/config.json',base/'B0/probe/config.json',base/'official_split_exclusion.json',
             base/'target_freeze.json',base/'teacher/rois.parquet']
    record=dict(source=str(base),reusable=True,initial_sha=initial['state_sha256'],B0=measured,
                selected_reader_epoch=best,reader_config=probe,student_config=cfg,
                sha256={str(p):sha(p) for p in sources},
                matched_protocol='Standard DINOv3-B/16, RGB384, bilinear ROIAlign24, R2, 10 epochs, seed0, Official Val2410',
                baseline_test_not_consulted=True)
    path=run/'baseline_audit.json'
    if path.exists():assert read(path)==record,'B0 baseline artifacts changed'
    else:write_json(path,record)
    protocol=dict(condition='N1',base_run=str(base),initial_sha=cfg['initial_sha'],seed=0,steps=12000,
                  sampling='Same COCO stream and exclusions as B0; GT + random + fallback ROIs',
                  inference='Strict-loaded standard encoder only; no auxiliary module',
                  val_selection='Max Official Val mean Object/Part across R2 epochs1..10; earlier tie',
                  gate=dict(operator='>',threshold=measured['overall'],display_threshold=55.450,
                            precision='Use unrounded B0 AP to avoid rounding-only gains',bootstrap_required=False),
                  test='One final evaluation of Val-selected encoder/Reader identity; never modify N1 after Test',
                  bootstrap=dict(split='val',samples=1000,required_for_gate=False),ADE20K_automatic=False)
    path=run/'evaluation_protocol.json'
    if path.exists():assert read(path)==protocol,'N1 evaluation protocol is immutable'
    else:write_json(path,protocol)
    link=run/'B0'
    if not link.exists():link.symlink_to(base/'B0',target_is_directory=True)
    link=run/'bootstrap_inputs/B0/val';link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():link.symlink_to(base/'bootstrap_inputs/B0/val',target_is_directory=True)
    report(run)
    return record


def report(run):
    run=Path(run);audit=read(run/'baseline_audit.json');base=audit['B0']
    result=dict(status='Awaiting N1 training configuration',baseline_reused=True,B0_val=base)
    if (run/'N1/config.json').exists():result['status']='Training or R2/Official Val in progress'
    selection=run/'N1/selection.json'
    if selection.exists():
        epoch=read(selection)['epoch']
        n1=metrics(run/'N1/official'/f'ep{epoch}/official_results.json')
        result.update(status='Official Val complete',N1_val=n1,delta_val_pp={k:n1[k]-base[k] for k in KEYS},reader_epoch=epoch)
    gate=run/'test_gate.json'
    if gate.exists():result['test_gate']=read(gate)
    test=run/'test/results.json'
    if test.exists():
        n1=read(test)['metrics']['N1'];ep=read(run/'test/fixed_protocol.json')['methods']['N1']['epoch']
        # Read B0 Test only for final reporting, after N1's frozen Test evaluation.
        base_test=Path(audit['source'])/'test/B0/official/official_results.json'
        b0=metrics(base_test)
        n1=metrics(run/'test/N1/official/official_results.json')
        result.update(status='Final Test complete',N1_test=n1,B0_test=b0,delta_test_pp={k:n1[k]-b0[k] for k in KEYS},test_reader_epoch=ep)
    write_json(run/'results.json',result)
    lines=['# N1: B0 + NeCo-style ROI Neighbor Ordering','',
           '사용자가 고정한 ROI-level 변형이며 원래 NeCo의 공식 재현이 아니다. 학습 설정은 `training_protocol.json`, `N1/config.json`에 보존한다. 추가 Fine/Pixel Decoder, Adapter는 없다.',
           '', 'B0는 Original pretrained 초기화, COCO GT+random/fallback,12k, RGB384 DINOv3-B/16, 표준 encoder-only export를 사용했다. 기존 B0의 Official Val 및10epoch R2 설정/초기화/제외 split/forward 검증을 재확인하여 재사용한다.',
           '',f"Test Gate: N1 Official Val Overall > {base['overall']:.12f} (B0 표시값55.450). bootstrap은 추가 진입 조건이 아니다. Test에는 Val 선택을 그대로 고정하며 한 번만 평가한다. ADE20K는 자동 실행하지 않는다.",
           '', 'Reader selection은 같은 Official Val2410 이미지에서 10epoch 중 Object/Part 평균 최대, 동점이면 앞 epoch다. 최종 EMA encoder는12k 고정이다. Test로 loss·method·epoch·B2 여부를 선택하지 않는다.',
           '', '| Split / Model | Overall | Obj AP | Obj col | Obj pat | Obj mat | Obj ref | Part AP | Part col | Part pat | Part mat | Part ref |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for name in ['Frozen','B0','B1','N1','C0']:
        if name=='N1':value=result.get('N1_val')
        else:
            source=Path(audit['source'])/name
            epoch=read(source/'selection.json')['epoch']
            value=metrics(source/'official'/f'ep{epoch}/official_results.json')
        rows.append(('Val '+name,value))
    rows.append(('Val N1−B0',result.get('delta_val_pp')))
    if 'N1_test' in result:rows.extend([('Test B0',result['B0_test']),('Test N1',result['N1_test']),('Test N1−B0',result['delta_test_pp'])])
    for label,values in rows:
        lines.append('| '+label+' | '+' | '.join(f'{values[k]:.3f}' if values else '미실행' for k in KEYS)+' |')
    lines+=['','`ref`는 기존 공식 evaluator의 `transparency` 출력이다. Val/Test 수치를 섞지 않는다.',
            'C0는 native ROI RGB를 재인코딩하는 Crop Oracle이며 표준 encoder-only 방법과 정보 조건이 다르다. Frozen은 학습0step, B0/B1/N1은 동일 Original 초기화에서12k다.',
            '', '검증: `baseline_audit.json`, `evaluation_protocol.json`. 결과: `test_gate.json`, `paired_bootstrap.json`, `test/fixed_protocol.json`, `results.json`.']
    lines+=['','## 고정 Loss와 실제 검증','',
            'Student full RGB384 → 기존 coverage-weighted pooled ROI feature를 사용한다. Teacher는 원본 RGB ROI crop→letterbox768→Frozen Original DINOv3의 B0 pooled cache이며 detach한다. 여러 이미지의 ROI를 함께 L2 normalize하여 cosine similarity를 구한다. PACO 속성 label은 encoder 학습에 사용하지 않는다.',
            'Self 및 같은 이미지 IoU>0.5를 먼저 제외하고 Teacher 순서의 상위4개를 positive, 1-based16~32위(양 끝 포함)를 negative로 고른다. Teacher cosine 차이>0.02인 조합만 비교한다. 같은 이미지라도 IoU≤0.5면 후보로 남긴다.',
            '`L_rank = mean softplus((S_neg − S_pos)/0.07)`; `L_N1 = L_crop + 0.5 L_global + lambda_rank L_rank`. Crop/Global 경로는 B0와 동일한 함수를 그대로 사용한다. lambda는 초기 동일 stream0 미니배치에서 encoder 전체 gradient norm 비율0.25로 한 번 보정한 뒤 고정한다. Teacher/Student similarity 행렬과 이웃 선택은 학습에만 사용하며 추론 경로에는 없다.']
    if (run/'lambda_rank.json').exists():
        pre=read(run/'lambda_rank.json');ps=pre['pair_statistics']
        result['preflight']=dict(lambda_rank=pre['lambda_rank'],gradients=pre['gradients'],pairs=ps)
        lines += ['',f"실측 lambda_rank={pre['lambda_rank']:.12f}; crop encoder gradient={pre['gradients']['crop']['encoder']:.8f}, rank={pre['gradients']['rank']['encoder']:.8f}; weighted ratio={pre['measured_gradient_ratio']:.8f}.",
                  f"초기 {ps['rois']} ROI: valid pairs {ps['valid_pairs']}/{ps['possible_pairs']} ({100*ps['pair_ratio']:.3f}%), anchor coverage {100*ps['anchor_coverage']:.1f}%. Teacher gap 유효 분위수={ps['teacher_gap_valid']}. 첫 batch에서 B0 Crop/Global bit equality, Teacher detach, initial weight 불변성, patch embedding 및12block 각각의 nonzero gradient를 확인했다.",
                  '`ranking_unit_verification.json`은 독립 NumPy reference와 ranks16~32, self/overlap 제외, IoU=0.5 포함, 빈 pair의 differentiable zero 및 Teacher gradient 차단을 검증한다. `lambda_rank.json`은 실제 GPU preflight다. 지원이0 또는 극소이면 threshold를 완화하지 않고 중단한다.']
    log=run/'N1/train_log.jsonl'
    if log.exists():
        training=[json.loads(s) for s in log.read_text().splitlines()]
        chosen=[r for r in training if r['step'] in [50,1000,3000,6000,9000,12000]]
        if training and training[-1] not in chosen:chosen.append(training[-1])
        lines+=['','## 학습 추이','','| Step | Total | Crop | Global | Rank | Valid pair ratio | Anchor coverage |','|---|---:|---:|---:|---:|---:|---:|']
        for r in chosen:
            lines += [f"| {r['step']} | {r['loss']:.6f} | {r['crop']:.6f} | {r['glob']:.6f} | {r['rank']:.6f} | {r['rank_pair_ratio']:.6f} | {r['rank_anchor_coverage']:.6f} |"]
        lines+=['','전체 raw trace는 `N1/train_log.jsonl`, `N1/neighbor_diagnostics.jsonl`, `N1/per_loss_gradients.jsonl`, `N1/block_gradients.jsonl`에 보존한다.']
    exported=run/'N1/export/export_verification.json'
    if exported.exists():
        v=read(exported)
        lines+=['','## 표준 Encoder Export','',f"Strict loading={v['strict_loading']}; parameter count={v['parameter_count']}; patch shape={v['patch_shape']}; forward bit equality={v['bit_equal_forward']}; additional modules={v['additional_modules']}. Final12k EMA를 export하고 같은표준 encoder를 R2에 사용한다. 원본 RGB는 이미지당 한 번만 인코딩한다. 실제 PACO canvas 계수는 `N1/preds/inference_verification.json`에 저장한다."]
    boot=run/'paired_bootstrap.json'
    if boot.exists():
        import numpy as np
        b=read(boot);intervals={}
        lines+=['','## Val paired-image bootstrap (1000회)','','선택된 Reader epoch에 조건부인95% CI이며 seed 변동을 대신하지 않는다. Test Gate에는 사용하지 않는다.','','| N1−B0 | Difference (pp) | 95% CI |','|---|---:|---:|']
        for label,key in [('Object','obj'),('Part','part'),('Object Pattern','obj_pattern_marking'),('Part Pattern','part_pattern_marking')]:
            delta=np.array(b['boot']['N1'][key])-np.array(b['boot']['B0'][key])
            lo,hi=np.percentile(delta,[2.5,97.5]);point=b['points']['N1'][key]-b['points']['B0'][key]
            intervals[label]=dict(delta=point,ci=[float(lo),float(hi)])
            lines += [f'| {label} | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] |']
        delta=.5*(np.array(b['boot']['N1']['obj'])+np.array(b['boot']['N1']['part'])-np.array(b['boot']['B0']['obj'])-np.array(b['boot']['B0']['part']))
        lo,hi=np.percentile(delta,[2.5,97.5]);point=result['delta_val_pp']['overall']
        lines += [f'| Overall | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] |']
        write_json(run/'uncertainty.json',intervals)
    write_json(run/'results.json',result)
    (run/'README.md').write_text('\n'.join(lines)+'\n')
    dest=ROOT/'experiments/fgtrain_n1';dest.mkdir(exist_ok=True)
    for name in ['README.md','results.json']:(dest/name).write_text((run/name).read_text())
    return result


class N1Pipeline(Pipeline):
    def report(self):
        with self.report_lock:report(self.run)

    def execute(self,base):
        self.stage('preflight_N1','src.fgtrain.ranking',['--run',self.run,'--base-run',base],'0')
        self.stage('train_N1','src.fgtrain.train',['--condition','N1','--run',base,'--cache',Path(base)/'teacher',
                   '--out',self.run/'N1','--rank-config',self.run/'lambda_rank.json','--steps',12000,'--workers',4],'0')
        self.report()
        self.evaluate()

    def evaluate(self):
        cfg=read(self.run/'N1/config.json');audit=read(self.run/'baseline_audit.json')
        assert cfg['steps']==12000 and cfg['seed']==0 and cfg['initial_sha']==audit['initial_sha']
        assert cfg['roi_rule']==audit['student_config']['roi_rule']
        for key in ['encoder_lr','llrd','wd','betas','warmup','schedule','ema_m0','global_images_per_step','rois_per_image','lambda_global']:
            assert cfg[key]==audit['student_config'][key], ('N1/B0 training mismatch',key)
        exported=read(self.run/'N1/done.json')['export']
        assert exported['strict_loading'] and exported['parameter_count']==85641216
        assert exported['additional_modules']==[] and exported['bit_equal_forward']
        assert exported['patch_shape'][1:]==[576,768]
        self.probe('N1',0)
        for f in self.cpu_futures:f.result()
        self.cpu.shutdown(wait=True)
        new=read(self.run/'N1/probe/config.json');old=audit['reader_config']
        for key in ['world','S','P','layers','n_train_rows','n_train_images','n_val_rows','n_val_images','steps_per_epoch','total_steps','n_params']:
            assert new[key]==old[key], ('N1/B0 Reader protocol mismatch',key)
        epoch=read(self.run/'N1/selection.json')['epoch']
        value=metrics(self.run/'N1/official'/f'ep{epoch}/official_results.json')['overall']
        threshold=read(self.run/'evaluation_protocol.json')['gate']['threshold']
        gate=dict(passed=value>threshold,overall=value,baseline=threshold,operator='>',bootstrap_required=False,
                  encoder_sha=sha(self.run/'N1/export/encoder.pt'),reader_sha=sha(self.run/'N1/probe/r2'/f'ep{epoch}.pt'),reader_epoch=epoch)
        path=self.run/'test_gate.json'
        if path.exists():assert read(path)==gate,'N1 Test gate and model identity cannot change'
        else:write_json(path,gate)
        self.report()
        if gate['passed']:
            target=self.run/'test';target.mkdir(exist_ok=True)
            test=Pipeline(target)
            test.run_final_test(self.run,methods=('N1',),report_module=None,once=True)
        # Bootstrap is a diagnostic, never a condition for Test admission.
        try:self.stage('paired_bootstrap_N1','src.fgtrain.bootstrap',['--run',self.run,'--conds','B0,N1'],official=True)
        except RuntimeError as e:write_json(self.run/'bootstrap_unavailable.json',dict(reason=str(e),gate_unchanged=True))
        self.report()
        write_json(self.run/'done.json',dict(status='done',test_ran=gate['passed'],ADE20K_ran=False,utc=time.strftime('%FT%TZ',time.gmtime())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(N1_RUN));p.add_argument('--base-run',default=str(RUN));p.add_argument('--evaluate',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args();prepare(a.run,a.base_run)
    if a.evaluate or a.execute:
        from src.fgtrain.ranking import CONFIG
        run=Path(a.run)
        fixed=dict(ranking=CONFIG,base_run=a.base_run,steps=12000,seed=0,initial_sha=read(run/'baseline_audit.json')['initial_sha'],
                   initialization='Original pretrained; independent from B0/B1',attribute_labels_used=False,extra_inference_modules=False)
        protocol=run/'training_protocol.json'
        if protocol.exists():assert read(protocol)==fixed,'N1 training design is immutable'
        else:write_json(protocol,fixed)
        assert not (Path(a.base_run)/'active.json').exists() or not any(j.get('gpus') for j in read(Path(a.base_run)/'active.json').values()),'Existing foreground ADE must finish first'
        write_json(run/'launch.json',dict(pid=os.getpid(),detached=True,command=sys.argv))
        write_json(run/'background_policy.json',dict(enabled=True,allow_background_with_stages=[],include_final_test=True,
                   ownership_prefix='fgtrain per-GPU priority: fgtrain_n1_s0:',
                   rule='Give GPU0 to N1 GPU stages; keep GPU1 and CPU-only intervals available to existing A/B'))
        (run/'test').mkdir(exist_ok=True)
        pipe=N1Pipeline(a.run)
        with (run/'resource_manager.log').open('a') as log:
            manager=subprocess.Popen([sys.executable,'-u','-m','src.fgtrain.resources','--run',a.run],cwd=ROOT,
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(run/'resource_manager_process.json',dict(pid=manager.pid))
        def stop(sig,frame):pipe.terminate_children();raise SystemExit(128+sig)
        for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
        try:
            if a.execute:pipe.execute(a.base_run)
            else:pipe.evaluate()
        except BaseException:
            pipe.terminate_children();write_json(Path(a.run)/'failed.json',dict(status='failed',traceback=traceback.format_exc()));raise
    else:print(json.dumps(dict(status='B0 and fixed N1 evaluation verified; run --execute for training',run=a.run)),flush=True)


if __name__=='__main__':main()
