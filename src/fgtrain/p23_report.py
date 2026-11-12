"""Only measured P2/P3 results; never reads PACO Test results."""
import json
from pathlib import Path

from src.fgtrain.common import ROOT, write_json
from src.fgtrain.n1 import metrics
from src.fgtrain.p1 import read


def report(root):
    root=Path(root);protocol=read(root/'training_protocol.json');base=Path(protocol['base_run']);p1=Path(protocol['p1_run'])
    results=dict(PACO_Test_ran=False,seed=0,conditions=['P2','P3'],status='running')
    lines=['# P2 / P3 — Uniform vs Region-Selective Patch Preservation','',
        'P2와 P3는 Original pretrained `vit_base_patch16_dinov3.lvd1689m`에서 각각 독립적으로 시작한다. Student는 RGB384 letterbox → 기존12block → 최종 LN576×768이다. Native RGB Crop768 target 및 Full384 global target은 기존 B0 cache/ROI stream 그대로다. Patch preservation teacher는 동일 Full384 tensor를 받는 frozen/eval Original이며 학습 내내 업데이트하지 않는다.',
        '', '`L = L_crop + 0.5 L_global + lambda_preserve L_patch`',
        '', 'Crop/Global은 기존 `src.fgtrain.train.terms`를 수정 없이 호출한다. `d_p=1−cos(normalize(Fs_p),normalize(F_original_p))`는 FP32로 계산하고 Teacher를 detach한다. CLS/register는 제외한다. Fractional letterbox coverage `v_p`에 대해 이미지별 `sum(v*w*d)/sum(v)`를 계산하고 유효 이미지별로 평균한다. Zero-valid 이미지는 제외하며 전체가 zero-valid면 Student graph에 연결된0을 반환한다.',
        '', 'P2는 `w=1`. P3는 실제 Crop loss의 GT+Random+fallback ROI **기하학적 Union**에 대한 유효 patch 면적 비율 `m`을 구하고, `raw=1−0.75*m`, `mean_raw=sum(v*raw)/sum(v)`, `w=raw/mean_raw`를 적용한다. 정규화 후에도 분모는 `sum(v)`다. Float64 canvas rectangle inclusion/exclusion은 중복·겹침을 한 번만 세며 독립 sweep-line 참조와 대조했다.',
        '', '추론에는 표준 DINOv3 encoder만 남는다. Teacher/ROI mask/loss는 encoder module tree에 추가되지 않는다. RGB는 이미지당 한 번만 인코딩하고 ROI RGB 재인코딩은 없다. P2는 물리GPU0, P3는 물리GPU1, 각각 내부cuda:0으로 학습한다. 기존 A/B는 안전한 checkpoint pause 후 배정하고 Reader/CPU 구간에는 backfill한다.',
        '', 'B0와 같은 Seed0,12k,AdamW,LR5e-5,LLRD0.75,WD0.05,betas0.9/0.999,600warmup+cosine,clip1,동일cosineEMA를 사용한다. 기본16image×4ROI stream에서 같은 제외 목록을 적용하므로 실제 유효 batch 크기도 B0와 같다. Final12k EMA만 평가하며 calibration6k EMA로 초기화하지 않는다.',
        '', '## 검증 및 사전 고정','',
        '`training_protocol.json`, `calibration_source_audit.json`, `source_manifest.json`, `data_manifest.json`, `protected_inputs.json`에 설정·소스·체크포인트·데이터 해시를 기록했다. B0에 별도 training_protocol.json은 없어서 실제 B0/config.json과 초기화/제외/cache 기록을 검증한다. P1 replay는 B0의6000step까지 동일12000 LR/EMA schedule로 실행된 EMA이며 BF16 replay의 미세한 수치 변동은 기존 calibration_replay_match.json에 남아 있다.']
    if (root/'lambda_preserve.json').exists():
        c=read(root/'lambda_preserve.json');results['calibration']=c
        lines+=['',f"P2에서 한 번 보정한 공통 lambda_preserve={c['lambda_preserve']:.12f}. Crop norm={c['unweighted_gradient_trace']['crop']['encoder']:.9f}, unweighted patch norm={c['unweighted_gradient_trace']['preserve']['encoder']:.9f}, 가중 ratio={c['measured_ratio']:.6f}. P3 별도 보정과 λ 탐색은 하지 않는다.",f"Calibration EMA SHA: `{c['calibration_state_sha256']}`. 동일 stream0 이미지 IDs: `{c['image_ids']}`."]
    if (root/'roi_union_preflight.json').exists():
        p=read(root/'roi_union_preflight.json');results['roi_union_preflight']={k:v for k,v in p.items() if k!='images'}
        lines+=['',f"ROI Union 사전 점검 {len(p['images'])}images, coverage quantiles={p['union_fraction_quantiles']}; P2/P3 평균 가중치 절대차<0.01 비율={p['nearly_uniform_fraction']:.6f}. Union이 거의 전체이면 P3가 P2로 수렴하는 한계가 있다.0.75 계수는 변경하지 않는다.",
                'Padding/inside/outside/boundary/정규화 가중치 및 차이는 `roi_union_preflight.json`; 시각화는 `preflight/union_*.png`.']
    lines+=['','## PACO Official Val + ADE20K','',
        'PACO: Official Val2410, GT Box+GT Category Oracle attribute AP, 기존 bilinear ROIAlign24, R2 seed0/init7,10epochs. Object/Part AP 평균 최대 epoch를 선택하고 동률은 이른 epoch를 사용한다. B0/Frozen/B1/P1도 같은 프로토콜의 완료 결과만 재사용한다. Test는 실행하거나 읽지 않는다.',
        '', 'ADE20K: train20210/Official Val2000,150classes, frozen final-LN encoder + Linear768→150, seed0,40k,world2/globalbatch16. 기존 `src.g1eval.ade`를 수정하지 않고 `src.fgtrain.ade`의 encoder loader만 사용한다. P2/P3 각각 새로운 동일 초기화 head를 동일 augmented batch로 학습한다. 최종iteration만 평가하고 Val head selection은 없다.',
        '', 'ADE 학습 전처리는 기존 short-side384×U(0.5,2),384crop,pad0,label255,hflip0.5,ImageNet normalization이다. 평가는 기존 short-side384 가변 aspect grid,16배수 padding,유효 logits crop 후 원래 GT 좌표로 resize한다. Class0은 ignore255,1..150→0..149. 동일 프로토콜 유지를 위해 평가는 square letterbox로 변경하지 않는다.',
        '', '| Method | Overall AP | Object AP | Part AP | Object Pattern | Part Pattern | Obj Color | Part Color | Obj Material | Part Material | ADE mIoU |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    ade=read(root/'ade/done.json')['results'] if (root/'ade/done.json').exists() else {}
    keys=['overall','object','part','obj_pattern','part_pattern','obj_color','part_color','obj_material','part_material','miou']
    rows={}
    for name in ['Frozen','B0','B1','P1','P2','P3']:
        folder=(p1/name if name=='P1' else base/name) if name in ['Frozen','B0','B1','P1'] else root/name
        r={}
        if (folder/'selection.json').exists():
            selected=read(folder/'selection.json');e=selected['epoch']
            official=folder/'official'/f'ep{e}/official_results.json'
            check=read(official);assert check['split']=='val' and check['n_eval_images']==2410 and check['alignment_mismatch']==0
            r=dict(metrics(official),reader_epoch=e)
        if name in ['Frozen','B0','B1']:r['miou']=read(base/'results.json')['metrics'][name]['miou']
        if name=='P1':r['miou']=read(p1/'ade/done.json')['results']['P1']['mIoU']
        if name in ade:r['miou']=ade[name]['mIoU']
        rows[name]=r
        lines+=['| '+name+' | '+' | '.join(f'{r[k]:.6f}' if k in r else '진행 중' for k in keys)+' |']
    results['metrics']=rows
    lines+=['','## P2/P3 비교 및 진단','']
    results['differences_pp']={}
    for name in ['P2','P3']:
        results['differences_pp'][name]={}
        for ref in ['Frozen','B0','P1']:
            delta={k:rows[name][k]-rows[ref][k] for k in keys if k in rows[name] and k in rows[ref]}
            results['differences_pp'][name][ref]=delta
            if delta:lines+=[f"{name}−{ref} (pp): `{json.dumps(delta)}`",'']
        diagnostic=root/name/'gradient_diagnostics.jsonl'
        if diagnostic.exists():
            ds=[json.loads(s) for s in diagnostic.read_text().splitlines()]
            results[name+'_gradient_diagnostics']=ds
            lines += [f'### {name} gradient','','| Step | Crop norm | Weighted preservation | Ratio | Gradient cosine | Patch drift | ROI inside drift | ROI outside drift |','|---|---:|---:|---:|---:|---:|---:|---:|']
            for q in ds:
                vals=[q['step'],q['crop']['encoder'],q['weighted_preservation_norm'],q['preservation_crop_ratio'],q['crop_preservation_cosine'],q['patch_distance'],q['roi_inside_distance'],q['roi_outside_distance']]
                lines+=['| '+' | '.join('정의 없음' if x is None else f'{x:.8g}' for x in vals)+' |']
            lines+=['','Patch embedding과 12개 Block의 raw/total norm, Loss, 이미지 ID는 각 `gradient_diagnostics.jsonl`에 기록한다. 초기 Student==Teacher이면 Preservation Gradient가 거의0이므로 이 시점의 Gradient Cosine은 해석하지 않는다. `train_log.jsonl`에 Loss50step 평균과 Global Similarity를 저장한다.','']
        export=root/name/'export/export_verification.json'
        if export.exists():
            e=read(export);results[name+'_export']=e
            lines += [f"{name}: strict={e['strict_loading']}, params={e['parameter_count']}, shape={e['patch_shape']}, forward equality={e['bit_equal_forward']}, same module tree={e['identical_module_tree']}, extra modules={e['additional_modules']}. EMA SHA `{e['state_sha256']}`.",'']
    if rows['P2'] and rows['P3']:
        delta={k:rows['P3'][k]-rows['P2'][k] for k in keys if k in rows['P3'] and k in rows['P2']}
        results['P3_minus_P2_pp']=delta
        if delta:lines += [f"P3−P2 (pp): `{json.dumps(delta)}`",'']
    if (root/'patch_drift.json').exists():
        r=read(root/'patch_drift.json');results['patch_drift']=r['mean']
        lines += ['동일한 excluded COCO dev126 이미지와 Full384 Tensor에서 Original/B0/P1/P2/P3를 비교한다. ROI 내부/외부는 정확한 Union의 fractional m/1−m으로 면적 가중하고, 면적0인 그룹은 None으로 기록한다. 이미지별 분포는 `patch_drift.json`에 저장한다.','',f"Measured drift means: `{json.dumps(r['mean'])}`",'']
    complete=all('miou' in rows[n] and 'overall' in rows[n] for n in ['P2','P3'])
    if complete:
        results['status']='done'
        p2,p3=rows['P2'],rows['P3']
        lines += ['## 결과 해석','',f"P2−P1 ADE={p2['miou']-rows['P1']['miou']:+.6f} pp.P3−P2 ADE={p3['miou']-p2['miou']:+.6f} pp, Object Pattern={p3['obj_pattern']-p2['obj_pattern']:+.6f} pp, Part Pattern={p3['part_pattern']-p2['part_pattern']:+.6f} pp.",'']
        if p3['miou']>=p2['miou'] and p3['obj_pattern']>=p2['obj_pattern'] and p3['part_pattern']>=p2['part_pattern']:
            lines+=['Seed0 점추정에서 P3가 P2의 ADE와 두 Pattern을 유지/개선했다. Region-Selective Preservation 가설을 지지하는 제한적 결과이며, 단일 Seed로 안정적인 개선을 확정하지 않는다.','']
        else:lines+=['ADE와 Object/Part Pattern의 변화 방향을 개별적으로 평가한다. P3가 P2의 두 성능을 동시에 개선했다고 일괄 주장할 수 없다. 단일 Seed의 작은 차이에 통계적 의미를 부여하지 않는다.','']
    lines+=['Feature/Gram Drift 감소만으로 ADE 보존의 성공을 판단하지 않는다. PACO와 ADE 실측이 모두 확보될 때까지 Method의 성공은 미확정이다. PACO Test, P4, 추가 Loss, λ/Mask Sweep, 새로운 Architecture를 자동 실행하지 않는다.','']
    if (root/'implementation_failures').exists():
        lines+=['## 구현 중 수정','',
                '최초 인계 시 Python3.10의 os.pidfd_open 미지원으로 실행이 중단됐다. 기존 프로젝트의 libc pidfd 호환 함수를 재사용하도록 수정했다. 이 시도에서는 Calibration/학습 업데이트가 없었으며 예약 해제 후 A/B가 재개됐다. 수정 전 Source Manifest와 로그는 implementation_failures에 보존했다. Loss, λ 설계, 데이터 Stream은 변경하지 않았다.','']
    # Keep this experiment's reports separate from every previous experiment.
    write_json(root/'results.json',results);(root/'README.md').write_text('\n'.join(lines))
    dest=ROOT/'experiments/fgtrain_p23';dest.mkdir(exist_ok=True)
    for name in ['README.md','results.json']:(dest/name).write_text((root/name).read_text())
    tsv='Method\t'+'\t'.join(keys)+'\n'+'\n'.join(n+'\t'+'\t'.join(str(rows[n].get(k,'')) for k in keys) for n in rows)+'\n'
    (root/'combined_results.tsv').write_text(tsv);(dest/'combined_results.tsv').write_text(tsv)
    for condition,path in protocol['run_directories'].items():
        run=Path(path)
        write_json(run/'results.json',dict(condition=condition,metrics=rows.get(condition,{}),PACO_Test_ran=False,shared_report=str(root/'README.md')))
        (run/'README.md').write_text(f'# {condition}\n\nIndependent Original initialization, physical GPU {0 if condition=="P2" else 1}.\n\nFull protocol, calibration and measured comparison: [{root.name}](../{root.name}/README.md).\n')
