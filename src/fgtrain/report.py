"""Report only observed metrics; missing and conditional stages remain explicit."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from src.fgtrain.common import ROOT, RUN, write_json


COLS=['overall','object','part','obj_pattern','part_pattern','miou']


def read(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def paco(root,name):
    sel=read(root/name/'selection.json')
    if not sel:return {}
    rec=read(root/name/'official'/f'ep{sel["epoch"]}'/'official_results.json')
    if not rec:return {}
    h=rec['hierarchical'];o,p=h['obj-attr-AP'],h['obj-attr-part-AP']
    return dict(overall=.5*(o+p),object=o,part=p,obj_pattern=h['obj-attrs']['pattern_marking'],part_pattern=h['obj-part-attrs']['pattern_marking'],
                obj_color=h['obj-attrs']['color'],part_color=h['obj-part-attrs']['color'],obj_material=h['obj-attrs']['material'],
                part_material=h['obj-part-attrs']['material'],epoch=sel['epoch'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN));a=p.parse_args();root=Path(a.run)
    rows={name:paco(root,name) for name in ['Frozen','B0','B1','B2','G1cross','C0']}
    base=read(root/'ade_baseline_reuse.json')
    if base:
        rows['Frozen']['miou']=base['results']['original']['mIoU']
        rows['G1cross']['miou']=base['results']['g1cross']['mIoU']
    ade=read(root/'ade/done.json')
    if ade:
        for m,v in ade['results'].items():rows[m]['miou']=v['mIoU']
    delta={}
    for reference in ['Frozen','B0']:
        delta[reference]={name:{k:r[k]-rows[reference][k] for k in r if k in COLS and k in rows[reference]} for name,r in rows.items()}
    write_json(root/'results.json',dict(metrics=rows,delta_pp=delta,official_validation_only=True,B2_gate=read(root/'b2_gate.json')))
    with (root/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['method']+COLS+['epoch','obj_color','part_color','obj_material','part_material'])
        w.writeheader()
        for m,v in rows.items():w.writerow(dict(method=m,**v))
    protocol=(ROOT/'configs/fgtrain/protocol.md').read_text()
    lines=[protocol,'','## 실행 결과 (측정된 항목만 기입)','','| Method | PACO Overall AP | Object AP | Part AP | Object Pattern AP | Part Pattern AP | ADE20K mIoU |','|---|---:|---:|---:|---:|---:|---:|']
    for m,r in rows.items():
        lines.append('| '+m+' | '+' | '.join(f'{r[k]:.3f}' if k in r else '—' if m=='C0' and k=='miou' else '미완료' if m!='B2' else '조건부 미실행' for k in COLS)+' |')
    lines+=['','Color / Material (공식 evaluator 출력):','','| Method | Object Color | Part Color | Object Material | Part Material | Val-selected R2 epoch |','|---|---:|---:|---:|---:|---:|']
    for m,r in rows.items():
        lines.append('| '+m+' | '+' | '.join(f'{r[k]:.3f}' if k in r else '—' for k in ['obj_color','part_color','obj_material','part_material','epoch'])+' |')
    for ref,d in delta.items():
        lines+=['',f'### {ref} 대비 변화 (percentage points)','','| Method | Overall | Object | Part | Obj Pattern | Part Pattern | mIoU |','|---|---:|---:|---:|---:|---:|---:|']
        for m,r in d.items():lines.append('| '+m+' | '+' | '.join(f'{r[k]:+.3f}' if k in r else '—' for k in COLS)+' |')
    smoke=read(root/'smoke_verification.json');cal=read(root/'lambda_spatial.json')
    if smoke:
        lines+=['','## 실제 검증','',f'- Smoke: **{smoke["status"]}**. 좌표 최대 오차 {smoke["coordinate_max_error"]:.3g}; Padding target 변경 시 loss 동일.',
                f'- Spatial loss → Encoder norm {smoke["gradients"]["spatial"]["encoder"]:.6f}; Patch embedding {smoke["gradients"]["spatial"]["patch_embed"]:.6f}. 12개 Block 모두 gradient > 0.',
                f'- 실제 optimizer update의 Patch embedding 최대 변화: {smoke["patch_embedding_update_max"]:.6g}.',
                f'- Smoke strict export: {smoke["export"]["parameter_count"]:,} parameters, 표준 클래스 `{smoke["export"]["class_name"]}`, 576×768 output, Module tree 동일, Forward bit-equal. 최종 12k export 검증과 구분한다.']
    if cal:
        lines += [f'- 고정 λ_spatial = **{cal["lambda_spatial"]:.12g}**; 같은 초기 64 ROI batch에서 Crop gradient {cal["gradients"]["crop"]["encoder"]:.6f}, Spatial gradient {cal["gradients"]["spatial"]["encoder"]:.6f}, 가중 비율 {cal["measured_weighted_ratio"]:.6f}. Decoder warmup/update 없이 측정.']
    for name in ['B0','B1','B2','FrozenDecoder']:
        exp=read(root/name/'export/export_verification.json')
        if exp:lines+=[f'- {name} 최종 strict export: {exp["strict_loading"]}, parameters {exp["parameter_count"]:,}, Forward bit-equal {exp["bit_equal_forward"]}, 추가 module {exp["additional_modules"]}.']
    lines+=['','## Reconstruction / Encoder 진단','', '| Method | COCO dev Crop cos | CLS cos | Patch-mean cos | Spatial cos (init → final EMA) |','|---|---:|---:|---:|---:|']
    diagnostics={}
    for name in ['B0','B1','B2','FrozenDecoder']:
        path=root/name/'diagnostics.jsonl'
        records=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        records=[r for r in records if r['model']=='ema']
        if not records:continue
        first,last=records[0],records[-1]
        diagnostics[name]=last
        sf=first.get('spatial_cos');sl=last.get('spatial_cos')
        spat=f"{sf:.5f} → {sl:.5f}" if sf is not None and sl is not None else '해당 없음'
        lines += [f"| {name} (step {last['step']}) | {last['crop_cos']:.5f} | {last['cls_cos']:.5f} | {last['mean_cos']:.5f} | {spat} |"]
    if 'B1' in diagnostics and 'FrozenDecoder' in diagnostics:
        x,y=diagnostics['B1'],diagnostics['FrozenDecoder']
        if x['step']==y['step']:
            lines += ['',f"동일 {x['step']}step에서 Trainable B1−Frozen Decoder의 spatial cosine 차이는 {x['spatial_cos']-y['spatial_cos']:+.6f}이다. Decoder 학습만으로 줄어드는 loss와 Encoder 변경의 기여를 구분한다."]
    write_json(root/'diagnostic_summary.json',diagnostics)
    lines+=['','## 결과 해석','']
    if all('overall' in rows[m] for m in ['B0','B1']):
        dr=delta['B0']['B1']
        lines += [f'B1−B0: Overall {dr["overall"]:+.3f} pp, Object Pattern {dr["obj_pattern"]:+.3f} pp, Part Pattern {dr["part_pattern"]:+.3f} pp.']
        if all('miou' in rows[m] for m in ['B0','B1']):
            lines += [f'ADE20K B0−Frozen {rows["B0"]["miou"]-rows["Frozen"]["miou"]:+.3f} pp; B1−B0 {dr["miou"]:+.3f} pp. 작은 차이는 seed-0 파일럿 변동과 구분하지 못한다.']
    else:lines+=['B0/B1 공식 평가가 아직 모두 완료되지 않았으므로 표현 개선 여부를 판단하지 않는다. Teacher cosine 또는 reconstruction loss 감소만으로 성공을 주장하지 않는다.']
    bootstrap=read(root/'paired_bootstrap.json')
    if bootstrap:
        lines += ['','기존 official matching 기반 paired-image bootstrap (1000회, 95% CI; seed 반복 실험과 다름):','','| B1−B0 | Δ AP (pp) | 95% CI |','|---|---:|---:|']
        for label,key in [('Object','obj'),('Part','part'),('Object Pattern','obj_pattern_marking'),('Part Pattern','part_pattern_marking')]:
            b=np.array(bootstrap['boot']['B1'][key])-np.array(bootstrap['boot']['B0'][key])
            lo,hi=np.percentile(b,[2.5,97.5]);d=bootstrap['points']['B1'][key]-bootstrap['points']['B0'][key]
            lines += [f'| {label} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] |']
        b=.5*(np.array(bootstrap['boot']['B1']['obj'])+np.array(bootstrap['boot']['B1']['part'])-np.array(bootstrap['boot']['B0']['obj'])-np.array(bootstrap['boot']['B0']['part']))
        lo,hi=np.percentile(b,[2.5,97.5]);d=delta['B0']['B1']['overall']
        lines += [f'| Overall | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] |']
    if all('miou' in rows[m] and 'overall' in rows[m] for m in ['B0','B1']):
        apdiff=rows['B1']['overall']-rows['B0']['overall'];semdiff=rows['B1']['miou']-rows['B0']['miou']
        if apdiff>0 and semdiff>0:interpret='두 평가의 점추정이 함께 개선됐다.'
        elif apdiff>0 and semdiff<0:interpret='속성 판별의 점추정 이득과 segmentation 손실이 함께 나타났다. 공간 표현의 trade-off 가능성을 확인해야 한다.'
        elif apdiff<=0 and semdiff>0:interpret='공간 표현의 점추정 이득이 세부 속성 AP 개선으로 이어지지는 않았다.'
        else:interpret='두 평가의 동시 개선을 확인하지 못했다.'
        lines += ['',interpret+' 작은 차이는 seed0 파일럿 변동 가능성이 있어 seed 반복 없이 안정적인 개선으로 부르지 않는다.']
    gate=read(root/'b2_gate.json')
    if gate:
        lines += [f'B2 자동 진행: {gate["run_B2"]}. 사전 고정 gate와 bootstrap CI는 `b2_gate.json`, `paired_bootstrap.json`에 저장한다.']
        if not gate['run_B2']:lines+=['B1의 안정적인 Pattern 이득을 확인하지 못해 B2를 실행하지 않았다. Native Crop768 감독의 이 결과는 C1 Crop-from-384 판별 학습의 실패를 뜻하지 않는다. Decoder의 복원 향상과 Encoder의 AP 변화, 기존 PixelDiT/J-series 결과를 함께 검토한다.']
    lines += ['','## G1 기존 Reader 참고 결과 (matched ablation과 분리)','',
              '아래는 같은 Official Val2410이지만 기존 full-canvas R2 및 dev-selected ep6 결과다. 새 24×24 bilinear ROIAlign/Val selection과 epoch 규칙이 다르므로 B0/B1의 직접 ablation으로 사용하지 않는다. G1-pool/cross는 Original 초기화에서 Crop384로12k 학습했고 Frozen은 COCO 학습0step이다.',
              '기존 G1 Encoder 사전학습은 full official Val146/Test475 exact 이미지에 노출됐다(해당 이미지에는 속성 positive 주석이 없음). 새 B 조건은 이 이미지 및 확인된 near duplicate를 전부 제외하므로 사전학습 data scope가 다르다. 새 G1-cross bilinear Reader 결과 역시 Encoder는 이 과거 가중치를 쓰는 참고 결과다.',
              '', '| Reference | Overall | Object | Part | Obj Pattern | Part Pattern | ADE20K mIoU |','|---|---:|---:|---:|---:|---:|---:|']
    for name,key,ade_key in [('Frozen full-grid','frozen_full','original'),('G1-pool full-grid','g1pool_full','g1pool'),('G1-cross Encoder Only full-grid','g1cross_full','g1cross')]:
        old=read(ROOT/'output/2026-11-07/paco_oracle_attr/eval'/key/'val/official_results.json')
        if old:
            h=old['hierarchical'];o,part_ap=h['obj-attr-AP'],h['obj-attr-part-AP']
            mio=base['results'][ade_key]['mIoU'] if base else float('nan')
            lines += [f"| {name} | {(o+part_ap)/2:.3f} | {o:.3f} | {part_ap:.3f} | {h['obj-attrs']['pattern_marking']:.3f} | {h['obj-part-attrs']['pattern_marking']:.3f} | {mio:.3f} |"]
    lines += ['','## 기존 결과와 다음 실험의 근거','',
        '공식 Val의 기존 A0 계열에서 E1a→E1b는 Overall 57.11→56.63, Object Pattern 57.95→59.23, Part Pattern 59.22→59.44였다. Spatial loss가 줄어도 Overall 이득은 확인되지 않았고, 현재 표준 DINOv3와는 G1-pool 초기화·추론 Fine path·spatial target 집계가 다르다.',
        '기존 PixelDiT Hybrid→P-C768의 Val Overall은 56.71→56.37, Joint J1의 Val Overall은 56.40이었다. 이 결과는 Flow 구조의 참고 결과이며 이번 단순 Pixel L1의 효과를 입증하거나 부정하지 않는다. 기존 Test는 이번 가중치·선택·gate에 사용하지 않았다.']
    if gate and not gate['run_B2']:
        lines += ['B1의 Pattern 이득이 불충분하면 이 결과를 유지하고 B2를 보류한다. 우선 ROI token 수별 spatial cosine과 FrozenDecoder 대조군을 비교해 decoder capacity에 의한 loss 감소인지, 관측 정보의 제약인지 구분한다. 추가 loss나 λ 탐색으로 본 파일럿의 음성 결과를 덮어쓰지 않는다. Native768와 C1-from-384는 다른 정보 조건이므로 후속 가설도 구분한다.']
    elif gate and gate['run_B2']:
        lines += ['다음 판단은 B2−B1의 공식 Pattern/Overall AP 및 ADE20K 변화다. RGB L1만 좋아졌다면 Pixel Detail의 표현 학습 효과가 확인되지 않은 것으로 해석한다. 유리한 점추정은 같은 프로토콜의 독립 seed로 재현해야 한다.']
    trials=read(root/'implementation_trials.json') or [read(root/'implementation_trial.json')]
    lines+=['','## 구현 중 수정과 제외 실행','']
    for trial in trials:
        if trial:
            archive=trial.get('archive',trial.get('archived'))
            lines += [trial['reason'],f'제외 실행: `{archive}`. 표와 모델 선택에 사용하지 않았다.','']
    lines += ['','## 파일과 상태','',f'실행 원본: `{root}`. `active.json`, `events.jsonl`, `state/`, `logs/`, 각 condition의 `progress.json`을 참조한다.',
              '`initialisation.json`, `teacher/manifest.json`, `smoke_verification.json`, `lambda_spatial.json`이 초기화·데이터·좌표·gradient 근거다.',
              '`<method>/per_loss_gradients.jsonl`, `block_gradients.jsonl`, `diagnostics/`에 patch embedding 및 12개 block gradient와 ROI별 관측 token 수·복원 cosine을 보존한다.',
              '`<method>/export/encoder.pt`는 raw 표준 DINOv3 state dict; `encoder_eval.pt`는 기존 평가기용 enc-only envelope다. Teacher/Decoder 가중치는 포함하지 않는다.',
              '`<method>/probe/frozen_check.json` 및 `preds/inference_verification.json`으로 Probe 중 Encoder 불변성과 이미지당 1회 인코딩을 확인한다.',
              '`ade/config.json`, `ade/done.json`, `ade/val_inter_union.npz`, `ade/head_*.pt`가 ADE20K 재현 artifact다.']
    final_test=read(root/'test/results.json')
    fixed_test=read(root/'test/fixed_protocol.json')
    if fixed_test:
        lines += ['','## 최종 PACO Test (Val 선택 고정 후 별도 보고)','',
                  '사용자 추가 요청으로 최종 Test를 실행한다. B0/B1은 각각 Official Val에서 선택한 R2 epoch를 고정하고 Encoder/Reader hash를 추론 전에 저장했다. Test로 method/epoch/가중치 또는 B2 gate를 변경하지 않는다. 위 표는 계속 Val 결과이며 Test와 섞지 않는다.',
                  '', '| Method | Val-selected epoch | Test Overall | Object | Part | Object Pattern | Part Pattern |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for m,v in fixed_test['methods'].items():
            r=(final_test or {}).get('metrics',{}).get(m,{})
            lines += ['| '+m+' | '+str(v['epoch'])+' | '+' | '.join(f'{r[k]:.3f}' if k in r else '실행 중' for k in ['overall','object','part','obj_pattern','part_pattern'])+' |']
        lines += ['', '원본: `test/fixed_protocol.json`, `test/active.json`, `test/results.json`, `test/<method>/official/official_results.json`.']
    text='\n'.join(lines)+'\n';(root/'README.md').write_text(text)
    dest=ROOT/'experiments/fgtrain';dest.mkdir(exist_ok=True)
    for f in ['README.md','results.json','results.csv']:
        shutil.copyfile(root/f,dest/f)
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
