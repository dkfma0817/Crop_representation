"""Final structural comparison from fixed official predictions and real costs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.siglip_extra.report import flat, TYPES, ATTRS
from src.siglip_extra.train import file_sha
from src.siglip_pm.report import md
from src.siglip_pm.gpu_report import summarize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--base',required=True);a=ap.parse_args();run,base=Path(a.run),Path(a.base)
    load=lambda p:json.loads(p.read_text());out=run/'results'
    models=['P8','P4','Native','Decode','M','B0','C0']
    E={c:load(run/f'eval_test/{c}_r2/test/official_results.json') for c in models}
    table=pd.DataFrame([flat(c,E[c]['hierarchical']) for c in models]);table.to_csv(out/'paco_ap.csv',index=False)
    boot=load(run/'boot_test.json');T=table.set_index('Model')
    keys={'Object AP':'obj','Part AP':'part'}
    for lv in ['Obj','Part']:
        for typ,label in TYPES:keys[f'{lv} {label}']=('obj_' if lv=='Obj' else 'part_')+typ
    pairs=[('P4','P8'),('Native','P8'),('Decode','Native'),('Decode','P8'),('P4','Decode'),('C0','P8'),('C0','P4'),('C0','Native'),('C0','Decode')]
    rows=[]
    for x,y in pairs:
        for metric,key in keys.items():
            delta=np.array(boot['boot'][x+'_r2'][key])-np.array(boot['boot'][y+'_r2'][key]);lo,hi=np.percentile(delta,[2.5,97.5])
            rows.append(dict(comparison=f'{x} − {y}',metric=metric,delta=float(T.loc[x,metric]-T.loc[y,metric]),ci95_low=float(lo),ci95_high=float(hi),B=boot['B']))
    C=pd.DataFrame(rows);C.to_csv(out/'contrasts.csv',index=False)
    gt=load(Path('/home/irteam/datasets/paco/paco_lvis_v1_test.json'));attributes={v['name']:v['id'] for v in gt['attributes']};categories={v['id']:v['name'] for v in gt['categories']};focus=[]
    for attribute in ATTRS:
        for level,part,field in [('Object',False,'obj-attr-AP-per-class'),('Part',True,'obj-attr-part-AP-per-class')]:
            positives=[v for v in gt['annotations'] if attributes[attribute] in v.get('attribute_ids',[]) and (':' in categories[v['category_id']])==part]
            for c in models:
                focus.append(dict(model=c,level=level,attribute=attribute,AP=E[c]['hierarchical'][field].get(attribute,float('nan')),
                                  positive_ROIs=len(positives),positive_categories=len({v['category_id'] for v in positives})))
    F=pd.DataFrame(focus);F.to_csv(out/'focus_attributes.csv',index=False)
    sizes=pd.concat([pd.read_csv(base/'results/roi_size_ap.csv').replace({'model':{'P':'P8'}})]+[pd.read_csv(out/f'roi_size_ap_{c}.csv') for c in ['P4','Native','Decode']],ignore_index=True)
    sizes.to_csv(out/'roi_size_ap.csv',index=False);sd=[]
    for x,y in [('P4','P8'),('Native','P8'),('Decode','Native'),('C0','P4')]:
        for lv in ['Object','Part']:
            for name in sizes.size_bin.unique():
                xx=sizes[(sizes.model==x)&(sizes.level==lv)&(sizes.size_bin==name)].iloc[0];yy=sizes[(sizes.model==y)&(sizes.level==lv)&(sizes.size_bin==name)].iloc[0]
                sd.append(dict(comparison=f'{x} − {y}',level=lv,size_bin=name,Pattern=float(xx.Pattern-yy.Pattern),Material=float(xx.Material-yy.Material)))
    S=pd.DataFrame(sd);S.to_csv(out/'roi_size_contrasts.csv',index=False)
    efficiency=pd.concat([pd.read_csv(out/'efficiency_P4.csv'),pd.read_csv(out/'efficiency_Decode.csv')],ignore_index=True)
    old_cost=pd.read_csv(base/'results/efficiency.csv').query('model=="P"').copy().replace({'model':{'P':'P8'}})
    efficiency=pd.concat([old_cost,efficiency],ignore_index=True);efficiency.to_csv(out/'efficiency.csv',index=False)
    training={c:dict(config=load(run/c/'config.json'),completion=load(run/c/'done.json'),smoke=load(run/c/'smoke.json'),
                    diagnostics=[json.loads(s) for s in (run/c/'diagnostics.jsonl').read_text().splitlines()]) for c in ['P4','Decode']}
    for c in training:
        assert training[c]['completion']['steps']==12000 and training[c]['completion']['teacher_unchanged']
        pd.read_json(run/c/'train_log.jsonl',lines=True).to_csv(out/f'{c}_train.csv',index=False)
    old_reader=load(base/'probe_P/config.json');checks={}
    for c in ['P4','Native','Decode']:
        cfg=load(run/f'probe_{c}/config.json');frozen=load(run/f'probe_{c}/frozen_check.json')
        assert frozen['unchanged'] and not frozen['requires_grad_any']
        for key in ['reader_params','steps_per_epoch','n_train_rows','n_train_images']:
            assert cfg[key]==old_reader[key],(c,key)
        checks[c]=dict(reader_parameters=cfg['reader_params'],frozen=frozen,steps_per_epoch=cfg['steps_per_epoch'])
    summarize(run)
    def contrast(x,y,metric):return C[(C.comparison==f'{x} − {y}')&(C.metric==metric)].iloc[0]
    def statement(x,y,metric):
        v=contrast(x,y,metric);verdict='양의 차이가 관측됨' if v.ci95_low>0 else '음의 차이가 관측됨' if v.ci95_high<0 else 'CI가0을 포함해 차이가 불확실함'
        return f'{metric} {v.delta:+.2f} AP [95% CI {v.ci95_low:.2f}, {v.ci95_high:.2f}]: {verdict}'
    answers={
        'Q1_P4_crop_gap':[statement('P4','P8',k) for k in ['Obj Material','Obj Pattern']],
        'Q2_small_Object':S[(S.comparison=='P4 − P8')&(S.level=='Object')].to_dict('records'),
        'Q3_native_vs_ROIAlign':[statement('Native','P8',k) for k in ['Object AP','Part AP','Obj Material','Obj Pattern']],
        'Q4_decoder_vs_native':[statement('Decode','Native',k) for k in ['Object AP','Part AP','Obj Material','Obj Pattern']],
        'Q5_P4_vs_decoder':[statement('P4','Decode',k) for k in ['Obj Material','Obj Pattern']],
        'Q6_compute':'효율표의 추가 비용과 AP를 함께 판단. 같은 입력 해상도라도 P4의 추가 spatial 연산과 Decoder의 추가 capacity를 분리한 인과 비교가 아님.'}
    provenance=dict(p8_checkpoint=str(base/'P/checkpoints/ema_last.pt'),p8_sha256=file_sha(base/'P/checkpoints/ema_last.pt'),
                    source={str(p):file_sha(p) for p in Path('src/siglip_struct').glob('*.py')},
                    encoder_checkpoints={c:dict(path=str(run/c/'checkpoints/ema_last.pt'),sha256=file_sha(run/c/'checkpoints/ema_last.pt')) for c in ['P4','Decode']})
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    result=dict(status='complete',input_resolution=384,seed=0,models=models,AP=table.to_dict('records'),contrasts=C.to_dict('records'),
                focus_attributes=json.loads(F.to_json(orient='records')),ROI_size=S.to_dict('records'),efficiency=json.loads(efficiency.to_json(orient='records')),
                training=training,reader_checks=checks,answers=answers,provenance=provenance)
    (run/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    focus_table=F[F.level=='Object'].pivot(index=['attribute','positive_ROIs','positive_categories'],columns='model',values='AP').reset_index()
    lines=['# SigLIP2 @384 구조 비교 결과','','P4는 공식 pretrained SigLIP2에서 새로12k 학습했습니다. Native/Decode는 기존P12k EMA를 고정했습니다. 기존 P8/M/B0/C0 결과를 그대로 재사용했고 새 checkpoint·예측·로그는 별도 폴더에 저장했습니다.','','## Official TEST AP','',md(table),'',
           f"공식max300/image 규칙으로 조건별 제외 prediction 수는 {dict((c,E[c]['dets_dropped_by_max_dets']) for c in models)} 입니다. 모든 조건은 같은 GT ROI와 category를 사용했습니다.",'','## Object 속성 및 positive 수 (탐색적)','',md(focus_table),'',
           '## 주요 paired image95% CI','',md(C[C.metric.isin(['Object AP','Part AP','Obj Material','Obj Pattern'])]),'','이미지 단위1000회 bootstrap이며 training seed 변동과 다중 비교 보정은 포함하지 않습니다. TEST는 VAL로 고정한 epoch에만 사용했고 속성별 결과로 hyperparameter를 고르지 않았습니다.','','## ROI 크기별 Pattern/Material','',md(S),'',
           '크기는GT bbox를longest-side384에 매핑한 면적입니다. tiny<32², small32²–64², medium64²–128², large≥128²로 기존 진단과 같습니다. 크기별 category 구성 차이와 작은 positive 수가 있어 작은ROI의 상대 개선은 탐색적 분석입니다.','','## 질문별 결과','',json.dumps(answers,ensure_ascii=False,indent=2),'','## Parameter / FLOPs / Memory / Throughput','',md(efficiency),'',
           'P4 window16×16token은64×64픽셀이고 shift8token은32픽셀입니다. P8의window8/shift4와 같은 물리적 범위입니다. Interaction은4×4평균/방송이며 기존 fusion을 유지했습니다. Fine patch conv는 더 작은 kernel로 parameter가 줄 수 있지만 token 수와 연산·저장량은 증가합니다. 효율 비교는 이 비용을 포함해 해석해야 합니다.','','Native의coarse768과fine384는 원본 저장 feature에서 선택합니다. Fine은 기존 frozenWd로768에 매핑하며, type은명시적metadata 및stride16/8에 따른5D geometry로 보존합니다. 새 trainable R2 type embedding은 없습니다. Padding용 token은R2에서 마스킹합니다.','','Decode는fullimage384를한번인코딩하고16개query가ROI+25%문맥의실제coarse/fine cell을읽습니다. RGB crop을추론에재인코딩하지않습니다. 새Decoder는COCO의기존mean cosine으로만학습한뒤고정하고PACO에서독립R2를학습했습니다. 모든조건의R2parameter,epoch,trainrows,학습step,optimizer및VAL선택규칙동일성을검증했습니다.','','추가Decodercapacity와16-tokenbottleneck, Native geometry의cellsize정보가함께바뀌므로 결과를순수한보간제거효과로단정하지않습니다. Native/Decoder 처리시간은대표val1image의실제ROI분포에따른값이며전데이터평균이아닙니다. FLOPs는2FLOPs/MAC이며LN·softmax·선택연산은제외했습니다.','','## 이전 G1-cross와 차이','','이전G1-cross는jointencoder/decoder,coarse-onlyfulltokens,576queries,MLPx4,전체image문맥및token/detail감독을사용했습니다. 이번은학습완료P12kSigLIP2encoder고정,coarse+fine,16queries,MLPx2,ROI+25%mask,기존mean-onlycosine입니다. G1checkpoint를사용하지않았습니다. 따라서이전실패의단순재현또는backbone-only비교가아닙니다.','','## 검증·학습·재개','',json.dumps({c:dict(completion=v['completion'],initial=v['diagnostics'][0],final=v['diagnostics'][-1]) for c,v in training.items()},ensure_ascii=False,indent=2),'',
           'CPU geometry/native-cell count/16-querygradient/4×4interaction검증후실제GPUforward/backwardsmoke를수행했습니다. Teachercache가동일SigLIP2인지확인하고실제원본cropteacher와cachecosine을검증했습니다. P4는microbatch4로16imageeffectivebatch를정확히유지하며ROI·global분모를원래전체batch로계산했습니다.','','`bash '+str(run/'scripts/resume.sh')+'`는완료stage를건너뜁니다. Encoder별재개는scripts/resume_P4.sh와resume_Decode.sh이며500step마다optimizer/EMAcheckpoint를저장합니다.','','새로운loss·768입력·PACOlabelrepresentation학습은사용하지않았습니다. 기존실험결과는수정하지않았습니다.','']
    report='\n'.join(lines);(run/'REPORT.md').write_text(report);p=run/'README.md';p.write_text(p.read_text().split('<!-- RESULTS -->')[0]+'\n<!-- RESULTS -->\n'+report)
    print(md(table),flush=True)


if __name__=='__main__':main()
