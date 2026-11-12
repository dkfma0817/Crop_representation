"""Standalone, offline analysis HTML from completed official results only."""
import argparse
from collections import Counter,defaultdict
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import time

import pandas as pd

from src.siglip_extra.report import flat

HERE=Path(__file__).parent
CORE=['P8','P4','Native','Decode','M','B0','C0']
LABELS={'P8':'P8 · Coarse–Fine','P4':'P4 · Fine 4px','Native':'Native · 실제 token 선택','Decode':'Decoder · 4×4 queries','M':'M · Multi-layer','B0':'Full B0 · Frozen Full','C0':'C0 · Crop Re-encoding','E1':'E1 · Contrastive +4k','E2':'E2 · Spatial +4k','Continue':'P-Continue +4k'}
COLORS={'P8':'#315bd9','P4':'#8050c7','Native':'#16879b','Decode':'#ce7437','M':'#637793','B0':'#88796d','C0':'#14815f','E1':'#477ca2','E2':'#a173a8','Continue':'#707780'}


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def lines(path):
    try:
        out=[]
        for s in Path(path).read_text().splitlines():
            try:out.append(json.loads(s))
            except ValueError:pass
        return out
    except OSError:return []


def clean(obj):
    if isinstance(obj,float) and not math.isfinite(obj):return None
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,list):return [clean(v) for v in obj]
    return obj


def atomic(path,text):
    path=Path(path);tmp=path.with_name(path.name+f'.tmp-{os.getpid()}');tmp.write_text(text);os.replace(tmp,path)


def svg_diagram(model,train):
    esc=html.escape
    title=LABELS[model]+(' — 학습' if train else ' — 추론')
    v=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 690" role="img" aria-label="'+esc(title)+'">',
       '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse"><path d="M0 0L8 4L0 8" fill="#8090a6"/></marker></defs>',
       '<rect width="1180" height="690" rx="18" fill="#fbfcff"/>',
       '<style>text{font-family:Arial,"Noto Sans CJK KR",sans-serif;fill:#20304b}.sub{font-size:13px;fill:#526580}.head{font-size:22px;font-weight:700}.txt{font-size:15px;font-weight:600}</style>',
       '<text x="30" y="37" class="head">'+esc(title)+'</text>']
    def box(x,y,w,h,text,kind='frozen'):
        bg,stroke={'frozen':('#eaf0ff','#9eafd9'),'learn':('#fff1df','#deb987'),'data':('#eff4f8','#b9c6d6'),'loss':('#e6f5ee','#90bca5')}[kind]
        v.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{bg}" stroke="{stroke}"/>')
        ss=text.split('\n')
        for i,s in enumerate(ss):v.append(f'<text x="{x+w/2}" y="{y+h/2-(len(ss)-1)*10+i*20+5}" text-anchor="middle" class="txt">{esc(s)}</text>')
    def edge(x,y,xx,yy,label=''):
        v.append(f'<path d="M{x} {y} L{xx} {yy}" fill="none" stroke="#8090a6" stroke-width="1.7" marker-end="url(#a)"/>')
        if label:v.append(f'<text x="{(x+xx)/2+5}" y="{(y+yy)/2-7}" class="sub">{esc(label)}</text>')
    def note(x,y,text):
        for i,s in enumerate(text.split('\n')):v.append(f'<text x="{x}" y="{y+i*21}" class="sub">{esc(s)}</text>')
    if model in ['P8','P4','E1','E2','Continue'] and train:
        fine=4 if model=='P4' else 8;grid=384//fine;window=64//fine;ratio=16//fine
        box(20,165,150,90,'원본 RGB\n384 × 384','data')
        box(190,75,170,75,'Coarse patch16\n24² × 768','learn')
        box(190,260,170,75,f'Fine patch{fine}\n{grid}² × 384','learn');edge(170,190,190,110);edge(170,230,190,298)
        for i in range(4):
            x=390+i*165
            box(x,75,145,75,f'Stage {i+1}\nViT 3 blocks','learn')
            box(x,260,145,75,f'Fine block {i+1}\nshift {0 if i%2==0 else window//2}','learn')
            edge(360 if i==0 else x-20,112,x,112);edge(360 if i==0 else x-20,297,x,297)
            edge(x+47,150,x+47,260);edge(x+100,260,x+100,150)
            for xx,label in [(x+33,'broadcast'),(x+118,f'pool {ratio}×{ratio}')]:
                v.append(f'<text x="{xx}" y="205" text-anchor="middle" transform="rotate(-90 {xx} 205)" class="sub">{label}</text>')
        box(380,425,260,80,'GT ROI coverage mean\nA + s_d · W_d(F)','learn');edge(1030,112,1075,395);edge(1030,297,1075,395)
        v.append('<path d="M1075 395 H510 V425" fill="none" stroke="#8090a6" stroke-width="1.7" marker-end="url(#a)"/>')
        box(730,425,230,80,'Crop cosine\n+ 0.5 × Global loss','loss');edge(640,465,730,465)
        box(20,425,290,80,'Native image embedding\n+ valid spatial mean\nFrozen Full Teacher와 비교','loss')
        v.append('<path d="M165 505 L165 532 L765 532 L765 505" fill="none" stroke="#8090a6" stroke-width="1.7" marker-end="url(#a)"/>')
        box(415,560,230,65,'원본 GT RGB crop\nletterbox 384','data');box(730,550,335,85,'Frozen 공식 SigLIP2 Teacher\ncoverage-weighted spatial mean','frozen');edge(645,592,730,592);edge(845,550,845,505)
        root='공식 pretrained coarse + 새 Fine 초기화' if model in ['P8','P4'] else '동일 완료 P12k EMA에서 추가4k 시작'
        note(25,95 if False else 375,f'{root} · Fine window {window} tokens =64px · shift={window//2} tokens =32px')
        if model=='E1':note(25,650,'추가: frozen teacher in-batch InfoNCE, τ=.07, λ=.1, 유사 negative 제외. 구조는 P8과 동일.')
        elif model=='E2':note(25,650,'추가: valid crop 2×2 local mean cosine, λ=.1. 작은 ROI spatial loss 마스킹. 구조는 P8과 동일.')
        elif model=='Continue':note(25,650,'추가 loss 없음. P12k + 기존 loss 4k. Encoder 학습에 PACO attribute label 사용 안 함.')
        else:note(25,650,'COCO 12k steps · effective batch16 · seed0 · PACO attribute label로 encoder 학습하지 않음')
    elif model=='M' and train:
        box(25,205,165,90,'Full RGB\n384 × 384','data')
        box(240,95,240,305,'공식 pretrained SigLIP2\n12 Transformer blocks\nEncoder 학습','learn');edge(190,250,240,250)
        for i,label in enumerate(['L4 → LayerNorm','L8 → LayerNorm','L12 → PostNorm']):
            y=90+i*130;box(535,y,225,75,label+'\n24² × 768','learn');edge(480,y+37,535,y+37)
            edge(760,y+37,825,270)
        box(825,215,320,110,'GT ROI coverage mean\nL12 + s4W4(L4) + s8W8(L8)\n768-dimensional fused mean','learn')
        box(825,450,320,85,'Crop cosine\n+0.5 × Global preservation','loss');edge(985,325,985,450)
        box(295,450,460,85,'원본 GT Crop → Frozen SigLIP2\ncoverage-weighted spatial mean','frozen');edge(755,492,825,492)
        box(25,450,220,85,'Native MAP embedding\n+ spatial mean 보존','loss')
        v.append('<path d="M135 535 L135 563 L855 563 L855 535" fill="none" stroke="#8090a6" stroke-width="1.7" marker-end="url(#a)"/>')
        note(30,615,'COCO12k: Encoder/fusion 학습. Fine path 없음. 이후 Encoder를 고정하고 PACO에서 새 R2를10epochs 학습.')
    elif model=='Decode':
        box(25,135,155,85,'Full RGB\n384 × 384','data');box(215,110,235,135,'Frozen P12k EMA\nCoarse 24² × 768\nFine 48² × 384','frozen');edge(180,177,215,177)
        box(500,115,235,115,'실제 native K/V\nROI +25% 문맥\nletterbox padding mask','frozen');edge(450,177,500,177)
        box(500,315,235,105,'GT ROI + 상대 위치\n4 × 4 query grid\n16 learned positions','learn' if train else 'frozen')
        box(800,150,275,125,'ROI Cross-Attention\n2 blocks ·768d ·8 heads\n16 ROI-conditioned tokens','learn' if train else 'frozen');edge(735,172,800,185);edge(735,355,850,275)
        if train:
            box(800,355,275,75,'16-token mean\n기존 cosine distillation','loss');edge(938,275,938,355)
            box(260,495,400,85,'Frozen 공식 SigLIP2 GT Crop Teacher\nvalid-region weighted spatial mean','frozen');edge(660,540,800,415)
            note(30,620,'1단계 COCO: Decoder만12k 학습. 새 loss 없음. Encoder/Teacher frozen. RGB crop은 Teacher 학습 target에만 사용.')
            note(30,647,'2단계 PACO: Encoder + Decoder 고정 → 독립 R2 10epochs. Attribute label은 R2에만 사용.')
        else:
            box(800,370,275,80,'Frozen R2 Reader\nObject / Part attribute heads','frozen');edge(938,275,938,370)
            box(800,510,275,75,'59 attribute scores\nGT category로 official 평가','data');edge(938,450,938,510)
            note(30,620,'Full image는한번인코딩. 추론에 RGB crop 재인코딩 없음. 16개 output이 정확한 local 관측16개를 뜻하지는 않음.')
    else:
        if model in ['P8','P4','E1','E2','Continue']:
            f=4 if model=='P4' else 8;g=384//f
            texts=['Full RGB\n384 × 384',f'Frozen P{f}\nCoarse24²×768\nFine{g}²×384','GT ROIAlign\nCoarse/Fine →24²', 'Fused768 tokens\nA+s_dW_d(F)','R2 2 blocks\nObject/Part heads']
            foot='동일 full image 한 번 인코딩 → ROI별 576개 보간 위치 → 동일 R2. Fine source의 실제 간격과 보간 token 수는 구분.'
        elif model=='M':
            texts=['Full RGB\n384 × 384','SigLIP2 12 blocks\nL4 / L8 / L12','3×24²×768\nGT ROIAlign24²','기존 multi-layer\nresidual fusion','R2 2 blocks\nObject/Part heads']
            foot='Fine path 없음. 학습은 COCO mean crop cosine + native global 보존12k. 공식 pretrained에서 시작.'
        elif model=='Native':
            texts=['Full RGB\n384 × 384','Frozen P12k\nCoarse / Fine','GT ROI 겹침\n실제 cell만 선택','Fine frozen W_d\nType + Geometry\n가변 token 수','R2 2 blocks\nObject/Part heads']
            foot='보간 576 token 없음. Coarse768 / Fine384 source, 기존 frozen Wd로768에 매핑. Native stride16/8을 geometry에 보존.'
        elif model=='B0':
            texts=['Full RGB\n384 × 384','Frozen 공식\nSigLIP2 Vision','24²×768 tokens\nGT ROIAlign24²','576 ROI tokens\ncoverage mask','R2 2 blocks\nObject/Part heads']
            foot='Representation learning 없음. PACO에서는 Encoder를 고정하고 새 R2만10epochs 학습.'
        else:
            texts=['GT 원본 RGB crop\nROI마다 별도','Resize / Letterbox\n384 × 384','Frozen 공식\nSigLIP2 Vision','Native crop\n24²×768 tokens','R2 2 blocks\nObject/Part heads']
            foot='ROI마다 crop을 재인코딩하는 reference. Full image당 encoder 비용과 직접 같은 단위로 비교할 수 없음.'
        for i,t in enumerate(texts):
            x=25+i*232;kind='data' if i==0 else 'learn' if train and (i==4 or model=='M') else 'frozen'
            box(x,200,205,135,t,kind)
            if i:edge(x-27,267,x,267)
        if train and model=='M':
            box(265,430,370,85,'COCO fused ROI mean cosine\n+ pretrained image embedding 보존','loss');box(735,430,370,85,'Frozen SigLIP2 GT Crop mean\nEncoder12k → 이후독립 R2','frozen');edge(635,472,735,472)
        elif train:
            box(710,430,395,85,'PACO attribute masked BCE\nR2만 10epochs 학습','loss');edge(1050,335,1050,430)
        else:
            box(710,430,395,85,'59 attribute scores\nGT category / Object·Part로 official 평가','data');edge(1050,335,1050,430)
        note(30,580,foot)
        note(30,612,'공식 VAL의 (Object AP + Part AP)/2로 epoch 선택. 선택된 epoch만 TEST 평가. Seed0.')
    note(30,680,'파랑: 고정/기존 feature · 주황: 이 단계에서 학습 · 초록: loss/target · 모든 RGB 입력384×384')
    v.append('</svg>');return ''.join(v)


def collect(run,base,extra):
    now=dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    payload=dict(updated=now.isoformat(),models={},contrasts=[],sizes=[],costs=[],core=CORE,labels=LABELS,colors=COLORS,run=str(run),metrics=list(flat('x',read(base/'eval_test/P_r2/test/official_results.json')['hierarchical']).keys())[1:])
    gt=read(Path('/home/irteam/datasets/paco/paco_lvis_v1_test.json'));cat={v['id']:v['name'] for v in gt['categories']};attr={v['id']:v['name'] for v in gt['attributes']}
    counts=Counter();categories=defaultdict(set)
    for ann in gt['annotations']:
        lv='Part' if ':' in cat[ann['category_id']] else 'Object'
        for aid in ann.get('attribute_ids',[]):counts[lv,attr[aid]]+=1;categories[lv,attr[aid]].add(ann['category_id'])
    mapping={c:(base,'P' if c=='P8' else c) for c in ['P8','M','B0','C0']}
    mapping.update({c:(run,c) for c in ['P4','Native','Decode']});mapping.update({c:(extra,c) for c in ['E1','E2','Continue']})
    for c,(root,key) in mapping.items():
        official=read(root/f'eval_test/{key}_r2/test/official_results.json')
        mean_root=base if c in ['P8','M','B0','C0'] else extra
        mean=read(mean_root/f'mean/eval_test/{key}_linear/test/official_results.json') if c not in ['P4','Native','Decode'] else None
        entry=dict(label=LABELS[c],group='structural' if c in CORE else 'loss4k',ap=flat(c,official['hierarchical']) if official else None,
                   mean=flat(c,mean['hierarchical']) if mean else None,attributes={},mean_attributes={},test_complete=official is not None)
        for reader,result in [('attributes',official),('mean_attributes',mean)]:
            if result:
                for lv,field in [('Object','obj-attr-AP-per-class'),('Part','obj-attr-part-AP-per-class')]:
                    entry[reader][lv]=[dict(attribute=name,AP=value,positive_ROIs=counts[lv,name],positive_categories=len(categories[lv,name])) for name,value in result['hierarchical'][field].items()]
        if c in ['P4','Decode']:
            logs=lines(run/c/'train_log.jsonl');diagnostics=lines(run/c/'diagnostics.jsonl');done=read(run/c/'done.json')
            entry['training']=dict(step=logs[-1]['step'] if logs else 0,target=12000,done=done,last=logs[-1] if logs else None,diagnostics=diagnostics,curve=logs[::max(1,len(logs)//150)])
        if c in ['P4','Native','Decode']:
            reader=read(run/f'probe_{c}/done.json');cfg=read(run/f'probe_{c}/config.json',{})
            tail=lines(run/f'logs/probe_{c}.log');last=next((v for v in reversed(tail) if 'epoch' in v),{})
            epochs=len(list((run/f'probe_{c}/r2').glob('ep*.pt')))
            entry['progress']=dict(reader_done=reader is not None,reader_epoch=10 if reader else last.get('epoch',epochs),reader_step=last.get('step',0),
                                   val_complete=len(list((run/f'eval_val/{c}_r2').glob('ep*/official_results.json'))),test_complete=official is not None,
                                   selection=read(run/f'selection_{c}.json',{}).get(c+'_r2'),reader_parameters=cfg.get('reader_params',{}).get('r2'))
            if c=='Native':entry['training']=dict(done=read(run/'Decode/done.json'),step=0,target=0,note='Representation learning 없음; 기존P12k 고정')
        payload['models'][c]=entry
    summary=base.parent.parent/'2026-11-09/siglip2_completed_summary'
    for root,name,reader in [(base,'results/contrasts.csv','ap'),(extra,'results/r2_contrasts.csv','ap'),(run,'results/contrasts.csv','ap'),(extra,'results/mean_probe_contrasts.csv','mean'),(summary,'mean_probe_contrasts.csv','mean')]:
        try:
            rows=pd.read_csv(root/name).to_dict('records')
            for row in rows:
                row['reader']=reader
                if root in [base,extra,summary]:row['comparison']=row['comparison'].replace(' − P',' − P8').replace('P − ','P8 − ')
                payload['contrasts'].append(row)
        except (OSError,ValueError):pass
    # Canonicalize the original pair names without turning P4 into P84.
    for row in payload['contrasts']:
        if row['comparison']=='P − M':row['comparison']='P8 − M'
    for p in [base/'results/roi_size_ap.csv']+list((run/'results').glob('roi_size_ap_*.csv')):
        try:
            for row in pd.read_csv(p).to_dict('records'):
                if row['model']=='P':row['model']='P8'
                payload['sizes'].append(row)
        except (OSError,ValueError):pass
    for p in [base/'results/efficiency.csv']+list((run/'results').glob('efficiency_*.csv')):
        try:
            for row in pd.read_csv(p).to_dict('records'):
                if row['model']=='P':row['model']='P8'
                payload['costs'].append(row)
        except (OSError,ValueError):pass
    for c in ['P4','Native','Decode']:
        if not any(v['model']==c for v in payload['costs']):
            cfg=read(run/('Decode' if c=='Native' else c)/'config.json',{})
            expected=read(run/'expected_costs.json',{})
            payload['costs'].append(dict(model=c,encoder_parameters=102953097 if c=='P4' else 103008393,additional_decoder_parameters=cfg.get('n_parameters') if c=='Decode' else 0,
                     encoder_GFLOPs=expected.get('P4_encoder_GFLOPs') if c=='P4' else expected.get('P8_encoder_GFLOPs'),stored_scalars=3981312 if c=='P4' else 1327104,
                     R2_parameters=10738294,provisional=True))
    try:
        text=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
        payload['gpu']=[dict(index=int(v.split(',')[0]),utilization=int(v.split(',')[1]),memory_MiB=int(v.split(',')[2])) for v in text.splitlines()]
    except (OSError,subprocess.SubprocessError):payload['gpu']=[]
    payload['failures']=[p.name for p in (run/'state').glob('*.failed.json')]
    payload['complete']=read(run/'done.json') is not None
    return clean(payload)


def render(run,base,extra,out):
    out.mkdir(parents=True,exist_ok=True);(out/'diagrams').mkdir(exist_ok=True)
    payload=collect(run,base,extra)
    from src.siglip_struct.dashboard_figures import write_figures
    write_figures(payload,out/'figures')
    for c in LABELS:
        for train in [True,False]:
            p=out/'diagrams'/f'{c}_{"train" if train else "inference"}.svg'
            diagram=svg_diagram(c,train)
            if not p.exists() or p.read_text()!=diagram:p.write_text(diagram)
    payload['diagrams']={c:{k:(out/'diagrams'/f'{c}_{k}.svg').read_text() for k in ['train','inference']} for c in LABELS}
    atomic(out/'data.json',json.dumps(payload,ensure_ascii=False,allow_nan=False))
    ap=[dict(**entry['ap'],Reader='R2') for entry in payload['models'].values() if entry['ap']]
    ap+=[dict(**entry['mean'],Reader='Mean linear') for entry in payload['models'].values() if entry['mean']]
    pd.DataFrame(ap).to_csv(out/'ap_tables.csv',index=False)
    attrs=[]
    for c,entry in payload['models'].items():
        for lv,values in entry['attributes'].items():attrs.extend([dict(model=c,level=lv,**v) for v in values])
    pd.DataFrame(attrs).to_csv(out/'all_attributes.csv',index=False)
    document=(HERE/'dashboard.html').read_text().replace('/*__STYLE__*/',(HERE/'dashboard.css').read_text()).replace('/*__SCRIPT__*/',(HERE/'dashboard.js').read_text()).replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<'+chr(92)+'/'))
    atomic(out/'index.html',document)
    print(json.dumps(dict(updated=payload['updated'],completed_TEST=[c for c,e in payload['models'].items() if e['test_complete']],out=str(out/'index.html'))),flush=True)


def main():
    ap=argparse.ArgumentParser()
    for key in ['run','base','extra','out']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--watch',action='store_true');a=ap.parse_args();run,base,extra,out=map(Path,[a.run,a.base,a.extra,a.out])
    while True:
        render(run,base,extra,out)
        if not a.watch or (run/'done.json').exists() or (run/'queue_failed.json').exists():break
        if time.time()>read(run/'budget.json',{}).get('deadline_unix',time.time())+300:break
        time.sleep(60)


if __name__=='__main__':main()
