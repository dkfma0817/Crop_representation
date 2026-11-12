"""Render the portable P4 neighbor audit from its saved diagnostic arrays."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from measure import OUT, MODELS, REGIMES, BANDS, KS, read, write

BLUE, NAVY, RED, GREY = '#2977b7', '#152e50', '#a95363', '#96a9bf'
COLORS = {'Original':NAVY, 'B0':RED, 'P2':BLUE, 'Frequency random':GREY, 'Class-balanced random':'#c0d3e4'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','axes.titlelocation':'left','axes.axisbelow':True,'grid.alpha':.18,
                     'axes.grid':True,'text.color':NAVY,'svg.fonttype':'none','svg.hashsalt':'p4-neighbor-audit'})


def esc(s):
    return html.escape(str(s), quote=True)


def pct(x, n=2):
    return '—' if x is None else f'{100*x:.{n}f}'


def num(x, n=3):
    return '—' if x is None else f'{x:.{n}f}'


def stat(s):
    return f'{pct(s["micro"])} / {pct(s["macro"])}'


def details(title, content):
    return f'<details><summary>{esc(title)}</summary>{content}</details>'


def table(columns, rows, sortable=False):
    cls = ' class="sortable"' if sortable else ''
    cols = ''.join(f'<th scope="col">{esc(c)}</th>' for c in columns)
    body = ''.join('<tr>'+''.join(f'<td>{esc(c)}</td>' for c in row)+'</tr>' for row in rows)
    return f'<div class="scroll"><table{cls}><thead><tr>{cols}</tr></thead><tbody>{body}</tbody></table></div>'


def inline_image(path, alt):
    mime = 'image/svg+xml' if path.suffix == '.svg' else 'image/jpeg'
    return f'<img src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{esc(alt)}" loading="lazy">'


def figure(fig, file, caption):
    path = OUT/(file+'.svg')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return f'<figure class="chart">{inline_image(path,caption)}<figcaption>{esc(caption)}</figcaption></figure>'


def dataset(d):
    pool_rows = []
    for r in REGIMES:
        p = d[r]['pool']
        count = list(p['per_class'].values())
        pool_rows.append((r,f'{p["patches"]:,}',f'{p["patches"]:,}',p['images'],p['max_patches_per_image'],
                          '–'.join(map(str,p['candidate_count_range'])),
                          f'{min(count)} / {np.median(count):.0f} / {max(count)}',pct(p['has_same_class']['micro'])))
    return table(['구획','Query Patch','후보 Patch','후보 이미지','이미지당 관측 최대','Query당 적격 후보 범위','Class당 Patch 최소 / 중앙 / 최대','동일 Class 후보 존재 %'],pool_rows) + details('제외한 39개 클래스와 구획별 지원 수',table(['Class ID','Class','Interior 이미지','Mixed 이미지','누락 사유'],[(x['class_id'],x['name'],x['interior'],x['mixed'],'한 구획 이상에서 8장 미만') for x in d['omitted_classes']],True))


def retrieval(d):
    names = ['Original','B0','P2','Frequency random','Class-balanced random']
    fig, axes = plt.subplots(1,2,figsize=(11,3.45),layout='constrained')
    for ax, region in zip(axes,REGIMES):
        for i,n in enumerate(names):
            ax.bar(np.arange(3)+(i-2)*.155, [100*d[region]['retrieval'][n][f'p{k}']['macro'] for k in KS],
                   width=.145, color=COLORS[n], label=n)
        ax.set(title=region,xlabel='Retrieved rank cutoff',ylabel='Macro precision (%)',xticks=range(3),xticklabels=['P@1','P@5','P@10'],ylim=(0,66))
    axes[1].legend(frameon=False,fontsize=8,loc='upper right')
    out = figure(fig,'retrieval_precision','구획별 Class-balanced Macro Precision. Exact Random Baseline은 1% 안팎입니다.')
    rows = [(r,n,*[stat(d[r]['retrieval'][n][k]) for k in ('p1','p5','p10','hit5','hit10','mrr')])
            for r in ('All',*REGIMES) for n in names]
    return out+table(['구획','모델 / 기준','P@1 %','P@5 %','P@10 %','Hit@5 %','Hit@10 %','MRR ×100'],rows)


def confidence_chart(d):
    fields = [('top1_cosine','Top-1 cosine'),('gap_1_5','Rank 1–5 gap'),('gap_1_10','Rank 1–10 gap')]
    fig, axes = plt.subplots(1,3,figsize=(11.7,3.25),layout='constrained')
    rows=[]
    for ax,(key,title) in zip(axes,fields):
        for region,color in zip(REGIMES,(NAVY,BLUE)):
            data=d[region]['confidence'][key]['groups']
            for mode,style in [('micro','-'),('macro','--')]:
                ax.plot(range(1,6),[100*x[mode] for x in data],marker='o',lw=1.7,ls=style,color=color,
                        label=region+' '+mode)
            rows.extend([(title,region,x['bin'],f'{x["lower"]:.4f}–{x["upper"]:.4f}',x['n'],x['classes'],stat(x)) for x in data])
        ax.set(title=title,xlabel='Within-regime quintile',ylabel='Teacher Top-1 same class (%)',xticks=range(1,6),ylim=(20,95))
    axes[0].legend(frameon=False,fontsize=8)
    out=figure(fig,'teacher_confidence','Teacher Confidence 5분위와 실제 Top-1 정답률. 실선 Micro, 점선 Macro')
    out+=details('Confidence 구간, 표본 수 및 정확한 수치',table(['Teacher 지표','구획','분위','경계','Query','Class','동일 클래스 % Micro / Macro'],rows))
    dist = plt.figure(figsize=(7.1,3.0),layout='constrained')
    a = dist.add_subplot(111)
    edges=np.linspace(0,1,11)
    dist_rows=[]
    for region,color in zip(REGIMES,(NAVY,BLUE)):
        v = np.array([x['p_at_1']['Original'] for x in d[region]['classes'].values()])
        a.stairs(np.histogram(v,bins=edges)[0],edges,color=color,label=region,lw=2)
        q = np.quantile(v,[0,.05,.25,.5,.75,.95,1])
        dist_rows.append((region,len(v),*[pct(x) for x in q]))
    a.set(xlabel='Per-class Teacher P@1',ylabel='Number of classes',title='Class-level reliability',xlim=(0,1))
    a.legend(frameon=False)
    return out+'<div class="columns">'+figure(dist,'class_reliability','Teacher Precision@1의 클래스별 분포')+table(['구획','Class','Min %','P05','P25','중앙값','P75','P95','Max %'],dist_rows)+'</div>'


def retention(d):
    fig, axes = plt.subplots(1,2,figsize=(10.8,3.25),layout='constrained')
    for ax,region in zip(axes,REGIMES):
        for name,color in [('B0',RED),('P2',BLUE)]:
            x = d[region]['retention'][name]
            ax.plot(range(3),[100*x[str(k)]['overlap']['micro'] for k in KS],marker='o',color=color,label=name+' all IDs')
            ax.plot(range(3),[100*x[str(k)]['same_class_id_retention']['pooled_teacher_hits'] for k in KS],marker='s',ls='--',color=color,label=name+' same-class IDs')
        ax.set(title=region,xlabel='Top-k',ylabel='Identity retention (%)',xticks=range(3),xticklabels=KS,ylim=(18,61))
        ax.legend(frameon=False,fontsize=8)
    out = figure(fig,'retention','Top-k Exact ID Overlap과 Teacher Same-class ID 유지율. 두 값의 분모가 다릅니다.')
    rows=[]
    for r in ('All',*REGIMES):
        for n in MODELS:
            for k in KS:
                x=d[r]['retention'][n][str(k)]
                y=x['same_class_id_retention']
                rows.append((r,n,k,stat(x['overlap']),stat(y),pct(y['pooled_teacher_hits']),f'{y["retained_teacher_hits"]:,} / {y["teacher_hits"]:,}'))
    out+=table(['구획','모델','k','전체 ID % Micro / Macro','Same-class ID % Micro / Macro','Same-class ID pooled %','유지한 Teacher Hit / 전체'],rows)
    outcomes=[]
    for r in REGIMES:
        for n in ('B0','P2'):
            q=d[r]['top1_outcomes'][n]
            outcomes.append((r,n,q['same_patch']['n'],*[stat(q[k]) for k in ('same_patch','new_same_class','different_class','corrected_teacher_error')]))
    out+='<h3>정확한 이웃 ID를 잃었을 때의 클래스 결과</h3>'+table(['구획','모델','Teacher Top-1 정답 수','동일 Patch %','다른 Same-class Patch %','다른 Class로 변경 %','Teacher 오답을 정정 %'],outcomes)
    return out


def band_label(i):
    lo,hi=BANDS[i]
    return f'{lo}–{hi}' if hi is not None else f'{lo}–N'


def violation(d, archive):
    fig, axes = plt.subplots(1,2,figsize=(11,3.4),layout='constrained')
    rows=[]
    for ax,region in zip(axes,REGIMES):
        groups=d[region]['ranking']['margin_groups']
        for name,color in [('B0',RED),('P2',BLUE)]:
            for key,style in [('all','--'),('same_different','-')]:
                ax.plot(range(1,6),[100*q[key]['violation'][name]['micro'] for q in groups],marker='o',ls=style,color=color,
                        label=f'{name} {"all" if key=="all" else "same→different"}')
        ax.set(title=region,xlabel='Teacher margin quintile',ylabel='Violation rate (%)',xticks=range(1,6),ylim=(0,48))
        ax.legend(frameon=False,fontsize=8)
        for q in groups:
            a,b=q['all'],q['same_different']
            rows.append((region,q['bin'],f'{q["lower"]:.4f}–{q["upper"]:.4f}',a['n'],b['n'],pct(a['same_different_share']['micro']),stat(a['violation']['B0']),stat(a['violation']['P2']),stat(b['violation']['B0']),stat(b['violation']['P2'])))
    out=figure(fig,'margin_violation','Teacher Margin 5분위에 따른 역전율. Mixed의 Same→Different는 Dominant-label Proxy입니다.')
    out+=table(['구획','분위','Margin 범위','전체 Pair','Same→Diff','Same→Diff 비율 %','B0 전체 %','P2 전체 %','B0 Same→Diff %','P2 Same→Diff %'],rows)
    ranked=[]
    for r in REGIMES:
        for band in d[r]['ranking']['by_band']:
            a=band['same_different']; b=band['band_high_margin_same_different']; h=band['band_high_margin']
            ranked.append((r,band_label(band['band']),a['n'],pct(band['all']['same_different_share']['micro']),stat(a['violation']['B0']),stat(a['violation']['P2']),num(band['band_p80'],4),b['n'],pct(h['same_different_share']['micro']),stat(b['violation']['B0']),stat(b['violation']['P2'])))
    out+='<h3>가까운 후보와 먼 후보를 구분한 역전율</h3>'+table(['구획','하위 k 순위','Same→Diff n','전체 중 Same→Diff %','B0 역전 %','P2 역전 %','구간 P80 Margin','P80 Same→Diff n','P80 중 Same→Diff %','P80 B0 역전 %','P80 P2 역전 %'],ranked)
    joint_rows=[]
    for r in REGIMES:
        x=d[r]['ranking']['joint_confident_near']
        a,b=x['all'],x['same_different']
        joint_rows.append((r,num(x['top1_cosine_p80'],4),num(x['near_margin_p80'],4),a['n'],b['n'],x['relation_counts']['same→same'],stat(a['same_different_share']),stat(b['violation']['B0']),stat(b['violation']['P2'])))
    out+='<p>가까운 후보의 신뢰도를 더 확인하기 위해 Teacher j=1, k=6–10 중 Top-1 cosine과 해당 구간 Margin이 각각 P80 이상인 교집합을 별도로 집계했습니다.</p>'+table(['구획','Top-1 Cosine P80','Near Margin P80','전체 n','Same→Diff n','Same→Same n','Same→Diff %','B0 역전 %','P2 역전 %'],joint_rows)
    rel_rows=[]
    for r in REGIMES:
        for name,x in d[r]['ranking']['by_relation'].items():
            qq=x['teacher_margin_quantiles']
            rel_rows.append((r,name,x['n'],stat(x['share']),num(qq[2],4),num(qq[3],4),num(qq[4],4),stat(x['violation']['B0']),stat(x['violation']['P2'])))
    out+='<h3>실제 클래스 관계로 나눈 Teacher Pair</h3>'+table(['구획','Teacher j→k','Pair','Teacher Pair 비율 %','Margin P25','중앙값','P75','B0 역전 %','P2 역전 %'],rel_rows)
    fig, axes = plt.subplots(1,2,figsize=(11,3.2),layout='constrained')
    for ax,r in zip(axes,REGIMES):
        prefix=r.lower()+'_'
        cos=archive[prefix+'cosine_pairs']
        relation=archive[prefix+'dominant_relation']
        m=(cos[0,...,0]-cos[0,...,1])
        bins=np.linspace(0,.8,33)
        for i,color in enumerate([BLUE,'#828bb4','#97a7b7',RED]):
            a=m[(relation==i)&(m>0)]
            ax.stairs(100*np.histogram(a,bins=bins)[0]/len(a),bins,label=['same→different','same→same','different→different','different→same'][i],color=color,lw=1.75)
        ax.set(title=r,xlabel='Teacher cosine margin',ylabel='Within-group bin mass (%)')
    axes[1].legend(frameon=False,fontsize=8)
    return out+figure(fig,'teacher_margin_distribution','관계별 Teacher Margin 분포. 위 표에서 각 분포의 분위수와 표본 수를 확인할 수 있습니다.')


def class_table(d):
    rows=[]
    for r in REGIMES:
        for id,x in d[r]['classes'].items():
            v=x['semantic_violation']['P2']; h=x['high_semantic_violation']['P2']
            rows.append((r,d['class_names'][int(id)],id,x['query_count'],pct(x['p_at_1']['Original']),
                         pct(x['p_at_5']['Original']),pct(x['p_at_10']['Original']),pct(x['p_at_10']['B0']),pct(x['p_at_10']['P2']),
                         pct(x['retention_at_10']['P2']),v['n'],pct(v['micro']),h['n'],pct(h['micro'])))
    return details('111개 클래스별 후보 수, Precision, Retention, Violation · 열 머리글로 정렬',table(['구획','Class','ID','Patch 수','Original P@1','Original P@5','Original P@10','B0 P@10','P2 P@10','P2 ID Ret@10','Same→Diff n','P2 역전 %','High Same→Diff n','P2 High 역전 %'],rows,True))


def example_patches(d, ar, ade_root):
    with (OUT/'patch_bank.csv').open() as f:
        patches={int(row['cache_row']):row for row in csv.DictReader(f)}
    raw=ar['interior_cosine_pairs']
    margins=raw[...,0]-raw[...,1]
    rel=ar['interior_dominant_relation']==0
    high=ar['interior_margin_quintile']==4
    near=np.zeros_like(rel)
    near[:,0,:]=True
    cutoff=d['Interior']['ranking']['by_band'][0]['band_p80']
    masks=[('B0에서 역전되고 P2에서 복원된 높은 Margin 관계',rel&high&(margins[1]<=0)&(margins[2]>0)),
           ('가까운 후보에서 P2가 놓친 구간 상위 Margin 관계',rel&near&(margins[0]>=cutoff)&(margins[2]<=0)),
           ('전체 상위 Margin에서도 P2가 놓친 관계',rel&high&(margins[2]<=0))]
    examples=[]
    for title, mask in masks:
        available=np.argwhere(mask)
        if len(available)==0:
            continue
        vals=margins[0][mask]
        idx=tuple(available[np.argmin(np.abs(vals-np.median(vals)))])
        rows=[int(ar[f'interior_pair_{x}_row'][idx]) for x in ('query','upper','lower')]
        examples.append(dict(title=title, pool_count=int(mask.sum()), rows=rows, roles=['Query','Teacher j','Teacher k'],
                             margins=dict(zip(MODELS, margins[(slice(None),*idx)].tolist())), teacher_ranks=[int(idx[-1])+1,int(ar['interior_lower_teacher_rank'][idx])],
                             cosine=dict(zip(MODELS,raw[(slice(None),*idx)].tolist()))))
    q=ar['interior_query_row']; t=ar['interior_top_rows'][0,:,0]; p=ar['interior_top_rows'][2,:,0]
    c=lambda rows: np.array([int(patches[int(i)]['class_id_zero_based']) for i in rows])
    score=ar['interior_top_scores'][0,:,0]
    wrong=(c(q)!=c(t))&(c(q)==c(p))&(score>=d['Interior']['confidence']['top1_cosine']['edges'][-2])
    if wrong.any():
        indices=np.flatnonzero(wrong)
        i=indices[np.argmin(np.abs(score[indices]-np.median(score[indices])))];rows=[int(q[i]),int(t[i]),int(p[i])]
        examples.append(dict(title='Teacher Top-1 유사도가 높아도 클래스가 다른 사례',pool_count=int(wrong.sum()),rows=rows,
                             roles=['Query','Teacher Top-1','P2 Top-1'],teacher_top1_cos=float(score[i]),p2_top1_cos=float(ar['interior_top_scores'][2,i,0])))
    cards=[]
    for i,x in enumerate(examples):
        figures=[]
        metadata=[]
        for j,(role,row) in enumerate(zip(x['roles'],x['rows'])):
            item=patches[row]; iid=item['image_id']
            with Image.open(ade_root/'images/training'/f'{iid}.jpg') as img:
                rgb=img.convert('RGB')
            scale=384/min(rgb.size)
            nw,nh=round(rgb.width*scale),round(rgb.height*scale)
            full=rgb.resize((nw,nh),Image.Resampling.BILINEAR)
            width=(nw+15)//16
            gy,gx=divmod(int(item['patch_index_row_major']),width)
            left,top=gx*16,gy*16
            assert left<nw and top<nh
            focus=full.crop((left-24,top-24,left+40,top+40)).resize((112,112),Image.Resampling.NEAREST)
            box_color=['#4dc0ef','#fbc95f','#f37e93'][j]
            ImageDraw.Draw(focus).rectangle((42,42,70,70),outline=box_color,width=3)
            thumb=ImageOps.contain(full,(194,154))
            sx,sy=thumb.width/nw,thumb.height/nh
            ImageDraw.Draw(thumb).rectangle((left*sx,top*sy,min(left+16,nw)*sx,min(top+16,nh)*sy),outline=box_color,width=2)
            canvas=Image.new('RGB',(320,160),'#eef4fa')
            canvas.paste(thumb,((194-thumb.width)//2,(160-thumb.height)//2))
            canvas.paste(focus,(202,24))
            path=OUT/f'example_{i+1}_{j+1}.jpg';canvas.save(path,quality=89)
            caption=f'{role} · {item["class"]} · purity {100*float(item["purity"]):.0f}%'
            figures.append(f'<figure>{inline_image(path,caption)}<figcaption>{esc(caption)}<br><span class="muted">{iid} · Patch ({gy}, {gx})</span></figcaption></figure>')
            metadata.append(dict(item,patch_grid_y=gy,patch_grid_x=gx))
        x['patches']=metadata
        if 'margins' in x:
            explanation=' · '.join(f'{m} margin {v:+.3f}' for m,v in x['margins'].items())
            explanation+=f' · Teacher ranks {x["teacher_ranks"][0]} / {x["teacher_ranks"][1]}'
        else:
            explanation=f'Teacher Top-1 cosine {x["teacher_top1_cos"]:.3f} · P2 Top-1 cosine {x["p2_top1_cos"]:.3f}'
        cards.append(f'<div class="case"><h4>{esc(x["title"])}</h4><p class="muted">{esc(explanation)} · 선택 조건을 충족한 관측 {x["pool_count"]:,}개</p><div class="tiles">'+''.join(figures)+'</div></div>')
    write(OUT/'examples.json',examples)
    return ''.join(cards)


CSS='''
:root{color-scheme:light;--ink:#172e4b;--navy:#112947;--blue:#24649b;--muted:#52687f;--edge:#d9e5f0;--pale:#eef6ff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:white;font:15px/1.68 "Gothic A1",Inter,system-ui,"Malgun Gothic",sans-serif}a{color:#175a91;text-underline-offset:3px}.wrap{max-width:1240px;margin:auto;padding:0 26px}header{border-top:7px solid var(--navy);background:linear-gradient(145deg,#eef5ff,white 77%);border-bottom:1px solid var(--edge)}header .wrap{padding-top:39px;padding-bottom:28px}.kicker{color:var(--blue);letter-spacing:.14em;font-weight:750;text-transform:uppercase;font-size:.76rem}h1{line-height:1.13;letter-spacing:-.04em;font-size:clamp(1.9rem,4.8vw,3.4rem);margin:.28em 0 .2em}h2{font-size:1.58rem;line-height:1.25;border-bottom:2px solid #bfd7ef;margin:0 0 15px;padding-bottom:11px}h3{font-size:1.13rem;margin:25px 0 7px}h4{font-size:.98rem;margin:0 0 3px}p{margin:.57rem 0 .93rem}.lede{max-width:940px;color:#3d5673;font-size:1.05rem}nav{display:flex;flex-wrap:wrap;gap:10px 19px;margin-top:18px;font-size:.89rem}nav a{font-weight:680}section{padding:27px 0 18px;scroll-margin:15px}main{padding-bottom:35px!important}.muted,figcaption{color:var(--muted);font-size:.85rem}.summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.summary article,.case,.chart{border:1px solid var(--edge);border-radius:9px;padding:13px;background:white;break-inside:avoid}.summary article{background:#f8fbff}.summary p{margin:.22em 0 0}.eyebrow{font-weight:750;color:var(--blue);font-size:.79rem}.callout{margin:13px 0;background:var(--pale);border-left:4px solid #347db6;padding:10px 16px;border-radius:0 7px 7px 0}.callout p:last-child{margin-bottom:.35rem}.columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;align-items:center}.scroll{overflow-x:auto;border:1px solid var(--edge);border-radius:8px;margin:10px 0 15px}table{border-collapse:collapse;width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}th,td{white-space:nowrap;padding:8px 10px;border-bottom:1px solid #e7eef5;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf3fc;color:#173b61}tbody tr:nth-child(even){background:#f7fafc}tbody tr:hover{background:#eaf5ff}.sortable th{cursor:pointer}.sortable th[aria-sort=ascending]::after{content:" ↑"}.sortable th[aria-sort=descending]::after{content:" ↓"}details{border:1px solid var(--edge);border-radius:8px;padding:9px 12px;margin:11px 0}summary{font-weight:680;cursor:pointer}figure{margin:0;min-width:0}figure img{width:100%;height:auto;display:block;border-radius:4px}.chart{margin:13px 0 15px}.chart img{max-height:510px;object-fit:contain}figcaption{text-align:center;margin-top:6px}.case{margin:10px 0}.case p{margin:0 0 7px}.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.case img{background:#eef4fa}code,pre{font:12px/1.6 ui-monospace,monospace}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fc;padding:12px}.footer{border-top:1px solid var(--edge);padding:15px 0;color:var(--muted)}li{margin:.38em 0}
@media(max-width:760px){.summary,.columns{grid-template-columns:1fr}.wrap{padding-left:16px;padding-right:16px}.tiles{gap:4px}figcaption{font-size:.73rem}}
@media print{nav{display:none}.wrap{max-width:100%;padding:0}.chart,.summary,.case{break-inside:avoid}}
'''
JS='''document.querySelectorAll('.sortable th').forEach(h=>{h.tabIndex=0;const t=h.closest('table');function sort(){const a=h.getAttribute('aria-sort')!=='ascending';t.querySelectorAll('th').forEach(x=>x.removeAttribute('aria-sort'));h.setAttribute('aria-sort',a?'ascending':'descending');const i=h.cellIndex;const rs=[...t.tBodies[0].rows];rs.sort((r,s)=>{const x=r.cells[i].textContent,y=s.cells[i].textContent;if(x==='—')return y==='—'?0:1;if(y==='—')return -1;const p=Number(x.replaceAll(',','')),q=Number(y.replaceAll(',',''));return (Number.isNaN(p)||Number.isNaN(q)?x.localeCompare(y):p-q)*(a?1:-1)});rs.forEach(r=>t.tBodies[0].append(r))}h.addEventListener('click',sort);h.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();sort()}})});'''


def render(d, ar, ade_root):
    interior,mixed=d['Interior'],d['Mixed']
    ri,rm=interior['ranking'],mixed['ranking']
    teacher=interior['retrieval']['Original']; p2=interior['retrieval']['P2']
    high=ri['high_margin_same_different']; near=ri['by_band'][0]; near_high=near['band_high_margin_same_different']
    near_joint=ri['joint_confident_near']; near_joint_sem=near_joint['same_different']
    high_sim=interior['confidence']['top1_cosine']['groups'][-1]; mixed_sim=mixed['confidence']['top1_cosine']['groups'][-1]
    all_overlap=interior['retention']['P2']['10']['overlap']['micro']; replacement=interior['retention']['P2']['10']['same_class_id_retention']
    relative=interior['top1_outcomes']['P2']; mixed_proxy=rm['same_different']
    joint=[]
    for m in MODELS:
        joint.append((m,num(d['previous_auc'][m]['Interior'],4),num(d['previous_auc'][m]['Mixed'],4),num(d['previous_ade'][m]),
                      stat(interior['retention'][m]['10']['overlap']),stat(mixed['retention'][m]['10']['overlap']),
                      stat(ri['same_different']['violation'][m]),stat(mixed_proxy['violation'][m]),stat(high['violation'][m])))
    menu=[('summary','1. 요약'),('protocol','2. 데이터'),('teacher','3. Teacher'),('student','4. Retention'),('ranking','5. Ranking'),('joint','6. 해석'),('p4','7. P4')]
    nav='<nav aria-label="보고서 탐색">'+''.join(f'<a href="#{anchor}">{title}</a>' for anchor,title in menu)+'</nav>'
    prov={**d['provenance'],'analysis_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')},'ade_root':str(ade_root)}
    ranks=table(['모델','Interior AUC','Mixed AUC','ADE20K mIoU','Interior Ret@10 %','Mixed Ret@10 %','Interior Semantic 역전 %','Mixed Proxy 역전 %','Interior High Margin Semantic 역전 %'],joint)
    high_cutoff=ri['high_margin_min']; high_count=high['n']; all_count=ri['same_different']['n']
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cross-image Semantic Neighbor Audit · P4</title><style>{CSS}</style></head><body>
<header><div class="wrap"><div class="kicker">DINOv3-B/16 · P4 pre-audit · ADE20K Train</div><h1>Cross-image Semantic<br>Neighbor Audit</h1><p class="lede">Original의 Patch 이웃이 실제 클래스를 얼마나 잘 반영하는지, P2에서 잃은 이웃 중 어떤 관계가 클래스 구별에 관련되는지 검증했습니다.</p>{nav}</div></header>
<main class="wrap">
<section id="summary"><h2>1. Executive Summary</h2><div class="summary">
<article><div class="eyebrow">H1 · Teacher Reliability</div><p>Interior Top-1 동일 클래스 비율은 <b>{pct(teacher['p1']['micro'],1)}%</b>이고 Class-balanced Macro는 <b>{pct(teacher['p1']['macro'],1)}%</b>입니다. Top-1 cosine 상위 20%에서는 각각 <b>{pct(high_sim['micro'],1)}%</b>와 <b>{pct(high_sim['macro'],1)}%</b>입니다.</p></article>
<article><div class="eyebrow">H2 · P2 Relationship Degradation</div><p>Interior Semantic Pair 역전율은 P2 <b>{pct(ri['same_different']['violation']['P2']['micro'])}%</b>, Teacher Margin 상위 20%에서 <b>{pct(high['violation']['P2']['micro'])}%</b>입니다. 가까운 6–10위와 비교하면 <b>{pct(near['same_different']['violation']['P2']['micro'])}%</b>입니다.</p></article>
<article><div class="eyebrow">P4 Recommendation</div><p><b>현재 Pair 규칙으로는 P4 학습 보류.</b> 가까운 경쟁 후보의 의미적 신뢰도가 확보되는 경우에 한해 좁은 범위의 P4 비교 실험을 고려할 근거가 있습니다.</p></article></div></section>
<section id="protocol"><h2>2. Dataset &amp; Retrieval Protocol</h2>
<p>기존 진단의 ADE20K Train 이미지 <b>{len(d['image_ids']):,}장</b>과 저장된 Feature Cache를 읽었습니다. 111개 공통 클래스에서 두 구획 합계 <b>{d['All']['patches']:,}개</b> Patch가 {d['All']['images_with_sampled_patches']:,}장에 분포했습니다. 이미지당 합산 관측 최대는 {d['All']['max_patches_per_image']}개입니다. 원래 Pool 제한은 이미지·클래스·구획당 1개, 클래스·구획당 최대 64장입니다. 두 구획 모두 최소 8장에 나타난 클래스가 포함됐습니다.</p>
{dataset(d)}
<p>Full RGB384는 원본 전체 이미지의 짧은 변을 384로 맞추는 기존 ADE 전처리입니다. 가로세로 비율을 유지하고 우·하단을 Patch16 배수로 Padding합니다. 최종 LN의 Patch Token에서 CLS와 Register를 제외했고, 캐시의 FP32 L2 정규화를 검증했습니다. GT가 128픽셀 이상인 Patch에서 Padding과 Ignore를 빼고 최빈 클래스 비율을 계산했습니다. Interior는 ≥90%, Mixed는 50–90% 미만입니다.</p>
<p>각 구획의 모든 저장 Patch가 Query이자 Candidate입니다. 같은 이미지에서 나온 모든 후보를 점수 계산 후 순위를 매기기 전에 제거합니다. 동일 점수는 저장 Row 순서로 정렬합니다. 같은 후보와 위치를 세 모델에 적용합니다. GT는 기존 샘플을 층화한 기록과 진단 정답 판정에만 사용했습니다. 검색과 Teacher Triplet 선택은 Cosine 점수와 사전에 고정한 Rank 규칙을 따릅니다.</p>
<p>Frequency Random은 같은 이미지 후보를 뺀 Patch Pool의 균등 비복원 순열입니다. Class-balanced Random은 적격 클래스를 균등 선택한 뒤 그 클래스 내 Patch를 균등 선택하는 독립 추출입니다. 둘 다 기대값을 정확하게 계산했습니다. Micro는 Query 또는 Pair 수, Macro는 지원 표본이 있는 각 클래스의 평균을 동일 가중합니다. All은 Interior·Mixed를 따로 검색한 관측을 합친 집계입니다.</p>
</section>
<section id="teacher"><h2>3. Teacher Semantic Reliability</h2>
<p>Original은 Interior <b>{pct(teacher['p1']['micro'])}%</b>, Mixed <b>{pct(mixed['retrieval']['Original']['p1']['micro'])}%</b> Top-1 Precision을 기록했습니다. Frequency Random Micro는 각각 {pct(interior['retrieval']['Frequency random']['p1']['micro'])}%, {pct(mixed['retrieval']['Frequency random']['p1']['micro'])}%이고, Class-balanced Random은 두 구획 모두 {pct(interior['retrieval']['Class-balanced random']['p1']['micro'])}%입니다. 모든 Query에 동일 클래스의 다른 이미지 후보가 존재했습니다.</p>
<p class="muted">표에서 슬래시 앞은 Micro, 뒤는 Macro입니다. Precision@k는 상위 k개 중 같은 클래스의 비율, Hit@k는 상위 k개에 같은 클래스가 한 개 이상 있는 Query 비율, MRR은 전체 적격 순위에서 첫 같은 클래스의 역순위 평균입니다. 지원 클래스와 실제 후보 수는 위 표에 있습니다.</p>
{retrieval(d)}
<h3>유사도와 Rank Gap의 신뢰도</h3><p>Top-1 cosine 상위 20%에서 Interior 정확도는 {pct(high_sim['micro'])}% Micro와 {pct(high_sim['macro'])}% Macro였습니다. Mixed의 동일 구간은 {pct(mixed_sim['micro'])}%와 {pct(mixed_sim['macro'])}%였습니다. Rank 1–5/1–10 Gap도 상위 구간에서 정답률이 높았고, 아래 표에서 구간별 수치와 지원 클래스를 확인할 수 있습니다. 신뢰도는 클래스와 Purity에 따라 달랐습니다.</p>
{confidence_chart(d)}{class_table(d)}
</section>
<section id="student"><h2>4. Student Neighbor Retention</h2>
<p>P2의 Interior Exact Top-10 Overlap은 {pct(all_overlap)}%였습니다. Teacher의 같은 클래스 Top-10 ID {replacement['teacher_hits']:,}개 중 {replacement['retained_teacher_hits']:,}개가 남았고, P2가 새로 찾은 같은 클래스 ID는 {replacement['new_same_class_hits']:,}개였습니다. 신규 같은 클래스 Hit 수는 사라진 Teacher Hit 수의 {pct(replacement['new_per_lost'],1)}%였습니다. 이 값은 서로 다른 ID 사이의 일대일 대응을 가정하지 않은 총계입니다.</p>
<p>Teacher Top-1이 정답인 Interior Query에서 P2는 같은 Patch를 {pct(relative['same_patch']['micro'])}%, 다른 Same-class Patch를 {pct(relative['new_same_class']['micro'])}%, 다른 Class를 {pct(relative['different_class']['micro'])}% 선택했습니다. Teacher가 오답이던 Query의 {pct(relative['corrected_teacher_error']['micro'])}%에서는 P2가 정답을 찾았습니다. Interior P@10은 Original {pct(teacher['p10']['micro'])}%, P2 {pct(p2['p10']['micro'])}%입니다.</p>
<p class="muted">Exact ID Overlap은 |Teacher Top-k ∩ Student Top-k|/k입니다. Same-class ID 유지율의 Query 평균은 Teacher Hit이 하나 이상인 Query에 한해 계산합니다. Pooled는 해당 ID의 총 유지 수를 Teacher Hit 총수로 나눈 값입니다. Top-1 정정률의 분모는 Teacher가 틀린 Query입니다.</p>
{retention(d)}
</section>
<section id="ranking"><h2>5. Ranking Violation</h2>
<p>각 Query의 Teacher 1–5위 j마다 하위 Rank 구간 6–10, 11–50, 51–200, 201–N에서 k를 하나씩 균등 추출했습니다. 같은 Pair {d['All']['ranking']['all']['n']:,}개를 세 모델에 적용했습니다. 두 후보 모두 Query 이미지와 다릅니다. Teacher margin M<sub>T</sub> = cos(T<sub>i</sub>, T<sub>j</sub>) − cos(T<sub>i</sub>, T<sub>k</sub>)가 양수이고 Student margin이 0 이하일 때 역전으로 셉니다. Teacher 동점 {d['All']['ranking']['all']['ties_excluded']}개는 분모에서 제외했습니다.</p>
<p>Interior의 Same→Different를 Semantic-relevant로 정의했습니다. j는 Query와 같은 클래스이고 k는 다른 클래스입니다. Same→Same과 Different→Different는 이 GT 기준으로 의미 있는 우선순위를 보장하지 않습니다. Different→Same은 Teacher가 클래스 기준으로 반대로 배치한 경우입니다. Mixed 전체는 Label 혼합 때문에 불확실하며, 표의 Same→Different는 Dominant-label Proxy입니다.</p>
<div class="callout"><p>Interior Semantic Pair {all_count:,}개의 P2 역전율은 <b>{pct(ri['same_different']['violation']['P2']['micro'])}%</b>입니다. 전체 Interior Margin 상위 20%의 기준은 M<sub>T</sub> ≥ {high_cutoff:.4f}이며, 그중 Semantic Pair {high_count:,}개에서 <b>{pct(high['violation']['P2']['micro'])}%</b>가 역전됐습니다. 상위 Margin Pair의 {pct(ri['high_margin_band_shares'][-1],1)}%는 하위 후보가 201위 이후입니다.</p>
<p>하위 후보를 6–10위로 제한하면 그 구간에서 계산한 P80 Margin은 {near['band_p80']:.4f}입니다. 해당 상위 구간의 Semantic Pair {near_high['n']:,}개에서도 P2 역전율은 <b>{pct(near_high['violation']['P2']['micro'])}%</b>였습니다. 가까운 순위에서 역전이 더 많다는 관측은 전체 상위 Margin의 낮은 역전율과 양립합니다.</p></div>
<p class="muted">슬래시는 Micro / Macro입니다. P80과 5분위는 표본의 분포를 설명하기 위한 고정 집계입니다. 전체 구획 기준 P80과 각 Rank 구간의 P80은 서로 다른 분모와 절대 Margin을 사용합니다.</p>
{violation(d, ar)}
<h3>실제 이미지 사례</h3><p>각 사례는 설명된 조건을 만족하는 집합에서 Teacher Margin 또는 Top-1 cosine이 그 집합 중앙값에 가장 가까운 관측입니다. 오른쪽은 표시한 Patch를 중심으로 확대한 RGB 문맥입니다. 전체 성능의 대표 표본으로 해석하지 않습니다.</p>
{example_patches(d, ar, ade_root)}
</section>
<section id="joint"><h2>6. Joint Interpretation</h2>
{ranks}
<p class="muted">AUC와 ADE20K mIoU는 기존 Dense Diagnosis의 고정 Train Pair 평가와 기존 Val Linear Probe 결과입니다. Retention과 Violation은 이번에 계산했습니다. Original의 100% 및 0%는 같은 Teacher 순위에 대한 자기 비교입니다. Mixed Proxy는 Interior Semantic 역전율과 구분합니다.</p>
<h3>직접 관측된 사실</h3><p>Teacher 이웃은 두 Random 기준보다 클래스 정보를 훨씬 많이 담았습니다. P2는 B0보다 검색 Precision과 Teacher 순위 유지가 높았고, 여전히 Original보다 낮았습니다. 의미 관련 관계의 역전은 가까운 후보에서 집중됐고, 전체 Margin 상위 구간에서는 드물었습니다. 단일 공통 직교 회전은 정규화 Cosine과 모든 Margin을 보존합니다. 따라서 관측된 양의 Teacher Margin 역전은 그러한 회전만으로 설명되지 않습니다.</p>
<h3>분석과 일치하는 해석</h3><p>Case 1과 Case 2가 섞여 있습니다. P2는 일부 실제 클래스 구별 관계를 잃었지만, Teacher ID를 바꾸면서 같은 클래스 Hit을 회복하는 경우가 많고 큰 Margin의 관계는 대부분 유지했습니다. 한편 높은 Teacher 유사도에서도 클래스 오류가 남아 있어 Case 3의 우려가 특히 Mixed와 일부 클래스에서 적용됩니다. Teacher 순위 전부를 복제하면 이미 맞는 관계와 Teacher 오답에도 손실을 줄 수 있습니다.</p>
<p>Original 대비 P2의 AUC와 ADE20K 격차는 실제로 남아 있습니다. 이번 분석은 그중 일부가 가까운 이미지 간 경쟁 후보의 순위에 나타난다는 근거입니다. 이 관계를 맞추면 ADE20K가 회복되는지, Crop Distillation의 PACO 이득을 유지하는지는 학습 비교가 필요합니다. Candidate Pool은 111개 클래스와 구획별 최대 64장으로 제한된 진단 집합이므로, 비균형 COCO 학습 Pool이나 제외한 클래스, Pattern·Texture 검색에 이 Precision을 그대로 적용할 수 없습니다.</p>
</section>
<section id="p4"><h2>7. P4 Recommendation</h2>
<div class="callout"><p><b>전역 Neighbor Ordering은 우선순위가 낮습니다.</b> 가까운 후보만 다루는 Teacher Margin 보존은 조건부로 한 번 검증할 근거가 있습니다. 이 진단만으로 P4의 학습 효과가 확인되지는 않았습니다.</p></div>
<ol><li><b>지지 근거</b> · 6–10위의 구간 상위 Margin에서도 의미 관련 역전이 {pct(near_high['violation']['P2']['micro'])}% 남았습니다. Student에서 가까운 의미 경쟁 후보를 구별하는 데 개선 여지가 있습니다.</li>
<li><b>반대 근거</b> · 전체 상위 Margin의 의미 관련 역전은 {pct(high['violation']['P2']['micro'])}%였습니다. 가까운 구간에서 Teacher Top-1 cosine과 Margin을 모두 P80 이상으로 제한해도 Same→Different는 {near_joint_sem['n']}/{near_joint['all']['n']} ({pct(near_joint['all']['same_different_share']['micro'])}%)였고, Same→Same은 {near_joint['relation_counts']['same→same']}개였습니다. 그 의미 관련 Pair 중 P2가 뒤집은 비율은 {pct(near_joint_sem['violation']['P2']['micro'])}%였습니다. 높은 Teacher 신뢰도는 가까운 Negative의 클래스 차이를 보장하지 않습니다.</li>
<li><b>최소 Loss 조건</b> · 검증한다면 P2를 기준으로, 다른 COCO 이미지에서 Original Teacher가 높은 신뢰도로 선택한 가까운 경쟁 후보에만 하나의 Margin 보존 항을 사용합니다. Student가 요구 Margin을 이미 충족한 관계는 벌점 없이 두는 단측 형태를 가정할 수 있습니다. 예를 들어 고정 비율 α에 대해 max(0, α·M<sub>T</sub> − M<sub>S</sub>)입니다. Confidence와 Rank 규칙 및 가중치는 COCO Train의 Teacher 점수로 사전에 정합니다. 이 보고서의 ADE 라벨, ID, Pair 파일은 COCO Pair 선택과 Loss 계산에 연결하지 않습니다.</li>
<li><b>최소 비교 실험</b> · 먼저 ADE Train의 오프라인 진단에서 라벨 비사용 Confidence 규칙의 의미 관련 비율과 지원 수를 확인합니다. 충분할 때 동일 초기화·데이터·예산의 P2와 국소 Margin 항 하나를 추가한 조건을 비교합니다. 가까운 Semantic 역전율과 기존 Train AUC, ADE Official Val, PACO Val Object/Part Pattern을 함께 확인합니다. Encoder는 표준 DINOv3-B/16과 Full RGB384 한 번 인코딩을 유지합니다.</li></ol>
</section>
<section id="provenance"><h2>재현 정보</h2><p>이미지 추출 Seed {d['fixed_protocol']['image_seed']}, 기존 Cache 후보 Seed {d['fixed_protocol']['cache_seed']}, 새 Rank Pair Seed {d['fixed_protocol']['pair_seed']}입니다. 체크포인트 SHA-256, Cache 해시와 전처리 코드 해시를 확인했습니다. 기존 Cache만 사용해 CPU에서 계산했습니다.</p>
{details('샘플 이미지 ID 전체','<pre>'+esc(json.dumps(d['image_ids'],indent=1))+'</pre>')}
{details('프로토콜, 코드 버전, Checkpoint 및 Cache Hash','<pre>'+esc(json.dumps({'protocol':d['fixed_protocol'],'provenance':prov},indent=2))+'</pre>')}
<p><a href="metrics.json">정확한 수치 JSON</a> · <a href="patch_bank.csv">Query/Candidate Patch ID와 클래스</a> · <a href="retrieval_and_triplets.npz">모델별 Rank·Cosine과 공통 Triplet</a> · <a href="examples.json">사례 ID</a> · <a href="measure.py">분석 코드</a> · <a href="render.py">Figure 및 HTML Source</a></p>
</section></main><footer class="footer"><div class="wrap">Cross-image Semantic Neighbor Audit · Local HTML</div></footer><script>{JS}</script></body></html>'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ade-root',type=Path,default=Path('/home/irteam/datasets/ade20k/ADEChallengeData2016'))
    args=p.parse_args()
    d=read(OUT/'metrics.json')
    with np.load(OUT/'retrieval_and_triplets.npz') as arrays:
        html_doc=render(d, arrays, args.ade_root)
    (OUT/'report.html').write_text(html_doc,encoding='utf-8')
    print(OUT/'report.html')


if __name__=='__main__':
    main()
