"""Create the portable Korean report and its reproducible source figures."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
from PIL import Image

from infer import OUT, NAMES, ade

COLORS = ['#162b4d', '#9c5260', '#5685a2', '#1764b0', '#806cb1']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#293e59','text.color':'#172c49','axes.titleweight':'bold','axes.titlelocation':'left',
                     'figure.facecolor':'white','axes.facecolor':'white','axes.grid':True,'axes.axisbelow':True,'grid.alpha':.17,
                     'svg.fonttype':'none','svg.hashsalt':'p23-dense-diagnosis'})


def esc(s):
    return html.escape(str(s), quote=True)


def f(x, digits=3):
    return '—' if x is None else f'{x:.{digits}f}'


def signed(x, digits=3):
    return '—' if x is None else f'{x:+.{digits}f}'


def table(headers, rows, sortable=False):
    th = ''.join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = []
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, tuple):
                value, style, tip = cell
                cells.append(f'<td style="{esc(style)}" title="{esc(tip)}">{esc(value)}</td>')
            else:
                cells.append(f'<td>{esc(cell)}</td>')
        body.append('<tr>'+''.join(cells)+'</tr>')
    cls = ' class="sortable"' if sortable else ''
    return f'<div class="scroll"><table{cls}><thead><tr>{th}</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'


def inline_image(path, alt):
    mime = {'.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg'}[path.suffix]
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<img src="data:{mime};base64,{data}" alt="{esc(alt)}" loading="lazy">'


def save(fig, name, suffix='svg'):
    path = OUT/f'{name}.{suffix}'
    fig.savefig(path, bbox_inches='tight', dpi=142, facecolor='white')
    plt.close(fig)
    return path


def plot(fig, name, caption):
    return f'<figure class="chart">{inline_image(save(fig,name),caption)}<figcaption>{esc(caption)}</figcaption></figure>'


def details(summary, content):
    return f'<details><summary>{esc(summary)}</summary>{content}</details>'


def color_weight(delta, span=12):
    frac = min(abs(delta)/span,1) if delta is not None else 0
    target = np.array([49,111,174] if (delta or 0)>=0 else [190,75,76])
    rgb = np.rint(255-(255-target)*(.80*frac)).astype(int)
    text = '#fff' if frac>.83 else '#132b47'
    return f'background:rgb({rgb[0]},{rgb[1]},{rgb[2]});color:{text}'


def maps_cell(v, norm, cmap='Blues'):
    cmap = matplotlib.colormaps[cmap].copy()
    cmap.set_bad('#e8edf3')
    rgb = (cmap(norm(v))[:,:,:3]*255).astype(np.uint8)
    return Image.fromarray(rgb).resize((384,384), Image.Resampling.NEAREST)


def mask_panel(mask):
    arrays = np.load(OUT/'roi_stats_arrays.npz')
    union_bins = np.linspace(0,1,21)
    weight_bins = np.linspace(0,4,41)
    union_hist = np.histogram(arrays['union_fraction'],bins=union_bins)[0]
    valid = arrays['area'] > 0
    exposure = arrays['area']*arrays['appearances'][:,None]
    weight_hist = np.histogram(arrays['normalized'][valid],bins=weight_bins,weights=exposure[valid])[0]
    fig, axes = plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    axes[0].stairs(100*union_hist/union_hist.sum(),union_bins,fill=True,color='#327db3',alpha=.75)
    axes[0].axvline(mask['union_quantiles'][3],color=COLORS[0],linestyle='--',label='Median')
    axes[0].set(title='ROI union coverage',xlabel='Fraction of image content',ylabel='Images per bin (%)')
    axes[0].legend(frameon=False)
    axes[1].stairs(100*weight_hist/weight_hist.sum(),weight_bins,fill=True,color='#327db3',alpha=.75)
    axes[1].axvline(1,color=COLORS[0],linestyle='--',label='P2 weight = 1')
    axes[1].set(title='P3 normalized weights',xlabel='Weight',ylabel='Valid area per bin (%)')
    axes[1].legend(frameon=False)
    output = plot(fig,'roi_distributions','이미지별 ROI Union 분포와 노출 횟수·유효 면적으로 가중한 P3 가중치 분포')
    hist_tables = table(['Coverage 구간','이미지 수'],[(f'{l:.2f}–{r:.2f}',int(n)) for l,r,n in zip(union_bins,union_bins[1:],union_hist)])
    hist_tables += table(['P3 Weight 구간','면적 비율 %'],[(f'{l:.1f}–{r:.1f}',f(100*n/weight_hist.sum(),2)) for l,r,n in zip(weight_bins,weight_bins[1:],weight_hist)])
    output += details('히스토그램 Bin별 수치',f'<div class="columns">{hist_tables}</div>')
    group = ['inside','boundary','outside']
    fig, ax = plt.subplots(figsize=(6.4,2.9),layout='constrained')
    bars = ax.bar(['ROI interior','ROI mixed cell','Outside ROI'],[mask['weight_by_group'][x] for x in group],color=['#446d9a','#7cadcf','#1f5990'])
    ax.bar_label(bars,fmt='%.3f',padding=3)
    ax.axhline(1,ls='--',lw=1.3,c=COLORS[0],label='P2 weight = 1')
    ax.set(ylabel='Mean normalized weight',ylim=(0,1.5))
    ax.legend(frameon=False)
    groups = table(['ROI 구획','Patch 수 %','유효 면적 %','P3 평균 Weight'],[(x,f(100*mask['fraction_by_patch_count'][x],2),f(100*mask['fraction_by_valid_area'][x],2),f(mask['weight_by_group'][x])) for x in group])
    output += f'<div class="columns">{plot(fig,"roi_groups","P3 영역별 가중치 평균")}{groups}</div>'
    output += table(['조건','Min','Mean','Max','Std','이미지별 Std 중앙값'],[(k,*[f(v[c]) for c in ('min','mean','max','std','median_image_std')]) for k,v in mask['weight_stats'].items()])
    images = []
    for example in mask['examples']:
        iid = example['image_id']
        with np.load(OUT/f'roi_{iid}_mask.npz') as masks:
            paths = [OUT/f'roi_{iid}_boxes.jpg']
            for key,max_value in [('union',1),('weight',4)]:
                path = OUT/f'roi_{iid}_{key}.png'
                vals = np.where(masks['valid']>0,masks[key],np.nan)
                maps_cell(vals,Normalize(0,max_value)).save(path)
                paths.append(path)
        pictures = ''.join(f'<figure>{inline_image(p,k)}<figcaption>{k}</figcaption></figure>' for p,k in zip(paths,['ROI boxes','Union 0–1','P3 Weight 0–4']))
        lo,hi = example['weights_range']
        images.append(f'<div class="case"><b>COCO {iid}</b><p class="muted">Union {100*example["coverage"]:.1f}% · Weight {lo:.2f}–{hi:.2f} · {esc(", ".join(example["kinds"]))}</p><div class="images three">{pictures}</div></div>')
    output += '<h3>실제 학습 이미지의 ROI와 가중치</h3><p class="muted">Union 분위수 10·40·70·95% 사례. 노랑 GT, 하늘색 Random, 분홍 fallback. 회색은 Padding입니다.</p><div class="columns">'+''.join(images)+'</div>'
    gradients = mask['gradients']
    fig, ax = plt.subplots(1,2,figsize=(10.8,3.15),layout='constrained')
    for name,col in [('P2',COLORS[3]),('P3',COLORS[4])]:
        rows = [r for r in gradients if r['model']==name and r['step']>1]
        ax[0].plot([r['step'] for r in rows],[r['weighted'] for r in rows],label=name,color=col,marker='o')
        ax[1].plot([r['step'] for r in rows],[r['cosine'] for r in rows],label=name,color=col,marker='o')
    for a,title,unit in zip(ax,['Weighted preservation gradient','Crop vs preservation gradient'],['Encoder norm','Cosine']):
        a.set(title=title,ylabel=unit,xlabel='Optimizer step',xticks=[6000,12000])
        a.legend(frameon=False)
    ax[1].axhline(0,color='#a4b3c2',lw=1)
    output += '<h3>실제 학습 Gradient</h3>'+plot(fig,'gradient_trace','보존 손실의 Encoder gradient norm과 Crop gradient와의 cosine')
    output += table(['모델','Step','Crop norm','Preservation raw','λ × Preservation','Ratio','Cosine'],
                    [(r['model'],r['step'],f(r['crop'],4),f'{r["raw"]:.4g}',f'{r["weighted"]:.4g}',f'{r["ratio"]:.4g}',f(r['cosine'],4)) for r in gradients])
    return output


def drift_panel(data):
    with np.load(OUT/'train_features.npz') as ar:
        stats = ar['image_stats']
    fig, ax = plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for a,idx,title,unit in [(ax[0],0,'Patch direction similarity','Cosine to Original'),(ax[1],2,'Within-image Gram drift','Weighted Gram MSE')]:
        boxes = a.boxplot([stats[:,i,idx] for i in range(1,4)],tick_labels=NAMES[1:],whis=(5,95),showfliers=False,patch_artist=True)
        for b,color in zip(boxes['boxes'],COLORS[1:4]):
            b.set(facecolor=color,alpha=.68,edgecolor='#234568')
        for median in boxes['medians']:
            median.set(color='#0d2848',linewidth=2)
        a.set(title=title,ylabel=unit)
    output = plot(fig,'feature_drift','ADE Train 이미지별 분포. 상자는 25–75%, 수염은 5–95% 분위수')
    rows = []
    for n in NAMES:
        for key in ('patch_cosine','normalized_l2','gram_mse'):
            v = data['drift'][n][key]
            rows.append((n,key,f(v['mean'],5),*[f(q,5) for q in v['quantiles']]))
    output += table(['모델','Metric','평균','P05','P25','중앙값','P75','P95'], rows)
    roi = data['reused_coco_roi_drift']
    output += '<h3>ROI가 있는 COCO Dev의 Feature Drift</h3><p>기존 COCO Dev 결과를 별도로 읽었습니다. ADE20K Train에는 동일한 Crop Distillation ROI가 없으므로 위 ADE 통계와 직접 합산하지 않습니다. ROI 안팎은 각 패치의 Union 면적과 그 여집합으로 나눕니다.</p>'
    output += table(['모델','내부 1−cos','외부 1−cos','Gram MSE'],[(n,*[f(roi['mean'][n][k],5) for k in ('roi_inside_distance','roi_outside_distance','gram_drift')]) for n in (*NAMES,'P3')])
    examples = sorted(OUT.glob('train_case_ADE*.npz'))
    vmax = max(np.nanquantile(np.load(path)['drift'][1:],.99) for path in examples)
    output += f'<h3>동일 이미지의 Spatial Drift</h3><p class="muted">색은 모든 예시에서 동일한 1−cos 범위 0–{vmax:.3f}를 사용합니다. 표시 상한을 넘는 패치는 가장 진하게 표시합니다. 회색은 유효 면적이 없는 패치입니다.</p>'
    for path in examples:
        iid = path.stem.removeprefix('train_case_')
        img = Image.open(Path(ade.ROOT)/'images/training'/f'{iid}.jpg').convert('RGB')
        with np.load(path) as row:
            cov = row['coverage']
            res = img.resize((round(img.width*384/min(img.size)),round(img.height*384/min(img.size))),Image.Resampling.BILINEAR)
            canvas = Image.new('RGB',(cov.shape[1]*16,cov.shape[0]*16),'#e8edf3')
            canvas.paste(res)
            canvas.thumbnail((360,300))
            rgb = OUT/f'drift_{iid}_rgb.jpg'
            canvas.save(rgb,quality=86)
            figures = [(rgb,'RGB')]
            for i,n in enumerate(NAMES[1:],1):
                p = OUT/f'drift_{iid}_{n}.png'
                z = np.where(cov>0,row['drift'][i],np.nan)
                cmap = matplotlib.colormaps['Blues'].copy()
                cmap.set_bad('#e8edf3')
                heat = Image.fromarray((cmap(Normalize(0,vmax)(z))[:,:,:3]*255).astype(np.uint8))
                heat.resize(canvas.size,Image.Resampling.NEAREST).save(p)
                figures.append((p,f'{n} · {np.average(row["drift"][i],weights=cov):.3f}'))
        output += image_case(iid,figures,'four')
    return output


def image_case(title, items, cls, extra=''):
    imgs = ''.join(f'<figure>{inline_image(p,t)}<figcaption>{esc(t)}</figcaption></figure>' for p,t in items)
    return f'<div class="case"><b>{esc(title)}</b>{extra}<div class="images {cls}">{imgs}</div></div>'


def semantic_panel(data):
    semantic = data['semantic']
    fig, axes = plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    for ax,region,title in zip(axes,['interior','mixed'],['Interior patch pairs','Mixed-label patch pairs']):
        series = data['semantic_distributions'][region]
        for i,n in enumerate(NAMES):
            for pair,style in [('within','-'),('between','--')]:
                ax.stairs(100*np.array(series[n][pair]),series['edges'],color=COLORS[i],linestyle=style,
                          label=n if pair=='within' else None)
        ax.set(title=title,xlabel='Cross-image patch cosine',ylabel='Class-balanced mass per bin (%)')
        ax.legend(frameon=False,fontsize=8)
    output = plot(fig,'semantic_pair_distributions','실선은 Same class, 점선은 Different class. 같은 패치 쌍을 모든 모델에 사용')
    rows = []
    for r in ('interior','mixed'):
        for n in NAMES:
            q = semantic[r][n]
            rows.append((r,n,f(q['within'],4),f(q['between'],4),f(q['separation'],4),f(q['auc'],4),f(data['drift'][n][r+'_cosine'],4)))
    output += table(['Train patch 구획','모델','Within-class cosine','Between-class cosine','Within−Between','Macro ROC-AUC','원본 대비 Cosine'],rows)
    for region in ('interior','mixed'):
        hist = data['semantic_distributions'][region]
        headers = ['Cosine bin'] + [f'{n} {pair} %' for n in NAMES for pair in ('within','between')]
        values = [(f'{left:.3f}–{right:.3f}',*[f(100*hist[n][pair][i],2) for n in NAMES for pair in ('within','between')])
                  for i,(left,right) in enumerate(zip(hist['edges'],hist['edges'][1:]))]
        output += details(f'{region} 분포의 Bin별 수치',table(headers,values))
    output += details('클래스별 Semantic Separation과 Val IoU',table(['클래스','구획','Train 이미지 수 (pool)','Original AUC','B0 AUC','P1 AUC','P2 AUC','P2−B0 IoU pp'],[
        (data['class_rows'][q['class_id']]['name'],r,q['n_images'],*[f(semantic[r][n]['classes'][i]['auc'],4) for n in NAMES],signed(data['class_rows'][q['class_id']]['b0_to_p2'],2))
        for r in ('interior','mixed') for i,q in enumerate(semantic[r]['P2']['classes'])],sortable=True))
    return output


def top_tables(data):
    classes = data['class_rows']
    damaged = sorted(classes,key=lambda q:q['delta_from_original']['B0'])[:10]
    recovered = sorted([q for q in classes if q['delta_from_original']['B0']<0 and q['b0_to_p2']>0],key=lambda q:-q['b0_to_p2'])[:10]
    residual = sorted(classes,key=lambda q:q['delta_from_original']['P2'])[:10]
    fig, axes = plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for a,subset,title,key,col in [(axes[0],damaged,'Largest B0 losses','loss','#b36066'),(axes[1],recovered,'Largest P2 recoveries','gain','#367cb4')]:
        delta = [q['delta_from_original']['B0'] if key=='loss' else q['b0_to_p2'] for q in subset]
        bars = a.barh([q['name'] for q in subset],delta,color=col,alpha=.87)
        a.invert_yaxis()
        a.bar_label(bars,labels=[signed(x,1) for x in delta],padding=3,fontsize=8)
        a.margins(x=.23)
        a.set(title=title,xlabel='Class IoU change (pp)')
    output = plot(fig,'class_extremes','B0 손상 순위와 B0에서 하락한 클래스 중 P2 회복 순위')
    headers = ['클래스','GT 이미지','Original','B0','P1','P2','B0−Original','P2−B0','P2−Original']
    def rows(subset):
        return [(q['name'],q['gt_images'],*[f(q['iou'][n],2) for n in NAMES],signed(q['delta_from_original']['B0'],2),signed(q['b0_to_p2'],2),signed(q['delta_from_original']['P2'],2)) for q in subset]
    for title,subset in [('B0에서 가장 많이 손상된 클래스 10개',damaged),('P2에서 가장 많이 회복된 클래스 10개',recovered),('P2에 가장 큰 손실이 남은 클래스 10개',residual)]:
        output += f'<h4>{title}</h4>'+table(headers,rows(subset))
    heat = []
    for q in sorted(classes,key=lambda x:x['delta_from_original']['B0']):
        deltas = [q['delta_from_original'][n] for n in ('B0','P1','P2')]+[q['b0_to_p2']]
        heat.append([q['name'],q['group'],q['gt_images'],*[f(q['iou'][n],2) for n in NAMES],
                     *[(signed(x,2),color_weight(x),f'{q["aliases"]} · Δ {signed(x,5)} pp · {q["gt_pixels"]:,} GT pixels') for x in deltas]])
    output += details('150개 클래스 IoU 변화 Heatmap · 열 머리글을 눌러 정렬',table(['클래스','구분','GT 이미지',*NAMES,'B0−Original','P1−Original','P2−Original','P2−B0'],heat,True))
    return output


def pixel_regions_chart(data):
    fig, axes = plt.subplots(1,2,figsize=(10.8,3.2),layout='constrained')
    for ax,key,title in zip(axes,['pixel_accuracy','miou'],['Pixel accuracy','Region-restricted class mIoU']):
        for i,name in enumerate(NAMES):
            values = [data['regions'][r][name][key] for r in ('boundary','interior')]
            ax.bar(np.arange(2)+(i-1.5)*.18,values,width=.17,color=COLORS[i],label=name)
        ax.set(title=title,ylabel='Percent',xticks=[0,1],xticklabels=['Boundary','Interior'])
        ax.legend(frameon=False,fontsize=8)
    return plot(fig,'boundary_interior','Val의 동일 GT 경계/내부 픽셀에서 집계한 Pixel Accuracy와 보조 mIoU. 정확한 수치는 아래 표')


def confusion_panel(data):
    pairs = data['hard_pairs']
    classes = list(dict.fromkeys(c for p in pairs[:8] for c in p['class_ids']))
    names = [data['class_rows'][i]['name'] for i in classes]
    with np.load(OUT/'val_confusion.npz') as ar:
        c = ar['confusion'][:,0]
    rates = c / np.maximum(c.sum(-1,keepdims=True),1) * 100
    fig, axes = plt.subplots(1,2,figsize=(11.5,5.1),layout='constrained')
    for a,i in zip(axes,[1,3]):
        values = rates[i][np.ix_(classes,classes)].copy()
        np.fill_diagonal(values,np.nan)
        pic = a.imshow(values,cmap='Blues',vmin=0,vmax=max(5,np.nanquantile(rates[1][np.ix_(classes,classes)][~np.eye(len(classes),dtype=bool)],.98)),interpolation='nearest')
        a.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,title=NAMES[i],xlabel='Predicted class',ylabel='GT class')
        plt.setp(a.get_xticklabels(),rotation=55,ha='right',fontsize=7)
        plt.setp(a.get_yticklabels(),fontsize=7)
        a.grid(False)
    fig.colorbar(pic,ax=axes,shrink=.68,label='GT-row error rate (%)')
    output = plot(fig,'confusion_b0_p2','B0 실제 혼동 상위 8쌍에 등장하는 클래스의 행 정규화 혼동률. 대각선은 숨김')
    rows = []
    for q in pairs:
        ratestr = [f'{f(q["directions"][n][0],1)} / {f(q["directions"][n][1],1)}' for n in NAMES]
        def aucstr(region):
            v = q['train_auc'].get(region,{})
            return f'{f(v.get("B0"),3)} → {f(v.get("P2"),3)}'
        rows.append([' ↔ '.join(q['labels']),*ratestr,aucstr('interior'),aucstr('mixed')])
    output += table(['Class A ↔ B','Original %','B0 %','P1 %','P2 %','Interior AUC B0→P2','Mixed AUC B0→P2'],rows)
    for m in (1,3):
        subset = rates[m][np.ix_(classes,classes)]
        output += details(f'{NAMES[m]} Heatmap 행렬 수치',table(['GT ↓ / Pred →',*names],[(n,*[f(x,2) if i!=j else '—' for j,x in enumerate(row)]) for i,(n,row) in enumerate(zip(names,subset))]))
    return output


def segmentation_panel(data):
    palette = np.full((256,3),214,np.uint8)
    for i in range(150):
        hue = (.61803398875*i)%1
        palette[i] = (255*matplotlib.colors.hsv_to_rgb([hue,.64,.84 if i%2 else .72])).astype(np.uint8)
    cases = json.loads((OUT/'val_cases.json').read_text())
    output = ''
    for item in cases.values():
        iid = item['image_id']
        img = Image.open(Path(ade.ROOT)/'images/validation'/f'{iid}.jpg').convert('RGB')
        img.thumbnail((330,270))
        rgb = OUT/f'seg_{iid}_rgb.jpg'
        img.save(rgb,quality=86)
        figures = [(rgb,'RGB')]
        with np.load(OUT/f'val_case_{iid}.npz') as masks:
            ignore = masks['label'] == ade.IGN
            counts = np.bincount(masks['label'][~ignore],minlength=150)
            for n in ('label',*NAMES):
                arr = masks[n].copy()
                arr[ignore] = ade.IGN
                p = OUT/f'seg_{iid}_{n}.png'
                Image.fromarray(palette[arr]).resize(img.size,Image.Resampling.NEAREST).save(p)
                figures.append((p,'GT' if n=='label' else n))
            colors = []
            for c in np.argsort(-counts)[:8]:
                if counts[c] == 0:
                    continue
                code = '#'+''.join(f'{x:02x}' for x in palette[c])
                colors.append(f'<span class="swatch" style="--swatch:{code}">{esc(data["class_rows"][c]["name"])}</span>')
        desc = f'<p class="muted">{esc(item["reason"])} · 공통 Union 클래스 {item["common_union_classes"]}개 · 이미지 Macro IoU Original / B0 / P2 {" / ".join(f(x,1) for x in item["macro_iou"])}</p>'
        output += image_case(iid,figures,'six',desc)+'<div class="legend">'+''.join(colors)+'</div>'
    return output


CSS = '''
:root{color-scheme:light;--ink:#152d4d;--muted:#536980;--line:#d9e4f0;--blue:#1e6199;--pale:#eef6ff;--white:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:white;color:var(--ink);font:15px/1.67 "Gothic A1",Inter,system-ui,"Malgun Gothic",sans-serif}
a{color:#16588d;text-decoration-thickness:1px;text-underline-offset:3px}header{border-top:7px solid #102b4d;background:linear-gradient(135deg,#f3f8ff,white 75%);border-bottom:1px solid var(--line)}
.wrap{max-width:1260px;margin:auto;padding:0 28px}header .wrap{padding-top:40px;padding-bottom:32px}.eyebrow{letter-spacing:.12em;text-transform:uppercase;color:#31618e;font-weight:700;font-size:.77rem}
h1{font-size:clamp(2rem,5vw,3.45rem);line-height:1.1;letter-spacing:-.043em;max-width:1000px;margin:.35em 0 .32em}h2{font-size:1.72rem;line-height:1.25;margin:0 0 17px;border-bottom:2px solid #bed7ef;padding-bottom:12px}h3{font-size:1.21rem;margin:29px 0 8px}h4{font-size:1rem;margin:17px 0 4px}p{margin:.56rem 0 1rem}.lede{max-width:920px;font-size:1.08rem;color:#364f6c}nav{display:flex;gap:18px;flex-wrap:wrap;padding:13px 0;font-size:.87rem}nav a{font-weight:650}section{padding:30px 0 20px;scroll-margin-top:14px}main{padding-bottom:40px!important}.muted,figcaption{color:var(--muted);font-size:.87rem}.columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start}.case{border:1px solid var(--line);border-radius:10px;padding:13px;margin:12px 0;background:white;break-inside:avoid}.case p{margin:0 0 8px}.images{display:grid;gap:8px;align-items:start;margin-top:8px}.three{grid-template-columns:repeat(3,minmax(0,1fr))}.four{grid-template-columns:repeat(4,minmax(0,1fr))}.six{grid-template-columns:repeat(6,minmax(0,1fr))}
img{width:100%;height:auto;display:block;border-radius:4px}figure{margin:0;min-width:0}figcaption{text-align:center;margin:5px 0 2px}.chart{padding:11px 10px 8px;border:1px solid var(--line);border-radius:9px;margin:13px 0;background:white;break-inside:avoid}.chart img{max-height:560px;object-fit:contain}
.scroll{overflow-x:auto;margin:10px 0 16px;border:1px solid var(--line);border-radius:8px}table{width:100%;border-collapse:collapse;font-size:.84rem;line-height:1.42;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;white-space:nowrap;text-align:right;border-bottom:1px solid #e5edf6}th:first-child,td:first-child{text-align:left}th{background:#eaf3fc;color:#153a61;font-weight:700}tbody tr:nth-child(even){background:#f7fafc}tbody tr:hover{background:#e8f3fe}.sortable th{cursor:pointer}.sortable th:focus-visible{outline:2px solid var(--blue);outline-offset:-2px}.sortable th[aria-sort=ascending]::after{content:" ↑"}.sortable th[aria-sort=descending]::after{content:" ↓"}
.callout{background:var(--pale);border-left:4px solid #3781bb;padding:14px 17px;margin:16px 0;border-radius:0 8px 8px 0}.callout p:last-child{margin-bottom:0}.label{font-size:.78rem;font-weight:750;letter-spacing:.035em;color:#25669b}details{margin:12px 0;padding:9px 12px;border:1px solid var(--line);border-radius:8px}summary{cursor:pointer;font-weight:650}code,pre{font:12px/1.6 ui-monospace,monospace}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fc;padding:14px}.legend{display:flex;flex-wrap:wrap;gap:6px 12px;margin:-3px 0 16px}.swatch{font-size:.79rem}.swatch:before{content:"";display:inline-block;width:.84em;height:.84em;margin-right:4px;border-radius:2px;background:var(--swatch)}footer{border-top:1px solid var(--line);padding:15px 0;color:var(--muted)}
@media(max-width:850px){.columns{grid-template-columns:1fr}.six{grid-template-columns:repeat(3,minmax(0,1fr))}.wrap{padding-left:17px;padding-right:17px}}
@media print{nav{display:none}section{break-inside:auto}.case,figure,.callout{break-inside:avoid}body{font-size:11pt}.wrap{max-width:100%;padding:0}.images{gap:4px}}
'''
JS = '''document.querySelectorAll('.sortable th').forEach((th,i)=>{const table=th.closest('table');const index=th.cellIndex;th.tabIndex=0;function sort(){const asc=th.getAttribute('aria-sort')!=='ascending';table.querySelectorAll('th').forEach(x=>x.removeAttribute('aria-sort'));th.setAttribute('aria-sort',asc?'ascending':'descending');const rows=[...table.tBodies[0].rows];rows.sort((a,b)=>{const x=a.cells[index].textContent,y=b.cells[index].textContent;const p=Number(x.replaceAll(',','')),q=Number(y.replaceAll(',',''));if(x==='—')return y==='—'?0:1;if(y==='—')return -1;return (Number.isNaN(p)||Number.isNaN(q)?x.localeCompare(y):p-q)*(asc?1:-1)});rows.forEach(r=>table.tBodies[0].append(r))}th.addEventListener('click',sort);th.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();sort()}})});'''


def main():
    data = json.loads((OUT/'diagnosis_metrics.json').read_text())
    mask = json.loads((OUT/'roi_mask_audit.json').read_text())
    interpretation = json.loads((OUT/'interpretation.json').read_text())
    official = data['official']
    scores = table(['모델','PACO Overall AP','Object Pattern AP','Part Pattern AP','ADE20K mIoU','ADE Δ vs Original'],[
        (n,f(v['overall']),f(v['obj_pattern']),f(v['part_pattern']),f(v['ade']),signed(v['ade']-official['Original']['ade'])) for n,v in official.items()])
    repro = table(['모델','기존 mIoU','재추론 mIoU','차이 pp','최대 Class IoU 차이 pp','I/U가 달랐던 이미지'],[
        (n,f(x['reference'],9),f(x['rerun'],9),signed(x['delta_pp'],9),f(x['max_class_iou_delta'],9),x['images_differing']) for n,x in data['val_reproduction'].items()])
    region = table(['모델','Boundary Pixel Acc','Interior Pixel Acc','Boundary mIoU','Interior mIoU','Thing mIoU','Stuff mIoU'],[
        (n,*[f(data['regions'][r][n][k],3) for k in ('pixel_accuracy','miou') for r in ('boundary','interior')],f(data['things_stuff'][n]['Thing']),f(data['things_stuff'][n]['Stuff'])) for n in NAMES])
    checkpoint_rows = [(n,q['encoder'],q['head'],q['checksum'],q['sha256'],q['head_sha256']) for n,q in data['checkpoints'].items()]
    nav = '<nav aria-label="보고서 탐색">'+''.join(f'<a href="#{k}">{v}</a>' for k,v in [('scores','기존 성능'),('mask','A. ROI Weight'),('drift','B1. Patch Drift'),('semantic','B2. Separability'),('errors','B3. ADE 오류'),('implications','C. 결론'),('provenance','재현 정보')])+'</nav>'
    boundary_n = data['regions']['boundary']['Original']['pixels']
    interior_n = data['regions']['interior']['Original']['pixels']
    def paras(name):
        return ''.join(f'<p>{esc(p)}</p>' for p in interpretation[name])
    content = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="P2/P3 frozen DINOv3 representation and ADE20K error analysis"><title>P2/P3 Dense Representation Diagnosis</title><style>{CSS}</style></head><body>
<header><div class="wrap"><div class="eyebrow">Research report · DINOv3-B/16 · P2 / P3</div><h1>Crop Distillation 이후의<br>Dense Representation</h1><div class="lede">{paras('lead')}</div>{nav}</div></header>
<main class="wrap">
<section id="scores"><h2>기존 공식 평가</h2>{scores}<p class="muted">PACO Official Val 2,410장, GT Box와 GT Category Oracle, R2 10 epochs. ADE20K Official Val 2,000장, 동일 frozen encoder + final 40k Linear Head. 점수는 각 평가 원본에서 읽었으며, 아래 진단 수치와 구분해 표시합니다.</p></section>
<section id="mask"><h2>A. P3의 ROI 가중치는 얼마나 달랐는가</h2><div class="callout">{paras('mask')}</div>
<p>기존 12k COCO Stream의 제외 정책을 적용한 뒤 고유 이미지 {mask['stream_population']:,}장 중 {len(mask['image_ids']):,}장을 균등 비복원 추출했습니다. 집계에서는 이 이미지들이 실제 Stream에 등장한 {mask['sample_occurrences']:,}회를 반영했습니다. Histogram의 Union Coverage는 이미지별 동일 가중치, Patch Weight는 Stream 노출 횟수 × 유효 면적을 사용합니다.</p>
<p><code>m = Area(Union(ROI)∩Patch∩Content) / Area(Patch∩Content)</code>이고 <code>w_raw = 1−0.75m</code>를 이미지별 유효 면적 가중 평균으로 나눴습니다. 기존 Float64 rectangle inclusion–exclusion 구현을 그대로 호출했습니다. Stream과 ROI 좌표의 SHA-256이 학습 Manifest와 일치했고 {mask['canvas_checks']}장의 실제 학습 데이터 경로에서 좌표·Padding·마스크가 일치했습니다. 기존 sweep-line 대조 최대 면적 오차는 {mask['union_check']['max_area_error']:.2g} px²이며, 이미지별 정규화 평균의 최대 오차는 {mask['normalized_mean_max_error']:.2g}입니다.</p>
{mask_panel(mask)}<p class="muted">P2/P3 공통 λ={mask['lambda_preserve']:.10f}. Step 1에는 Teacher와 Student가 같은 초기화이므로 거의 0인 보존 Gradient의 Cosine 해석을 생략합니다. ROI 안팎 혼합 면적을 분할한 평균 Weight는 각각 {mask['fractional_inside_weight']:.3f}, {mask['fractional_outside_weight']:.3f}입니다.</p></section>
<section id="drift"><h2>B1. 최종 Patch Feature가 원본에서 얼마나 이동했는가</h2>{paras('drift')}
<p>ADE20K Train {len(data['ade_train_ids']):,}장에 기존 평가와 동일한 short-side 384, aspect ratio 유지, Patch16 배수 우·하단 Padding을 사용했습니다. 각 모델에서 같은 좌표의 최종 LN Patch Token을 사용하고 CLS/Register는 제외했습니다. 부분 Padding Patch는 실제 면적 비율만 포함합니다. L2는 L2 정규화한 벡터 간 거리이며, Gram MSE는 이미지 내 모든 정규화 Patch Pair의 cosine 행렬 차이를 유효 면적 곱으로 가중한 평균입니다.</p>{drift_panel(data)}</section>
<section id="semantic"><h2>B2. 클래스 구별 정보가 보존됐는가</h2><div class="callout">{paras('semantic')}</div>
<p>ADE20K Train 마스크를 동일 Canvas에 Nearest로 정렬하고, 유효 GT가 128픽셀 이상인 16×16 Patch에서 최빈 클래스 비율을 계산했습니다. Interior는 Purity ≥90%, Mixed는 50–90% 미만입니다. Padding과 Ignore Label 255를 모두 제외했습니다. 전체 패치는 Interior {data['train_patch_counts'][1]:,}개, Mixed {data['train_patch_counts'][2]:,}개, 기준 미달 {data['train_patch_counts'][0]:,}개였습니다.</p>
<p>두 구획 모두 8장 이상의 서로 다른 이미지에서 관측된 {len(data['eligible_train_classes'])}개 클래스를 사용했습니다. 클래스·구획당 이미지별 후보 한 개, 최대 64장입니다. 클래스당 256개의 Anchor를 고정하고 Positive는 같은 클래스, Negative는 다른 클래스를 균등 선택했습니다. Positive와 Negative는 모두 Anchor와 다른 이미지에 있습니다. ROC-AUC는 cosine으로 같은 클래스 쌍을 더 높게 매기는 정도이며, 클래스별 AUC를 동일 가중치로 평균했습니다.</p>
{semantic_panel(data)}</section>
<section id="errors"><h2>B3. ADE20K Val에서 실제로 남은 오류</h2>{paras('errors')}
<h3>기존 점수 재현</h3>{repro}<p>기존 Linear Head와 final Encoder를 사용해 동일 좌표에서 Logit을 두 번 bilinear 보간했습니다. Class IoU는 전체 2,000장의 Intersection/Union을 클래스별로 먼저 합산합니다. 위 재현 표는 저장된 원본 I/U와 다시 얻은 I/U를 직접 비교합니다.</p>
<h3>경계와 내부 · Thing과 Stuff</h3><p>원본 GT 좌표에서 유효한 4-neighbor의 라벨이 바뀌는 양쪽 픽셀을 경계로 잡고 Chebyshev 반경 2픽셀로 확장했습니다. 유효 픽셀 안에서 Boundary {boundary_n:,}개, Interior {interior_n:,}개입니다. Pixel Accuracy는 각 구획 픽셀 전체에 대한 정확도입니다. 영역 mIoU는 같은 GT 정의로 픽셀을 고른 뒤 해당 영역의 FP/FN만 합산한 보조 지표입니다. Thing/Stuff는 SceneParse150과 Instance100 간 공개 Panoptic 변환 매핑을 사용한 Class Macro 평균입니다. <a href="{data['things_stuff_mapping']['implementation']}">Panoptic 변환 구현</a> · <a href="{data['things_stuff_mapping']['mapping']}">클래스 매핑</a></p>
{pixel_regions_chart(data)}{region}{top_tables(data)}
<h3>실제로 혼동한 클래스 쌍</h3><p>B0 Val의 A→B, B→A 비율을 평균해 상위 12쌍을 골랐습니다. 각 클래스는 GT ≥10,000픽셀, ≥10장입니다. 표의 각 수치는 A→B / B→A입니다. AUC는 해당 클래스 쌍의 Train Patch만 사용하며 구획당 두 클래스가 각각 최소 8장에서 관측된 경우 계산했습니다. Val 혼동은 오류 설명에 사용했습니다.</p>{confusion_panel(data)}
<h3>동일 이미지의 Segmentation 사례</h3><p>저장된 이미지별 I/U에서 공통 Union 클래스 ≥6, Original 정답 픽셀 ≥20,000 조건으로 회복·잔여 손상·P2 회귀가 큰 사례를 하나씩 골랐습니다. 회색은 Ignore입니다. 이 사례는 전체 결과를 대변하지 않습니다.</p>{segmentation_panel(data)}</section>
<section id="implications"><h2>C. 다음 실험을 위한 해석</h2><h3>P2가 보존한 정보</h3>{paras('c1')}<h3>P3의 공간 선택성</h3>{paras('c2')}<h3>아직 남은 손실</h3>{paras('c3')}<h3>다음 Method 가설과 최소 검증</h3>{paras('c4')}</section>
<section id="provenance"><h2>재현 정보</h2><p>기준 Commit <code>{esc(data['code']['git_head'])}</code>, PyTorch {esc(data['code']['torch'])}, timm {esc(data['code']['timm'])}. 샘플링 seed {data['seed']}, Train 후보와 Pair seed는 각각 +1과 +2입니다. 추론은 기존 Encoder/Head의 고정 가중치로 수행했습니다. 정확한 split, Image ID, 코드·데이터·가중치 Hash를 아래 자료에 저장했습니다. <a href="{data['ade_benchmark']}">MIT Scene Parsing Benchmark</a></p>
{details('Encoder와 Linear Head 경로 및 SHA-256',table(['모델','Encoder','Head','기존 Checksum','Encoder SHA-256','Head SHA-256'],checkpoint_rows))}
{details('샘플 Image ID · Train, Val, COCO', '<pre>'+esc(json.dumps(dict(ade_train=data['ade_train_ids'],ade_val=data['ade_val_ids'],coco_train=mask['image_ids'],coco_dev=data['reused_coco_roi_drift']['image_ids']),indent=1))+'</pre>')}
{details('코드 Hash와 샘플링 정의','<pre>'+esc(json.dumps(dict(code=data['code'],sampling=data['sampling'],stream=mask['train_source_hashes'],thing_stuff=data['things_stuff_mapping']),indent=2))+'</pre>')}
<p><a href="diagnosis_metrics.json">진단 JSON</a> · <a href="roi_mask_audit.json">ROI Audit JSON</a> · <a href="val_confusion.npz">전체 150×150 Val Confusion / I-U</a> · <a href="train_features.npz">Train 후보 Feature / Image Drift</a> · <a href="inference.log">추론 로그</a></p>
</section></main><footer><div class="wrap">P2/P3 Dense Representation Diagnosis · Local HTML · DINOv3-B/16</div></footer><script>{JS}</script></body></html>'''
    (OUT/'report.html').write_text(content,encoding='utf-8')
    print(OUT/'report.html')


if __name__ == '__main__':
    main()
