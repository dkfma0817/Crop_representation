"""Standalone scientific figures accompanying the interactive HTML."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_figures(data,out):
    out=Path(out);out.mkdir(exist_ok=True)
    content={c:dict(ap=v['ap'],diagnostics=v.get('training',{}).get('diagnostics'),attrs=v['attributes']) for c,v in data['models'].items()}
    sig=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest();marker=out/'data_signature.txt'
    if marker.exists() and marker.read_text()==sig:return
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    models=data['core'];labels={'Decode':'Decoder','B0':'Full B0'}
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');y=np.arange(len(models))
    for j,(metric,color,shift) in enumerate([('Object AP','#315bd9',-.18),('Part AP','#aebdd0',.18)]):
        values=[data['models'][c]['ap'].get(metric) if data['models'][c]['ap'] else np.nan for c in models]
        ax.barh(y+shift,values,height=.32,color=color,label=metric)
        for yy,v in zip(y+shift,values):
            if np.isfinite(v):ax.text(v+.4,yy,f'{v:.2f}',va='center',fontsize=9)
    for i,c in enumerate(models):
        if data['models'][c]['ap'] is None:ax.text(2,i,'Official TEST pending',va='center',color='#7c8ba2',fontsize=9)
    ax.set(yticks=y,yticklabels=[labels.get(c,c) for c in models],xlabel='Official PACO GT-ROI attribute AP',xlim=(0,65),title='Frozen representation + independently trained R2 (seed0)');ax.invert_yaxis();ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15)
    for suffix in ['png','pdf']:fig.savefig(out/f'overall_test_ap.{suffix}',dpi=180)
    plt.close(fig)
    attrs=['ceramic','fabric','paper','striped','perforated','text'];students=['P8','P4','Native','Decode','M','B0']
    lookup=lambda c:{r['attribute']:r for r in data['models'][c]['attributes'].get('Object',[])}
    crop=lookup('C0');matrix=np.full((len(students),len(attrs)),np.nan)
    for i,c in enumerate(students):
        own=lookup(c)
        for j,k in enumerate(attrs):
            if k in own and own[k]['AP'] is not None:matrix[i,j]=own[k]['AP']-crop[k]['AP']
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained');cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#eceff4');im=ax.imshow(matrix,cmap=cmap,vmin=-20,vmax=20,aspect='auto')
    ax.set(yticks=np.arange(len(students)),yticklabels=[labels.get(c,c) for c in students],xticks=np.arange(len(attrs)),xticklabels=[f'{k}\nN={crop[k]["positive_ROIs"]}, C={crop[k]["positive_categories"]}' for k in attrs],title='Object attributes: Student minus Crop AP (exploratory)')
    for i in range(len(students)):
        for j in range(len(attrs)):
            v=matrix[i,j];ax.text(j,i,f'{v:+.2f}' if np.isfinite(v) else 'pending',ha='center',va='center',fontsize=10,color='white' if np.isfinite(v) and abs(v)>13 else '#26364c')
    fig.colorbar(im,ax=ax,label='AP difference; blue > Crop, orange/red < Crop',shrink=.85)
    for suffix in ['png','pdf']:fig.savefig(out/f'crop_gap_focus.{suffix}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.1),layout='constrained')
    for c in ['P4','Decode']:
        rows=data['models'][c].get('training',{}).get('diagnostics',[])
        if rows:ax.plot([r['step'] for r in rows],[r['crop_cosine'] for r in rows],'o-',label=labels.get(c,c),color=data['colors'][c])
    ax.set(xlabel='Representation optimization steps',ylabel='Student mean / frozen crop teacher cosine',title='COCO dev alignment (48 images); cosine is not attribute AP',ylim=(-.05,1));ax.legend();ax.grid(alpha=.15)
    for suffix in ['png','pdf']:fig.savefig(out/f'teacher_alignment.{suffix}',dpi=180)
    plt.close(fig);marker.write_text(sig)
