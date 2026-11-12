"""Standalone numeric figures for the saved reports; no new experiments."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--extra',required=True);a=ap.parse_args();base=Path(a.base);extra=Path(a.extra)
 (base/'figures').mkdir(exist_ok=True);(extra/'figures').mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
 for c,color in [('P','#c55a32'),('M','#256f9b')]:
  t=pd.read_csv(base/f'results/{c}_train_log.csv');d=pd.read_csv(base/f'results/{c}_diagnostics.csv');d=d[d.model.isin(['init','ema'])]
  ax[0].plot(t.step,1-t.l_crop,label=c,color=color);ax[1].plot(t.step,t.l_global,label=c,color=color);ax[2].plot(d.step,d.q_roi_cos,'o-',label=c,color=color)
 for axis,title,y in zip(ax,['Training crop alignment','Global preservation loss','COCO dev crop alignment (EMA)'],['Crop cosine (batch mean)','Global cosine distance','Fused ROI / teacher cosine']):axis.set(title=title,xlabel='Representation steps',ylabel=y);axis.legend();axis.grid(alpha=.15)
 fig.savefig(base/'figures/training.png',dpi=180);fig.savefig(base/'figures/training.pdf');plt.close(fig)
 res=json.load(open(extra/'results.json'));models=res['models'];cols=['Obj Material','Obj Pattern','Part Material','Part Pattern'];contrasts=pd.DataFrame(res['mean_contrasts'])
 fig,axes=plt.subplots(1,4,figsize=(15,3.8),constrained_layout=True)
 colors={'E1':'#287bad','E2':'#cd7133','Continue':'#777777','C0':'#45935c'}
 for axis,col in zip(axes,cols):
  cs=[c for c in ['Continue','E1','E2','C0'] if c in models]
  for i,c in enumerate(cs):
   rr=contrasts[(contrasts.comparison==f'{c} − P')&(contrasts.metric==col)].iloc[0]
   if 'ci95_low' in rr and pd.notna(rr.ci95_low):axis.vlines(i,rr.ci95_low,rr.ci95_high,color=colors[c],lw=2)
   axis.plot(i,rr.delta,'o',color=colors[c]);axis.annotate(f'{rr.delta:+.2f}',(i,rr.delta),xytext=(3,5),textcoords='offset points',fontsize=9)
  axis.axhline(0,color='black',lw=.7);axis.set(xticks=range(len(cs)),xticklabels=cs,title=col,ylabel='Official AP difference from P12k');axis.grid(axis='y',alpha=.15)
 fig.suptitle('Mean feature linear probe: paired image 95% CIs; one training seed')
 fig.savefig(extra/'figures/mean_relative_ap.png',dpi=180);fig.savefig(extra/'figures/mean_relative_ap.pdf');plt.close(fig)
 focus=pd.DataFrame(res['focus_attributes']);focus=focus[(focus.reader=='R2')&(focus.level=='Object')]
 fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
 for axis,attr in zip(axes.flat,['ceramic','fabric','paper','striped','perforated','text']):
  z=focus[focus.attribute==attr].set_index('model');names=[c for c in ['P','Continue','E1','E2','C0'] if c in models];v=[z.loc[c,'AP'] for c in names]
  axis.bar(names,v,color=[colors.get(c,'#554b72') for c in names]);axis.set(ylim=(0,100),ylabel='Object attribute AP',title=f"{attr} (positive ROIs={int(z.iloc[0].positive_ROIs)}, categories={int(z.iloc[0].positive_categories)})")
  for i,value in enumerate(v):axis.text(i,value+1,f'{value:.1f}',ha='center',fontsize=9)
 fig.suptitle('Exploratory object attributes; C0 is the frozen crop reference')
 fig.savefig(extra/'figures/focus_attributes.png',dpi=180);fig.savefig(extra/'figures/focus_attributes.pdf');plt.close(fig)
 print('Saved standalone PNG/PDF numeric figures.',flush=True)
if __name__=='__main__':main()
