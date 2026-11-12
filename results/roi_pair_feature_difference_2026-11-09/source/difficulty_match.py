"""Descriptive sensitivity: fixed nearest-margin A/B matching; no learned model."""
from analyze import OUT,MODELS,READERS,METRICS,Bootstrap,values_for,interval,dump
import numpy as np
import pandas as pd
import torch

def main():
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    pa=pd.read_parquet(OUT/'pairs.parquet');manifest=pd.read_parquet(OUT/'roi_manifest.parquet');boot=Bootstrap(pa,manifest)
    results=[];matches=[];thresholds=[]
    for model in MODELS:
        full=pd.read_parquet(OUT/f'pair_metrics_{model}.parquet')
        for reader in READERS:
            group=full[f'{reader}_group'].to_numpy();margin=full[f'{reader}_teacher_margin'].to_numpy();values=values_for(full,reader)
            for (level,typ),rr in pa.groupby(['level','attribute_type']).groups.items():
                # Threshold table is exploratory; continuous values remain primary.
                for grp in ['A','B','C']:
                    idx=np.asarray(rr);idx=idx[group[idx]==grp]
                    if not len(idx):continue
                    for rcut in [.7,.8,.9]:
                        for dcut in [.3,.5,.7]:
                            yes=((full.magnitude_ratio.to_numpy()[idx]>=rcut)&(full.difference_direction_cosine.to_numpy()[idx]<dcut)).astype(float)
                            estimate=boot.mean(idx,yes)[:,0]
                            thresholds.append(dict(model=model,reader=reader,level=level,attribute_type=typ,group=grp,ratio_min=rcut,direction_max=dcut,n_pairs=len(idx),**interval(estimate)))
                for caliper in [.05,.1,.2]:
                    match=[]
                    for key,rows in pa.loc[rr].groupby(['unit_id','pos_size_bin','neg_size_bin']).groups.items():
                        rows=np.asarray(rows);ia=rows[group[rows]=='A'];ib=rows[group[rows]=='B']
                        if not len(ia) or not len(ib):continue
                        ib=ib[np.lexsort((ib,margin[ib]))];mb=margin[ib]
                        near=np.searchsorted(mb,margin[ia]);lo=np.clip(near-1,0,len(ib)-1);hi=np.clip(near,0,len(ib)-1)
                        chosen=np.where(np.abs(mb[lo]-margin[ia])<=np.abs(mb[hi]-margin[ia]),lo,hi)
                        for a,b in zip(ia,ib[chosen]):
                            if abs(margin[a]-margin[b])<=caliper:match.append((int(key[0]),int(a),int(b)))
                    if not match:continue
                    tab=pd.DataFrame(match,columns=['unit_id','A_index','B_index'])
                    counts=tab.unit_id.value_counts();tab=tab[tab.unit_id.isin(counts[counts>=5].index)]
                    if not len(tab):continue
                    units=tab.unit_id.unique();attr={u:int(pa.attribute_id.iloc[pa.index[pa.unit_id==u][0]]) for u in units}
                    attrN={a:sum(attr[u]==a for u in units) for a in set(attr.values())}
                    sumA=np.zeros((1001,len(METRICS)));sumB=sumA.copy();sumW=sumA.copy()
                    for u,rows in tab.groupby('unit_id'):
                        ia=rows.A_index.to_numpy();ib=rows.B_index.to_numpy();ma=boot.mean(ia,values[ia]);mb=boot.mean(ib,values[ib])
                        alpha=1/len(attrN)/attrN[attr[u]];valid=np.isfinite(ma)&np.isfinite(mb)
                        sumA+=np.nan_to_num(ma)*valid*alpha;sumB+=np.nan_to_num(mb)*valid*alpha;sumW+=valid*alpha
                    ma=sumA/np.maximum(sumW,1e-8);mb=sumB/np.maximum(sumW,1e-8);delta=ma-mb
                    for j,metric in enumerate(METRICS):results.append(dict(model=model,reader=reader,level=level,attribute_type=typ,caliper=caliper,metric=metric,mean_A=ma[0,j],mean_B=mb[0,j],**interval(delta[:,j]),n_matched_A=len(tab),n_unique_B=tab.B_index.nunique(),n_units=len(units),n_attributes=len(attrN),fraction_A_retained=len(tab)/max(1,int((group[np.asarray(rr)]=='A').sum()))))
                    tab['model']=model;tab['reader']=reader;tab['level']=level;tab['attribute_type']=typ;tab['caliper']=caliper;tab['teacher_margin_difference']=margin[tab.A_index]-margin[tab.B_index]
                    matches.append(tab)
                print('matched',model,reader,level,typ,flush=True)
    pd.DataFrame(results).to_csv(OUT/'nearest_margin_matched_A_minus_B.csv',index=False)
    pd.concat(matches,ignore_index=True).to_csv(OUT/'difficulty_matches.csv.gz',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'threshold_sensitivity.csv',index=False)
    dump(OUT/'difficulty_match_protocol.json',dict(seed=0,fit=False,matching='For each A pair choose nearest B teacher logit margin within exact category-attribute and both endpoint size bins; deterministic nearest with replacement; calipers .05/.1/.2 logit; retain units >=5 matched A. B repetition retains same image bootstrap weights. Hierarchical equal-attribute/equal-category means; A count within unit uniform. Fixed match assignment in bootstrap; does not include uncertainty of rematching.',primary_caliper=.1,exploratory=True))

if __name__=='__main__':main()
