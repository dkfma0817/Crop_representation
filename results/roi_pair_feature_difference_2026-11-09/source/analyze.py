"""Pair geometry, existing-head diagnostics and image-vertex bootstrap. No fitting."""
import os,sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

ROOT=Path('/home/irteam/Crop_representation');OUT=ROOT/'results/roi_pair_feature_difference_2026-11-09'
MODELS=['P384','P768','P768-centered'];READERS=['mean','r2'];B=1000;EPS=1e-8
METRICS=['teacher_difference_norm','student_difference_norm','magnitude_ratio','difference_direction_cosine','individual_ts_cosine','teacher_pair_cosine','student_pair_cosine','teacher_margin','student_margin','teacher_probability_margin','student_probability_margin','raw_magnitude_ratio','raw_difference_direction_cosine','teacher_axis_student_rank_correct','student_axis_teacher_rank_correct','teacher_axis_student_delta_cosine','teacher_axis_teacher_delta_cosine','axis_cosine','ratio_below_08','ratio_at_least_08_direction_below_05']

def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),EPS)
def dot(a,b):return np.einsum('ij,ij->i',a,b)
def dump(p,x):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
        if isinstance(x,np.integer):return int(x)
        if isinstance(x,np.bool_):return bool(x)
        return x
    p.write_text(json.dumps(clean(x),indent=2,allow_nan=False))

class Bootstrap:
    def __init__(self,pairs,manifest):
        images=np.sort(manifest.image_id.unique());rng=np.random.default_rng(0)
        counts=rng.multinomial(len(images),np.full(len(images),1/len(images)),size=B)
        counts=np.concatenate([np.ones((1,len(images)),int),counts])
        np.savez_compressed(OUT/'bootstrap_image_counts.npz',image_ids=images,counts=counts.astype(np.int16))
        p=np.searchsorted(images,pairs.pos_image_id);n=np.searchsorted(images,pairs.neg_image_id)
        dev='cuda' if torch.cuda.is_available() else 'cpu'
        self.dev=dev;ct=torch.as_tensor(counts,dtype=torch.float64,device=dev)
        self.w=ct[:,p]*ct[:,n]
        same=np.flatnonzero(p==n)
        if len(same):self.w[:,same]=ct[:,p[same]]
        assert np.allclose(self.w[0].cpu().numpy(),1)
    def mean(self,idx,x):
        idx=np.asarray(idx,dtype=np.int64);x=np.asarray(x,dtype=np.float64)
        if x.ndim==1:x=x[:,None]
        assert len(idx)==len(x)
        if len(idx)==0:return np.full((B+1,x.shape[1]),np.nan)
        w=self.w[:,idx];finite=np.isfinite(x)
        v=torch.as_tensor(np.nan_to_num(x),dtype=torch.float64,device=self.dev)
        den=w@torch.as_tensor(finite.astype(float),dtype=torch.float64,device=self.dev)
        num=w@v
        y=torch.where(den>0,num/den,torch.nan)
        return y.cpu().numpy()

def interval(v):
    return dict(mean=float(v[0]),ci_low=float(np.nanquantile(v[1:],.025)),ci_high=float(np.nanquantile(v[1:],.975)),valid_bootstraps=int(np.isfinite(v[1:]).sum()))

def effective_axis(path):
    state=torch.load(path,map_location='cpu',weights_only=False)['reader']
    # Common RR.Heads is level-specific. Infer actual key names and assert shape.
    result=[]
    for lv in ['object','part']:
        keys=[k for k in state if k.endswith('.weight') and lv in k and state[k].shape==(59,768)]
        if not keys:
            keys=[k for k in state if k.endswith('.weight') and state[k].shape==(59,768)]
            assert len(keys)==2,[(k,tuple(v.shape)) for k,v in state.items()]
            key=keys[0 if lv=='object' else 1]
        else:assert len(keys)==1;key=keys[0]
        result.append(state[key].numpy()/state['sd'][0 if lv=='object' else 1].numpy())
    return np.stack(result).astype(np.float64)

def geometry(name,features,pa,manifest,prov):
    t=features['C0'].astype(np.float64);s=features[name].astype(np.float64)
    tn,sn=normalize(t),normalize(s);p=pa.pos_row.to_numpy();n=pa.neg_row.to_numpy()
    dt=tn[p]-tn[n];ds=sn[p]-sn[n]
    nt=np.linalg.norm(dt,axis=1);ns=np.linalg.norm(ds,axis=1)
    dtr=t[p]-t[n];dsr=s[p]-s[n]
    nt_raw=np.linalg.norm(dtr,axis=1);ns_raw=np.linalg.norm(dsr,axis=1)
    a=pa.attribute_id.to_numpy();lv=(pa.level=='part').astype(int).to_numpy()
    at=effective_axis(prov['conditions']['C0']['mean']['checkpoint'])[lv,a]
    ast=effective_axis(prov['conditions'][name]['mean']['checkpoint'])[lv,a]
    v=pd.DataFrame(dict(teacher_difference_norm=nt,student_difference_norm=ns,magnitude_ratio=ns/(nt+EPS),difference_direction_cosine=dot(ds,dt)/np.maximum(nt*ns,EPS),individual_ts_cosine=(dot(sn[p],tn[p])+dot(sn[n],tn[n]))/2,positive_ts_cosine=dot(sn[p],tn[p]),negative_ts_cosine=dot(sn[n],tn[n]),teacher_pair_cosine=dot(tn[p],tn[n]),student_pair_cosine=dot(sn[p],sn[n]),raw_magnitude_ratio=ns_raw/(nt_raw+EPS),raw_difference_direction_cosine=dot(dsr,dtr)/np.maximum(ns_raw*nt_raw,EPS),teacher_axis_student_rank_correct=(dot(at,dsr)>0).astype(float),student_axis_teacher_rank_correct=(dot(ast,dtr)>0).astype(float),teacher_axis_student_delta_cosine=dot(normalize(at),normalize(ds)),teacher_axis_teacher_delta_cosine=dot(normalize(at),normalize(dt)),axis_cosine=dot(normalize(at),normalize(ast)),teacher_axis_student_raw_margin=dot(at,dsr),student_axis_teacher_raw_margin=dot(ast,dtr)))
    v['ratio_below_08']=(v.magnitude_ratio<.8).astype(float)
    v['ratio_at_least_08_direction_below_05']=((v.magnitude_ratio>=.8)&(v.difference_direction_cosine<.5)).astype(float)
    v['model']=name;v['pair_id']=pa.pair_id
    for reader in READERS:
        tc=np.load(prov['conditions']['C0'][reader]['scores']);sc=np.load(prov['conditions'][name][reader]['scores'])
        tm=tc[p,a].astype(float)-tc[n,a];sm=sc[p,a].astype(float)-sc[n,a]
        v[f'{reader}_teacher_margin']=tm;v[f'{reader}_student_margin']=sm
        v[f'{reader}_teacher_probability_margin']=expit(tc[p,a])-expit(tc[n,a])
        v[f'{reader}_student_probability_margin']=expit(sc[p,a])-expit(sc[n,a])
        v[f'{reader}_group']=np.select([(tm>0)&(sm<=0),(tm>0)&(sm>0),(tm<=0)&(sm>0)],['A','B','C'],default='D')
    assert np.allclose(nt**2,2-2*v.teacher_pair_cosine,atol=1e-12)
    assert np.allclose(ns**2,2-2*v.student_pair_cosine,atol=1e-12)
    assert np.isfinite(v.select_dtypes('number')).all().all()
    return v

def values_for(g,reader):
    v=g.copy()
    for k in ['teacher_margin','student_margin','teacher_probability_margin','student_probability_margin']:v[k]=v[f'{reader}_{k}']
    return v[METRICS].to_numpy()

def summary_table(boot,pa,geoms):
    summary=[];counts=[];contrasts=[];unit_summary=[];distribution={}
    for model,g in geoms.items():
        for reader in READERS:
            values=values_for(g,reader);gr=g[f'{reader}_group'].to_numpy()
            for (level,typ),rows in pa.groupby(['level','attribute_type']).groups.items():
                base=np.asarray(rows);full_n=len(base)
                contexts=[('all',base),('supported_npos5',base[pa.old_supported.to_numpy()[base]])]
                contexts.extend((f'positive_size_{size}',base[pa.pos_size_bin.to_numpy()[base]==size]) for size in ['0-16','16-32','32-64','64+'])
                saved={}
                for context,rr in contexts:
                    for group in ['ALL','A','B','C','D']:
                        idx=rr if group=='ALL' else rr[gr[rr]==group]
                        if not len(idx):continue
                        sims=boot.mean(idx,values[idx]);saved[context,group]=sims
                        meta=dict(model=model,reader=reader,level=level,attribute_type=typ,context=context,group=group,n_pairs=len(idx),n_units=int(pa.unit_id.iloc[idx].nunique()),n_images=int(len(set(pa.pos_image_id.iloc[idx])|set(pa.neg_image_id.iloc[idx]))),n_unique_rois=int(len(set(pa.pos_row.iloc[idx])|set(pa.neg_row.iloc[idx]))),pair_fraction=len(idx)/len(rr),n_same_image=int(pa.same_image.iloc[idx].sum()))
                        counts.append(meta)
                        for j,metric in enumerate(METRICS):
                            x=values[idx,j]
                            summary.append(dict(**meta,metric=metric,**interval(sims[:,j]),q05=float(np.quantile(x,.05)),q25=float(np.quantile(x,.25)),median=float(np.median(x)),q75=float(np.quantile(x,.75)),q95=float(np.quantile(x,.95))))
                        if context=='all':
                            distribution[f'{model}|{reader}|{level}|{typ}|{group}']={metric:dict(quantiles=np.quantile(values[idx,j],np.linspace(0,1,101)).tolist()) for j,metric in enumerate(METRICS)}
                    if (context,'A') in saved and (context,'B') in saved:
                        diff=saved[context,'A']-saved[context,'B']
                        for j,metric in enumerate(METRICS):contrasts.append(dict(model=model,reader=reader,level=level,attribute_type=typ,context=context,adjustment='unadjusted_pair_mean',metric=metric,**interval(diff[:,j])))
                for unit,rr in pa.iloc[base].groupby('unit_id').groups.items():
                    rr=np.asarray(rr)
                    for group in ['A','B','C','D']:
                        idx=rr[gr[rr]==group]
                        if not len(idx):continue
                        d=dict(model=model,reader=reader,unit_id=unit,group=group,n_pairs=len(idx))
                        d.update({metric:float(np.mean(values[idx,j])) for j,metric in enumerate(METRICS)})
                        unit_summary.append(d)
                print('summary',model,reader,level,typ,flush=True)
    pd.DataFrame(summary).to_csv(OUT/'group_summary.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'group_counts.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'unadjusted_A_minus_B.csv',index=False)
    pd.DataFrame(unit_summary).to_csv(OUT/'category_attribute_group_summary.csv',index=False)
    dump(OUT/'distributions.json',distribution)

def adjusted(boot,pa,geoms):
    output=[];details=[]
    # Category-attribute, positive AND negative size, and within-unit teacher difficulty tertiles.
    for model,g in geoms.items():
        for reader in READERS:
            gr=g[f'{reader}_group'].to_numpy();values=values_for(g,reader)
            temp=pa.copy();temp['teacher_margin']=g[f'{reader}_teacher_margin']
            temp['difficulty_bin']=temp.groupby('unit_id').teacher_margin.transform(lambda x:np.searchsorted(np.quantile(x[x>0],[1/3,2/3]) if (x>0).any() else [0,0],x,side='right'))
            for (level,typ),rr in temp.groupby(['level','attribute_type']).groups.items():
                for adjustment,columns in [('category_attribute',['unit_id']),('category_attribute_both_sizes',['unit_id','pos_size_bin','neg_size_bin']),('category_attribute_both_sizes_teacher_margin',['unit_id','pos_size_bin','neg_size_bin','difficulty_bin'])]:
                    strata=[]
                    for key,rows in temp.loc[rr].groupby(columns,observed=True).groups.items():
                        rows=np.asarray(rows);ia=rows[gr[rows]=='A'];ib=rows[gr[rows]=='B']
                        if len(ia)<5 or len(ib)<5:continue
                        unit=int(temp.unit_id.iloc[rows[0]]);attribute=int(temp.attribute_id.iloc[rows[0]])
                        strata.append(dict(unit=unit,attribute=attribute,nA=len(ia),nB=len(ib),ia=ia,ib=ib,key=str(key)))
                    if not strata:continue
                    # Fixed A-target stratum weights within each unit, equal categories within attribute,
                    # equal attributes: analogous to project M2 hierarchy, common overlap only.
                    unitA={u:sum(h['nA'] for h in strata if h['unit']==u) for u in set(h['unit'] for h in strata)}
                    attrunits={a:len(set(h['unit'] for h in strata if h['attribute']==a)) for a in set(h['attribute'] for h in strata)}
                    aggA=np.zeros((B+1,len(METRICS)));aggB=aggA.copy();aggW=aggA.copy()
                    for h in strata:
                        alpha=h['nA']/unitA[h['unit']]/attrunits[h['attribute']]/len(attrunits)
                        ma=boot.mean(h['ia'],values[h['ia']]);mb=boot.mean(h['ib'],values[h['ib']]);valid=np.isfinite(ma)&np.isfinite(mb)
                        aggA+=np.nan_to_num(ma)*valid*alpha;aggB+=np.nan_to_num(mb)*valid*alpha;aggW+=valid*alpha
                        details.append(dict(model=model,reader=reader,level=level,attribute_type=typ,adjustment=adjustment,stratum=h['key'],unit_id=h['unit'],n_A=h['nA'],n_B=h['nB'],weight=alpha))
                    ma=aggA/np.maximum(aggW,EPS);mb=aggB/np.maximum(aggW,EPS);diff=ma-mb
                    for j,metric in enumerate(METRICS):output.append(dict(model=model,reader=reader,level=level,attribute_type=typ,adjustment=adjustment,metric=metric,mean_A=ma[0,j],mean_B=mb[0,j],**interval(diff[:,j]),n_strata=len(strata),n_units=len(unitA),n_attributes=len(attrunits),n_A=sum(h['nA'] for h in strata),n_B=sum(h['nB'] for h in strata),fraction_A_retained=sum(h['nA'] for h in strata)/max(1,int((gr[np.asarray(rr)]=='A').sum())),min_bootstrap_retained_weight=float(aggW[1:,j].min())))
                print('adjusted',model,reader,level,typ,flush=True)
    pd.DataFrame(output).to_csv(OUT/'controlled_A_minus_B.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'controlled_stratum_counts.csv',index=False)

def paired_comparisons(boot,pa,geoms):
    results=[];transitions=[]
    for reader in READERS:
        g384=geoms['P384'][f'{reader}_group'].to_numpy()
        teacher_good=geoms['P384'][f'{reader}_teacher_margin'].to_numpy()>0
        for (level,typ),rr in pa.groupby(['level','attribute_type']).groups.items():
            rr=np.asarray(rr)
            contexts=[('all_common_pairs',rr),('C0_correct',rr[teacher_good[rr]]),('P384_A_fixed',rr[g384[rr]=='A'])]
            contexts.extend((f'size_{sz}',rr[pa.pos_size_bin.to_numpy()[rr]==sz]) for sz in ['0-16','16-32','32-64','64+'])
            for name,idx in contexts:
                if not len(idx):continue
                for left,right in [('P768','P384'),('P768-centered','P768'),('P768-centered','P384')]:
                    metric_names=['magnitude_ratio','difference_direction_cosine','individual_ts_cosine','student_difference_norm','teacher_axis_student_rank_correct']
                    delta=geoms[left].loc[idx,metric_names].to_numpy()-geoms[right].loc[idx,metric_names].to_numpy()
                    smleft=geoms[left][f'{reader}_student_margin'].to_numpy()[idx]
                    smright=geoms[right][f'{reader}_student_margin'].to_numpy()[idx]
                    delta=np.column_stack([delta,(smleft>0).astype(float)-(smright>0).astype(float)])
                    metric_names+=['student_rank_correct_fraction']
                    sims=boot.mean(idx,delta)
                    for j,metric in enumerate(metric_names):results.append(dict(reader=reader,level=level,attribute_type=typ,context=name,left=left,right=right,n_pairs=len(idx),metric=metric,**interval(sims[:,j])))
                if name=='P384_A_fixed':
                    for model in MODELS[1:]:
                        gr=geoms[model][f'{reader}_group'].to_numpy()[idx]
                        for target in ['A','B','C','D']:
                            sims=boot.mean(idx,(gr==target).astype(float))[:,0]
                            transitions.append(dict(reader=reader,level=level,attribute_type=typ,source='P384 A',target_model=model,target_group=target,n_pairs=len(idx),**interval(sims)))
    pd.DataFrame(results).to_csv(OUT/'paired_model_comparisons.csv',index=False)
    pd.DataFrame(transitions).to_csv(OUT/'P384_A_transitions.csv',index=False)

def main():
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    start=time.monotonic();pa=pd.read_parquet(OUT/'pairs.parquet');manifest=pd.read_parquet(OUT/'roi_manifest.parquet');prov=json.load(open(OUT/'provenance.json'))
    features={};audits={}
    for model in ['C0']+MODELS:
        z=np.load(OUT/'features'/f'{model}.npz',allow_pickle=True)
        for k in ['ann_id','image_id','category_id','level']:assert np.array_equal(z[k],manifest[k].values)
        assert np.array_equal(z['labels'],np.load(OUT/'labels_val.npy'))
        features[model]=z['features'];assert features[model].shape==(len(manifest),768)
        audits[model]=json.load(open(OUT/'features'/f'{model}_audit.json'))
        assert audits[model]['encoder_unchanged'] and audits[model]['mean_probe_replay_exact']
    geoms={};roi_rows=[];rng=np.random.default_rng(0)
    for model in MODELS:
        g=geometry(model,features,pa,manifest,prov);geoms[model]=g
        joined=pd.concat([pa,g.drop(columns='pair_id')],axis=1)
        joined.to_parquet(OUT/f'pair_metrics_{model}.parquet',index=False)
        joined.to_csv(OUT/f'pair_metrics_{model}.csv.gz',index=False)
        tn=normalize(features['C0'].astype(float));sn=normalize(features[model].astype(float));matched=dot(tn,sn);shuffled=dot(tn,sn[rng.permutation(len(sn))]);permuted=dot(tn,sn[:,rng.permutation(768)])
        for i in range(len(manifest)):roi_rows.append(dict(model=model,row=i,ann_id=int(manifest.ann_id.iloc[i]),individual_cosine=matched[i],shuffled_roi_cosine=shuffled[i],permuted_channel_cosine=permuted[i],teacher_norm=float(np.linalg.norm(features['C0'][i])),student_norm=float(np.linalg.norm(features[model][i]))))
        print('geometry',model,'ratio',g.magnitude_ratio.mean(),'direction',g.difference_direction_cosine.mean(),flush=True)
    pd.DataFrame(roi_rows).to_csv(OUT/'roi_alignment.csv',index=False)
    boot=Bootstrap(pa,manifest)
    summary_table(boot,pa,geoms);adjusted(boot,pa,geoms);paired_comparisons(boot,pa,geoms)
    # Per-unit valid A/B/C/D counts for every model/readout, with full annotation inventory.
    inventory=pd.read_csv(OUT/'category_attribute_inventory.csv');count=[]
    for model,g in geoms.items():
        for reader in READERS:
            table=pd.crosstab(pa.unit_id,g[f'{reader}_group']).reindex(columns=['A','B','C','D'],fill_value=0).reset_index()
            table=table.merge(inventory,on='unit_id');table['model']=model;table['reader']=reader;count.append(table)
    pd.concat(count).to_csv(OUT/'category_attribute_group_counts.csv',index=False)
    dump(OUT/'analysis_checks.json',dict(status='complete',B=B,seed=0,no_training=True,all_roi_ids_categories_labels_match=True,all_mean_probe_scores_replayed_exactly=True,all_encoders_unchanged=True,difference_norm_cosine_identity_pass=True,zero_teacher_differences={m:int((g.teacher_difference_norm<EPS).sum()) for m,g in geoms.items()},zero_student_differences={m:int((g.student_difference_norm<EPS).sum()) for m,g in geoms.items()},bootstrap='Same multinomial image draws across all models/groups. Pair endpoints jointly weighted; same-image diagonal once. Fixed sampled graph; no independent-pair resampling.',seconds=round(time.monotonic()-start,2)))
    print('COMPLETE',round(time.monotonic()-start,2),flush=True)

if __name__=='__main__':main()
