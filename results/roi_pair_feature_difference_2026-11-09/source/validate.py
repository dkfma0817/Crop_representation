"""Independent numerical/integrity checks of the completed diagnosis."""
from pathlib import Path
import ast,json,hashlib
import numpy as np
import pandas as pd
from PIL import Image

OUT=Path('/home/irteam/Crop_representation/results/roi_pair_feature_difference_2026-11-09')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    prov=json.load(open(OUT/'provenance.json'));integrity=[]
    for p,before in prov['input_sha256'].items():
        after=sha(p);assert before==after,p
        integrity.append(dict(path=p,before=before,after=after,unchanged=True))
    pretrained=Path('/home/irteam/Crop_representation/cache/hf_hub/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/model.safetensors')
    assert pretrained.exists()
    prov['conditions']['C0']['pretrained_weight_file']=str(pretrained)
    prov['conditions']['C0']['pretrained_weight_sha256']=sha(pretrained)
    (OUT/'provenance.json').write_text(json.dumps(prov,indent=2))
    pa=pd.read_parquet(OUT/'pairs.parquet');df=pd.read_parquet(OUT/'roi_manifest.parquet');Y=np.load(OUT/'labels_val.npy')
    p,n,a=pa.pos_row.to_numpy(),pa.neg_row.to_numpy(),pa.attribute_id.to_numpy()
    assert (Y[p,a]==1).all() and (Y[n,a]==0).all()
    assert np.array_equal(df.category_id.to_numpy()[p],df.category_id.to_numpy()[n])
    assert not pa.duplicated(['unit_id','pos_row','neg_row']).any()
    assert not pa.same_image.any()
    assert np.array_equal(pa.pos_size_bin,pd.cut(pa.pos_short384,[0,16,32,64,np.inf],right=False,labels=['0-16','16-32','32-64','64+']).astype(str))
    caches={};feature_checks={};metric_errors={}
    rng=np.random.default_rng(0);idx=rng.choice(len(pa),100,replace=False)
    for name in ['C0','P384','P768','P768-centered']:
        z=np.load(OUT/'features'/f'{name}.npz',allow_pickle=True)
        for k in ['ann_id','image_id','category_id','level']:assert np.array_equal(z[k],df[k].values)
        assert np.array_equal(z['labels'],Y)
        c=prov['conditions'][name]
        assert np.array_equal(z['replayed_scores'],np.load(c['mean']['scores']))
        assert np.array_equal(z['replayed_r2_scores'],np.load(c['r2']['scores']))
        audit=json.load(open(OUT/'features'/f'{name}_audit.json'))
        assert audit['encoder_unchanged'] and audit['mean_probe_replay_exact'] and audit['r2_replay_exact']
        assert not audit['requires_grad_any'] and not audit['grads_present']
        before=np.load(OUT/'features'/f'{name}_first_pass.npy')
        assert np.array_equal(before,z['features'])
        feature_checks[name]=dict(first_second_feature_pass_exact=True,mean_replay_exact=True,r2_replay_exact=True,encoder_unchanged=True,encoder_state_sha256=audit['encoder_state_sha256_after'],feature_sha256=hashlib.sha256(z['features'].tobytes()).hexdigest())
        x=z['features'].astype(np.float64);caches[name]=x/np.linalg.norm(x,axis=1,keepdims=True)
    for name in ['P384','P768','P768-centered']:
        d=pd.read_parquet(OUT/f'pair_metrics_{name}.parquet')
        assert np.array_equal(d.pair_id,pa.pair_id)
        dt=caches['C0'][p[idx]]-caches['C0'][n[idx]];ds=caches[name][p[idx]]-caches[name][n[idx]]
        tn=np.linalg.norm(dt,axis=1);sn=np.linalg.norm(ds,axis=1)
        ratio=sn/(tn+1e-8);direction=np.sum(ds*dt,axis=1)/(sn*tn)
        re=float(np.max(np.abs(ratio-d.magnitude_ratio.values[idx])));de=float(np.max(np.abs(direction-d.difference_direction_cosine.values[idx])))
        assert re<1e-12 and de<1e-12
        metric_errors[name]=dict(ratio_error=re,direction_error=de)
        for r in ['mean','r2']:
            tc=np.load(prov['conditions']['C0'][r]['scores']);sc=np.load(prov['conditions'][name][r]['scores'])
            tm=tc[p,a].astype(float)-tc[n,a];sm=sc[p,a].astype(float)-sc[n,a]
            group=np.select([(tm>0)&(sm<=0),(tm>0)&(sm>0),(tm<=0)&(sm>0)],['A','B','C'],default='D')
            assert np.array_equal(group,d[f'{r}_group'])
    # Independently recompute one complete image bootstrap interval using NumPy.
    d=pd.read_parquet(OUT/'pair_metrics_P384.parquet');idx=np.flatnonzero((pa.level=='object')&(pa.attribute_type=='material')&(d.mean_group=='A'))
    z=np.load(OUT/'bootstrap_image_counts.npz');counts=z['counts'].astype(float);images=z['image_ids']
    w=counts[:,np.searchsorted(images,pa.pos_image_id.values[idx])]*counts[:,np.searchsorted(images,pa.neg_image_id.values[idx])]
    bmeans=w@d.magnitude_ratio.values[idx]/w.sum(1)
    g=pd.read_csv(OUT/'group_summary.csv');target=g[(g.model=='P384')&(g.reader=='mean')&(g.level=='object')&(g.attribute_type=='material')&(g.context=='all')&(g.group=='A')&(g.metric=='magnitude_ratio')].iloc[0]
    lo,hi=np.quantile(bmeans[1:],[.025,.975])
    assert np.allclose([bmeans[0],lo,hi],[target['mean'],target.ci_low,target.ci_high],atol=1e-12)
    match=pd.read_csv(OUT/'difficulty_matches.csv.gz')
    assert (match.teacher_margin_difference.abs()<=match.caliper+1e-12).all()
    tables=['group_summary','controlled_A_minus_B','nearest_margin_matched_A_minus_B','paired_model_comparisons','threshold_sensitivity']
    table_checks={}
    for name in tables:
        t=pd.read_csv(OUT/f'{name}.csv');assert (t.ci_low<=t.ci_high).all();assert np.isfinite(t[['mean','ci_low','ci_high']]).all().all()
        table_checks[name]=dict(rows=len(t),min_valid_bootstraps=int(t.valid_bootstraps.min()),all_finite=True)
    figchecks=[]
    for f in sorted((OUT/'figures').glob('*.png')):
        im=np.asarray(Image.open(f));assert im.std()>10 and min(im.shape[:2])>400
        pdf=f.with_suffix('.pdf');assert pdf.read_bytes().startswith(b'%PDF-')
        figchecks.append(dict(path=str(f.relative_to(OUT)),shape=list(im.shape),nonblank=True,pdf=True))
    assert len(figchecks)==5
    for f in (OUT/'source').glob('*.py'):ast.parse(f.read_text())
    result=dict(status='PASS',existing_input_files_checked=len(integrity),original_inputs_all_unchanged=True,input_integrity=integrity,feature_checks=feature_checks,manual_pair_metric_checks=metric_errors,bootstrap_independent_numpy_check=True,bootstrap_example=dict(mean=bmeans[0],low=lo,high=hi),all_pair_roi_labels_categories_match=True,no_duplicate_pairs=True,same_image_pairs=0,nearest_margin_calipers_pass=True,tables=table_checks,figures=figchecks,source_syntax_pass=True)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status='PASS',input_files=len(integrity),all_scores_exact=True,pair_metrics_manual=True,bootstrap_independent=True,figures=len(figchecks))),flush=True)

if __name__=='__main__':main()
