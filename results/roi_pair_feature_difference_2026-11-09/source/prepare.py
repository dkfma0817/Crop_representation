"""Read-only provenance audit and common PACO Val pair selection; seed 0."""
import os, sys, json, hashlib, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/home/irteam/Crop_representation')
LEGACY=Path('/home/irteam/sg/crop-rep')
OUT=ROOT/'results/roi_pair_feature_difference_2026-11-09'
sys.path.insert(0,str(LEGACY))
os.chdir(LEGACY)
import src.readout.common as RC
from src.ivc.common import EXCL

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def selected(directory,reader):
    d=directory/reader
    rows=[json.loads(s) for s in (d/'curves_dev.jsonl').read_text().splitlines()]
    best=max((r for r in rows if r['epoch']>=1),key=lambda r:(r['dev']['M0']['overall'],-r['epoch']))
    ep=best['epoch']
    val=next(json.loads(s) for s in (d/'curves.jsonl').read_text().splitlines() if json.loads(s)['epoch']==ep)
    return dict(epoch=ep,checkpoint=str(d/f'ep{ep}.pt'),scores=str(d/f'scores_val_ep{ep}.npy'),dev_M0=best['dev']['M0']['overall'],val_M2=val['val']['M2'])

def main():
    data=RC.load_data(EXCL)
    df,Y=data['val']; df=df.reset_index(drop=True)
    assert len(df)==11109 and df.image_id.nunique()==2051
    assert df.ann_id.is_unique and (df.split=='val').all()
    attrs=json.load(open(ROOT/'cache/paco_lp/attributes.json'))
    df['row']=np.arange(len(df))
    df['short384']=np.minimum(df.r-df.l,df.b-df.t)*384/np.maximum(df.W,df.H)
    df['size_bin']=pd.cut(df.short384,[0,16,32,64,np.inf],right=False,labels=['0-16','16-32','32-64','64+']).astype(str)
    df.to_parquet(OUT/'roi_manifest.parquet',index=False)
    df[['row','ann_id','image_id','category_id','category','level','short384','size_bin']].to_csv(OUT/'roi_manifest.csv',index=False)
    np.save(OUT/'labels_val.npy',Y)
    (OUT/'attributes.json').write_text(json.dumps(attrs,indent=2))
    causal=ROOT/'output/2026-11-08/hcf_material_causal'
    conds={}
    for name,folder,size in [('C0',ROOT/'output/2026-11-06/ivc_v3/probe_c0',384),('P384',causal/'r384_cosine/probe',384),('P768',causal/'r768_cosine/probe',768),('P768-centered',causal/'r768_centered/probe',768)]:
        cfg=json.load(open(folder/'config.json'))
        assert cfg['n_val_rows']==len(df) and cfg['n_val_images']==df.image_id.nunique()
        assert cfg['job']['seed']==0 and cfg['job']['backbone']=='dinov3'
        encoder=cfg['job'].get('encoder_ckpt')
        conds[name]=dict(probe_dir=str(folder),full_size=size,encoder_checkpoint=encoder,mean=selected(folder,'r0'),r2=selected(folder,'r2' if name=='C0' else 'r2fused'),config=cfg,feature_dimension=768)
        for reader in ['mean','r2']:
            x=np.load(conds[name][reader]['scores'])
            assert x.shape==Y.shape and np.isfinite(x).all()
    # Exactly the project M2 unit inclusion (>=1 pos, >=40 explicit neg, no other(*) attributes).
    units=RC.ET.EM.build_units(df,Y,RC.ytr_by_level(data),np.ones(len(df),bool))['cat']
    groups={a:k for k,ids in attrs['type_to_idx'].items() for a in ids}
    rng=np.random.default_rng(0); pairs=[]; inventory=[]
    for key,rows,a in units:
        level,cat,_=key; typ=groups[a]
        if typ not in ['material','pattern_marking']:continue
        pos=rows[Y[rows,a]==1];neg=rows[Y[rows,a]==0]
        total=len(pos)*len(neg); count=min(4096,total)
        flat=np.arange(total) if total<=4096 else rng.choice(total,count,replace=False)
        pi=pos[flat//len(neg)];ni=neg[flat%len(neg)]
        uid=len(inventory)
        inventory.append(dict(unit_id=uid,level=level,category_id=cat,category=str(df.loc[rows[0],'category']),attribute_id=a,attribute=attrs['names'][a],attribute_type=typ,n_positive=len(pos),n_negative=len(neg),possible_pairs=total,sampled_pairs=count,old_supported=len(pos)>=5 and len(neg)>=20))
        for p,n in zip(pi,ni):pairs.append((uid,int(p),int(n),'primary'))
        # Same-image census (bounded to 512/unit), only pairs absent from primary.
        existing=set(zip(pi.tolist(),ni.tolist()));extra=[]
        for image in sorted(set(df.image_id.values[pos])&set(df.image_id.values[neg])):
            pp=pos[df.image_id.values[pos]==image];nn=neg[df.image_id.values[neg]==image]
            extra.extend((int(p),int(n)) for p in pp for n in nn if (p,n) not in existing)
        if len(extra)>512:extra=[extra[i] for i in np.sort(rng.choice(len(extra),512,replace=False))]
        for p,n in extra:pairs.append((uid,p,n,'same_image_supplement'))
    un=pd.DataFrame(inventory);pa=pd.DataFrame(pairs,columns=['unit_id','pos_row','neg_row','sample'])
    pa.insert(0,'pair_id',np.arange(len(pa)))
    for col in ['level','category_id','category','attribute_id','attribute','attribute_type','old_supported']:
        pa[col]=un.set_index('unit_id').loc[pa.unit_id,col].values
    p=pa.pos_row.to_numpy();n=pa.neg_row.to_numpy();a=pa.attribute_id.to_numpy()
    assert (Y[p,a]==1).all() and (Y[n,a]==0).all()
    assert np.array_equal(df.category_id.values[p],df.category_id.values[n])
    for end,idx in [('pos',p),('neg',n)]:
        for col in ['ann_id','image_id','short384','size_bin']:pa[f'{end}_{col}']=df[col].values[idx]
    pa['same_image']=pa.pos_image_id==pa.neg_image_id
    assert not pa.duplicated(['unit_id','pos_row','neg_row']).any()
    un.to_csv(OUT/'category_attribute_inventory.csv',index=False)
    pa.to_parquet(OUT/'pairs.parquet',index=False);pa.to_csv(OUT/'pairs.csv.gz',index=False)
    files=[LEGACY/'experiments/hcf/diagnosis.html',LEGACY/'results/hcf_oracle_gap_2026-11-08/material_ranking_audit.json',LEGACY/'results/hcf_oracle_gap_2026-11-08/matched_mean_probe_val.json',Path('/tmp/hcf_material_audit.py'),ROOT/'cache/paco_lp/manifest_val.parquet',ROOT/'cache/paco_lp/labels_val.npy']
    files.extend(LEGACY/f for f in ['src/hcf/probe.py','src/hcf/model.py','src/ivc/bbdata.py','src/ivc/common.py','src/ivc/train_bb.py','src/viz/c0_pca.py','src/readout/common.py','src/hcf/distill.py','src/hcf/causal_distill.py'])
    for c in conds.values():
        if c['encoder_checkpoint']:files.append(Path(c['encoder_checkpoint']))
        for r in ['mean','r2']:files.extend(Path(c[r][k]) for k in ['checkpoint','scores'])
        files.append(Path(c['probe_dir'])/'config.json')
    hashes={str(p):sha(p) for p in files}
    # Preserve existing analysis sources, without editing the originals.
    for p in files[:4]:shutil.copy2(p,OUT/'source'/p.name)
    checks=dict(val_rows=len(df),val_images=int(df.image_id.nunique()),labels_shape=list(Y.shape),unique_ann_ids=bool(df.ann_id.is_unique),explicit_pair_labels=True,same_category=True,pairs=len(pa),primary_pairs=int((pa['sample']=='primary').sum()),same_image_pairs=int(pa.same_image.sum()),unit_count=len(un),no_test_predictions_loaded=True,no_training=True)
    provenance=dict(conditions=conds,checks=checks,input_sha256=hashes,prior_pair_reuse='Prior ranking used official Test; no Val pair IDs saved. Test IDs and scores not reused. Val common pairs regenerated using same within-category explicit-label concept; unique uniform cap4096/unit seed0; supplemental same-image census cap512/unit. All comparisons share exact pair list.',checkpoint_mismatch='Requested P384/P768 refer to causal r384_cosine/r768_cosine, NOT original HTML hcf_pilot/P. C0 original frozen pretrained DINOv3 crop re-encoding reused.',pooling='Exactly existing R0.pooled: coverage-weighted mean of 24x24 crop-grid features, fp32, then diagnostic L2 normalization. No diagnostic z-score. Existing probe applies own train per-level z-score before linear heads.',bootstrap='Image vertex multinomial resampling B1000 seed0. Cross-image pair weight=m_i*m_j; same-image pair weight=m_i. All pairs sharing any ROI/image share counts. Fixed selected pair graph; conditional diagnostic, sparse graph/pair selection and seed variation not included.')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(checks),flush=True)
    print(un.groupby(['level','attribute_type']).agg(units=('unit_id','size'),pairs=('sampled_pairs','sum')).to_string(),flush=True)

if __name__=='__main__':main()
