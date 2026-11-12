"""Measured fixed384 throughput and explicit dense/attention FLOP convention."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd
import torch

from src.siglip_pm.model import n_params
from src.siglip_pm.model import checksum
from src.siglip_pm.efficiency import costs
from src.siglip_struct.readout import Frozen
from src.siglip_struct.data import BBImageDS
from src.siglip_struct.geometry import native_tokens


def encoder_cost(kind):
    if kind!='P4':return costs('P')['encoder_GFLOPs']
    n,nf,d,f=576,9216,768,384
    extra=nf*f*3*4**2+4*(12*nf*f*f+2*nf*256*f)+4*(n*f*d+nf*d*f)
    return costs('B0')['encoder_GFLOPs']+2*extra/1e9


def decoder_cost(t):
    d,q=768,16
    key_geom=t*(6*d+d*d)
    query_geom=q*(8*d+d*d)
    cross=2*((2*q+2*t)*d*d+2*q*t*d+4*q*d*d)
    return 2*(key_geom+query_geom+cross+q*d*d)/1e9


def r2_cost(t):
    d=768
    geometry=t*(5*d+d*d)+(4*d+d*d)
    blocks=2*((2+2*t)*d*d+2*t*d+4*d*d)
    return 2*(geometry+blocks+59*d)/1e9


def timing(fn):
    values=[]
    for i in range(25):
        torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize()
        if i>=5:values.append((time.perf_counter()-t)*1000)
    return statistics.median(values)


def run(a):
    import yaml
    run=Path(a.run);modes=['P4'] if a.kind=='P4' else ['Native','Decode'];rows=[]
    manifest=pd.read_parquet('output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet')
    ds=BBImageDS(manifest,np.zeros((len(manifest),59),np.int8),False,'bil',384,16)
    import src.readout.common as RC
    b=RC.to_dev(ds[[0]],'cuda')
    for mode in modes:
        job=yaml.safe_load((run/f'scripts/probe_{mode}.yaml').read_text());model=Frozen(job,'cuda')
        encoder=model.encoder
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            ms=timing(lambda:encoder.forward_all(b['x']))
            o=encoder.forward_all(b['x'])
            scalars=o['coarse'][0].numel()+o['mem']['fine'][0].numel()
            if mode=='P4':
                # The live first worker was loaded before the metadata field
                # became nullable. P4 is trainable, so 'frozen encoder unchanged'
                # is inapplicable; verify its actual saved coarse weights changed.
                completion=json.loads((run/'P4/done.json').read_text())
                config=json.loads((run/'P4/config.json').read_text())
                final_sum=checksum(encoder.vit.parameters())
                completion.update(encoder_unchanged=None,coarse_encoder_trained=final_sum!=config['initial_encoder_checksum'],final_coarse_checksum=final_sum)
                (run/'P4/done.json').write_text(json.dumps(completion,indent=2))
                memory=dict(coarse=o['coarse'][0].float().cpu().reshape(24,24,768),fine=o['mem']['fine'][0].float().cpu().reshape(96,96,384))
                (run/'feature_examples').mkdir(exist_ok=True)
                torch.save(memory,run/'feature_examples/P4.pt')
                row=dict(model=mode,encoder_parameters=n_params(encoder),additional_decoder_parameters=0,
                         encoder_GFLOPs=encoder_cost(mode),encoder_ms=ms,readout_tokens=576,
                         R2_GFLOPs_per_ROI=r2_cost(576),ROI_fusion_GFLOPs_per_ROI=2*576*384*768/1e9)
            else:
                o['fine_projected']=encoder.Wd(o['mem']['fine'])
                tok,_,_,mask=native_tokens(o,encoder,b['img_index'],b['box_c'],b['valid_rect'],0 if mode=='Native' else .25)
                lengths=mask.sum(-1).float()
                if mode=='Native':
                    fn=lambda:native_tokens(o,encoder,b['img_index'],b['box_c'],b['valid_rect'])
                    read_ms=timing(fn);decoder_gflops=0.;extra_params=0;tokens=float(lengths.mean())
                else:
                    fn=lambda:model.decoder(o,encoder,b['img_index'],b['box_c'],b['valid_rect'])
                    read_ms=timing(fn);decoder_gflops=decoder_cost(float(lengths.mean()));extra_params=n_params(model.decoder);tokens=16
                    z=fn().float();s=torch.nn.functional.normalize(z,dim=-1)
                    off=~torch.eye(16,dtype=torch.bool,device='cuda')
                    (run/'decoder_token_diagnostic.json').write_text(json.dumps(dict(mean_pairwise_slot_cosine=float((s@s.transpose(-1,-2))[:,off].mean()),spatial_feature_std=float(z.std(1).mean()),note='Mean-only supervision does not guarantee position-specific slot semantics.'),indent=2))
                row=dict(model=mode,encoder_parameters=n_params(encoder),additional_decoder_parameters=extra_params,
                         encoder_GFLOPs=encoder_cost(mode),encoder_ms=ms,readout_tokens=tokens,
                         selected_key_tokens_mean=float(lengths.mean()),selected_key_tokens_p90=float(torch.quantile(lengths,.9)),
                         selection_or_decoder_ms_per_ROI=read_ms/len(b['box_c']),decoder_GFLOPs_per_ROI=decoder_gflops,
                         frozen_fine_projection_GFLOPs_per_image=2*2304*384*768/1e9,R2_GFLOPs_per_ROI=r2_cost(tokens))
        cfg=json.loads((run/f'probe_{mode}/config.json').read_text());row['R2_parameters']=cfg['reader_params']['r2']
        row.update(stored_scalars=scalars,feature_MiB_fp32=scalars*4/2**20,feature_MiB_fp16=scalars*2/2**20,
                   timing_sample_images=1,timing_sample_ROIs=len(b['box_c']),flop_convention='2FLOPs/MAC; dense/conv/attention; excludes LN/activation/softmax/selection. Native counts are sample-dependent; timing includes Python/native selection overhead.')
        rows.append(row);del model;torch.cuda.empty_cache()
    (run/f'efficiency_{a.kind}.json').write_text(json.dumps(rows,indent=2));pd.DataFrame(rows).to_csv(run/f'results/efficiency_{a.kind}.csv',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--base',required=True);ap.add_argument('--kind',choices=['P4','Decode'],required=True);a=ap.parse_args()
    from src.siglip_extra.resources import gpu_lease
    with gpu_lease(a.run):run(a)
