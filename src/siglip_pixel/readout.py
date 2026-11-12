"""Adapters only: the existing SigLIP2 R2 / Mean probe training and inference for the S1/S2 P-architecture checkpoints.

usage: python -m src.siglip_pixel.readout {r2,mean,infer,mean_infer} <the existing module's arguments>
"""
import sys

import src.siglip_pm.probe as BASE

ORIGINAL_LOAD = BASE.load_frozen


def load_frozen(job, dev):
    mapped = dict(job)
    if job['cond'] in ('S1', 'S2'):
        mapped['cond'] = 'P'
    model = ORIGINAL_LOAD(mapped, dev)
    model.roi_mode = job['cond']
    return model


def main():
    kind = sys.argv.pop(1)
    if kind == 'r2':
        BASE.load_frozen = load_frozen
        BASE.main()
    elif kind == 'mean':
        import src.siglip_pm.mean_probe as MEAN
        MEAN.BASE.load_frozen = load_frozen
        MEAN.main()
    elif kind == 'infer':
        import src.siglip_extra.infer as INFER
        INFER.load_frozen = load_frozen
        INFER.main()
    elif kind == 'mean_infer':
        import src.siglip_extra.mean_infer as MEAN_INFER
        MEAN_INFER.load_frozen = load_frozen
        MEAN_INFER.main()
    else:
        raise SystemExit(f'unknown kind {kind}')


if __name__ == '__main__':
    main()
