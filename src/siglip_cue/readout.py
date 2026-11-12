"""Adapters only: the existing SigLIP2 R2 training and inference for M1 (P architecture) and M2 (P + Detail, native image supplied per image).

usage: python -m src.siglip_cue.readout {r2,infer} <the existing module's arguments>
"""
import os
import sys

from PIL import Image
import torch

from src.siglip_cue.data import native_canvas, native_geo
from src.siglip_cue.model import load_m2
import src.siglip_pm.bbdata as BBD
import src.siglip_pm.probe as BASE

ORIGINAL_LOAD = BASE.load_frozen
ORIGINAL_ENCODE = BASE.encode


def load_frozen(job, dev):
    if job['cond'] == 'M2':
        model, ck = load_m2(job['ckpt'], dev)
        assert ck['kind'] == 'ema'
    else:
        mapped = dict(job, cond='P') if job['cond'] == 'M1' else dict(job)
        model = ORIGINAL_LOAD(mapped, dev)
    model.roi_mode = job['cond']
    return model


class BBNativeDS(BBD.BBImageDS):
    """The existing 'bil' dataset; additionally the unresized image of each group, with the same horizontal flip as the 384 canvas."""

    def __getitem__(self, idx):
        state = torch.get_rng_state()
        out = super().__getitem__(idx)
        torch.set_rng_state(state)
        native, geo = [], []
        for gi in idx:
            rows = self.groups[gi]
            img = Image.open(self.p[rows[0]]).convert('RGB')
            W, H = img.size
            if bool(self.flip and torch.rand(1).item() < .5):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            native.append(native_canvas(__import__('numpy').asarray(img)))
            geo.append(native_geo(W, H))
        out.update(native=torch.stack(native), native_geo=torch.stack(geo))
        return out


def dataset(job_cond):
    return BBNativeDS if job_cond == 'M2' else BBD.BBImageDS


def encode_native(model, b, keys, drop_fine=False):
    if getattr(model, 'roi_mode', '') != 'M2':
        return ORIGINAL_ENCODE(model, b, keys, drop_fine)
    model.set_pending(b['native'], b['native_geo'])
    try:
        return ORIGINAL_ENCODE(model, b, keys, drop_fine)
    finally:
        model.clear_pending()


def main():
    kind = sys.argv.pop(1)
    if '--split' in sys.argv and sys.argv[sys.argv.index('--split') + 1] == 'test' and os.environ.get('SIGLIP_CUE_ALLOW_TEST') != '1':
        raise SystemExit('PACO Official Test needs SIGLIP_CUE_ALLOW_TEST=1, set only by src.siglip_cue.queue_test (user request 2026-11-09: Test at the Val-selected epoch).')
    if '--job' in sys.argv:
        import yaml
        cond = yaml.safe_load(open(sys.argv[sys.argv.index('--job') + 1]))['cond']
    else:
        cond = ''
    if kind == 'r2':
        BASE.load_frozen = load_frozen
        BASE.encode = encode_native
        BASE.BBImageDS = dataset(cond)
        BASE.main()
    elif kind == 'infer':
        import src.siglip_extra.infer as INFER
        INFER.load_frozen = load_frozen
        INFER.encode = encode_native
        INFER.BBImageDS = dataset(cond)
        INFER.main()
    else:
        raise SystemExit(f'unknown kind {kind}')


if __name__ == '__main__':
    main()
