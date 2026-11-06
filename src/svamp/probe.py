"""PACO probe for the spatial-variance pilot = src.ivc.train (11-05 IVC / xdec probe protocol) with install-time hooks only:
  * src.msd.common.build_decoder: arch 'svamp_<cond>' -> src.svamp.feature_transform.SVTransform (no learned weights)
  * env SVAMP_CROPS=1 (O2 only): MSDImageDS always builds the C0 crops of the SAME images of the batch (train / dev / val
    each from its own split) and passes them to the transform as geom['crops'] (never to the reader)

  SVAMP_CROPS=1 torchrun --nproc_per_node 2 -m src.svamp.probe --job <yaml> --out <dir> --exclude <json>
"""
import os

import src.msd.common as MS
from src.svamp.feature_transform import SVTransform

_orig_build = MS.build_decoder


def build_decoder(arch="coordmlp"):
    if str(arch).startswith("svamp_"):
        return SVTransform(arch[len("svamp_"):])
    return _orig_build(arch)


MS.build_decoder = build_decoder

if os.environ.get("SVAMP_CROPS") == "1":
    _init, _get = MS.MSDImageDS.__init__, MS.MSDImageDS.__getitem__

    def __init__(self, df, Y, flip, want_crops):
        _init(self, df, Y, flip, True)

    def __getitem__(self, item):
        out = _get(self, item)
        out["geom"]["crops"] = out.pop("crops")
        return out

    MS.MSDImageDS.__init__, MS.MSDImageDS.__getitem__ = __init__, __getitem__

if __name__ == "__main__":
    import src.ivc.train as IT
    IT.main()
