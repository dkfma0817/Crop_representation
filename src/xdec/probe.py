"""PACO probe for the decoder-architecture pilot = src.ivc.train (11-05 IVC / MSD probe protocol, R0 + R2, per-epoch
checkpoints, dev-selected epoch) with two install-time hooks and NOTHING else changed:
  * src.xdec.geometry.install(): batches carry roi_frame (needed by the E-series position encoding)
  * src.msd.common.build_decoder: arch 'xdec_<cond>' -> src.xdec.models (E1-E7, R00)
Encoder and decoder are frozen inside src.ivc.train (requires_grad False, checksum before / after -> frozen_check.json).

  torchrun --nproc_per_node 2 -m src.xdec.probe --job <yaml> --out <dir> --exclude <json>
"""
import src.msd.common as MS
from src.xdec import geometry as XG
from src.xdec import models as XM

XG.install()
_orig_build = MS.build_decoder


def build_decoder(arch="coordmlp"):
    if str(arch).startswith("xdec_"):
        return XM.build_from_arch(arch)
    return _orig_build(arch)


MS.build_decoder = build_decoder

if __name__ == "__main__":
    import src.ivc.train as IT
    IT.main()
