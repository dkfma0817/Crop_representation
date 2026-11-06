"""PACO probe for the E6 multi-layer memory pilot = src.xdec.probe (src.ivc.train, roi_frame geometry) + arch 'e6ml_<A|B>'."""
import src.msd.common as MS
import src.xdec.probe  # noqa: F401  (installs geometry + the xdec_ build hook)
from src.e6ml import models as EM

_prev = MS.build_decoder


def build_decoder(arch="coordmlp"):
    if str(arch).startswith("e6ml_"):
        return EM.build_from_arch(arch)
    return _prev(arch)


MS.build_decoder = build_decoder

if __name__ == "__main__":
    import src.ivc.train as IT
    IT.main()
