"""Official-protocol Dev AP inference for decoder-study paths = src.paco_devap.infer_dev (unchanged code path) with the
encoder loader replaced by src.decstudy.paths.load_path (bilinear / residual / existing decoder / encoder-only).

  CUDA_VISIBLE_DEVICES=<gpu> python -m src.decstudy.infer_dev --probe <probe dir> --name <name> --epochs 6 --out <dir>
"""
import src.paco_devap.infer_dev as ID
from src.decstudy.infer import job_of
from src.decstudy.paths import load_path


def load_encoder(cfg, dev):
    vit, dec, info = load_path(job_of(cfg), dev)
    return vit, dec, info


ID.load_encoder = load_encoder

if __name__ == "__main__":
    ID.main()
