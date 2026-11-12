"""ADE20K linear segmentation of M and P: the unchanged src.g1eval.ade protocol (same head, batches, 40k iterations, eval); only the encoder list differs.
Feature = final-LN coarse tokens (24 x 24 at 384), i.e. the encoder's general representation; the Fine memory is not read."""
import os

import src.g1eval.ade as A
from src.hcf.model import load_hcf

RUN = "/home/irteam/Crop_representation/output/2026-11-08/hcf_pilot"
A.MODELS = ("M", "P")
A.G1 = RUN
A.PACO_PROBE = {"M": "probe_M", "P": "probe_P"}
A.load_encoder = lambda m, dev: load_hcf(os.path.join(RUN, m, "checkpoints", "ema_last.pt"), dev)[0]

if __name__ == "__main__":
    A.main()
