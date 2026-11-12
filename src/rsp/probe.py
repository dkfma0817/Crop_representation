"""Run the EXISTING R2 probe (src.tsplit.train_bb) and official-manifest inference (src.tsplit.official_infer) for mode 'RSP' by
swapping only their encoder (RSPRoiEncoder) and dataset (RSPContentDS = ContentDS + GT-box evidence / canvas box). Data, dev split,
sampler, flips, reader, optimiser, schedule, epochs, Val scoring and per-epoch checkpoints are the unchanged existing code.

  python -m src.rsp.probe train_bb --job <probe>/job.yaml --out <probe> --exclude ... --workers 8
  python -m src.rsp.probe official_infer --root <eval> --probe <probe> --name RSPPIX --split val
"""
import sys

import torch
import yaml

from src.rsp.data import RSPContentDS
from src.rsp.encoder import RSPRoiEncoder


def _want_ev(job_path):
    job = yaml.safe_load(open(job_path))
    if job.get("mode") != "RSP":
        return False
    return bool(torch.load(job["split_ckpt"], map_location="cpu", weights_only=False)["pix"])


def main():
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "train_bb":
        import src.tsplit.train_bb as M
        job = sys.argv[sys.argv.index("--job") + 1]
        RSPContentDS.WANT_EV = _want_ev(job)
        M.RoiEncoder, M.BBImageDS = RSPRoiEncoder, RSPContentDS
    elif cmd == "official_infer":
        import src.tsplit.official_infer as M
        probe = sys.argv[sys.argv.index("--probe") + 1]
        RSPContentDS.WANT_EV = _want_ev(f"{probe}/job.yaml")
        M.RoiEncoder, M.ContentDS = RSPRoiEncoder, RSPContentDS
    else:
        raise SystemExit(f"unknown command {cmd}")
    M.main()


if __name__ == "__main__":
    main()
