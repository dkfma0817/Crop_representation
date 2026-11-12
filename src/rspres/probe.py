"""Existing R2 probe / official inference with the RSP-Residual read (ResRoiEncoder) and RSPContentDS (evidence only for 'RSPRES').

  python -m src.rspres.probe train_bb --job <probe>/job.yaml --out <probe> --exclude ... --workers 8
  python -m src.rspres.probe official_infer --root <eval> --probe <probe> --name RSPRES --split val
"""
import sys

import yaml

from src.rsp.data import RSPContentDS
from src.rspres.encoder import ResRoiEncoder


def main():
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "train_bb":
        import src.tsplit.train_bb as M
        job = sys.argv[sys.argv.index("--job") + 1]
        M.RoiEncoder, M.BBImageDS = ResRoiEncoder, RSPContentDS
    elif cmd == "official_infer":
        import src.tsplit.official_infer as M
        job = sys.argv[sys.argv.index("--probe") + 1] + "/job.yaml"
        M.RoiEncoder, M.ContentDS = ResRoiEncoder, RSPContentDS
    else:
        raise SystemExit(f"unknown command {cmd}")
    RSPContentDS.WANT_EV = yaml.safe_load(open(job))["mode"] == "RSPRES"
    M.main()


if __name__ == "__main__":
    main()
