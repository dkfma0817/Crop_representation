"""Run the EXISTING R2 probe (src.tsplit.train_bb) and official-manifest inference (src.tsplit.official_infer) with the dense teacher
read (DTRoiEncoder) and RSPContentDS (ContentDS + canvas GT box 'boxc'; no evidence). Everything else is the unchanged existing code.

  python -m src.dtev.probe train_bb --job <probe>/job.yaml --out <probe> --exclude ... --workers 8
  python -m src.dtev.probe official_infer --root <eval> --probe <probe> --name DTM8 --split val
"""
import sys

from src.dtev.encoder import DTRoiEncoder
from src.rsp.data import RSPContentDS


def main():
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    RSPContentDS.WANT_EV = False
    if cmd == "train_bb":
        import src.tsplit.train_bb as M
        M.RoiEncoder, M.BBImageDS = DTRoiEncoder, RSPContentDS
    elif cmd == "official_infer":
        import src.tsplit.official_infer as M
        M.RoiEncoder, M.ContentDS = DTRoiEncoder, RSPContentDS
    else:
        raise SystemExit(f"unknown command {cmd}")
    M.main()


if __name__ == "__main__":
    main()
