"""Run the unchanged established 384 ADE20K probe on exported standard encoders only."""
import os
from pathlib import Path
import sys
import src.g1eval.ade as A
from src.fgtrain.common import RUN, encoder


def main():
    # Wrapper-specific options removed before calling the existing parser/evaluator.
    import argparse
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--run',default=str(RUN));p.add_argument('--models',default='B0,B1')
    args,rest=p.parse_known_args()
    root=Path(args.run)
    A.MODELS=tuple(args.models.split(','))
    A.G1=str(root)
    A.PACO_PROBE={m:f'{m}/probe' for m in A.MODELS}
    A.load_encoder=lambda m,dev:encoder(root/m/'export/encoder.pt',dev)
    sys.argv=[sys.argv[0]]+rest
    A.main()

if __name__=='__main__':main()
