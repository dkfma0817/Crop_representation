"""Thin CLI binding to the project's existing paired official bootstrap."""
import argparse
from pathlib import Path
import sys
import src.pacoattr.boot as B


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--conds',default='B0,B1');a=p.parse_args()
    B.RUN=a.run
    sys.argv=['bootstrap','--split','val','--conds',a.conds,'--evaldir','bootstrap_inputs','--B','1000','--procs','16','--out',str(Path(a.run)/'paired_bootstrap.json')]
    B.main()

if __name__=='__main__':main()
