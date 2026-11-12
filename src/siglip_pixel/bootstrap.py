"""The existing PACO paired image bootstrap for S1/S2 and the reused baselines (R2 and Mean probe)."""
import argparse
import json
from pathlib import Path
import sys

import src.pacoattr.boot as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--mean', action='store_true')
    a = ap.parse_args()
    root = Path(a.run) / 'mean' if a.mean else Path(a.run)
    key = 'linear' if a.mean else 'r2'
    names = [f'{c}_{key}' for c in ['P8', 'S1', 'S2', 'B0', 'C0']]
    B.RUN = str(root)
    sys.argv = ['boot', '--split', 'test', '--conds', ','.join(names), '--B', '1000', '--evaldir', 'eval_test', '--procs', '24', '--out', str(root / 'boot_test.json')]
    B.main()
    r = json.loads((root / 'boot_test.json').read_text())
    for c in names:
        h = json.loads((root / f'eval_test/{c}/test/official_results.json').read_text())['hierarchical']
        assert abs(r['points'][c]['obj'] - h['obj-attr-AP']) < 1e-9 and abs(r['points'][c]['part'] - h['obj-attr-part-AP']) < 1e-9


if __name__ == '__main__':
    main()
