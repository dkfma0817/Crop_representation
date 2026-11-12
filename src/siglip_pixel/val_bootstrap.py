"""The existing PACO paired image bootstrap on official Val predictions (R2 or Mean probe), B=1000."""
import argparse
import json
from pathlib import Path
import sys

import src.pacoattr.boot as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--key', required=True, choices=('r2', 'linear'))
    a = ap.parse_args()
    root = Path(a.root)
    names = [f'{c}_{a.key}' for c in ['P8', 'S1', 'S2', 'B0', 'C0']]
    B.RUN = str(root)
    out = root / 'boot_val.json'
    sys.argv = ['boot', '--split', 'val', '--conds', ','.join(names), '--B', '1000', '--evaldir', 'eval_boot', '--procs', '24', '--out', str(out)]
    B.main()
    r = json.loads(out.read_text())
    for c in names:
        h = json.loads((root / f'eval_boot/{c}/val/official_results.json').read_text())['hierarchical']
        assert abs(r['points'][c]['obj'] - h['obj-attr-AP']) < 1e-9 and abs(r['points'][c]['part'] - h['obj-attr-part-AP']) < 1e-9


if __name__ == '__main__':
    main()
