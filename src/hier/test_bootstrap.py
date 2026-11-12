"""The existing PACO paired image bootstrap on official Test (B=1000) for any fixed model list; used by the H0/H1 and S1/S2 follow-up Test queues."""
import argparse
import json
from pathlib import Path
import sys

import src.pacoattr.boot as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--key', required=True, choices=('r2', 'linear'))
    ap.add_argument('--models', required=True)
    a = ap.parse_args()
    root = Path(a.root)
    names = [f'{c}_{a.key}' for c in a.models.split(',')]
    out = root / 'boot_test.json'
    assert not out.exists(), out
    B.RUN = str(root)
    sys.argv = ['boot', '--split', 'test', '--conds', ','.join(names), '--B', '1000', '--evaldir', 'eval_test', '--procs', '24', '--out', str(out)]
    B.main()
    r = json.loads(out.read_text())
    for c in names:
        h = json.loads((root / f'eval_test/{c}/test/official_results.json').read_text())['hierarchical']
        assert abs(r['points'][c]['obj'] - h['obj-attr-AP']) < 1e-9 and abs(r['points'][c]['part'] - h['obj-attr-part-AP']) < 1e-9


if __name__ == '__main__':
    main()
