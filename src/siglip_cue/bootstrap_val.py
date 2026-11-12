"""Paired image bootstrap (B=1000) on PACO Official Val, GT-ROI / GT-Category oracle, at each model's selected epoch.

Uses the existing src.pacoattr.boot re-accumulation unchanged; the only addition is that per-attribute AP is kept for every resample (keys obj_attr::<name>, part_attr::<name>).
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from paco.evaluation.utils.paco_utils import heirachrichal_APs
import src.pacoattr.boot as B

MODELS = ['B0', 'P8', 'M1', 'M2', 'C0']


def hier_with_attributes(P):
    ev = SimpleNamespace(lvis_gt=B.G['gt'], lvis_dt=SimpleNamespace(dataset={'attr_type_to_attr_idxs': B.G['gt'].dataset['attr_type_to_attr_idxs']}), eval_joint_attr={'precision': P[..., None]})
    r = heirachrichal_APs(ev)
    out = dict(obj=r['obj-attr-AP'], part=r['obj-attr-part-AP'], **{f'obj_{k}': v for k, v in r['obj-attrs'].items()}, **{f'part_{k}': v for k, v in r['obj-part-attrs'].items()})
    out.update({f'obj_attr::{k}': v for k, v in r['obj-attr-AP-per-class'].items()})
    out.update({f'part_attr::{k}': v for k, v in r['obj-attr-part-AP-per-class'].items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    root = Path(a.run)
    names = [f'{c}_r2' for c in MODELS]
    B.RUN = str(root)
    B.hier = hier_with_attributes
    out = root / 'boot_val.json'
    sys.argv = ['boot', '--split', 'val', '--conds', ','.join(names), '--B', '1000', '--evaldir', 'eval_val_boot', '--procs', '24', '--out', str(out)]
    B.main()
    r = json.loads(out.read_text())
    for c in names:
        h = json.loads((root / f'eval_val_boot/{c}/val/official_results.json').read_text())['hierarchical']
        assert abs(r['points'][c]['obj'] - h['obj-attr-AP']) < 1e-9 and abs(r['points'][c]['part'] - h['obj-attr-part-AP']) < 1e-9


if __name__ == '__main__':
    main()
