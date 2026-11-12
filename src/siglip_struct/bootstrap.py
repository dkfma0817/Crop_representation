import argparse
import json
from pathlib import Path
import sys
import src.pacoattr.boot as B


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();B.RUN=a.run
    names=[c+'_r2' for c in ['P8','P4','Native','Decode','M','B0','C0']]
    sys.argv=['boot','--split','test','--conds',','.join(names),'--B','1000','--evaldir','eval_test','--procs','24','--out',str(Path(a.run)/'boot_test.json')]
    B.main();r=json.loads((Path(a.run)/'boot_test.json').read_text())
    for c in names:
        h=json.loads((Path(a.run)/f'eval_test/{c}/test/official_results.json').read_text())['hierarchical']
        assert abs(r['points'][c]['obj']-h['obj-attr-AP'])<1e-9 and abs(r['points'][c]['part']-h['obj-attr-part-AP'])<1e-9
