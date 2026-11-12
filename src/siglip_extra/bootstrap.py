"""Paired image bootstrap reusing official matching, for R2 AND the mean probe."""
import argparse,json,sys
from pathlib import Path
import src.pacoattr.boot as B

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--mean',action='store_true');ap.add_argument('--control',action='store_true');a=ap.parse_args()
 root=Path(a.run)/'mean' if a.mean else Path(a.run);models=['P','E1','E2','C0']+(['Continue'] if a.control else []);key='linear' if a.mean else 'r2';B.RUN=str(root)
 sys.argv=['boot','--split','test','--conds',','.join(f'{c}_{key}' for c in models),'--B','1000','--evaldir','eval_test','--procs','24','--out',str(root/'boot_test.json')];B.main()
 r=json.load(open(root/'boot_test.json'))
 for c in models:
  h=json.load(open(root/f'eval_test/{c}_{key}/test/official_results.json'))['hierarchical']
  assert abs(r['points'][f'{c}_{key}']['obj']-h['obj-attr-AP'])<1e-9
  assert abs(r['points'][f'{c}_{key}']['part']-h['obj-attr-part-AP'])<1e-9
if __name__=='__main__':main()
