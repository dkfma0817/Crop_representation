"""Established official matching + paired image resampling, with point equality gate."""
import argparse,json,sys
import src.pacoattr.boot as B

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--B',type=int,default=1000);a=ap.parse_args()
 B.RUN=a.run
 sys.argv=['boot','--split','test','--conds','P_r2,M_r2,B0_r2,C0_r2','--B',str(a.B),'--evaldir','eval_test','--procs','24','--out',f'{a.run}/boot_test.json']
 B.main()
 r=json.load(open(f'{a.run}/boot_test.json'))
 for c in ['P','M','B0','C0']:
  h=json.load(open(f'{a.run}/eval_test/{c}_r2/test/official_results.json'))['hierarchical']
  assert abs(r['points'][f'{c}_r2']['obj']-h['obj-attr-AP'])<1e-9
  assert abs(r['points'][f'{c}_r2']['part']-h['obj-attr-part-AP'])<1e-9
if __name__=='__main__':main()
