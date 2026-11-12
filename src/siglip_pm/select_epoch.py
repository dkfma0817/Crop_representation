"""Official VAL only. Same rule for every independently trained reader."""
import glob,json,sys,os
run=sys.argv[1];sel={};curves={}
for d in sorted(glob.glob(f'{run}/eval_val/*')):
 name=os.path.basename(d);c={}
 for e in range(1,11):
  h=json.load(open(f'{d}/ep{e}/official_results.json'))['hierarchical']
  c[e]=dict(obj=h['obj-attr-AP'],part=h['obj-attr-part-AP'])
 ep=max(c,key=lambda e:(round((c[e]['obj']+c[e]['part'])/2,10),-e))
 sel[name]=dict(epoch=ep,val_obj=c[ep]['obj'],val_part=c[ep]['part'],rule='max (official VAL object AP + part AP)/2; earlier epoch on ties')
 curves[name]=c
json.dump(sel,open(f'{run}/selection.json','w'),indent=1)
json.dump(curves,open(f'{run}/val_curves.json','w'),indent=1)
print(json.dumps(sel),flush=True)
