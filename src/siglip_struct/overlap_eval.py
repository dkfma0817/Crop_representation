"""Evaluate each completed reader while the slower reader finishes unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback

from src.siglip_struct.stages import Command


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run);cmd=Command(run)
    deadline=json.loads((run/'budget.json').read_text())['deadline_unix']
    def one(c,gpu):
        try:
            ready=run/'state'/f'infer_{c}_val.done.json'
            while not ready.exists():
                if time.time()>deadline:raise TimeoutError(c+' VAL not ready within budget')
                if (run/'queue_failed.json').exists():raise RuntimeError('Main queue failed')
                time.sleep(2)
            def official(split,e):
                args=['--split',split,'--pred',run/f'preds/{c}_r2/ep{e}/probs_{split}.npz',
                      '--out',run/f'eval_{split}/{c}_r2'/('test' if split=='test' else f'ep{e}')]
                if split=='test':args+=['--boot']
                cmd(f'official_{c}_{split}_{e}','src.pacoattr.official_eval',args,official=True)
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda e:official('val',e),range(1,11)))
            h={e:json.loads((run/f'eval_val/{c}_r2/ep{e}/official_results.json').read_text())['hierarchical'] for e in range(1,11)}
            best=max(h,key=lambda e:(round((h[e]['obj-attr-AP']+h[e]['obj-attr-part-AP'])/2,10),-e))
            selection={f'{c}_r2':dict(epoch=best,val_obj=h[best]['obj-attr-AP'],val_part=h[best]['obj-attr-part-AP'],rule='max (official VAL object AP + part AP)/2; earlier epoch on ties')}
            path=run/f'selection_{c}.json';path.write_text(json.dumps(selection,indent=2))
            cmd(f'infer_{c}_test','src.siglip_struct.readout',['infer','--run',run,'--job',run/f'scripts/probe_{c}.yaml','--split','test','--select',path,'--workers',8],gpu)
            official('test',best)
            # Same fixed TEST predictions; run the requested size diagnostic
            # early as well. The size module can use the per-model selection.
            cmd(f'roi_size_{c}','src.siglip_struct.size',['--run',run,'--model',c,'--select',path],official=True)
            (run/'state'/f'early_eval_{c}.done.json').write_text(json.dumps(dict(status='done',model=c,epoch=best,time=time.time()),indent=2))
        except Exception:
            (run/'state'/f'early_eval_{c}.failed.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2));raise
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(one,c,g) for c,g in [('P4',2),('Native',3),('Decode',3)]]
        for f in futures:f.result()


if __name__=='__main__':main()
