"""Acquire existing background scheduler's checkpoint pause, then validate on real data."""
import json
from pathlib import Path
import subprocess
import sys
import os
from src.fgtrain.common import RUN, write_json
from src.brixel_eval.ade_run import CONTROL, active, await_paused


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN));args=p.parse_args()
    root=Path(args.run);flag=CONTROL/'PAUSE_all';record=root/'resource_ownership.json'
    if not record.exists():
        owned=not flag.exists()
        write_json(root/'background_before.json',active())
        if owned:
            flag.write_text('fgtrain_s0 owns checkpoint pause; released by src.fgtrain.pipeline on completion/failure\n')
        write_json(record,dict(owned=owned,payload=flag.read_text(),flag=str(flag)))
    write_json(root/'preflight_state.json',dict(phase='waiting_for_background_checkpoint',pid=os.getpid()))
    write_json(root/'background_pause_verification.json',await_paused(json.loads((root/'background_before.json').read_text())))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1')
    stages=[('initial_teacher','src.fgtrain.cache',['--root',str(root/'teacher'),'--initial',str(root/'initial_encoder.pt'),'--ids-file',str(root/'first_stream_ids.json'),'--workers','4']),
            ('dev_teacher','src.fgtrain.cache',['--root',str(root/'teacher'),'--initial',str(root/'initial_encoder.pt'),'--split','dev','--workers','4']),
            ('smoke','src.fgtrain.checks',['--run',str(root),'--cache',str(root/'teacher')]),
            ('calibration','src.fgtrain.train',['--run',str(root),'--cache',str(root/'teacher'),'--out',str(root/'calibration'),'--calibrate'])]
    for name,module,args in stages:
        write_json(root/'preflight_state.json',dict(phase=name,pid=os.getpid()))
        with (root/f'{name}.log').open('a') as log:
            result=subprocess.run([sys.executable,'-u','-m',module]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(root/'preflight_state.json',dict(phase=name,status='failed',returncode=result.returncode))
            raise RuntimeError(name)
    from src.fgtrain.audit import freeze_targets
    freeze_targets(root)
    write_json(root/'preflight_state.json',dict(status='passed',phase='complete',pid=os.getpid()))

if __name__=='__main__':main()
