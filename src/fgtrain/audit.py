"""Target immutability and source integrity checks, independent of model metrics."""
import hashlib
import json
from pathlib import Path
import select
import os
from src.fgtrain.common import write_json


def freeze_targets(run):
    run=Path(run)
    sha={}
    for name in ['pooled','g_cls','g_mean']:
        h=hashlib.sha256()
        with (run/'teacher'/f'{name}.npy').open('rb') as file:
            for block in iter(lambda:file.read(4*1024*1024),b''):h.update(block)
        sha[name]=h.hexdigest()
    write_json(run/'target_freeze.json',dict(sha256=sha,rule='Finite pooled/global targets stay immutable for independent B0/B1'))


def target_audit(run):
    run=Path(run)
    expected=json.loads((run/'target_freeze.json').read_text())['sha256']
    actual={}
    for name in expected:
        h=hashlib.sha256()
        with (run/'teacher'/f'{name}.npy').open('rb') as file:
            for block in iter(lambda:file.read(4*1024*1024),b''):h.update(block)
        actual[name]=h.hexdigest()
    assert actual==expected,('pooled/global teacher targets changed',expected,actual)
    return dict(unchanged=True,sha256=actual)


def await_spatial_cache(run):
    """Wait on the owned cache worker's pidfd if a decoder control reaches it early."""
    import numpy as np
    run=Path(run)
    if np.load(run/'teacher/completed.npy').all():return
    active=json.loads((run/'active.json').read_text())
    task=active.get('teacher_full')
    if task:
        try:
            fd=os.pidfd_open(task['pid'])
        except ProcessLookupError:fd=None
        if fd is not None:
            try:select.select([fd],[],[])
            finally:os.close(fd)
    assert np.load(run/'teacher/completed.npy').all(),'Spatial cache not complete; decoder training is not allowed'
