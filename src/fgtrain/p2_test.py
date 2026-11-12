"""User-requested final P2 Test reporting with immutable Val selection."""
import hashlib
import json
from pathlib import Path
import signal
import sys
import traceback

from src.fgtrain.common import ROOT, write_json
from src.fgtrain.pipeline import Pipeline


SOURCE = ROOT / 'output/2026-11-11/fgtrain_p23_s0'
TARGET = SOURCE / 'test'
REPORT = ROOT / 'results/p2_paco_test'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    TARGET.mkdir(exist_ok=True)
    pipe = Pipeline(TARGET)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))
    try:
        selection = json.loads((SOURCE / 'P2/selection.json').read_text())
        assert selection['epoch'] == 8 and selection['test_used'] is False
        inputs = [SOURCE / 'P2/selection.json', SOURCE / 'P2/export/encoder.pt',
                  SOURCE / 'P2/export/encoder_eval.pt', SOURCE / 'P2/export/export_verification.json',
                  SOURCE / 'P2/probe/r2/ep8.pt', SOURCE / 'P2/probe/config.json',
                  SOURCE / 'results.json', SOURCE / 'P2/config.json',
                  ROOT / 'src/fgtrain/pipeline.py', ROOT / 'src/fgtrain/infer.py',
                  ROOT / 'src/pacoattr/official_eval.py', Path(__file__)]
        hashes = {str(path): sha(path) for path in inputs}
        freeze = TARGET / 'protected_inputs.json'
        if freeze.exists():
            assert json.loads(freeze.read_text()) == hashes, 'P2 Test source changed'
        else:
            write_json(freeze, hashes)
        pipe.run_final_test(SOURCE, methods=('P2',), report_module=None, once=True)
        inference = json.loads((TARGET / 'P2/preds/inference_verification.json').read_text())
        assert inference['status'] == 'passed' and inference['split'] == 'test'
        assert inference['reader_epochs'] == [8] and inference['standard_encoder']
        assert inference['counts']['images'] == inference['counts']['encoder_canvases']
        assert not inference['roi_rgb_reencoding'] and not inference['teacher_instantiated']
        official = json.loads((TARGET / 'P2/official/official_results.json').read_text())
        assert official['split'] == 'test' and official['alignment_mismatch'] == 0
        assert all(sha(path) == digest for path, digest in hashes.items())
        result = json.loads((TARGET / 'results.json').read_text())
        h = official['hierarchical']
        result['metrics']['P2'].update(obj_ref=h['obj-attrs']['transparency'],
                                      part_ref=h['obj-part-attrs']['transparency'])
        result.update(n_eval_images=official['n_eval_images'], inference=inference,
                      protected_inputs_unchanged=True, protected_inputs=hashes,
                      official_results=str(TARGET / 'P2/official/official_results.json'))
        write_json(TARGET / 'reporting_results.json', result)
        REPORT.mkdir(parents=True, exist_ok=True)
        write_json(REPORT / 'results.json', result)
        columns = ['object','obj_color','obj_pattern','obj_material','obj_ref',
                   'part','part_color','part_pattern','part_material','part_ref']
        header = ['Model','Obj AP','Obj AP col','Obj AP pat','Obj AP mat','Obj AP ref',
                  'Part AP','Part AP col','Part AP pat','Part AP mat','Part AP ref']
        scores = result['metrics']['P2']
        (REPORT / 'results.tsv').write_text('\t'.join(header) + '\nP2\t' +
            '\t'.join(f'{scores[k]:.6f}' for k in columns) + '\n')
        write_json(TARGET / 'reporting_done.json', dict(status='done', split='test',
            reader_epoch=8, protected_inputs_unchanged=True, results=str(REPORT / 'results.json')))
        print(json.dumps(result['metrics'], ensure_ascii=False), flush=True)
    except BaseException:
        write_json(TARGET / 'failed.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        pipe.terminate_children()
        pipe.cpu.shutdown(wait=True)


if __name__ == '__main__':
    main()
