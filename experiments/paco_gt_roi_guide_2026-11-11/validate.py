"""Validate report data, sources and offline links. Does not run any model."""
import ast
import csv
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


class Document(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.links=[]; self.scripts=[]

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'a' and 'href' in a:
            self.links.append(a['href'])
        if tag in ['script', 'img', 'link']:
            self.scripts.append(a)


def main():
    html = (OUT / 'index.html').read_text()
    doc = Document(); doc.feed(html)
    assert len(doc.ids) == len(set(doc.ids))
    broken = []
    for link in doc.links:
        if link == 'validation.json':
            continue
        if link.startswith('#'):
            if link[1:] not in doc.ids:
                broken.append(link)
        elif not (OUT / link).exists():
            broken.append(link)
    assert not broken, broken
    assert not re.search(r'/\*__[A-Z]+__\*/', html)
    assert all(not a.get('src', '').startswith(('http:', 'https:')) for a in doc.scripts)
    d = json.loads((OUT / 'data.json').read_text())
    assert len(d['names']) == 59 and sum(map(len, d['types'].values())) == 59
    assert len(d['examples']) == 6 and len(d['images']) == 4
    assert len(list(csv.DictReader((OUT/'example_predictions.csv').open()))) == 354
    assert all(all(v[k] for k in ['ann_id', 'image_id', 'category_id']) for v in d['checks'].values())
    assert all(v['sigmoid_max_error'] < 1e-6 for v in d['checks'].values())
    changed = [s['path'] for s in d['sources'] if hashlib.sha256(Path(s['path']).read_bytes()).hexdigest() != s['sha256']]
    assert not changed, changed
    # Use the exact existing project AP kernel, extracted without its legacy import side effects.
    kernel = ROOT/'third_party/research/2026-10-24/dinov2_oracle_region_decoder/paco_lp_official_metric/code/pom_common.py'
    tree = ast.parse(kernel.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'ap_unit')
    namespace = {'np': np, 'REC_THRS': np.linspace(0,1,101)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(kernel),'exec'), namespace)
    fn = namespace['ap_unit']
    cases = {'initial':[.91,.83,.72,.41,.36], 'perfect':[.95,.25,.85,.15,.75], 'reversed':[.45,.95,.35,.85,.25],
             'compressed':[.491,.483,.472,.441,.436], 'ties':[.5,.5,.5,.5,.5]}
    known = np.array([1,0,1,0,1],bool)
    expected = {key:float(fn(np.array(scores),known,np.ones((1,5)),('coco',))['coco'][0]) for key,scores in cases.items()}
    assert abs(expected['perfect']-1)<1e-12
    assert abs(expected['reversed']-.6)<1e-12
    assert abs(expected['initial']-expected['compressed'])<1e-12
    assert abs(expected['ties']-.6)<1e-12
    browser_status = 'PENDING'
    random_case_count = 0
    random_error = 0.0
    browser_file = OUT/'qa/browser_qa.json'
    if browser_file.exists():
        browser = json.loads(browser_file.read_text())
        if browser['html_sha256'] == hashlib.sha256(html.encode()).hexdigest():
            assert browser['status']=='PASS'
            for view in ['desktop', 'mobile']:
                for case in browser[view]['random_ap_cases']:
                    scores = np.array(case['scores'][:5])
                    correct = float(fn(scores,known,np.ones((1,5)),('coco',))['coco'][0])
                    error = abs(correct-case['ap'])
                    assert error < 1e-12, (case, correct)
                    random_error = max(random_error, error)
                    random_case_count += 1
            browser_status = 'PASS (1440 desktop / 390 mobile, offline images and Korean font, ROI controls, AP, split/model controls, print restore, no JS errors)'
        else:
            browser_status = 'STALE: rerun qa/browser_qa.py on the current HTML'
    result = dict(status='PASS',html_bytes=len(html.encode()),html_sha256=hashlib.sha256(html.encode()).hexdigest(),
                  sections=len(re.findall(r'<section\b',html)),unique_ids=len(doc.ids),broken_links=broken,
                  sources_checked=len(d['sources']),source_changes=changed,examples=6,prediction_csv_rows=354,
                  whole_manifest_identity_checks=d['checks'],
                  original_project_ap_kernel_expected=expected,
                  browser_qa=browser_status,browser_random_ap_cases_compared_to_existing_kernel=random_case_count,
                  browser_random_ap_max_error=random_error,
                  new_training=False,new_gpu_inference=False)
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
