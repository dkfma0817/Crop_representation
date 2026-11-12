"""Exercise the standalone report in existing headless Chromium, without running an encoder."""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from cdp import CDP

OUT = Path(__file__).resolve().parents[1]
CHROME = Path('/home/irteam/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell')
DEPS = Path('/tmp/p23-browser.IZySXp/root')


def main():
    profile = tempfile.mkdtemp(prefix='paco-guide-browser-')
    env = dict(os.environ)
    if DEPS.exists():
        env['LD_LIBRARY_PATH'] = ':'.join([str(DEPS/'usr/lib/x86_64-linux-gnu'), str(DEPS/'lib/x86_64-linux-gnu'), env.get('LD_LIBRARY_PATH','')])
        env['FONTCONFIG_FILE'] = str(DEPS.parent/'fonts.conf')
    log = (OUT/'qa/browser.log').open('w')
    proc = subprocess.Popen([str(CHROME),'--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--no-first-run','--remote-allow-origins=http://localhost',
                             '--remote-debugging-address=127.0.0.1','--remote-debugging-port=0',
                             '--user-data-dir='+profile,'about:blank'], env=env, stdout=log, stderr=log)
    try:
        active = Path(profile)/'DevToolsActivePort'
        for _ in range(100):
            if active.exists():
                break
            if proc.poll() is not None:
                raise RuntimeError((OUT/'qa/browser.log').read_text())
            time.sleep(.1)
        port = int(active.read_text().splitlines()[0])
        tabs = json.load(urlopen(f'http://127.0.0.1:{port}/json/list',timeout=5))
        c = CDP(tabs[0]['webSocketDebuggerUrl'])
        c.call('Page.enable'); c.call('Runtime.enable'); c.call('Log.enable')
        result = {}
        expected = json.loads((OUT/'validation.json').read_text())['original_project_ap_kernel_expected']
        for name,w,h in [('desktop',1440,1000),('mobile',390,844)]:
            c.call('Emulation.setDeviceMetricsOverride',dict(width=w,height=h,deviceScaleFactor=1,mobile=name=='mobile'))
            c.call('Page.navigate',dict(url='file://'+str(OUT/'index.html')))
            c.js("new Promise((resolve,reject)=>{let start=Date.now(),timer=setInterval(()=>{if(window.PACO_GUIDE){clearInterval(timer);resolve()}else if(Date.now()-start>12000){clearInterval(timer);reject('Report did not initialize')}},40)}).then(()=>document.fonts.ready).then(()=>Promise.all(Object.values(PACO_GUIDE.DATA.images).map(src=>new Promise((r,j)=>{const im=new Image();im.onload=()=>r(true);im.onerror=j;im.src=src})))).then(()=>new Promise(r=>setTimeout(r,200)))")
            c.js("document.documentElement.style.scrollBehavior='auto'")
            state = c.js("({title:document.title,innerWidth,scrollWidth:document.documentElement.scrollWidth,sections:document.querySelectorAll('section').length,initialRows:document.querySelectorAll('#attribute-table tbody tr').length,fonts:[...document.fonts].map(f=>({family:f.family,status:f.status})),bodyOverflow:document.documentElement.scrollWidth>innerWidth})")
            assert not state['bodyOverflow'], state
            assert all(f['status']=='loaded' for f in state['fonts']), state
            shot = c.call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
            (OUT/'qa'/f'{name}.png').write_bytes(base64.b64decode(shot['data']))
            state['roi_checks'] = c.js("(()=>{const out=[];for(let i=0;i<6;i++){const r=document.getElementById('roi-select');r.value=i;r.dispatchEvent(new Event('change'));const meta=document.getElementById('roi-meta').textContent;out.push({index:i,matchingID:meta.includes(String(PACO_GUIDE.DATA.examples[i].ann_id)),images:document.querySelectorAll('#original-svg image,#full-svg image,#crop-svg image').length,rows:document.querySelectorAll('#attribute-table tbody tr').length})}const a=document.getElementById('attr-type');a.value='all';a.dispatchEvent(new Event('change'));const allRows=document.querySelectorAll('#attribute-table tbody tr').length;document.getElementById('show-related').click();const relatedRects=document.querySelectorAll('#original-svg rect').length;document.getElementById('show-grid').click();const gridHidden=document.querySelectorAll('#full-svg path').length===0;return {rois:out,allRows,relatedRects,gridHidden}})()")
            assert state['roi_checks']['allRows']==59
            assert all(r['matchingID'] and r['images']==3 and r['rows']==14 for r in state['roi_checks']['rois'])
            assert state['roi_checks']['gridHidden']
            state['ap_presets'] = c.js("(()=>{let out={};document.querySelectorAll('[data-preset]').forEach(b=>{b.click();out[b.dataset.preset]=Number(document.getElementById('demo-ap').textContent)/100;});return out})()")
            for key,value in state['ap_presets'].items():
                assert abs(value-expected[key])<.000051,(key,value,expected[key])
            state['ap_invariants'] = c.js("(()=>{document.querySelector('[data-preset=initial]').click();const before=document.getElementById('demo-ap').textContent;const u=document.getElementById('demo-score-5');u.value=.01;u.dispatchEvent(new Event('input'));const afterUnknown=document.getElementById('demo-ap').textContent;const t=document.getElementById('threshold');t.value=.95;t.dispatchEvent(new Event('input'));const afterThreshold=document.getElementById('demo-ap').textContent;const ties=PACO_GUIDE.apDetails([.5,.5,.5,.5,.5,.9],[1,0,1,0,1,-1]).ap;return {before,afterUnknown,afterThreshold,ties}})()")
            assert len({state['ap_invariants'][x] for x in ['before','afterUnknown','afterThreshold']})==1
            assert abs(state['ap_invariants']['ties']-.6)<1e-10
            # Compare actual JS against the pre-existing Python AP kernel over deterministic random cases with ties.
            state['random_ap_cases'] = c.js("(()=>{let cases=[];let seed=17;for(let k=0;k<100;k++){let scores=[];for(let i=0;i<6;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;scores.push((seed%10)/10)}cases.push({scores,ap:PACO_GUIDE.apDetails(scores,[1,0,1,0,1,-1]).ap})}return cases})()")
            state['results'] = c.js("(()=>{const out=[];for(let split of ['val','test'])for(let key of ['c0','p768']){const s=document.getElementById('split-select'),m=document.getElementById('result-model');s.value=split;m.value=key;s.dispatchEvent(new Event('change'));out.push({split,key,rows:document.querySelectorAll('#result-table tbody tr').length,summary:document.getElementById('result-summary').textContent,bars:document.querySelectorAll('#actual-chart rect').length})}return out})()")
            assert all(r['rows']==2 and r['bars']==8 for r in state['results'])
            state['print_restore'] = c.js("(()=>{let ds=[...document.querySelectorAll('details')],before=ds.map(d=>d.open);dispatchEvent(new Event('beforeprint'));let opened=ds.every(d=>d.open);dispatchEvent(new Event('afterprint'));return {opened,restored:ds.every((d,i)=>d.open===before[i])}})()")
            assert all(state['print_restore'].values())
            # Save representative visualizations in their initial state.
            c.js("document.getElementById('roi-select').value='0';document.getElementById('attr-type').value='material';document.getElementById('show-grid').checked=true;document.getElementById('show-related').checked=false;PACO_GUIDE.renderROI();document.getElementById('inference').scrollIntoView();")
            shot=c.call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False));(OUT/'qa'/f'{name}_roi.png').write_bytes(base64.b64decode(shot['data']))
            c.js("document.querySelector('[data-preset=initial]').click();document.getElementById('threshold').value='.5';PACO_GUIDE.renderDemo();document.getElementById('ap').scrollIntoView();")
            shot=c.call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False));(OUT/'qa'/f'{name}_ap.png').write_bytes(base64.b64decode(shot['data']))
            state['expanded_width'] = c.js("(()=>{document.querySelectorAll('details').forEach(d=>d.open=true);return document.documentElement.scrollWidth})()")
            assert state['expanded_width']<=w
            result[name]=state
        errors=[e for e in c.events if e.get('method')=='Runtime.exceptionThrown' or e.get('method')=='Log.entryAdded' and e.get('params',{}).get('entry',{}).get('level')=='error']
        assert not errors,errors
        result['console_errors']=errors
        result['html_sha256']=hashlib.sha256((OUT/'index.html').read_bytes()).hexdigest()
        result['status']='PASS'
        (OUT/'qa/browser_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        c.call('Browser.close')
        print(json.dumps({k:{x:v[x] for x in ['bodyOverflow','ap_presets','ap_invariants','print_restore']} for k,v in result.items() if k in ['desktop','mobile']},ensure_ascii=False,indent=2))
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait()
        log.close()
        shutil.rmtree(profile,ignore_errors=True)


if __name__=='__main__':
    main()
