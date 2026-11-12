import requests,socket,json,base64,hashlib,os,struct,time
from urllib.parse import urlparse
from pathlib import Path
OUT=Path('/home/irteam/Crop_representation/experiments/daily_2026-11-09');OUT.joinpath('qa').mkdir(exist_ok=True)
class CDP:
 def __init__(self,url):
  p=urlparse(url);self.s=socket.create_connection((p.hostname,p.port));self.s.settimeout(10);self.seq=0;self.events=[];key=base64.b64encode(os.urandom(16)).decode()
  self.s.sendall((f'GET {p.path} HTTP/1.1\r\nHost: {p.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nOrigin: http://localhost\r\n\r\n').encode());r=b''
  while b'\r\n\r\n' not in r:r+=self.s.recv(1)
  assert b'101' in r,r
 def read(self,n):
  b=b''
  while len(b)<n:
   chunk=self.s.recv(n-len(b));assert chunk,'Websocket EOF';b+=chunk
  return b
 def receive(self):
  h=self.read(2);n=h[1]&127
  if n==126:n=struct.unpack('!H',self.read(2))[0]
  elif n==127:n=struct.unpack('!Q',self.read(8))[0]
  mask=self.read(4) if h[1]&128 else b'';data=self.read(n)
  if mask:data=bytes(x^mask[i%4] for i,x in enumerate(data))
  return json.loads(data)
 def call(self,method,params={}):
  self.seq+=1;data=json.dumps(dict(id=self.seq,method=method,params=params)).encode();n=len(data);mask=os.urandom(4)
  head=bytes([129,128|n]) if n<126 else bytes([129,254])+struct.pack('!H',n) if n<65536 else bytes([129,255])+struct.pack('!Q',n)
  self.s.sendall(head+mask+bytes(x^mask[i%4] for i,x in enumerate(data)))
  while True:
   r=self.receive()
   if r.get('id')==self.seq:
    assert 'error' not in r,r
    return r.get('result',{})
   self.events.append(r)
 def js(self,code):
  r=self.call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True))
  if 'exceptionDetails' in r:raise ValueError(r)
  return r['result'].get('value')
url=requests.get('http://127.0.0.1:9229/json/list',timeout=3).json()[0]['webSocketDebuggerUrl'];c=CDP(url)
c.call('Page.enable');c.call('Runtime.enable');c.call('Log.enable')
results={}
for name,w,h in [('desktop',1440,1000),('mobile',390,844)]:
 c.call('Emulation.setDeviceMetricsOverride',dict(width=w,height=h,deviceScaleFactor=1,mobile=name=='mobile'))
 c.call('Page.navigate',dict(url='file://'+str(OUT/'index.html')))
 c.js("new Promise(r=>{const h=setInterval(()=>{if(document.readyState==='complete'&&document.querySelectorAll('.experiment').length===17){clearInterval(h);r()}},30)}).then(()=>Promise.all([...document.images].map(i=>{i.loading='eager';return i.decode().catch(()=>null)}))).then(()=>document.fonts.ready).then(()=>new Promise(r=>setTimeout(r,300)))")
 state=c.js("({title:document.title,innerWidth,scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight,visibleCards:[...document.querySelectorAll('.experiment')].filter(x=>!x.hidden).length,brokenImages:[...document.images].filter(x=>!x.complete||!x.naturalWidth).map((x,i)=>({index:i,srcPrefix:x.src.slice(0,40)})),badOverflow:[...document.querySelectorAll('body *')].filter(x=>{let r=x.getBoundingClientRect();return r.width>0&&(r.right>innerWidth+1||r.left< -1)&&!x.closest('.table-scroll,.chart-scroll,.source-links')}).slice(0,12).map(x=>({tag:x.tagName,cls:x.className,width:x.getBoundingClientRect().width}))})")
 shot=c.call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False));(OUT/'qa'/f'{name}_1440.png' if name=='desktop' else OUT/'qa/mobile_390.png').write_bytes(base64.b64decode(shot['data']))
 # Real controls, counts and protocol data binding.
 filters=c.js("(()=>{let select=document.getElementById('group-filter');let opts=[...select.options].map(x=>x.value);document.querySelector('[data-family-filter=\"SigLIP2\"]').click();let only=[...document.querySelectorAll('.experiment')].filter(x=>!x.hidden).every(x=>x.dataset.family==='SigLIP2');document.getElementById('experiment-search').value='PixelDiT';document.getElementById('experiment-search').dispatchEvent(new Event('input'));let found=[...document.querySelectorAll('.experiment')].filter(x=>!x.hidden).map(x=>x.id);document.getElementById('reset-filters').click();let restored=[...document.querySelectorAll('.experiment')].filter(x=>!x.hidden).length;return {onlySigLIP:only,pixelSearch:found,restored,groupOptions:opts}})()")
 protocol=c.js("(()=>{let p=document.getElementById('benchmark-protocol'),out=[];for(let o of [...p.options]){p.value=o.value;p.dispatchEvent(new Event('change'));out.push({value:p.value,rows:document.querySelectorAll('#benchmark-table tbody tr').length,caption:document.getElementById('benchmark-caption').textContent,svgBars:document.querySelectorAll('#benchmark-chart rect').length});}p.value='official_test';p.dispatchEvent(new Event('change'));let m=document.getElementById('benchmark-metric');m.value='pattern';m.dispatchEvent(new Event('change'));return out})()")
 state['filters']=filters;state['protocols']=protocol;state['expandedLayout']=c.js("(()=>{const ds=[...document.querySelectorAll('details')],old=ds.map(x=>x.open);ds.forEach(d=>d.open=true);const width=document.documentElement.scrollWidth;ds.forEach((d,i)=>d.open=old[i]);return {scrollWidth:width,noBodyOverflow:width<=innerWidth}})()");state['embeddedFontCheck']=c.js("document.fonts.check('16px ReportKorean')");state['fontFaces']=c.js("document.fonts.ready.then(()=>[...document.fonts].map(f=>({family:f.family,status:f.status})))");state['printRestore']=c.js("(()=>{let ds=[...document.querySelectorAll('details')];ds[0].open=true;let before=ds.map(d=>d.open);dispatchEvent(new Event('beforeprint'));let all=ds.every(d=>d.open);dispatchEvent(new Event('afterprint'));let restored=ds.every((d,i)=>d.open===before[i]);ds[0].open=false;return {allExpanded:all,exactRestore:restored}})()");c.js("[...document.querySelectorAll('details')].forEach(d=>d.open=true);window.scrollTo(0,document.getElementById('exp-siglip-structure').getBoundingClientRect().top+scrollY-135)");shot=c.call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False));(OUT/'qa'/f'{name}_expanded.png').write_bytes(base64.b64decode(shot['data']));results[name]=state
results['html_sha256']=hashlib.sha256((OUT/'index.html').read_bytes()).hexdigest();results['fontEnvironment']='Only DejaVu system fallback exposed; Hangul must render from embedded ReportKorean WOFF';results['consoleErrors']=[e for e in c.events if e.get('method')=='Runtime.exceptionThrown' or e.get('method')=='Log.entryAdded' and e.get('params',{}).get('entry',{}).get('level')=='error']
(OUT/'qa/browser_qa.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps({v:{k:val for k,val in r.items() if k not in ['protocols','filters']} for v,r in results.items() if v in ['desktop','mobile']},ensure_ascii=False,indent=2))
c.call('Browser.close')
