"""Minimal local Chrome DevTools websocket client, reused from the daily report QA."""
import socket,json,base64,os,struct
from urllib.parse import urlparse
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
