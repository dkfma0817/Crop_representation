'use strict';
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const $ = id => document.getElementById(id);
const esc = x => String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (x,n=2) => Number(x).toFixed(n);
const TYPES = {color:'Color',pattern_marking:'Pattern / Marking',material:'Material',transparency:'Transparency'};
const COLORS = {obj:'#146b60',part:'#4663ad',unknown:'#7a8583',neg:'#a44142'};

function labelTag(y) {
  return y===1?'<span class="badge-pos">Positive · 1</span>':y===0?'<span class="badge-neg">Negative · 0</span>':'<span class="badge-unk">Unknown · -1</span>';
}
function lb(w,h,S=384){const s=S/Math.max(w,h),nw=Math.min(S,Math.max(1,Math.round(w*s))),nh=Math.min(S,Math.max(1,Math.round(h*s)));return {nw,nh,ox:Math.floor((S-nw)/2),oy:Math.floor((S-nh)/2)};}
function grid(n=24){let lines='';for(let i=1;i<n;i++){const p=i*384/n;lines+=`<path d="M${p} 0V384M0 ${p}H384"/>`;}return `<g stroke="#fff" stroke-opacity=".36" stroke-width=".65">${lines}</g>`;}
function imageTag(href,x,y,w,h){return `<image href="${href}" x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="none"/>`;}
function renderROI(){
  const e=DATA.examples[Number($('roi-select').value)||0],src=DATA.images[e.image_key],b=e.box;
  const showGrid=$('show-grid').checked,showRelated=$('show-related').checked;
  const stroke=Math.max(e.W,e.H)/180;
  $('original-svg').setAttribute('viewBox',`0 0 ${e.W} ${e.H}`);
  let original=`<title>${esc(e.category)} · ann ${e.ann_id}</title>${imageTag(src,0,0,e.W,e.H)}`;
  if(showRelated) original+=e.related.filter(r=>r.ann_id!==e.ann_id).map(r=>`<rect x="${r.bbox[0]}" y="${r.bbox[1]}" width="${r.bbox[2]-r.bbox[0]}" height="${r.bbox[3]-r.bbox[1]}" fill="none" stroke="${r.level==='part'?'#7395fb':'#f5de79'}" stroke-width="${stroke*.7}" stroke-dasharray="${stroke*2} ${stroke*1.4}"/>`).join('');
  original+=`<rect x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" fill="#d4e885" fill-opacity=".08" stroke="#d4e885" stroke-width="${stroke*1.5}"/>`;
  $('original-svg').innerHTML=original;
  const full=lb(e.W,e.H),crop=lb(b[2]-b[0],b[3]-b[1]);
  const x0=full.ox+b[0]*full.nw/e.W,y0=full.oy+b[1]*full.nh/e.H;
  const bw=(b[2]-b[0])*full.nw/e.W,bh=(b[3]-b[1])*full.nh/e.H;
  $('full-svg').innerHTML=`<title>Full384 canvas · 24 × 24 patch grid</title><rect width="384" height="384" fill="#aeb5ad"/>${imageTag(src,full.ox,full.oy,full.nw,full.nh)}${showGrid?grid():''}<rect x="${x0}" y="${y0}" width="${bw}" height="${bh}" fill="#d4e885" fill-opacity=".12" stroke="#d4e885" stroke-width="2.4"/>`;
  const sx=crop.nw/(b[2]-b[0]),sy=crop.nh/(b[3]-b[1]);
  $('crop-svg').innerHTML=`<title>C0 RGB crop384 canvas · 24 × 24 patch grid</title><defs><clipPath id="crop-content"><rect x="${crop.ox}" y="${crop.oy}" width="${crop.nw}" height="${crop.nh}"/></clipPath></defs><rect width="384" height="384" fill="#aeb5ad"/><g clip-path="url(#crop-content)">${imageTag(src,crop.ox-b[0]*sx,crop.oy-b[1]*sy,e.W*sx,e.H*sy)}</g>${showGrid?grid():''}<rect x="${crop.ox}" y="${crop.oy}" width="${crop.nw}" height="${crop.nh}" fill="none" stroke="#d4e885" stroke-width="2"/>`;
  $('roi-meta').textContent=`${e.level==='part'?'Part':'Object'} · category ${e.category} (id ${e.category_id}) · ann_id ${e.ann_id} · image_id ${e.image_id} · 원본 ${e.W}×${e.H} · 추론용 정수 box [${b.join(', ')}] · 원래 GT bbox xywh [${e.bbox_gt.map(x=>fmt(x,1)).join(', ')}]`;
  const type=$('attr-type').value,attrs=type==='all'?DATA.names.map((_,i)=>i):DATA.types[type];
  $('attribute-table').querySelector('tbody').innerHTML=attrs.map(i=>{
    const y=e.label[i],c=e.predictions.c0,p=e.predictions.p768;
    const bar=(score,color='')=>`<div>${fmt(score,4)}</div><div class="pbar ${color}" aria-hidden="true"><span style="width:${score*100}%"></span></div>`;
    return `<tr class="${y===-1?'unknown-row':''}"><td>${esc(DATA.names[i])}</td><td>${labelTag(y)}</td><td class="num">${fmt(c.logits[i],3)}</td><td class="num">${bar(c.probs[i])}</td><td class="num">${fmt(p.logits[i],3)}</td><td class="num">${bar(p.probs[i],'blue')}</td><td>${y===1?'이 속성 점수가 높을수록 유리':y===0?'이 속성 점수가 낮을수록 유리':'미주석: negative로 사용하지 않음'}</td></tr>`;
  }).join('');
}

const initial=[.91,.83,.72,.41,.36,.96];
let demoScores=initial.slice();
const demoLabels=[1,0,1,0,1,-1];
const demoNames=['ROI A','ROI B','ROI C','ROI D','ROI E','ROI U'];
function apDetails(scores,labels){
  const order=scores.map((s,i)=>({s,i,y:labels[i]})).filter(x=>x.y>=0).sort((a,b)=>b.s-a.s||a.i-b.i);
  const nPos=order.filter(x=>x.y===1).length;
  let tp=0,fp=0;const rows=[],points=[];
  for(let k=0;k<order.length;k++){
    const x=order[k];tp+=x.y;fp+=1-x.y;
    rows.push({...x,tp,fp,precision:tp/(tp+fp),recall:tp/nPos});
    if(k===order.length-1||order[k+1].s!==x.s)points.push({recall:tp/nPos,precision:tp/(tp+fp)});
  }
  const interp=Array.from({length:101},(_,i)=>Math.max(0,...points.filter(p=>p.recall>=i/100-1e-12).map(p=>p.precision)));
  return {order,rows,points,interp,ap:interp.reduce((a,b)=>a+b,0)/101};
}
function drawPR(info){
  const X=r=>55+400*r,Y=p=>278-235*p;
  let svg='<title>단일 category / attribute의 precision recall 곡선</title>';
  for(let i=0;i<=5;i++){
    const v=i/5;svg+=`<path d="M${X(v)} ${Y(0)}V${Y(1)}M${X(0)} ${Y(v)}H${X(1)}" stroke="#d8dfd9" stroke-width="1"/><text x="${X(v)}" y="299" font-size="11" text-anchor="middle" fill="#596d71">${fmt(v,1)}</text><text x="43" y="${Y(v)+4}" font-size="11" text-anchor="end" fill="#596d71">${fmt(v,1)}</text>`;
  }
  const path=info.interp.map((p,i)=>`${i?'L':'M'}${X(i/100)} ${Y(p)}`).join(' ');
  svg+=`<path d="M${X(0)} ${Y(0)} L${X(0)} ${Y(info.interp[0])} ${path.slice(path.indexOf(' L')+1)} L${X(1)} ${Y(0)} Z" fill="#d4e885" fill-opacity=".5"/>`;
  svg+=`<path d="${path}" stroke="#146b60" stroke-width="2.5" fill="none"/>`;
  const raw=[{recall:0,precision:1},...info.points];
  svg+=`<polyline points="${raw.map(p=>`${X(p.recall)},${Y(p.precision)}`).join(' ')}" stroke="#4663ad" stroke-width="1.6" fill="none" stroke-dasharray="4 3"/>`;
  svg+=info.points.map(p=>`<circle cx="${X(p.recall)}" cy="${Y(p.precision)}" r="4" fill="#4663ad"/>`).join('');
  svg+=`<text x="255" y="325" font-size="12" text-anchor="middle" fill="#596d71">Recall</text><text x="17" y="160" font-size="12" text-anchor="middle" fill="#596d71" transform="rotate(-90 17 160)">Precision</text>`;
  $('pr-plot').innerHTML=svg;
}
function renderDemo(){
  const info=apDetails(demoScores,demoLabels),threshold=Number($('threshold').value);
  $('demo-ap').textContent=fmt(info.ap*100);
  $('threshold-out').textContent=fmt(threshold);
  $('threshold-count').textContent=`Known ROI 중 ${info.order.filter(x=>x.s>=threshold).length} / 5개를 positive로 표시`;
  demoScores.forEach((v,i)=>{$('demo-score-'+i).value=v;$('demo-out-'+i).textContent=fmt(v);});
  $('ranking-table').querySelector('tbody').innerHTML=info.rows.map((r,k)=>`<tr><td>${k+1}</td><td>${demoNames[r.i]}</td><td>${labelTag(r.y)}</td><td>${fmt(r.s)}</td><td>${r.tp} / ${r.fp}</td><td>${fmt(r.precision,3)}</td><td>${fmt(r.recall,3)}</td><td>${r.s>=threshold?'Positive 표시':'Negative 표시'}</td></tr>`).join('')+`<tr class="unknown-row"><td>제외</td><td>ROI U</td><td>${labelTag(-1)}</td><td>${fmt(demoScores[5])}</td><td colspan="4">Score와 무관하게 이 단순 ranking 계산에서 제외</td></tr>`;
  drawPR(info);
}

function renderResults(){
  const split=$('split-select').value,key=$('result-model').value;
  const selected=DATA.results.find(r=>r.split===split&&r.key===key);
  $('result-summary').innerHTML=`<div class="metric-grid"><div class="metric"><span>Object AP</span><b>${fmt(selected.object)}</b></div><div class="metric"><span>Part AP</span><b>${fmt(selected.part)}</b></div><div class="metric"><span>Overall*</span><b>${fmt(selected.overall)}</b></div></div><p class="small">평가 이미지 ${selected.n_images.toLocaleString()}장 · 입력 detection ${selected.n_given.toLocaleString()}개<br>유지 ${selected.n_kept.toLocaleString()}개 · max_dets 제외 ${selected.dropped.toLocaleString()}개<br>Area=all 유효 joint pair ${selected.n_valid.toLocaleString()}개 / 공식 목록 5,797개<br>참고: 별도 flat joint AP = ${fmt(selected.flat_joint_ap)}</p>`;
  $('result-source').innerHTML=`원자료: <a href="../../${esc(selected.source)}">official_results.json</a>`;
  $('result-table').querySelector('tbody').innerHTML=DATA.results.filter(r=>r.split===split).map(r=>`<tr><td>${esc(r.name)}</td><td>${r.split==='val'?'Official Val':'Official Test'}</td><td class="num">${fmt(r.object)}</td><td class="num">${fmt(r.part)}</td><td class="num">${fmt(r.overall)}</td><td>${r.key==='c0'?'Held-out train dev M0':'Official Val mean(Object, Part)'}</td></tr>`).join('');
  let svg='<title>기존 공식 평가의 속성 type별 AP</title>';
  const left=145,right=490,scale=(right-left)/100;
  for(let i=0;i<=5;i++){const v=i*20,x=left+v*scale;svg+=`<path d="M${x} 25V278" stroke="#d8dfd9"/><text x="${x}" y="302" text-anchor="middle" font-size="11" fill="#596d71">${v}</text>`;}
  Object.keys(TYPES).forEach((type,i)=>{
    const y=42+i*61,o=selected.object_types[type],p=selected.part_types[type];
    svg+=`<text x="130" y="${y+18}" text-anchor="end" font-size="12" fill="#172c32">${esc(TYPES[type])}</text><rect x="${left}" y="${y}" width="${o*scale}" height="16" rx="2" fill="#146b60"/><rect x="${left}" y="${y+21}" width="${p*scale}" height="16" rx="2" fill="#4663ad"/><text x="${left+o*scale+6}" y="${y+12}" font-size="11" fill="#146b60">${fmt(o)}</text><text x="${left+p*scale+6}" y="${y+33}" font-size="11" fill="#4663ad">${fmt(p)}</text>`;
  });
  $('actual-chart').innerHTML=svg;
}

function downloadCSV(){
  const fields=['ann_id','image_id','category','level','attribute','type','label','c0_logit','c0_probability','p768_logit','p768_probability'];
  const rows=[fields];
  DATA.examples.forEach(e=>DATA.names.forEach((name,i)=>{
    const type=Object.keys(DATA.types).find(k=>DATA.types[k].includes(i));
    rows.push([e.ann_id,e.image_id,e.category,e.level,name,type,e.label[i],e.predictions.c0.logits[i],e.predictions.c0.probs[i],e.predictions.p768.logits[i],e.predictions.p768.probs[i]]);
  }));
  const csv='\uFEFF'+rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');
  a.href=url;a.download='paco_gt_roi_example_predictions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

$('roi-select').innerHTML=DATA.examples.map((e,i)=>`<option value="${i}">${esc(e.category)} · ${e.level==='part'?'Part':'Object'} · ann ${e.ann_id}</option>`).join('');
['roi-select','attr-type','show-grid','show-related'].forEach(id=>$(id).addEventListener('change',renderROI));
$('score-controls').innerHTML=demoScores.map((v,i)=>`<div class="score-edit"><label for="demo-score-${i}"><span>${demoNames[i].replace('ROI ','')}</span><span class="${demoLabels[i]===1?'badge-pos':demoLabels[i]===0?'badge-neg':'badge-unk'}">${demoLabels[i]===1?'Positive':demoLabels[i]===0?'Negative':'Unknown'}</span><input aria-label="${demoNames[i]} 점수" id="demo-score-${i}" class="score-input" type="range" min="0.01" max="0.99" step="0.001" value="${v}"><output id="demo-out-${i}" for="demo-score-${i}">${fmt(v)}</output></label></div>`).join('');
demoScores.forEach((v,i)=>$('demo-score-'+i).addEventListener('input',ev=>{demoScores[i]=Number(ev.target.value);document.querySelectorAll('[data-preset]').forEach(b=>b.setAttribute('aria-pressed','false'));renderDemo();}));
document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>{
  const k=button.dataset.preset;
  demoScores=k==='perfect'?[.95,.25,.85,.15,.75,.99]:k==='reversed'?[.45,.95,.35,.85,.25,.99]:k==='compressed'?initial.map(x=>.4+.1*x):initial.slice();
  document.querySelectorAll('[data-preset]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
  renderDemo();
}));
$('threshold').addEventListener('input',renderDemo);
['split-select','result-model'].forEach(id=>$(id).addEventListener('change',renderResults));
$('print-btn').addEventListener('click',()=>window.print());
$('download-btn').addEventListener('click',downloadCSV);
let detailState=[];
window.addEventListener('beforeprint',()=>{detailState=[...document.querySelectorAll('details')].map(d=>d.open);document.querySelectorAll('details').forEach(d=>d.open=true);});
window.addEventListener('afterprint',()=>document.querySelectorAll('details').forEach((d,i)=>d.open=detailState[i]));
renderROI();renderDemo();renderResults();
window.PACO_GUIDE={apDetails,renderROI,renderDemo,renderResults,getDemoScores:()=>demoScores.slice(),DATA};
