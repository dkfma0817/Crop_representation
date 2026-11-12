'use strict';
const D=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=2)=>Number.isFinite(x)?x.toFixed(n):'—',signed=x=>Number.isFinite(x)?(x>0?'+':'')+x.toFixed(2):'—';
const empty=s=>`<div class="empty">${esc(s)}</div>`,color=c=>D.colors[c]||'#718096',short=c=>({B0:'Full B0',Decode:'Decoder',Continue:'Continue+4k'}[c]||c);
const q=new URLSearchParams(location.search);let archMode=q.get('mode')==='inference'?'inference':'train';
function models(){return $('group').value==='all'?Object.keys(D.models):D.core;}
function metric(c,k,reader=$('reader').value){return D.models[c]?.[reader]?.[k]??null;}
function cell(x){return `<td>${fmt(x)}</td>`;}
function svg(w,h,body){return `<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img">${body}</svg>`;}
function options(id,list,selected){$(id).innerHTML=list.map(c=>`<option value="${esc(c)}">${esc(D.labels[c]||c)}</option>`).join('');$(id).value=selected||list[0];}
function legend(items){return `<div class="legend">${items.map(([t,c])=>`<span><i class="dot" style="background:${c}"></i>${esc(t)}</span>`).join('')}</div>`;}
function bars(items,{left=95,width=560,max=100,dual=false}={}){
  if(!items.length)return empty('공식 결과 대기 중');const row=dual?52:35,h=items.length*row+35,plot=width-left-50;
  let b='';for(let i=0;i<=4;i++){const x=left+plot*i/4;b+=`<path d="M${x} 8V${h-20}" stroke="#e8edf4"/><text x="${x}" y="${h-2}" text-anchor="middle">${fmt(max*i/4,0)}</text>`;}
  items.forEach((o,i)=>{let y=10+i*row;b+=`<text x="${left-10}" y="${y+17}" text-anchor="end">${esc(o.name)}</text>`;
    const values=dual?[[o.a,o.color],[o.b,'#c2ccda']]:[[o.a,o.color]];
    values.forEach(([v,c],j)=>{const yy=y+j*19;if(v==null){b+=`<text x="${left+5}" y="${yy+14}">평가 중</text>`;return;}const w=Math.max(0,v/max*plot);b+=`<rect x="${left}" y="${yy}" width="${w}" height="14" fill="${c}" rx="3"/><text class="value" x="${left+w+6}" y="${yy+12}">${fmt(v)}</text>`;});
  });return svg(width,h,b);
}
function renderStatus(){
  $('stamp').textContent=`데이터 갱신: ${D.updated.slice(0,19).replace('T',' ')} KST · HTML은60초마다 다시 저장됩니다. 열린 창은 새로고침하면 최신 값을 읽습니다.`;
  $('footer-stamp').textContent=D.updated.slice(0,19).replace('T',' ')+' KST';
  $('gpu').innerHTML=(D.gpu||[]).map(g=>`<span class="pill">GPU ${g.index} · ${g.utilization}% · ${(g.memory_MiB/1024).toFixed(1)} GiB</span>`).join('');
  $('status-cards').innerHTML=['P4','Native','Decode'].map(c=>{const e=D.models[c],t=e.training||{},p=e.progress||{},done=t.done,step=t.step||0;
    const train=c==='Native'?'고정된 P12k 재사용':`${step.toLocaleString()} /12,000 step`;
    return `<article class="status-card"><div class="small">${esc(D.labels[c])}</div><strong>${e.test_complete?'TEST 완료':p.reader_done?'공식 평가 진행':`Reader ${p.reader_epoch||0}/10 epoch`}</strong><div class="progress-track"><i style="width:${e.test_complete?100:p.reader_done?85:(p.reader_epoch||0)*6+20}%;background:${color(c)}"></i></div><div class="badges"><span class="pill ${done?'ok':''}">${train}</span><span class="pill ${p.reader_done?'ok':'pending'}">R2 ${p.reader_epoch||0}/10</span><span class="pill ${p.val_complete===10?'ok':'pending'}">VAL ${p.val_complete||0}/10</span><span class="pill ${e.test_complete?'ok':'pending'}">TEST ${e.test_complete?'완료':'대기'}</span></div><p class="small">${c==='Native'?'Representation 추가 학습 없음 · ':done?.seconds?`Representation 학습 ${(done.seconds/60).toFixed(1)}분 · `:''}${p.selection?`선택 epoch ${p.selection.epoch}`:'Official VAL로 epoch 선택 예정'}</p></article>`;
  }).join('');
  const records=['P4','Decode'].map(c=>({c,values:D.models[c].training?.diagnostics||[]}));let s='';const w=570,h=245,l=53,t=15,pw=465,ph=185;
  for(let i=0;i<=4;i++){let y=t+ph*(1-i/4);s+=`<path d="M${l} ${y}H${l+pw}" stroke="#e6ecf4"/><text x="${l-9}" y="${y+4}" text-anchor="end">${(i/4).toFixed(2)}</text>`;}
  for(const r of records){const pts=r.values.map(v=>`${l+pw*v.step/12000},${t+ph*(1-Math.max(0,v.crop_cosine))}`);if(pts.length){s+=`<polyline points="${pts.join(' ')}" fill="none" stroke="${color(r.c)}" stroke-width="2.5"/>`;r.values.forEach(v=>{const x=l+pw*v.step/12000,y=t+ph*(1-Math.max(0,v.crop_cosine));s+=`<circle cx="${x}" cy="${y}" r="4" fill="${color(r.c)}"><title>${r.c} step${v.step}: ${v.crop_cosine.toFixed(4)}</title></circle>`;});}}
  [0,3000,6000,9000,12000].forEach(v=>s+=`<text x="${l+pw*v/12000}" y="${h-15}" text-anchor="middle">${v/1000}k</text>`);
  $('cos-chart').innerHTML=legend(records.map(r=>[r.c,color(r.c)]))+svg(w,h,s);
  const p=metric('P8','Object AP','ap'),crop=metric('C0','Object AP','ap');let finding=`<div class="callout">기존 P8의 Object AP는 ${fmt(p)}, Crop은 ${fmt(crop)}입니다. 격차는 ${fmt(crop-p)} AP입니다.`;
  const a=metric('P4','Obj Material','ap'),b=metric('P8','Obj Material','ap');finding+=a!=null?` P4의 Object Material 변화는 ${signed(a-b)} AP입니다.`:' P4/Native/Decoder의 공식 TEST 결과는 아직 비교표에 확정하지 않았습니다.';
  finding+=' Cosine 개선을 AP 개선으로 간주하지 않습니다.</div>';$('current-findings').innerHTML=finding;
}
function renderResults(){
  const ms=models(),reader=$('reader').value;
  $('ap-table').innerHTML=`<thead><tr><th>Model</th>${D.metrics.map(k=>`<th>${esc(k)}</th>`).join('')}<th>상태</th></tr></thead><tbody>`+ms.map(c=>{const row=D.models[c][reader];return `<tr class="${c==='C0'?'reference':''}"><td class="model" style="--accent:${color(c)}">${esc(short(c))}</td>${D.metrics.map(k=>cell(row?.[k])).join('')}<td>${row?'Official TEST':reader==='mean'?'미측정':'평가 중'}</td></tr>`;}).join('')+'</tbody>';
  $('overall-chart').innerHTML=legend([['Object AP','#315bd9'],['Part AP','#c2ccda']])+bars(ms.map(c=>({name:short(c),a:metric(c,'Object AP'),b:metric(c,'Part AP'),color:color(c)})),{dual:true,max:65});
  const comparisons=['P8 − M','C0 − P8','P4 − P8','Native − P8','Decode − Native'];
  let rows=[];for(const name of comparisons){const [x,y]=name.split(' − ');for(const k of ['Object AP','Obj Material','Obj Pattern']){const v=metric(x,k),u=metric(y,k);if(v==null||u==null)continue;const ci=[...D.contrasts].reverse().find(r=>r.reader===reader&&r.comparison===name&&r.metric===k);rows.push(`<tr><td>${esc(name)}</td><td>${esc(k)}</td><td>${signed(v-u)}</td><td>${ci?`[${fmt(ci.ci95_low)}, ${fmt(ci.ci95_high)}]`:'계산 대기'}</td></tr>`);}}
  $('contrast-table').innerHTML=rows.length?`<table><thead><tr><th>비교</th><th>지표</th><th>ΔAP</th><th>95% CI</th></tr></thead><tbody>${rows.join('')}</tbody></table>`:empty('해당 reader의 비교 결과 대기 중');
  let heat=`<table><thead><tr><th>Student − C0</th>${D.metrics.map(k=>`<th>${esc(k)}</th>`).join('')}</tr></thead><tbody>`;
  for(const c of ms.filter(c=>c!=='C0')){heat+=`<tr><td class="model" style="--accent:${color(c)}">${esc(short(c))}</td>`;for(const k of D.metrics){const a=metric(c,k),b=metric('C0',k),diff=a!=null&&b!=null?a-b:null;const alpha=diff==null?0:Math.min(.28,Math.abs(diff)/20*.28);heat+=`<td class="gap ${diff>=0?'pos':'neg'}" style="background:${diff==null?'#fafbfd':diff>=0?`rgba(49,91,217,${alpha})`:`rgba(204,116,55,${alpha})`}">${signed(diff)}</td>`;}heat+='</tr>';}
  $('heatmap').innerHTML=heat+'</tbody></table>';renderAttributes();
}
function attrRows(){const c=$('attr-model').value,lv=$('level').value,key=$('reader').value==='ap'?'attributes':'mean_attributes';const reference=new Map((D.models.C0[key]?.[lv]||[]).map(r=>[r.attribute,r]));return (D.models[c][key]?.[lv]||[]).map(r=>({...r,Crop:reference.get(r.attribute)?.AP??null,gap:reference.get(r.attribute)?.AP!=null&&r.AP!=null?r.AP-reference.get(r.attribute).AP:null}));}
function renderAttributes(){
  const c=$('attr-model').value,lv=$('level').value,all=attrRows(),term=$('search').value.toLowerCase(),min=Number($('min-pos').value)||0,order=$('attr-sort').value;
  let rows=all.filter(r=>r.attribute.toLowerCase().includes(term)&&r.positive_ROIs>=min);
  rows.sort((a,b)=>order==='name'?a.attribute.localeCompare(b.attribute):order==='gap'?(a.gap??Infinity)-(b.gap??Infinity):order==='gap-desc'?(b.gap??-Infinity)-(a.gap??-Infinity):order==='AP'?(b.AP??-1)-(a.AP??-1):b.positive_ROIs-a.positive_ROIs);
  $('attr-count').textContent=`${rows.length}개 속성 · 전체 ${all.length}개 중 필터링`;
  $('attribute-table').innerHTML=`<thead><tr><th>Attribute</th><th>${esc(short(c))} AP</th><th>C0 AP</th><th>Student − Crop</th><th>Positive ROI</th><th>Positive category</th><th>진단</th></tr></thead><tbody>`+rows.map(r=>`<tr><td class="${['fabric','ceramic','paper','striped','perforated','text'].includes(r.attribute)?'focus':''}">${esc(r.attribute)}</td>${cell(r.AP)}${cell(r.Crop)}<td class="gap ${r.gap>=0?'pos':'neg'}">${signed(r.gap)}</td><td>${r.positive_ROIs.toLocaleString()}</td><td>${r.positive_categories}</td><td>${r.positive_ROIs<100?'<span class="pill pending">positive 적음</span>':'탐색적'}</td></tr>`).join('')+'</tbody>';
  if(!rows.length)$('attribute-table').innerHTML+='<caption class="empty">측정 결과 없음 또는 필터 조건에 맞는 속성이 없습니다.</caption>';
  const focus=['ceramic','fabric','paper','striped','perforated','text'].map(n=>all.find(r=>r.attribute===n)).filter(Boolean);
  $('focus-title').textContent=`${short(c)} vs C0 · ${lv}`;
  $('focus-chart').innerHTML=focus.length?legend([[short(c),color(c)],['C0 Crop','#a7cfbd']])+bars(focus.map(r=>({name:r.attribute,a:r.AP,b:r.Crop,color:color(c)})),{dual:true,max:100}):empty('이 조건의 공식 속성 AP 대기 중');
  const valid=all.filter(r=>r.gap!=null),weak=[...valid].sort((a,b)=>a.gap-b.gap).slice(0,3),strong=[...valid].sort((a,b)=>b.gap-a.gap).slice(0,3);
  let insight=valid.length?`<p><b>Crop 대비 가장 큰 부족:</b><br>${weak.map(r=>`${esc(r.attribute)} ${signed(r.gap)} AP (N=${r.positive_ROIs})`).join('<br>')}</p><p><b>상대적으로 격차가 작거나 앞선 속성:</b><br>${strong.map(r=>`${esc(r.attribute)} ${signed(r.gap)} AP`).join('<br>')}</p>`:empty('아직 결과가 없습니다.');
  const cer=all.find(r=>r.attribute==='ceramic');if(cer)insight+=`<div class="callout">Ceramic: Student ${fmt(cer.AP)} / Crop ${fmt(cer.Crop)}. Teacher도 이 속성에서 높은 AP를 보장하지 않습니다.</div>`;
  insight+='<p class="note">순위는 AP 점추정치 기준입니다. 개별 속성의 confidence interval이나 category 난이도를 통제한 우위 검정이 아닙니다.</p>';$('attribute-insights').innerHTML=insight;
}
function renderSizes(){
  const c=$('size-model').value,lv=$('size-level').value,k=$('size-metric').value,ref=$('size-reference').value;
  const bins=['tiny_lt4_tokens','small_4to16_tokens','medium_16to64_tokens','large_ge64_tokens'],names=['tiny A<32²','small 32²–64²','medium 64²–128²','large A≥128²'];
  const get=(m,b)=>D.sizes.find(r=>r.model===m&&r.level===lv&&r.size_bin===b);
  const rows=bins.map((b,i)=>({name:names[i],a:get(c,b)?.[k],b:get(ref,b)?.[k],color:color(c),n:get(c,b)?.roi_count}));
  $('size-chart').innerHTML=rows.some(r=>r.a!=null)?legend([[short(c),color(c)],[short(ref),'#c2ccda']])+bars(rows,{dual:true,max:100,left:135}):empty('새 조건의 크기별 공식 분석은 TEST 평가 후 계산합니다.');
  $('size-table').innerHTML=`<table><thead><tr><th>ROI 크기</th><th>${esc(short(c))}</th><th>${esc(short(ref))}</th><th>차이</th><th>GT ROI 수</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.name)}</td>${cell(r.a)}${cell(r.b)}<td>${r.a!=null&&r.b!=null?signed(r.a-r.b):'—'}</td><td>${r.n?.toLocaleString()??'—'}</td></tr>`).join('')}</tbody></table>`;
}
const archNotes={P8:'Coarse/Fine을 매 stage 양방향 갱신합니다. 학습은 fused ROI mean과 frozen crop teacher mean을 정렬하고 native image embedding을 보존합니다.',P4:'P8 weight를 변환하지 않습니다. 공식 pretrained coarse + 새로운4px Fine으로12k 학습합니다. Window16/shift8은64px/32px이며 interaction은4×4입니다.',Native:'Representation learning을 더 하지 않습니다. 완료 P12k에서 실제 겹치는 coarse/fine cell을 선택하고, 같은 R2를 새로 학습합니다. 보간 위치576개를 만들지 않습니다.',Decode:'1단계: P12k encoder를 고정하고 COCO에서16-query decoder만12k 학습합니다. 2단계: Encoder/Decoder를 고정하고 PACO에서 같은 R2를 새로 학습합니다.',M:'L4/L8/L12 feature를768 채널로 융합합니다. Fine patch embedding과 Fine Transformer가 없습니다.',B0:'공식 pretrained full-image encoder는 고정합니다. Representation learning 없이 GT ROIAlign feature로 R2만 학습합니다.',C0:'원본 RGB GT crop을 ROI마다384로 재인코딩합니다. Crop의 관찰 scale과 내용은 full-image ROI feature와 다릅니다.',E1:'P8 구조는 동일합니다. P12k EMA에서 원래 cosine/global loss와 teacher-based negative filtering InfoNCE를 추가해4k 학습한 완료 결과입니다.',E2:'P8 구조는 동일합니다. P12k EMA에서 원래 mean/global loss에 valid crop2×2 local cosine을 추가해4k 학습한 완료 결과입니다.',Continue:'동일한 P12k EMA에서 새 loss 없이4k 더 학습한 학습량 Control입니다. E1/E2의 새 loss 효과를 해석할 때 함께 비교합니다.'};
function renderArch(){const c=$('arch-model').value;$('diagram').innerHTML=D.diagrams[c][archMode];$('arch-note').textContent=archNotes[c];$('diagram-download').href=`diagrams/${c}_${archMode}.svg`;$('train-tab').classList.toggle('active',archMode==='train');$('infer-tab').classList.toggle('active',archMode==='inference');}
function renderCosts(){const unique=new Map();for(const r of D.costs)unique.set(r.model,r);let rows='';for(const c of D.core){const r=unique.get(c);if(!r)continue;const params=r.encoder_parameters??r.parameters;const bytes=r.feature_MiB_fp32??r.stored_scalars*4/1048576;rows+=`<tr><td class="model" style="--accent:${color(c)}">${esc(short(c))}${r.provisional?' *':''}</td><td>${params?fmt(params/1e6):'—'}</td><td>${fmt((r.additional_decoder_parameters||0)/1e6)}</td>${cell(r.encoder_GFLOPs)}<td>${fmt(bytes)}</td><td>${fmt(r.encoder_ms??r.median_encode_ms)}</td><td>${r.readout_tokens!=null?fmt(r.readout_tokens,0):'576'}</td><td>${fmt(r.selection_or_decoder_ms_per_ROI)}</td></tr>`;}
  $('cost-table').innerHTML=`<table><thead><tr><th>Model</th><th>Encoder M params</th><th>Decoder M params</th><th>Encoder GFLOPs /image</th><th>Feature MiB FP32</th><th>Encode ms /sample</th><th>R2 tokens /ROI</th><th>Native/Dec ms /ROI</th></tr></thead><tbody>${rows}</tbody></table><p class="note" style="padding:0 12px">* 구조식으로 계산한 provisional 값. 실제 측정 완료 시 갱신합니다. C0는 crop당 비용이므로 이 표의 image당 비용에서 제외했습니다.</p>`;
}
function init(){
  if(q.has('reader'))$('reader').value=q.get('reader');
  options('attr-model',Object.keys(D.models),q.get('model')||'P8');options('size-model',D.core.filter(c=>c!=='C0'),q.get('model')||'P8');options('arch-model',Object.keys(D.models),q.get('model')||'P4');
  if(q.has('level'))$('level').value=q.get('level');
  $('diagram-gallery').innerHTML=D.core.map(c=>`<article class="gallery-card" data-model="${c}"><h4 style="color:${color(c)}">${esc(D.labels[c])}</h4><div class="preview">${D.diagrams[c].inference}</div><a href="diagrams/${c}_train.svg" download>학습 SVG ↓</a><a href="diagrams/${c}_inference.svg" download>추론 SVG ↓</a></article>`).join('');
  document.querySelectorAll('.gallery-card').forEach(el=>el.addEventListener('click',e=>{if(e.target.closest('a'))return;$('arch-model').value=el.dataset.model;renderArch();$('architectures').scrollIntoView();}));
  $('reader').addEventListener('change',renderResults);$('group').addEventListener('change',renderResults);
  for(const id of ['attr-model','level','search','min-pos','attr-sort'])$(id).addEventListener(id==='search'?'input':'change',renderAttributes);
  for(const id of ['size-model','size-level','size-metric','size-reference'])$(id).addEventListener('change',renderSizes);
  $('arch-model').addEventListener('change',renderArch);$('train-tab').onclick=()=>{archMode='train';renderArch();};$('infer-tab').onclick=()=>{archMode='inference';renderArch();};
  renderStatus();renderResults();renderSizes();renderArch();renderCosts();document.body.dataset.reportReady='true';
  if(q.has('shot'))setTimeout(()=>$(q.get('shot'))?.scrollIntoView({behavior:'instant',block:'start'}),150);
}
try{init();}catch(e){document.body.dataset.reportError=e.stack;console.error(e);}
