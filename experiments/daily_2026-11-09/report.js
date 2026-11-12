
'use strict';
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id);
let selectedFamily='all';
const fmt=(x,d=2)=>x==null?'—':Number(x).toFixed(d);
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const descriptions={
 official_test:'공식 PACO Test AP · R2 · 각 조건의 공식 Val (Object+Part)/2 최대 epoch을 고정해 평가. C0와 Frozen/G1은 기존 공식 참조를 재사용합니다. 아래 차이는 점추정이며 이 표에 CI가 없습니다.',
 official_val:'공식 PACO Val AP · R2 · reader epoch 선택에 사용한 데이터입니다. Test와 같은 모델이어도 split이 다르며 project M2 Val과도 직접 비교하지 않습니다.',
 project_val_r2:'PACO Val · project M2 AP · R2 · train 내부 dev 1,000장의 M0로 epoch 선택. 공식 evaluator 결과가 아닙니다. 순정 native/bilinear readout도 구분합니다.',
 project_val_mean:'PACO Val · project M2 AP · Mean + Linear · train 내부 dev M0 선택. Encoder는 고정했지만 선형 probe는 PACO 정답으로 학습했습니다.'
};
function benchmark(){
 const protocol=$('benchmark-protocol').value,metric=$('benchmark-metric').value,rows=DATA.benchmarks[protocol];
 $('benchmark-caption').textContent=descriptions[protocol];
 $('benchmark-table').querySelector('tbody').innerHTML=rows.map(r=>'<tr class="'+(r.name.startsWith('C0')?'reference':r.name.includes('P768')?'highlight':'')+'"><th scope="row">'+escapeHTML(r.name)+'</th>'+['overall','object','part','pattern','material'].map(k=>'<td>'+fmt(r[k])+'</td>').join('')+'<td>'+fmt(r.epoch,0)+'</td></tr>').join('');
 const w=1020,h=rows.length*37+46,left=270,right=62,plot=w-left-right,max=Math.max(60,Math.ceil(Math.max(...rows.map(r=>r[metric]||0))/10)*10);
 $('benchmark-chart').setAttribute('viewBox','0 0 '+w+' '+h);
 let s='<title>'+escapeHTML(descriptions[protocol])+'</title><desc>지표 '+metric+'. '+rows.map(r=>r.name+' '+fmt(r[metric])).join('; ')+'</desc>';
 for(let v=0;v<=max;v+=10){let x=left+plot*v/max;s+='<line x1="'+x+'" x2="'+x+'" y1="5" y2="'+(h-35)+'" stroke="#e0e7e5"/><text x="'+x+'" y="'+(h-12)+'" text-anchor="middle" fill="#6b7c85" font-size="12">'+v+'</text>';}
 rows.forEach((r,i)=>{let y=i*37+10,value=r[metric]||0,color=r.name.startsWith('C0')?'#233b50':r.name.includes('中心')||r.name.includes('중심화')?'#b27525':r.name.includes('순정')?'#8c9ba1':'#007e78';s+='<text x="'+(left-14)+'" y="'+(y+17)+'" text-anchor="end" font-size="13" fill="#263e49">'+escapeHTML(r.name)+'</text><rect x="'+left+'" y="'+y+'" width="'+(plot*value/max)+'" height="24" rx="2" fill="'+color+'"/><text x="'+(left+plot*value/max+9)+'" y="'+(y+17)+'" font-size="13" font-weight="650" fill="#263e49">'+fmt(value)+'</text>';});
 $('benchmark-chart').innerHTML=s;
}
function filter(){
 const query=$('experiment-search').value.toLowerCase().trim(),group=$('group-filter').value,status=$('status-filter').value;
 let n=0;document.querySelectorAll('.experiment').forEach(card=>{let show=(selectedFamily==='all'||card.dataset.family===selectedFamily)&&(group==='all'||card.dataset.group===group)&&(status==='all'||card.dataset.status===status)&&(!query||card.textContent.toLowerCase().includes(query)||card.id.toLowerCase().includes(query));card.hidden=!show;if(show)n++;});
 $('visible-count').textContent=n+' / '+DATA.experiments.length+' 묶음';$('empty-results').hidden=n>0;
 document.querySelectorAll('[data-family-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.familyFilter===selectedFamily)));
}
function reset(){selectedFamily='all';$('experiment-search').value='';$('group-filter').value='all';$('status-filter').value='all';filter();}
$('benchmark-protocol').addEventListener('change',benchmark);$('benchmark-metric').addEventListener('change',benchmark);
['experiment-search','group-filter','status-filter'].forEach(id=>$(id).addEventListener(id==='experiment-search'?'input':'change',filter));
document.querySelectorAll('[data-family-filter]').forEach(b=>b.addEventListener('click',()=>{selectedFamily=b.dataset.familyFilter;filter();}));
document.querySelectorAll('[data-quick-group]').forEach(b=>b.addEventListener('click',()=>{$('group-filter').value=b.dataset.quickGroup;filter();$('experiment-grid').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth',block:'start'});}));
$('reset-filters').addEventListener('click',reset);
$('download-json').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(DATA,null,2)],{type:'application/json;charset=utf-8'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='crop_representation_2026-11-09.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});
let beforePrint=[];
window.addEventListener('beforeprint',()=>{beforePrint=Array.from(document.querySelectorAll('details')).map(d=>[d,d.open]);beforePrint.forEach(([d])=>d.open=true);});
window.addEventListener('afterprint',()=>beforePrint.forEach(([d,open])=>d.open=open));
$('print-report').addEventListener('click',()=>window.print());
benchmark();filter();
window.REPORT_READY={experiments:DATA.experiments.length,protocol:$('benchmark-protocol').value,snapshot:DATA.snapshot_utc};
