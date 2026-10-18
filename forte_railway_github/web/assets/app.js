const $=id=>document.getElementById(id);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const input=$('videoInput'),key=$('apiKey'),drop=$('drop');
const make=(tag,cls,txt)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(txt!==undefined)e.textContent=String(txt);return e;};
const PROC={0:'Приём наличных от клиента',1:'Внутренний пересчёт наличных',2:'Вход в кассовое помещение',3:'Выход из кассового помещения',4:'Приём и обработка упаковки с наличными'};
const OBJ={cash:'наличные',cash_bag:'инкассаторский пакет',counting_machine:'счётная машинка',person:'человек',document:'документ',phone:'телефон',laptop_open:'ноутбук (открыт)',laptop_closed:'ноутбук (закрыт)',door_open:'дверь (открыта)',door_closed:'дверь (закрыта)',safe:'сейф',other:'прочее'};
const ACT={door_open:'дверь открыта',door_closed:'дверь закрыта',laptop_open:'ноутбук открыт',laptop_closed:'ноутбук закрыт',cash_handover:'передача наличных',document_handover:'передача документов',cash_counting_machine:'пересчёт на машинке',cash_storage:'уборка наличных в хранилище',cashbag_handover:'передача инкассаторского пакета',computer_interaction:'работа за компьютером',safe_closing:'закрытие сейфа',cashier_leaving:'кассир уходит'};
const tr=(m,k)=>m[k]||m[String(k).trim().toLowerCase().replace(/ /g,'_')]||k;
key.value=sessionStorage.getItem('forte_api_key')||'';
key.addEventListener('change',()=>sessionStorage.setItem('forte_api_key',key.value.trim()));
input.addEventListener('change',()=>{$('chosen').textContent=input.files[0]?.name||'Файл не выбран';});
for(const s of ['dragenter','dragover'])drop.addEventListener(s,e=>{e.preventDefault();drop.classList.add('dragover');});
for(const s of ['dragleave','drop'])drop.addEventListener(s,e=>{e.preventDefault();drop.classList.remove('dragover');});
drop.addEventListener('drop',e=>{if(e.dataTransfer.files.length){input.files=e.dataTransfer.files;$('chosen').textContent=input.files[0].name;}});
async function req(path,init={},binary=false){
 if(!key.value.trim())throw Error('Введите X-API-Key');
 sessionStorage.setItem('forte_api_key',key.value.trim());
 const rsp=await fetch(path,{...init,headers:{...(init.headers||{}),'X-API-Key':key.value.trim()}});
 if(!rsp.ok){let err;try{err=await rsp.json();}catch{err={detail:`HTTP ${rsp.status}`}}throw Error(err.detail||err.error||`HTTP ${rsp.status}`);}
 return binary?rsp.blob():rsp.json();
}
function render(report,details){
 const r=$('result');r.classList.remove('hidden');
 r.classList.remove('show');void r.offsetWidth;r.classList.add('show'); // retrigger entrance
 const viol=report.verdict==='violation';
 const vb=$('verdict');vb.textContent=viol?'ВОЗМОЖНОЕ НАРУШЕНИЕ':'НЕ ОБНАРУЖЕНО';
 vb.classList.toggle('warn',viol);
 vb.classList.remove('pop');void vb.offsetWidth;vb.classList.add('pop'); // retrigger pop
 const p=details.process||{},exp=details.process_expectations||null;
 $('processName').textContent=p.name||'Не определён';
 $('processConf').textContent=p.confidence!==undefined?`достоверность ${Math.round(p.confidence*100)}%`:'';
 $('eventCount').textContent=String(details.events?.length||0);
 $('processingTime').textContent=details.processing_sec!==undefined?`${details.processing_sec} с`:'—';
 // Почему это такой процесс — факты
 const why=$('whyProcess');why.replaceChildren();
 if(exp?.recognize_ru)why.append(make('li','key','Как распознаётся: '+exp.recognize_ru));
 for(const part of String(p.reason||'').split(';')){const t=part.trim();if(t)why.append(make('li','',t));}
 if(!why.children.length)why.append(make('li','muted','Недостаточно данных для объяснения'));
 // Шкала версий процесса
 const sb=$('procScores');sb.replaceChildren();
 const scores=p.scores||{},mx=Math.max(0.001,...Object.values(scores).map(Number));
 for(const k of Object.keys(PROC)){const v=Number(scores[k]||0);const row=make('div','scorebar');
  row.append(make('span','sb-name',PROC[k]));
  const track=make('span','sb-track'),fill=make('span','sb-fill');
  const pct=Math.round(v/mx*100);fill.style.width='0%';requestAnimationFrame(()=>{fill.style.width=pct+'%';}); // grow-in
  if(String(k)===String(p.class_id))fill.classList.add('best');
  track.append(fill);row.append(track,make('span','sb-val',v.toFixed(2)));sb.append(row);}
 // Кадры-обоснования процесса
 const pe=$('procEvidence');pe.replaceChildren();
 for(const f of details.process_evidence||[]){if(!f.image_base64)continue;
  const fig=make('figure','ev'),img=make('img');img.src='data:image/jpeg;base64,'+f.image_base64;img.alt=f.caption||'frame';
  fig.append(img,make('figcaption','',`${f.caption||''} · ${Number(f.timestamp_sec||0).toFixed(1)} с`));pe.append(fig);}
 pe.classList.toggle('hidden',!pe.children.length);
 // Что требует процесс
 const ch=$('procChecks');ch.replaceChildren();
 for(const c of exp?.checks_ru||[])ch.append(make('li','',c));
 if(!ch.children.length)ch.append(make('li','muted','Для этого процесса условия не заданы'));
 // Что фактически наблюдается
 const ob=$('observed');ob.replaceChildren();
 const counts=details.object_counts||{};
 const line=Object.entries(counts).filter(([k])=>k!=='other').sort((a,b)=>b[1]-a[1]).map(([k,n])=>`${tr(OBJ,k)} ×${n}`).join(', ');
 ob.append(make('li','',line?('В кадре обнаружено: '+line):'Характерные объекты не обнаружены'));
 const states=[...new Set((details.events||[]).filter(e=>e.source==='detector_state').map(e=>tr(ACT,e.action)))];
 if(states.length)ob.append(make('li','','Наблюдаемые состояния: '+states.join(', ')));
 // Итог
 const conc=$('conclusion');
 if(viol){conc.className='conclusion warn';conc.textContent='Обнаружены кандидаты на нарушения — см. список ниже. Каждый вывод требует проверки человеком.';}
 else{conc.className='conclusion ok';conc.textContent='Нарушений в пределах наблюдаемого фрагмента не выявлено: эвристические проверки (оставленные наличные / открытый ноутбук / открытая дверь при уходе и порядок действий) не сработали. Это не гарантирует соблюдение всех процедур.';}
 // Действия
 const events=$('events');events.replaceChildren();
 for(const ev of details.events||[]){const box=make('div','event'),a=Number(ev.start_sec||0),b=Number(ev.end_sec||0);
  box.append(make('small','',`${a.toFixed(1)}–${b.toFixed(1)} с`),make('strong','',tr(ACT,ev.action)),make('span','',`${ev.source||'model'} · conf ${ev.confidence??'n/a'}`));events.append(box);}
 if(!events.children.length)events.append(make('div','empty','Нет уверенно распознанных действий'));
 // Нарушения
 const violations=$('violations');violations.replaceChildren();
 for(const v of report.violations||[]){const box=make('div','violation'),encoded=v.evidence_frames?.[0]?.image_base64;
  if(encoded && encoded.length<2000000){const img=make('img');img.src='data:image/jpeg;base64,'+encoded;img.alt='Evidence frame';box.append(img);}
  const content=make('div');content.append(make('strong','',`${v.type} · ${Number(v.timecode_sec||0).toFixed(1)} сек`),make('p','',v.description));box.append(content);violations.append(box);}
 if(!violations.children.length)violations.append(make('div','empty','Признаки нарушений не выявлены в пределах наблюдаемого фрагмента.'));
}
async function showJob(jobId){
 const j=await req(`/api/v1/jobs/${jobId}`);
 if(j.status==='done'){const d=await req(`/api/v1/jobs/${jobId}/details`);render(j,d);$('result').scrollIntoView({behavior:'smooth'});}
 else if(j.status==='failed')throw Error(j.error||'Обработка не выполнена');
 else throw Error('Анализ ещё выполняется');
}
async function download(path,name){const blob=await req(path,{},true);const a=make('a');a.href=URL.createObjectURL(blob);a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
async function history(){
 const data=await req('/api/v1/history?limit=50');const root=$('history');root.replaceChildren();
 for(const r of data.jobs){const item=make('div','history-item');const left=make('div','history-info');left.append(make('strong','',r.filename),make('small','',`${r.status} · ${(new Date(r.created_at*1000)).toLocaleString()} · ${r.process_name||'—'}`));
 const buttons=make('div','history-buttons');
 const open=make('button','secondary-btn','Открыть');open.onclick=()=>showJob(r.id).catch(e=>alert(e.message));buttons.append(open);
 if(r.status==='done'){const report=make('button','secondary-btn','JSON');report.onclick=()=>download(`/api/v1/jobs/${r.id}/report`,`${r.id}.json`).catch(e=>alert(e.message));buttons.append(report);}
 if(r.video_retained){const video=make('button','secondary-btn','Видео');video.onclick=()=>download(`/api/v1/jobs/${r.id}/video`,r.filename).catch(e=>alert(e.message));buttons.append(video);}
 const remove=make('button','secondary-btn danger-btn','Удалить');remove.onclick=async()=>{if(!confirm('Удалить видео и отчёт этого задания?'))return;try{await req(`/api/v1/jobs/${r.id}`,{method:'DELETE'});await history();}catch(e){alert(e.message)}};buttons.append(remove);
 item.append(left,buttons);root.append(item);
 }
 if(!data.jobs.length)root.append(make('div','empty','История пока пустая'));
}
$('refresh').addEventListener('click',()=>history().catch(e=>alert(e.message)));
$('submit').addEventListener('click',async()=>{
 $('error').textContent='';if(!input.files[0]){$('error').textContent='Выберите видео';return;}
 $('submit').disabled=true;$('status-card').classList.remove('hidden');$('result').classList.add('hidden');
 const body=new FormData();body.append('file',input.files[0]);
 try{const created=await req('/api/v1/analyze',{method:'POST',body});const id=created.job_id;
  $('statusText').textContent=`Job ID: ${id}`;
  const started=Date.now();
  while(Date.now()-started<305000){const state=await req(`/api/v1/jobs/${id}`);
   if(state.status==='done'){const details=await req(`/api/v1/jobs/${id}/details`);render(state,details);await history();return;}
   if(state.status==='failed')throw Error(state.error||'Ошибка обработки');
   const progress=await req(`/api/v1/jobs/${id}/details`);$('statusTitle').textContent=progress.stage||'Ожидание GPU-worker';await sleep(2500);
  }
  throw Error('Превышено ожидание. Проверьте историю задания.');
 }catch(e){$('error').textContent=e.message||String(e);}
 finally{$('status-card').classList.add('hidden');$('submit').disabled=false;}
});

async function showWorkerStatus(){
 if(!key.value.trim())return;
 try{const info=await req('/api/v1/model-status');$('workerStatus').textContent=info.worker_online?'GPU WORKER ONLINE':'GPU WORKER OFFLINE';}
 catch{$('workerStatus').textContent='CHECK API KEY';}
}
key.addEventListener('change',showWorkerStatus);
$('refresh').addEventListener('click',showWorkerStatus);
showWorkerStatus();
