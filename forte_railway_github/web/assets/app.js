const $=id=>document.getElementById(id);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const input=$('videoInput'),key=$('apiKey'),drop=$('drop');
const make=(tag,cls,txt)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(txt!==undefined)e.textContent=String(txt);return e;};
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
 $('result').classList.remove('hidden');
 $('verdict').textContent=report.verdict==='violation'?'ВОЗМОЖНОЕ НАРУШЕНИЕ':'НЕ ОБНАРУЖЕНО';
 $('verdict').classList.toggle('warn',report.verdict==='violation');
 $('processName').textContent=details.process?.name||'Не определён';
 $('processWhy').textContent=details.process?.reason||'';
 $('eventCount').textContent=String(details.events?.length||0);
 $('processingTime').textContent=details.processing_sec!==undefined?`${details.processing_sec} с`:'—';
 const events=$('events');events.replaceChildren();
 for(const ev of details.events||[]){const box=make('div','event');const a=Number(ev.start_sec||0),b=Number(ev.end_sec||0);box.append(make('small','',`${a.toFixed(1)}–${b.toFixed(1)} с`),make('strong','',ev.action||'action'),make('span','',`${ev.source||'model'} · conf ${ev.confidence??'n/a'}`));events.append(box);}
 if(!events.children.length)events.append(make('div','empty','Нет уверенно распознанных действий'));
 const violations=$('violations');violations.replaceChildren();
 for(const v of report.violations||[]){const box=make('div','violation');const encoded=v.evidence_frames?.[0]?.image_base64;
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
