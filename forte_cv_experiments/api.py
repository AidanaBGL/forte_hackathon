"""Optional local FastAPI for Datathon. Run after training a temporal experiment.

  pip install fastapi uvicorn python-multipart
  export FORTE_API_KEY=your-long-secret
  export FORTE_MODEL=runs/exp2_temporal/model.joblib
  uvicorn api:app --host 0.0.0.0 --port 8080

Models stay local. Video uploads are deleted after processing. 280s worker timeout
plus 20s buffer for HTTP serialization; after a timeout the worker may complete
internally but the job remains failed. Run with one worker for GPU resource safety.
"""
import base64,os,shutil,tempfile,time,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import cv2,joblib,numpy as np
from fastapi import FastAPI,UploadFile,File,Header,HTTPException,Depends
from src.features import VisionExtractor
from src.experiments import aggregate

API_KEY=os.environ.get('FORTE_API_KEY')
MODEL_FILE=os.environ.get('FORTE_MODEL','runs/exp2_temporal/model.joblib')
MAX_SECONDS=float(os.environ.get('FORTE_MAX_SECONDS','280'))
app=FastAPI(title='ForteBank Procedure Screening (experimental)')
pool=ThreadPoolExecutor(max_workers=1)
lock=Lock();infer_lock=Lock();jobs={}
state={'model':None,'extractor':None,'settings':None}


def authenticate(x_api_key: str|None=Header(default=None,alias='X-API-Key')):
    if not API_KEY:raise HTTPException(status_code=503,detail='Set FORTE_API_KEY')
    if x_api_key!=API_KEY:raise HTTPException(status_code=401,detail='Invalid API key')

@app.on_event('startup')
def load_model():
    saved=joblib.load(MODEL_FILE)
    if saved['method']!='temporal':
        raise RuntimeError('API requires temporal experiment for meaningful evidence timecode')
    from pathlib import Path
    feat_dir=Path(saved['features_dir'])
    settings=__import__('json').loads((feat_dir/'settings.json').read_text())
    roi=','.join(str(q) for q in settings['roi']) if settings.get('roi') else ''
    state.update(model=saved,settings=settings,extractor=VisionExtractor(
        encoder=settings['encoder'],detector=settings['detector'],hands=settings['hands'],roi=roi,
        hand_model=os.environ.get('FORTE_HAND_MODEL',''),
        custom_weights=os.environ.get('FORTE_CUSTOM_WEIGHTS',''),max_side=settings.get('max_side',640)))


def process(job_id,path):
    t0=time.monotonic()
    try:
        with infer_lock:
            ext=state['extractor'];s=state['settings'];m=state['model']
            tt,x,duration=ext.process_video(path,s['window_sec'],s['stride_sec'],s['frames_per_window'])
            score=m['model'].predict_proba(x)[:,1]
            best=int(np.argmax(score));prob=float(score[best])
            threshold=float(m['threshold']);viol=prob>=threshold
            ts=float(np.mean(tt[best]))
            violations=[]
            if viol:
                cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_MSEC,ts*1000)
                ok,frame=cap.read();cap.release()
                evidence=[]
                if ok:
                    _,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,78])
                    evidence=[{'timestamp_sec':round(ts,2),
                               'image_base64':base64.b64encode(buf.tobytes()).decode('ascii')}]
                violations=[{'type':'Возможное отклонение (тип не классифицирован)',
                             'timecode_sec':round(ts,2),
                             'description':'Слабая ML-модель отметила интервал как подозрительный. '+
                                           'Не является доказательством юридического нарушения.',
                             'evidence_frames':evidence}]
            if time.monotonic()-t0>MAX_SECONDS:
                result={'status':'failed','error':'Processing exceeded time limit'}
            else:
                result={'status':'done','verdict':'violation' if viol else 'no_violation',
                        'violations':violations}
            with lock:jobs[job_id].update(result)
    except Exception as e:
        with lock:jobs[job_id].update({'status':'failed','error':f'{type(e).__name__}: {e}'})
    finally:
        Path(path).unlink(missing_ok=True)

@app.post('/api/v1/analyze',dependencies=[Depends(authenticate)])
def analyze(file:UploadFile=File(...)):
    if not file.filename.lower().endswith('.mov'):
        raise HTTPException(status_code=400,detail='Expected .mov')
    path=Path(tempfile.gettempdir())/f'forte_{uuid.uuid4().hex}.mov'
    with path.open('wb') as f:shutil.copyfileobj(file.file,f)
    job_id=uuid.uuid4().hex[:12]
    with lock:jobs[job_id]={'job_id':job_id,'status':'processing','submitted_monotonic':time.monotonic()}
    pool.submit(process,job_id,path)
    return {'job_id':job_id}

@app.get('/api/v1/jobs/{job_id}',dependencies=[Depends(authenticate)])
def get_job(job_id:str):
    with lock:
        if job_id not in jobs:raise HTTPException(status_code=404,detail='Unknown job')
        j=jobs[job_id].copy()
        if j['status']=='processing' and time.monotonic()-j['submitted_monotonic']>MAX_SECONDS:
            j.update(status='failed',error='Queue or inference exceeded time limit')
            jobs[job_id].update(status='failed',error=j['error'])
    j.pop('submitted_monotonic',None)
    return j
