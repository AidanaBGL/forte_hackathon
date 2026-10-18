"""Railway CPU gateway. Never imports torch, YOLO or video AI dependencies."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Config, ROOT
from .storage import Store

ID_PATTERN=re.compile(r'^[0-9a-f]{16}$')


class ResultReport(BaseModel):
    result: dict
    details: dict = Field(default_factory=dict)


class WorkerHeartbeat(BaseModel):
    worker_id: str = "v100-0"
    detector_classes: list[str] = Field(default_factory=list)
    action_classes: list[str] = Field(default_factory=list)


class StageReport(BaseModel):
    stage: str


def public_row(row):
    return {k:row.get(k) for k in ('id','filename','status','stage','verdict','process_name','created_at',
                                   'finished_at','size_bytes','duration_sec','error','video_retained')}


def create_app(config: Config|None=None):
    cfg=config or Config()
    store=Store(cfg.storage_dir)

    @asynccontextmanager
    async def lifespan(app:FastAPI):
        cfg.validate()
        os.umask(0o077)
        store.init()
        yield

    app=FastAPI(title='Forte Vision — Railway Gateway',version='1.0.0',lifespan=lifespan)
    app.state.store=store

    def check_client(x_api_key:str|None=Header(default=None,alias='X-API-Key')):
        if not x_api_key or not hmac.compare_digest(x_api_key,cfg.api_key):
            raise HTTPException(401,'Invalid or missing X-API-Key')

    def check_worker(x_worker_key:str|None=Header(default=None,alias='X-Worker-Key')):
        if not x_worker_key or not hmac.compare_digest(x_worker_key,cfg.internal_key):
            raise HTTPException(401,'Invalid worker credentials')

    def row_or_404(job_id):
        if not ID_PATTERN.fullmatch(job_id):raise HTTPException(404,'Job not found')
        row=store.one(job_id)
        if not row:raise HTTPException(404,'Job not found')
        return row

    def require_lease(job_id,lease):
        if not lease or not store.is_lease(job_id,lease):
            raise HTTPException(409,'Job lease expired or invalid')

    def expire():store.expire(cfg.job_timeout_sec)

    @app.get('/')
    def index():return FileResponse(ROOT/'web'/'index.html')

    @app.get('/health')
    def health():return {'status':'ok','component':'railway-gateway'}

    @app.post('/api/v1/analyze',dependencies=[Depends(check_client)],status_code=202)
    async def analyze(file:UploadFile=File(...)):
        expire()
        suffix=Path(file.filename or '').suffix.lower()
        if suffix not in ({'.mov','.mp4'} if cfg.allow_mp4 else {'.mov'}):
            raise HTTPException(415,'Expected .mov or .mp4 video')
        if store.count_pending()>=cfg.max_pending:
            raise HTTPException(503,'Processing queue is full')
        store.uploads.mkdir(parents=True,exist_ok=True)
        path=store.uploads/f'{uuid.uuid4().hex}{suffix}'
        digest=hashlib.sha256()
        total=0
        try:
            with path.open('xb') as f:
                while True:
                    chunk=await file.read(1024*1024)
                    if not chunk: break
                    total+=len(chunk)
                    if total>cfg.max_upload_mb*1024*1024:
                        raise HTTPException(413,'Video too large')
                    digest.update(chunk)
                    f.write(chunk)
            if not total:raise HTTPException(400,'Empty video')
            job_id=store.create(Path(file.filename).name[:180],path,digest.hexdigest(),total)
            return {'job_id':job_id}
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get('/api/v1/jobs/{job_id}',dependencies=[Depends(check_client)])
    def get_job(job_id:str):
        expire()
        row=row_or_404(job_id)
        if row['status']=='done':
            payload=store.load_result(row)
            if not payload:raise HTTPException(503,'Report temporarily unavailable')
            return {k:payload[k] for k in ('job_id','status','verdict','violations')}
        if row['status']=='failed':
            return {'job_id':job_id,'status':'failed','error':row['error'] or 'Processing failed'}
        # The competition API only accepts processing/done/failed.
        return {'job_id':job_id,'status':'processing'}

    @app.get('/api/v1/jobs/{job_id}/details',dependencies=[Depends(check_client)])
    def get_details(job_id:str):
        expire()
        row=row_or_404(job_id)
        if row['status']!='done':
            return {'job_id':job_id,'status':row['status'],'stage':row['stage'],'error':row['error']}
        payload=store.load_result(row)
        if not payload:raise HTTPException(503,'Report temporarily unavailable')
        details=dict(payload.get('details') or {})
        # Raw object detections can be large; keep them in the persisted report for download.
        details.pop('observations',None)
        details.pop('action_windows',None)
        return {'job_id':job_id,'status':'done',**details}

    @app.get('/api/v1/history',dependencies=[Depends(check_client)])
    def history(limit:int=50):
        expire()
        return {'jobs':[public_row(row) for row in store.all(min(max(1,limit),100))]}

    @app.get('/api/v1/jobs/{job_id}/report',dependencies=[Depends(check_client)])
    def download_report(job_id:str):
        row=row_or_404(job_id)
        if row['status']!='done' or not row['result_path'] or not Path(row['result_path']).is_file():
            raise HTTPException(404,'Report unavailable')
        return FileResponse(row['result_path'],media_type='application/json',filename=f'{job_id}-report.json')

    @app.get('/api/v1/jobs/{job_id}/video',dependencies=[Depends(check_client)])
    def download_video(job_id:str):
        row=row_or_404(job_id)
        if not row['video_retained'] or not Path(row['upload_path']).is_file():
            raise HTTPException(404,'Video was not retained')
        return FileResponse(row['upload_path'],filename=row['filename'],media_type='application/octet-stream')

    @app.delete('/api/v1/jobs/{job_id}',dependencies=[Depends(check_client)])
    def delete_job(job_id:str):
        row_or_404(job_id)
        try:store.delete(job_id)
        except ValueError as e:raise HTTPException(409,str(e))
        return {'deleted':True,'job_id':job_id}

    @app.get('/api/v1/model-status',dependencies=[Depends(check_client)])
    def model_status():
        # Railway itself never loads checkpoints; worker does.
        workers=[{'worker_id':w['worker_id'],'online':time.time()-w['last_seen']<30,
                  'last_seen':w['last_seen'], 'detector_classes':json.loads(w['detector_classes']),
                  'action_classes':json.loads(w['action_classes'])} for w in store.workers()]
        return {'mode':'remote GPU worker','gateway_gpu':False,'result_storage':True,
                'worker_online':any(w['online'] for w in workers),'workers':workers}

    @app.post('/internal/heartbeat',dependencies=[Depends(check_worker)])
    def worker_heartbeat(body:WorkerHeartbeat):
        store.presence(body.worker_id,body.detector_classes,body.action_classes)
        return {'ok':True}

    @app.post('/internal/claim',dependencies=[Depends(check_worker)])
    def claim(worker_id:str='v100-0'):
        expire()
        task=store.claim(worker_id[:80])
        return {'job':task}

    @app.get('/internal/jobs/{job_id}/video',dependencies=[Depends(check_worker)])
    def worker_video(job_id:str,x_job_lease:str|None=Header(default=None,alias='X-Job-Lease')):
        row=row_or_404(job_id)
        require_lease(job_id,x_job_lease)
        if not Path(row['upload_path']).is_file():raise HTTPException(404,'Source video missing')
        return FileResponse(row['upload_path'],media_type='application/octet-stream')

    @app.post('/internal/jobs/{job_id}/stage',dependencies=[Depends(check_worker)])
    def update_stage(job_id:str,body:StageReport,x_job_lease:str|None=Header(default=None,alias='X-Job-Lease')):
        row_or_404(job_id)
        require_lease(job_id,x_job_lease)
        store.stage(job_id,x_job_lease,body.stage)
        return {'ok':True}

    @app.post('/internal/jobs/{job_id}/result',dependencies=[Depends(check_worker)])
    def submit_result(job_id:str,body:ResultReport,x_job_lease:str|None=Header(default=None,alias='X-Job-Lease')):
        row_or_404(job_id)
        require_lease(job_id,x_job_lease)
        result=body.result
        if result.get('verdict') not in ('violation','no_violation') or not isinstance(result.get('violations'),list):
            raise HTTPException(422,'Result must have verdict and violations list')
        if len(result['violations'])>30:raise HTTPException(422,'Too many candidate violations')
        for v in result['violations']:
            if not all(k in v for k in ('type','timecode_sec','description','evidence_frames')):
                raise HTTPException(422,'Invalid violation schema')
            if len(v['evidence_frames'])>3:
                raise HTTPException(422,'Maximum 3 evidence frames per violation')
        ok=store.complete(job_id,x_job_lease,result,body.details,cfg.save_videos)
        if not ok:raise HTTPException(409,'Job no longer active')
        return {'ok':True}

    @app.post('/internal/jobs/{job_id}/fail',dependencies=[Depends(check_worker)])
    def worker_failed(job_id:str,body:dict,x_job_lease:str|None=Header(default=None,alias='X-Job-Lease')):
        row_or_404(job_id)
        require_lease(job_id,x_job_lease)
        ok=store.fail(job_id,x_job_lease,str(body.get('error','Unknown failure')),cfg.save_videos)
        if not ok:raise HTTPException(409,'Job no longer active')
        return {'ok':True}

    app.mount('/assets',StaticFiles(directory=str(ROOT/'web'/'assets')),name='assets')
    return app


app=create_app()
