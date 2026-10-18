from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient

from backend.config import Config
from backend.main import create_app


@pytest.fixture
def env(tmp_path):
    cfg=Config(api_key='client-key-with-32-characters-AAA', worker_key='worker-key-with-32-characters-BBB',
               storage_dir=tmp_path,save_videos=True,max_upload_mb=10,max_pending=3,job_timeout_sec=300)
    with TestClient(create_app(cfg)) as client:
        yield client,cfg


def upload(c,cfg,name='X-1.mov'):
    return c.post('/api/v1/analyze',headers={'X-API-Key':cfg.api_key},files={'file':(name,b'fake video payload','video/quicktime')})


def test_auth_and_health(env):
    c,cfg=env
    assert c.get('/health').status_code==200
    assert c.post('/api/v1/analyze',files={'file':('a.mov',b'123')}).status_code==401
    assert c.get('/api/v1/history').status_code==401
    assert c.post('/internal/claim').status_code==401
    assert c.get('/').status_code==200


def test_complete_roundtrip_and_archive(env):
    c,cfg=env
    r=upload(c,cfg)
    assert r.status_code==202,r.text
    job=r.json()['job_id']
    assert c.get(f'/api/v1/jobs/{job}',headers={'X-API-Key':cfg.api_key}).json()['status']=='processing'
    claim=c.post('/internal/claim',headers={'X-Worker-Key':cfg.worker_key}).json()['job']
    assert claim['id']==job
    headers={'X-Worker-Key':cfg.worker_key,'X-Job-Lease':claim['lease']}
    assert c.get(f'/internal/jobs/{job}/video',headers=headers).content==b'fake video payload'
    assert c.post(f'/internal/jobs/{job}/stage',headers=headers,json={'stage':'YOLO26 objects'}).status_code==200
    details={'process':{'name':'Приём наличных','process_id':0},
             'events':[{'action':'cash_handover','start_sec':7,'end_sec':10}],
             'processing_sec':19.8}
    result={'verdict':'violation','violations':[{'type':'suspected_cash_handover','timecode_sec':7,
             'description':'Needs review','evidence_frames':[{'timestamp_sec':7,'image_base64':'AA=='}]}]}
    r=c.post(f'/internal/jobs/{job}/result',headers=headers,json={'result':result,'details':details})
    assert r.status_code==200,r.text
    pub=c.get(f'/api/v1/jobs/{job}',headers={'X-API-Key':cfg.api_key}).json()
    assert pub=={'job_id':job,'status':'done',**result}
    view=c.get(f'/api/v1/jobs/{job}/details',headers={'X-API-Key':cfg.api_key}).json()
    assert view['events'][0]['action']=='cash_handover'
    history=c.get('/api/v1/history',headers={'X-API-Key':cfg.api_key}).json()['jobs']
    assert len(history)==1 and history[0]['process_name']=='Приём наличных'
    assert c.get(f'/api/v1/jobs/{job}/report',headers={'X-API-Key':cfg.api_key}).status_code==200
    assert c.get(f'/api/v1/jobs/{job}/video',headers={'X-API-Key':cfg.api_key}).content==b'fake video payload'
    assert c.delete(f'/api/v1/jobs/{job}',headers={'X-API-Key':cfg.api_key}).status_code==200
    assert c.get(f'/api/v1/jobs/{job}',headers={'X-API-Key':cfg.api_key}).status_code==404
    assert not any((cfg.storage_dir/'uploads').iterdir())


def test_lease_security(env):
    c,cfg=env
    job=upload(c,cfg).json()['job_id']
    claim=c.post('/internal/claim',headers={'X-Worker-Key':cfg.worker_key}).json()['job']
    h={'X-Worker-Key':cfg.worker_key,'X-Job-Lease':'forged'}
    assert c.get(f'/internal/jobs/{job}/video',headers=h).status_code==409
    assert c.post(f'/internal/jobs/{job}/result',headers=h,json={'result':{'verdict':'no_violation','violations':[]},'details':{}}).status_code==409
    assert c.delete(f'/api/v1/jobs/{job}',headers={'X-API-Key':cfg.api_key}).status_code==409
    h['X-Job-Lease']=claim['lease']
    assert c.post(f'/internal/jobs/{job}/fail',headers=h,json={'error':'test error'}).status_code==200
    assert c.get(f'/api/v1/jobs/{job}',headers={'X-API-Key':cfg.api_key}).json()['status']=='failed'


def test_save_off(tmp_path):
    cfg=Config(api_key='client-key-with-32-characters-AAA',worker_key='worker-key-with-32-characters-BBB',
               storage_dir=tmp_path,save_videos=False)
    with TestClient(create_app(cfg)) as c:
        job=upload(c,cfg).json()['job_id']
        claim=c.post('/internal/claim',headers={'X-Worker-Key':cfg.worker_key}).json()['job']
        headers={'X-Worker-Key':cfg.worker_key,'X-Job-Lease':claim['lease']}
        assert c.post(f'/internal/jobs/{job}/result',headers=headers,
                      json={'result':{'verdict':'no_violation','violations':[]},'details':{}}).status_code==200
        assert c.get(f'/api/v1/jobs/{job}/video',headers={'X-API-Key':cfg.api_key}).status_code==404
        assert not any((tmp_path/'uploads').iterdir())


def test_upload_validations(env):
    c,cfg=env
    h={'X-API-Key':cfg.api_key}
    assert c.post('/api/v1/analyze',headers=h,files={'file':('a.txt',b'abc')}).status_code==415
    assert c.post('/api/v1/analyze',headers=h,files={'file':('a.mov',b'')}).status_code==400
    assert c.get('/api/v1/jobs/../../../etc/passwd',headers=h).status_code in (404,422)


def test_worker_presence_uses_actual_action_labels(env):
    c,cfg=env
    before=c.get('/api/v1/model-status',headers={'X-API-Key':cfg.api_key}).json()
    assert not before['worker_online']
    worker={'X-Worker-Key':cfg.worker_key}
    assert c.post('/internal/heartbeat',headers=worker,json={'worker_id':'v100-0',
       'detector_classes':['cash','person','cash_bag'],
       'action_classes':['cash_handover','safe_closing','document_handover']}).status_code==200
    status=c.get('/api/v1/model-status',headers={'X-API-Key':cfg.api_key}).json()
    assert status['worker_online']
    assert status['workers'][0]['action_classes']==['cash_handover','safe_closing','document_handover']
