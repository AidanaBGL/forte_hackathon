"""Runs on the user's own NVIDIA V100 server; polls Railway outbound-only.

Railway receives and temporarily holds uploaded videos. This worker downloads them,
performs actual YOLO26 + Action Recognition locally, and posts JSON evidence back.
No hosted AI inference API is used.
"""
from __future__ import annotations
import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import requests

log=logging.getLogger('forte.worker')


def request_or_raise(session, method, url, **kwargs):
    response=session.request(method,url,**kwargs)
    response.raise_for_status()
    return response


def run_worker(once=False):
    gateway=os.environ.get('FORTE_GATEWAY_URL','').rstrip('/')
    # FORTE_WORKER_KEY is optional; without it the worker authenticates with FORTE_API_KEY
    # (the gateway's /internal routes fall back to the same key — see Config.internal_key).
    secret=os.environ.get('FORTE_WORKER_KEY') or os.environ.get('FORTE_API_KEY','')
    if not gateway.startswith(('https://','http://')) or len(secret)<16:
        raise RuntimeError('Set FORTE_GATEWAY_URL and FORTE_API_KEY (or FORTE_WORKER_KEY)')
    if gateway.startswith('http://') and not gateway.startswith('http://localhost') and not gateway.startswith('http://127.0.0.1'):
        raise RuntimeError('Use HTTPS for a remote Railway gateway')
    if not os.environ.get('FORTE_DETECTOR_WEIGHTS'):
        raise RuntimeError('Missing FORTE_DETECTOR_WEIGHTS — no pretrained mock fallback allowed')
    if not os.environ.get('FORTE_ACTION_CHECKPOINT'):
        raise RuntimeError('Missing FORTE_ACTION_CHECKPOINT — configure your trained action model')

    # Set environment defaults BEFORE importing Settings (dataclass defaults are evaluated on import).
    os.environ.setdefault('FORTE_API_KEY','local-v100-pipeline-key')
    os.environ.setdefault('FORTE_JOB_TIMEOUT_SEC','265')
    os.environ.setdefault('FORTE_MAX_DURATION_SEC','180')
    from .vision.settings import Settings
    from .vision.inference import FortePipeline
    settings=Settings()
    pipeline=FortePipeline(settings)
    log.info('Ready: detector=%s, action classes=%s',settings.detector_weights,pipeline.action.classes)
    session=requests.Session()
    session.headers.update({'X-Worker-Key':secret})
    worker_id=os.environ.get('FORTE_WORKER_ID','v100-worker-0')
    interval=max(1,int(os.environ.get('FORTE_POLL_SECONDS','2')))
    last_heartbeat=0.0

    while True:
        try:
            if time.monotonic()-last_heartbeat>=10:
                request_or_raise(session,'POST',f'{gateway}/internal/heartbeat',json={
                    'worker_id':worker_id,
                    'detector_classes':sorted(pipeline.cls_names),
                    'action_classes':list(pipeline.action.classes),
                },timeout=(10,15))
                last_heartbeat=time.monotonic()
            rsp=request_or_raise(session,'POST',f'{gateway}/internal/claim',params={'worker_id':worker_id},timeout=(10,20)).json()
            task=rsp.get('job')
            if not task:
                if once:return
                time.sleep(interval)
                continue
            job_id=task['id'];lease=task['lease']
            headers={'X-Job-Lease':lease}
            log.info('Claimed %s %s',job_id,task['filename'])
            with tempfile.TemporaryDirectory(prefix='forte-worker-') as temp:
                extension=Path(task['filename']).suffix.lower()
                path=Path(temp)/('input'+extension)
                with request_or_raise(session,'GET',f'{gateway}/internal/jobs/{job_id}/video',headers=headers,
                                      stream=True,timeout=(15,120)) as response:
                    with path.open('wb') as file:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:file.write(chunk)
                if path.stat().st_size!=int(task['size_bytes']):
                    raise RuntimeError('Uploaded video size mismatch during download')

                def report_stage(s):
                    log.info('%s: %s',job_id,s)
                    try:
                        request_or_raise(session,'POST',f'{gateway}/internal/jobs/{job_id}/stage',headers=headers,
                                         json={'stage':s},timeout=(5,15))
                    except Exception:
                        log.warning('Could not update progress stage for %s',job_id)

                try:
                    result,details=pipeline.analyze(path,on_stage=report_stage)
                    request_or_raise(session,'POST',f'{gateway}/internal/jobs/{job_id}/result',headers=headers,
                                     json={'result':result,'details':details},timeout=(15,60))
                    log.info('Done %s: %s; %d findings',job_id,result['verdict'],len(result['violations']))
                except Exception as e:
                    log.exception('Inference failed: %s',job_id)
                    try:
                        request_or_raise(session,'POST',f'{gateway}/internal/jobs/{job_id}/fail',headers=headers,
                                         json={'error':f'{type(e).__name__}: {str(e)[:350]}'},timeout=(5,20))
                    except Exception:
                        log.error('Could not report failure to Railway: %s',job_id)
            if once:return
        except (requests.RequestException,OSError) as e:
            log.exception('Worker transport error: %s',e)
            if once:raise
            time.sleep(min(20,interval*3))


def main():
    p=argparse.ArgumentParser(description='Forte Vision outbound GPU worker')
    p.add_argument('--once',action='store_true',help='Process at most one job then exit')
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    run_worker(once=args.once)

if __name__=='__main__':main()
