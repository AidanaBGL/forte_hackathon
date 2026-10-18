"""Illustrative (NOT normative) temporal hypotheses for manually annotated actions.

Rule engine operates on ordered action events and keeps UNKNOWN state distinct
from OPEN/CLOSED. The output is for REVIEW, never an automatic legal verdict.
"""
import json
from pathlib import Path


def merge_segments(timeline,min_conf=0.45):
    a=sorted([x for x in timeline if float(x.get('confidence',1))>=min_conf],
             key=lambda x:float(x['start_sec']))
    merged=[]
    for x in a:
        start=float(x['start_sec']);end=float(x['end_sec']);name=x['action']
        if merged and merged[-1]['action']==name and start<=merged[-1]['end_sec']+0.05:
            merged[-1]['end_sec']=max(end,merged[-1]['end_sec'])
            merged[-1]['confidence']=max(merged[-1]['confidence'],float(x.get('confidence',1)))
        else:
            merged.append({'start_sec':start,'end_sec':end,'action':name,
                           'confidence':float(x.get('confidence',1))})
    return merged


def check_hypotheses(timeline):
    events=merge_segments(timeline)
    state='UNKNOWN';retrieved_at=None;warnings=[]
    for e in events:
        a=e['action']; t=e['start_sec']
        if a=='safe_opening':state='OPEN';retrieved_at=None
        elif a=='safe_closing':state='CLOSED';retrieved_at=None
        elif a=='cash_retrieval' and state=='OPEN':retrieved_at=t
        elif a in ('cash_counting_manual','cash_counting_machine'):
            if state=='OPEN' and retrieved_at is not None:
                warnings.append(dict(rule_id='H01',timecode_sec=t,
                 status='hypothesis_only',source='user-observed workflow, unverified internal procedure',
                 description='Пересчёт после извлечения наличных, пока сейф визуально открыт. Требуется проверка применимого порядка.'))
        elif a=='cashier_leaving':
            if state=='OPEN':
                warnings.append(dict(rule_id='H02',timecode_sec=t,
                    status='hypothesis_only',source='user-observed workflow, unverified internal procedure',
                    description='Кассир покидает рабочее место при визуально открытой дверце сейфа. Требуется проверка применимого порядка.'))
    return {'segments':events,'review_candidates':warnings,'legal_verdict':'not_assessed'}


def run(in_dir,out_dir):
    d=Path(in_dir);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    n=0
    for f in sorted(d.glob('X-*.json')):
        events=json.loads(f.read_text(encoding='utf-8'))
        res=check_hypotheses(events)
        (out/f.name).write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
        n+=1
    print('Hypothesis review reports:',n,'in',out)
