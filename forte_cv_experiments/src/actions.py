"""Train supervised step/action classifier AFTER human labeling action_segments.csv.
Video-level 'violation' or a timecode is NOT a valid action label.
"""
import csv,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .experiments import read_feat
from .dataset import load_manifest,vid_id


def read_segments(path):
    d={}
    with open(path,encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if not row.get('video_id') or not row.get('action'):continue
            k=vid_id(row['video_id']);start=float(row['start_sec']);end=float(row['end_sec'])
            if end<=start:raise ValueError('Invalid segment '+repr(row))
            d.setdefault(k,[]).append((start,end,row['action'].strip()))
    return d


def make_labels(times,segments):
    result=[]
    for begin,end in times:
        scores={}
        for s,e,name in segments:
            overlap=max(0,min(float(end),e)-max(float(begin),s))
            ratio=overlap/max(0.001,float(end-begin))
            if ratio>=0.45:
                scores[name]=max(scores.get(name,0.0),ratio)
        if len(scores)==1: result.append(next(iter(scores)))
        else:result.append(None)
    return result


def train_action_classifier(manifest_file,feature_dir,segments_file,out_path):
    manifest=load_manifest(manifest_file)
    rows={r['video_id']:r for r in manifest}
    seg=read_segments(segments_file)
    X=[];y=[];samples=[]
    for video,sub in seg.items():
        if video not in rows:raise ValueError('Annotation has unknown video '+video)
        if rows[video]['split']=='test':
            raise ValueError('DO NOT train on held-out test action annotations: '+video)
        if rows[video]['duplicate_of']:continue
        times,features=read_feat(feature_dir,video)
        labels=make_labels(times,sub)
        for i,label in enumerate(labels):
            if label is not None:
                X.append(features[i]);y.append(label);samples.append(video)
    if len(set(y))<2:raise RuntimeError('Please annotate at least 2 action classes in 2+ train videos')
    if len(set(samples))<2:raise RuntimeError('Need samples from 2+ independent train videos')
    counts={k:y.count(k) for k in set(y)}
    print('Action clips used:',len(y),'classes:',counts)
    model=Pipeline([('scaler',StandardScaler()),
                    ('lr',LogisticRegression(max_iter=2000,C=0.25,class_weight='balanced'))])
    model.fit(np.asarray(X),np.asarray(y))
    out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'feature_dir':str(feature_dir),'action_names':sorted(set(y))},out)
    print('Saved action classifier:',out)
    print('WARNING: Action training accuracy is not validation accuracy. Do not trust predictions before held-out action evaluation.')


def predict_actions(manifest_file,feature_dir,weights,out_dir,split='test'):
    rows=load_manifest(manifest_file)
    model=joblib.load(weights)['model']
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    for r in rows:
        if split!='all' and r['split']!=split:continue
        tt,x=read_feat(feature_dir,r['video_id'])
        ps=model.predict_proba(x);ix=ps.argmax(axis=1)
        pred=[{'start_sec':round(float(t[0]),3),'end_sec':round(float(t[1]),3),
               'action':str(model.classes_[k]),'confidence':round(float(ps[i,k]),4)}
              for i,(t,k) in enumerate(zip(tt,ix))]
        (out/(r['video_id']+'.json')).write_text(json.dumps(pred,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Action timelines written to',out)
