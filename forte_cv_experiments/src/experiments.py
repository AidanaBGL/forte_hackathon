"""Video classification and weak point-supervised temporal anomaly baselines.

IMPORTANT: video labels do not say which action or legal violation occurred. The
point-supervised model learns 'looks similar to known violation moments', NOT
banking regulation correctness or semantic action types.
"""
import csv, json, time
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix,accuracy_score,precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from .dataset import load_manifest


def read_feat(folder,video):
    with np.load(Path(folder)/(video+'.npz')) as data:
        return data['times'],data['X']


def aggregate(x):
    return np.concatenate([np.mean(x,axis=0),np.std(x,axis=0),np.max(x,axis=0)]).astype(np.float32)


def clf():
    return Pipeline([
        ('impute',SimpleImputer(strategy='median',add_indicator=False)),
        ('scaler',StandardScaler()),
        ('lr',LogisticRegression(C=0.25,class_weight='balanced',max_iter=1500,random_state=42))
    ])


def training_rows(manifest,feat_folder):
    # Never touch held-out test features or labels during fitting or threshold tuning.
    train=[r for r in manifest if r['split']=='train' and not r['duplicate_of']]
    absent=[r['video_id'] for r in train if not (Path(feat_folder)/(r['video_id']+'.npz')).exists()]
    if absent:raise FileNotFoundError('Missing train features for: '+','.join(absent))
    return train


def temporal_training_samples(rows,folder):
    xx=[];yy=[]
    for r in rows:
        times, features=read_feat(folder,r['video_id'])
        if int(r['label'])==0:
            xx.extend(features)
            yy.extend([0]*len(features))
        else:
            if not r['timecode_sec']:
                continue
            gt=float(r['timecode_sec'])
            # Supervise windows that actually contain the point event; DO NOT
            # label all other windows negative (there may be other violations).
            inds=np.where((times[:,0]<=gt)&(times[:,1]>=gt))[0]
            if len(inds)==0:
                nearest=int(np.argmin(abs(times.mean(axis=1)-gt)))
                inds=[nearest]
            for i in inds:
                xx.append(features[i]);yy.append(1)
    if not xx:raise RuntimeError('No weakly labelled windows')
    x=np.asarray(xx,np.float32);y=np.asarray(yy,np.int32)
    if len(np.unique(y))!=2:raise RuntimeError('Need positives and negatives')
    return x,y


def fit(rows,features_dir,method):
    if method=='video':
        X=np.stack([aggregate(read_feat(features_dir,r['video_id'])[1]) for r in rows])
        y=np.array([int(r['label']) for r in rows])
    elif method=='temporal':
        X,y=temporal_training_samples(rows,features_dir)
    else:raise ValueError('method must be video or temporal')
    model=clf().fit(X,y)
    return model


def predict(model,features_dir,r,method):
    ts,x=read_feat(features_dir,r['video_id'])
    if method=='video':
        prob=float(model.predict_proba(aggregate(x)[None])[0,1])
        # VIDEO-level features do not localize. No guessed precise timecode.
        return prob,None,[]
    probs=model.predict_proba(x)[:,1]
    best=int(np.argmax(probs))
    best_time=float(ts[best].mean())
    windows=[{'start_sec':round(float(t[0]),3),'end_sec':round(float(t[1]),3),
              'score':round(float(p),6)} for t,p in zip(ts,probs)]
    return float(probs.max()),best_time,windows


def pick_threshold(y,prob):
    # Set decision threshold on TRAIN OOF only, never test.
    y=np.array(y,dtype=int);p=np.array(prob,dtype=float)
    candidates=np.unique(np.r_[0.1,np.linspace(0.15,0.9,31),p])
    best=(float('-inf'),0.5)
    for t in candidates:
        pred=(p>=t).astype(int)
        precision,recall,f2,_=precision_recall_fscore_support(y,pred,average='binary',
                                                zero_division=0,beta=2.0)
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        fpr=fp/(fp+tn) if fp+tn else 0
        # Balance recall and false alarms; avoid trivial all-positive decisions.
        tpr=tp/(tp+fn) if tp+fn else 0.0
        balanced_accuracy=0.5*(tpr+(1-fpr))
        score=0.70*balanced_accuracy+0.30*f2
        if score>best[0]:best=(score,float(t))
    return best[1]


def cv_threshold(train,folder,method,folds=4):
    y=np.asarray([int(r['label']) for r in train])
    groups=np.asarray([r['sha256'] for r in train])
    n=min(folds,int(np.bincount(y).min()))
    if n<2:raise ValueError('Need multiple videos per class for CV')
    oof=np.zeros(len(train),float)
    splitter=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=42)
    for i,(ti,vi) in enumerate(splitter.split(np.zeros((len(y),1)),y,groups)):
        tr=[train[k] for k in ti]
        if len({int(r['label']) for r in tr})!=2:
            raise ValueError('CV fold has one class only. Reduce --folds.')
        model=fit(tr,folder,method)
        for k in vi:oof[k]=predict(model,folder,train[k],method)[0]
    threshold=pick_threshold(y,oof)
    return threshold,oof


def metrics(ys,probs,threshold):
    y=np.asarray(ys,dtype=int);p=np.asarray(probs,dtype=float)
    pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    out={'n_videos':len(y),'threshold':threshold,'accuracy':float(accuracy_score(y,pred)),
         'precision':float(precision),'violation_recall':float(recall),'f1':float(f1),
         'false_positive_rate':float(fp/(fp+tn)) if (fp+tn) else None,
         'tp':tp,'tn':tn,'fp':fp,'fn':fn}
    if len(np.unique(y))==2:out['roc_auc']=float(roc_auc_score(y,p))
    return out


def evidence_frame(video,time_sec,save_path):
    import cv2
    c=cv2.VideoCapture(str(video))
    c.set(cv2.CAP_PROP_POS_MSEC,max(0,float(time_sec))*1000)
    ok,frame=c.read();c.release()
    if not ok:return False
    save_path.parent.mkdir(parents=True,exist_ok=True)
    return bool(cv2.imwrite(str(save_path),frame,[cv2.IMWRITE_JPEG_QUALITY,86]))


def run_experiment(manifest_path,features_dir,out_dir,method='temporal',folds=4):
    rows=load_manifest(manifest_path)
    train=training_rows(rows,features_dir)
    test=[r for r in rows if r['split']=='test']
    if len(test)!=4:raise ValueError('Expected four held out video examples')
    if any(not (Path(features_dir)/(r['video_id']+'.npz')).exists() for r in test):
        raise FileNotFoundError('Extract feature files for all held-out test videos before evaluation')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    threshold,oof=cv_threshold(train,features_dir,method,folds)
    train_cv=metrics([int(r['label']) for r in train],oof,threshold)
    model=fit(train,features_dir,method)
    joblib.dump({'model':model,'threshold':threshold,'method':method,
                 'features_dir':str(Path(features_dir).resolve())},out/'model.joblib')
    print('TRAIN cross-validation:',json.dumps(train_cv,ensure_ascii=False,indent=2))
    preds=[];times=[]
    for r in test:
        p,tc,windows=predict(model,features_dir,r,method)
        decision=int(p>=threshold)
        truth=int(r['label'])
        evidence_path=''
        if tc is not None:
            fn=out/'evidence'/f'{r["video_id"]}_candidate.jpg'
            if evidence_frame(r['path'],tc,fn): evidence_path=str(fn)
        preds.append({'video_id':r['video_id'],'gt_violation':truth,
                      'gt_time_sec':r['timecode_sec'],'pred_violation':decision,
                      'score':round(p,5),'threshold':round(threshold,5),
                      'candidate_time_sec':'' if tc is None else round(tc,2),
                      'abs_time_error_sec':'' if tc is None or not r['timecode_sec'] else round(abs(tc-float(r['timecode_sec'])),2),
                      'evidence_path':evidence_path})
        times.append(windows)
        if windows:(out/f'{r["video_id"]}_timeline.json').write_text(json.dumps(windows,indent=2),encoding='utf-8')
        print(f'TEST {r["video_id"]}: gt={truth}, pred={decision}, p={p:.3f}, candidate={tc}')
    with open(out/'test_predictions.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(preds[0]));writer.writeheader();writer.writerows(preds)
    test_m=metrics([p['gt_violation'] for p in preds],
                   [p['score'] for p in preds],threshold)
    full={'method':method,'train_cv':train_cv,'test':test_m,
          'training_videos':len(train),'heldout_test_videos':[r['video_id'] for r in test],
          'run_seconds':round(time.perf_counter()-t0,2),'notes':
          'Video method has no event localization; temporal method uses point labels, NOT action types or legal validation.'}
    (out/'metrics.json').write_text(json.dumps(full,indent=2,ensure_ascii=False),encoding='utf-8')
    print('TEST metrics:',json.dumps(test_m,ensure_ascii=False,indent=2))
    print('Saved:',out/'test_predictions.csv')
    return full
