"""Prepare/Train/Evaluate Video COCO temporal action classifiers.

Never uses test set to select hyperparameters. Evaluation on test requires separate command.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,f1_score
from torch import nn
from torch.utils.data import DataLoader,WeightedRandomSampler
from tqdm import tqdm

from data import prepare,load_manifest,RoboflowClips
from models import make_model,criterion_for


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # cudnn deterministic improves repeatability but can reduce speed.
    torch.backends.cudnn.benchmark = True


def evaluate(model,loader,device,n_classes):
    model.eval(); logits_by_idx={}; labels_by_idx={};
    t0=time.perf_counter()
    with torch.inference_mode():
        for x,y,idx in tqdm(loader,desc='eval',leave=False):
            x=x.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',enabled=device.type=='cuda',dtype=torch.float16):
                logits=model(x).float().cpu().numpy()
            for i,z,gt in zip(idx.tolist(),logits,y.tolist()):
                logits_by_idx.setdefault(i,[]).append(z)
                labels_by_idx[i]=gt
    all_idx=sorted(logits_by_idx)
    lg=np.stack([np.stack(logits_by_idx[i]).mean(axis=0) for i in all_idx])
    pred=lg.argmax(axis=1)
    gold=np.asarray([labels_by_idx[i] for i in all_idx])
    probs=torch.softmax(torch.tensor(lg),dim=1).numpy()
    met={'accuracy':float(accuracy_score(gold,pred)),
         'macro_f1':float(f1_score(gold,pred,labels=list(range(n_classes)),average='macro',zero_division=0)),
         'balanced_accuracy':float(balanced_accuracy_score(gold,pred)),
         'weighted_f1':float(f1_score(gold,pred,average='weighted',zero_division=0)),
         'segments':int(len(gold)),'wall_seconds':round(time.perf_counter()-t0,2)}
    return met,all_idx,gold,pred,probs


def save_eval(folder,metrics,idx,gold,pred,probs,df,classes,prefix):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    df=df.reset_index(drop=True)
    rows=[]
    for n,i in enumerate(idx):
        row=df.iloc[i]
        rows.append({'video':row.video_name,'video_path':row.path,
            'start_sec':row.start_sec,'end_sec':row.end_sec,
            'gt_action':classes[int(gold[n])], 'pred_action':classes[int(pred[n])],
            'confidence':round(float(probs[n,pred[n]]),5),
            'correct':int(gold[n]==pred[n]),'segment_key':row.segment_key})
    pd.DataFrame(rows).to_csv(folder/f'{prefix}_predictions.csv',index=False)
    # Report presence of each action per VIDEO, not just per temporal segment.
    by_video={}
    for r in rows:
        slot=by_video.setdefault(r['video_path'],{'video':r['video'],'gt':set(),'pred':set()})
        slot['gt'].add(r['gt_action']);slot['pred'].add(r['pred_action'])
    vrows=[];g_multi=[];p_multi=[]
    for path,slot in sorted(by_video.items()):
        gt_set=slot['gt'];pred_set=slot['pred']
        g_multi.append([int(c in gt_set) for c in classes]);p_multi.append([int(c in pred_set) for c in classes])
        vrows.append({'video':slot['video'],'gt_actions':' | '.join(sorted(gt_set)),
                      'pred_actions':' | '.join(sorted(pred_set)),
                      'exact_set_match':int(gt_set==pred_set)})
    pd.DataFrame(vrows).to_csv(folder/f'{prefix}_video_predictions.csv',index=False)
    from sklearn.metrics import f1_score
    metrics['video_count']=len(vrows)
    metrics['video_exact_match']=float(np.mean([r['exact_set_match'] for r in vrows]))
    metrics['video_micro_f1']=float(f1_score(np.asarray(g_multi),np.asarray(p_multi),average='micro',zero_division=0))
    pd.DataFrame(confusion_matrix(gold,pred,labels=list(range(len(classes)))),
                 columns=classes,index=classes).to_csv(folder/f'{prefix}_confusion.csv')
    report=classification_report(gold,pred,labels=list(range(len(classes))),
                                 target_names=classes,zero_division=0,output_dict=True)
    pd.DataFrame(report).transpose().to_csv(folder/f'{prefix}_per_class.csv')
    with open(folder/f'{prefix}_metrics.json','w',encoding='utf-8') as f:
        json.dump(metrics,f,indent=2,ensure_ascii=False)


def loaders(df,model_name,frames,size,batch,workers,aug,train_views,eval_views):
    norm='imagenet' if model_name.endswith('_gru') else 'kinetics'
    train=RoboflowClips(df,frames,size,'train',aug=aug,normalization=norm,repeat_train=train_views)
    val=RoboflowClips(df,frames,size,'val',views=eval_views,normalization=norm)
    kwargs=dict(batch_size=batch,num_workers=workers,pin_memory=torch.cuda.is_available())
    return (DataLoader(train,shuffle=True,**kwargs),DataLoader(val,shuffle=False,**kwargs))


def train_run(args):
    seed_all(args.seed)
    df,info=load_manifest(args.manifest)
    classes=info['idx_to_class']; n_classes=len(classes)
    run=Path(args.out);run.mkdir(parents=True,exist_ok=True)
    params=vars(args).copy();params['classes']=classes
    (run/'config.json').write_text(json.dumps(params,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('DEVICE:',device,'MODEL:',args.model,'ACTIONS:',len(classes))
    model,norm=make_model(args.model,n_classes,pretrained=not args.no_pretrained,
                          freeze_backbone=args.freeze_backbone)
    model=model.to(device)
    train_dl,val_dl=loaders(df,args.model,args.frames,args.size,args.batch,args.workers,
                            args.aug,args.train_views,args.eval_views)
    train_df=df[df.split=='train']
    loss_fn=criterion_for(args.loss,n_classes,train_df.class_id.tolist(),device,
                          smoothing=args.smoothing,gamma=args.gamma)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=-1.0;no_improve=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]; t0=time.perf_counter()
        for x,y,_ in tqdm(train_dl,desc=f'Epoch {epoch}/{args.epochs}',leave=False):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=device.type=='cuda',dtype=torch.float16):
                logits=model(x)
                loss=loss_fn(logits,y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters,max_norm=2.0)
            scaler.step(optimizer);scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val,idx,gold,pred,probs=evaluate(model,val_dl,device,n_classes)
        train_loss=float(np.mean(losses)) if losses else float('nan')
        history.append({'epoch':epoch,'train_loss':train_loss,**val,
                        'epoch_sec':round(time.perf_counter()-t0,1)})
        print(f'Epoch {epoch:02d}: loss={train_loss:.3f}, val F1={val["macro_f1"]:.3f}, '
              f'val ACC={val["accuracy"]:.3f}, {history[-1]["epoch_sec"]}s',flush=True)
        score=val['macro_f1']
        if score>best+1e-5:
            best=score;no_improve=0
            torch.save({'state_dict':model.state_dict(),'model':args.model,'classes':classes,
                        'frames':args.frames,'size':args.size,'normalization':norm,
                        'freeze_backbone':args.freeze_backbone,'epoch':epoch,
                        'seed':args.seed,'val_metrics':val},run/'best.pt')
            save_eval(run,val,idx,gold,pred,probs,df[df.split=='val'],classes,'val')
        else:
            no_improve+=1
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        if no_improve>=args.patience:
            print('Early stopping');break
    print('BEST VAL macro-F1:',best,'best checkpoint:',run/'best.pt')


def test_run(args):
    seed_all(args.seed)
    df,info=load_manifest(args.manifest)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    classes=cp['classes']
    if classes!=info['idx_to_class']:
        raise ValueError('Checkpoint class ordering differs from manifest!')
    model,_=make_model(cp['model'],len(classes),pretrained=False,
                       freeze_backbone=cp.get('freeze_backbone'))
    model.load_state_dict(cp['state_dict'],strict=True)
    model=model.to(device)
    norm=cp['normalization']
    ds=RoboflowClips(df,frames=cp['frames'],size=cp['size'],split='test',
                     views=args.eval_views,normalization=norm)
    dl=DataLoader(ds,batch_size=args.batch,num_workers=args.workers,pin_memory=device.type=='cuda')
    metrics,idx,gold,pred,probs=evaluate(model,dl,device,len(classes))
    save_eval(args.out,metrics,idx,gold,pred,probs,df[df.split=='test'],classes,'test')
    print('TEST (only after selecting model on VAL):',json.dumps(metrics,indent=2))


def parser():
    p=argparse.ArgumentParser(description='Video COCO action classification benchmarks')
    sp=p.add_subparsers(dest='cmd',required=True)
    pr=sp.add_parser('prepare')
    pr.add_argument('--data',required=True,help='Directory with mp4+Video COCO JSON in root or train/valid/test')
    pr.add_argument('--manifest',default='runs/data',help='Output manifest dir')
    pr.add_argument('--seed',type=int,default=42)
    pr.add_argument('--test-count',type=int,default=4)
    pr.add_argument('--val-fraction',type=float,default=.2)
    pr.add_argument('--resplit-by-hash',action='store_true',
                    help='Ignore Roboflow folders and rebuild leakage-safe train/val/test by SHA-256; use when identical videos span source splits')
    pr.add_argument('--test-videos',nargs='*',default=None,help='Optionally force 4 video file names to test split')
    tr=sp.add_parser('train')
    tr.add_argument('--manifest',default='runs/data')
    tr.add_argument('--model',choices=['mobilenet_gru','resnet18_gru','r3d18','mc3_18','r2plus1d18'],default='mobilenet_gru')
    tr.add_argument('--out',default='runs/mobilenet_ce')
    tr.add_argument('--loss',choices=['ce','weighted_ce','smooth_ce','focal'],default='ce')
    tr.add_argument('--aug',choices=['none','light','strong'],default='light')
    tr.add_argument('--epochs',type=int,default=20)
    tr.add_argument('--patience',type=int,default=5)
    tr.add_argument('--lr',type=float,default=3e-4)
    tr.add_argument('--weight-decay',type=float,default=1e-4)
    tr.add_argument('--batch',type=int,default=4)
    tr.add_argument('--frames',type=int,default=16)
    tr.add_argument('--size',type=int,default=112)
    tr.add_argument('--train-views',type=int,default=2)
    tr.add_argument('--eval-views',type=int,default=3)
    tr.add_argument('--freeze-backbone',action=argparse.BooleanOptionalAction,default=None)
    tr.add_argument('--no-pretrained',action='store_true')
    tr.add_argument('--smoothing',type=float,default=.1)
    tr.add_argument('--gamma',type=float,default=2.)
    tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--workers',type=int,default=2)
    tr.add_argument('--cpu',action='store_true')
    te=sp.add_parser('test')
    te.add_argument('--manifest',default='runs/data')
    te.add_argument('--checkpoint',required=True)
    te.add_argument('--out',default='runs/final_test')
    te.add_argument('--eval-views',type=int,default=3)
    te.add_argument('--batch',type=int,default=4)
    te.add_argument('--workers',type=int,default=2)
    te.add_argument('--seed',type=int,default=42)
    te.add_argument('--cpu',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if args.cmd=='prepare':
        names=[v.strip() for arg in (args.test_videos or []) for v in arg.split(',') if v.strip()]
        summary=prepare(args.data,args.manifest,args.seed,args.test_count,args.val_fraction,names or None,args.resplit_by_hash)
        print(json.dumps(summary,indent=2,ensure_ascii=False))
        if summary['whole_video_segment_fraction']>.5:
            print('WARNING: >50% of action segments cover nearly the whole video. '
                  'This is video-level supervision; NOT adequate for precise action timecodes.')
    elif args.cmd=='train':train_run(args)
    elif args.cmd=='test':test_run(args)


if __name__=='__main__':main()
