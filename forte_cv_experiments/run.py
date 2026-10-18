"""Run ForteBank baselines. Commands: prepare, extract, train, train-actions, predict-actions."""
import argparse
import sys
from pathlib import Path
from src.dataset import prepare,load_manifest,TEST_IDS
from src.features import VisionExtractor,save_features
from src.experiments import run_experiment
from src.actions import train_action_classifier,predict_actions


def main():
    p=argparse.ArgumentParser(description='ForteBank local vision experiments')
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare',help='unpack archive, read XLSX and split 4 holdout videos')
    a.add_argument('--zip',default='videos.zip')
    a.add_argument('--labels',default='open_labels.xlsx')
    a.add_argument('--out',default='data')
    a.add_argument('--test-ids',nargs=4,default=TEST_IDS)

    a=sub.add_parser('extract',help='compute cached visual window features')
    a.add_argument('--manifest',default='data/manifest.csv')
    a.add_argument('--features-dir',required=True)
    a.add_argument('--encoder',choices=['none','resnet18','r3d18'],default='none')
    a.add_argument('--detector',choices=['none','coco','world','custom'],default='none')
    a.add_argument('--custom-weights',default='')
    a.add_argument('--hands',choices=['none','mediapipe'],default='none')
    a.add_argument('--hand-model',default='')
    a.add_argument('--device',default='auto')
    a.add_argument('--roi',default='',help='normalized x1,y1,x2,y2; e.g. 0.15,0.12,0.9,0.98')
    a.add_argument('--max-side',type=int,default=640,help='Max decoded frame side; 960 helps small objects')
    a.add_argument('--window-sec',type=float,default=4.0)
    a.add_argument('--stride-sec',type=float,default=2.0)
    a.add_argument('--frames-per-window',type=int,default=8)
    a.add_argument('--limit',type=int,default=0,help='only extract N videos (for smoke test)')
    a.add_argument('--max-windows',type=int,default=0,help='only N windows per video (for smoke test)')
    a.add_argument('--overwrite',action='store_true')

    a=sub.add_parser('train',help='train video- and point-supervised anomaly baselines; evaluate heldout 4')
    a.add_argument('--manifest',default='data/manifest.csv')
    a.add_argument('--features-dir',required=True)
    a.add_argument('--out',required=True)
    a.add_argument('--method',choices=['video','temporal'],default='temporal')
    a.add_argument('--folds',type=int,default=4)

    a=sub.add_parser('train-actions',help='requires manual action_segments.csv')
    a.add_argument('--manifest',default='data/manifest.csv')
    a.add_argument('--features-dir',required=True)
    a.add_argument('--segments',default='configs/action_segments.csv')
    a.add_argument('--out',default='runs/action_model.joblib')

    a=sub.add_parser('predict-actions',help='predict action timelines for heldout videos')
    a.add_argument('--manifest',default='data/manifest.csv')
    a.add_argument('--features-dir',required=True)
    a.add_argument('--model',default='runs/action_model.joblib')
    a.add_argument('--out',default='runs/action_predictions')
    a.add_argument('--split',choices=['train','test','all'],default='test')

    args=p.parse_args()
    if args.cmd=='prepare':
        prepare(args.zip,args.labels,args.out,args.test_ids)
    elif args.cmd=='extract':
        if args.frames_per_window<4:raise ValueError('Minimum four frames per window')
        ex=VisionExtractor(encoder=args.encoder,detector=args.detector,hands=args.hands,
                           hand_model=args.hand_model,device=args.device,roi=args.roi,
                           custom_weights=args.custom_weights,max_side=args.max_side)
        try:
            save_features(load_manifest(args.manifest),args.features_dir,ex,
                     window_sec=args.window_sec,stride_sec=args.stride_sec,
                     frames_per_window=args.frames_per_window,
                     limit=args.limit,max_windows=args.max_windows,overwrite=args.overwrite)
        finally:ex.close()
    elif args.cmd=='train':
        run_experiment(args.manifest,args.features_dir,args.out,method=args.method,folds=args.folds)
    elif args.cmd=='train-actions':
        train_action_classifier(args.manifest,args.features_dir,args.segments,args.out)
    elif args.cmd=='predict-actions':
        predict_actions(args.manifest,args.features_dir,args.model,args.out,args.split)

if __name__=='__main__':main()
