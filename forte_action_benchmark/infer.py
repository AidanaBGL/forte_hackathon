"""Sliding-window exploratory action timeline. Not legally definitive video timestamp localization."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from data import read_video_clip,IMAGENET_MEAN,IMAGENET_STD,KINETICS_MEAN,KINETICS_STD
from models import make_model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--video',required=True)
    ap.add_argument('--window-sec',type=float,default=4.0)
    ap.add_argument('--stride-sec',type=float,default=2.0)
    ap.add_argument('--threshold',type=float,default=.5)
    ap.add_argument('--out',default='runs/inference.json')
    ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--allow-weak-intervals',action='store_true',
                    help='Explicitly acknowledge full-video labels cannot teach precise localization')
    ap.add_argument('--manifest',default='runs/data')
    args=ap.parse_args()
    manifest_file=Path(args.manifest)/'dataset_info.json'
    if manifest_file.is_file():
        meta=json.loads(manifest_file.read_text())
        weak=meta.get('whole_video_segment_fraction',0)
        if weak>.5 and not args.allow_weak_intervals:
            raise SystemExit(f'{weak:.0%} labels cover whole videos. Sliding windows may produce fake timecodes. '
                             'Relabel actual action start/end or use --allow-weak-intervals for exploratory results only.')
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model,_=make_model(cp['model'],len(cp['classes']),pretrained=False,
                       freeze_backbone=cp.get('freeze_backbone'))
    model.load_state_dict(cp['state_dict']);model.to(device).eval()
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():raise FileNotFoundError(args.video)
    fps=cap.get(cv2.CAP_PROP_FPS);frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration=frames/max(fps,1);cap.release()
    if duration<=0:raise ValueError('Empty video')
    if args.window_sec<=0 or args.stride_sec<=0:raise ValueError('window/stride must >0')
    windows=[];pos=0.
    while pos<duration:
        end=min(pos+args.window_sec,duration)
        if end-pos<.25:break
        windows.append((pos,end))
        if end==duration:break
        pos+=args.stride_sec
    mean,std=(IMAGENET_MEAN,IMAGENET_STD) if cp['normalization']=='imagenet' else (KINETICS_MEAN,KINETICS_STD)
    results=[]
    with torch.inference_mode():
        for start,end in windows:
            rgb=read_video_clip(args.video,start,end,cp['frames'],cp['size'])
            x=(rgb.astype(np.float32)/255-mean[None,None,None,:])/std[None,None,None,:]
            tensor=torch.from_numpy(x.transpose(3,0,1,2).copy())[None].to(device)
            with torch.autocast('cuda',enabled=device.type=='cuda',dtype=torch.float16):
                probs=model(tensor).float().softmax(dim=1)[0].cpu().numpy()
            top=int(probs.argmax())
            results.append({'start_sec':round(start,2),'end_sec':round(end,2),
                            'action':cp['classes'][top],'confidence':round(float(probs[top]),4),
                            'uncertain':bool(probs[top]<args.threshold)})
    output={'video':str(Path(args.video).resolve()),'duration_sec':duration,
            'method':'sliding_window_action_classifier',
            'warning':'Candidate time windows only. Full-video action annotations do not support precise timestamps.',
            'windows':results}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',out)
    for r in results:print(f'{r["start_sec"]:5.1f}-{r["end_sec"]:5.1f}s {r["action"]:<26} {r["confidence"]:.2f}')

if __name__=='__main__':main()
